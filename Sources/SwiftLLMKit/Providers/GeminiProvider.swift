import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "Gemini")

/// LLM provider for Google's Gemini API (POST /v1beta/models/{model}:generateContent).
///
/// Key differences from OpenAI-compatible APIs:
/// - Messages use `contents` with `parts` arrays instead of `messages` with `content` strings
/// - The assistant role is called `model`
/// - System instructions are a separate top-level field
/// - Tool definitions use `functionDeclarations` instead of OpenAI's `functions` wrapper
/// - Tool calls are `functionCall` parts; results are `functionResponse` parts
/// - Auth via `x-goog-api-key` header so the key never appears in the URL
///   (request logs capture URLs but not headers, so this keeps keys out of logs)
struct GeminiProvider: LLMProvider {
    private let configuration: ModelConfiguration
    private let provider: ModelProvider
    private let readAPIKey: @Sendable () -> String
    private let verboseLogging: Bool
    private let session: URLSession

    public init(
        configuration: ModelConfiguration,
        provider: ModelProvider,
        readAPIKey: @Sendable @escaping () -> String,
        verboseLogging: Bool = false,
        session: URLSession = llmURLSession
    ) {
        self.configuration = configuration
        self.provider = provider
        self.readAPIKey = readAPIKey
        self.verboseLogging = verboseLogging
        self.session = session
    }

    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        maxOutputTokensOverride: Int? = nil
    ) async throws -> LLMResponse {
        // Build URL: {endpoint}/models/{model}:generateContent
        let base = provider.endpoint.path.hasSuffix("/")
            ? provider.endpoint
            : provider.endpoint.appendingPathComponent("")
        let url = base
            .appendingPathComponent("models/\(configuration.model):generateContent")

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let apiKey = readAPIKey()
        if !apiKey.isEmpty {
            request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")
        }

        let body = try buildRequestBody(messages: messages, tools: tools, maxOutputTokensOverride: maxOutputTokensOverride)
        // .sortedKeys keeps wire bytes stable across requests for any
        // downstream prefix-caching the provider applies.
        let requestData = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        request.httpBody = requestData

        logger.debug("Request: POST \(url.absoluteString, privacy: .public) model=\(configuration.model, privacy: .public)")
        if verboseLogging {
            LLMRequestLogger.logRequest(label: "Gemini", url: url, model: configuration.model, body: body, rawData: requestData)
        }

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw LLMProviderError.invalidResponse
        }

        if verboseLogging {
            LLMRequestLogger.logResponse(label: "Gemini", statusCode: httpResponse.statusCode, data: data)
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let responseBody = String(data: data, encoding: .utf8) ?? "unknown"
            logger.error("HTTP \(httpResponse.statusCode, privacy: .public) from \(url.absoluteString, privacy: .public) body=\(responseBody, privacy: .public)")
            throw LLMProviderError.httpError(statusCode: httpResponse.statusCode, body: responseBody, url: url)
        }

        return try parseResponse(data: data)
    }

    // MARK: - Request building

    func buildRequestBody(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        maxOutputTokensOverride: Int? = nil
    ) throws -> [String: Any] {
        var systemParts: [String] = []
        var conversationMessages: [LLMMessage] = []

        for message in messages {
            // .developer is treated as system for Gemini (no native support).
            // Both collapse into the top-level systemInstruction field.
            if (message.role == .system || message.role == .developer),
               let text = message.content.textValue {
                systemParts.append(text)
            } else {
                conversationMessages.append(message)
            }
        }

        // Gemini's `functionResponse.name` field must match the original
        // `functionCall.name` — Gemini uses name (NOT ID — IDs aren't part of
        // its wire schema at all) for parallel-call pairing. Walk the
        // conversation once to build [toolCallID → functionName] so the
        // .toolResult encoder below can look up the right name.
        // Serial single-call conversations also work without this (Gemini
        // falls back to positional pairing), but parallel calls fail without
        // correct names.
        let toolNameByCallID = Self.buildToolNameLookup(conversationMessages)

        // Gemini requires strict user/model alternation. Merge consecutive same-role
        // messages (e.g. a tool result followed by a user message) into one content entry.
        let rawContents = try conversationMessages.map {
            try encodeContent($0, toolNameByCallID: toolNameByCallID)
        }
        let mergedContents = Self.mergeConsecutiveSameRole(rawContents)

        let effectiveMaxTokens = maxOutputTokensOverride ?? configuration.maxTokens
        var generationConfig: [String: Any] = ["maxOutputTokens": effectiveMaxTokens]
        if let temperature = configuration.temperature {
            generationConfig["temperature"] = temperature
        }
        var body: [String: Any] = [
            "contents": mergedContents,
            "generationConfig": generationConfig
        ]

        if !systemParts.isEmpty {
            body["systemInstruction"] = [
                "parts": [["text": systemParts.joined(separator: "\n\n")]]
            ] as [String: Any]
        }

        if !tools.isEmpty {
            body["tools"] = [
                [
                    "functionDeclarations": tools.map { tool in
                        [
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters.mapValues(\.rawValue)
                        ] as [String: Any]
                    }
                ] as [String: Any]
            ]
            body["toolConfig"] = [
                "functionCallingConfig": [
                    "mode": "AUTO"
                ]
            ] as [String: Any]
        }

        // Apply caller-provided overrides last so they win over any defaults
        // this method set. Deep-merges dict-valued keys — e.g. setting
        // generationConfig.thinkingConfig won't wipe our temperature /
        // maxOutputTokens defaults under that same key.
        if let overrides = configuration.extraJSONOverrides {
            mergeJSONOverrides(&body, with: overrides)
        }

        return body
    }

    /// Builds `[toolCallID → functionName]` from prior assistant turns in the
    /// conversation. Walks `.toolCalls` and `.mixed` content; ignores others.
    /// Used by `encodeContent` to populate `functionResponse.name` correctly
    /// (Gemini matches functionResponse to functionCall by name, not ID).
    static func buildToolNameLookup(_ messages: [LLMMessage]) -> [String: String] {
        var map: [String: String] = [:]
        for message in messages {
            switch message.content {
            case .toolCalls(let calls):
                for call in calls { map[call.id] = call.name }
            case .mixed(_, let calls):
                for call in calls { map[call.id] = call.name }
            case .text, .toolResult:
                break
            }
        }
        return map
    }

    /// Encodes one LLMMessage as a Gemini `contents` entry.
    ///
    /// For assistant turns with thoughtSignatures carried in
    /// `message.continuation`, re-attaches each signature to its
    /// corresponding part by index. Critical for Gemini 2.5's thinking
    /// continuity: without the prior turn's thoughtSignature, multi-turn
    /// tool-use conversations silently drop results.
    private func encodeContent(
        _ message: LLMMessage,
        toolNameByCallID: [String: String]
    ) throws -> [String: Any] {
        let role = geminiRole(for: message)

        // Gemini thoughtSignatures from a prior response, keyed by part index.
        // Only meaningful for assistant turns (Gemini calls them "model").
        let signatures: [String: String] = (message.role == .assistant)
            ? (message.continuation?.geminiThoughtSignatures ?? [:])
            : [:]

        func attachSignature(_ part: [String: Any], at index: Int) -> [String: Any] {
            guard let sig = signatures[String(index)] else { return part }
            var withSig = part
            withSig["thoughtSignature"] = sig
            return withSig
        }

        switch message.content {
        case .text(let text):
            var parts: [[String: Any]] = []
            if let images = message.images, !images.isEmpty {
                for image in images {
                    parts.append([
                        "inlineData": [
                            "mimeType": image.mimeType,
                            "data": image.data.base64EncodedString()
                        ]
                    ])
                }
            }
            parts.append(["text": text])
            // Attach signatures by emitted-part index (matches the source-side
            // parsing which also keyed by part index).
            parts = parts.enumerated().map { attachSignature($1, at: $0) }
            return ["role": role, "parts": parts]

        case .toolCalls(let calls):
            var parts: [[String: Any]] = try calls.map { call in
                [
                    "functionCall": [
                        "name": call.name,
                        "args": try parseJSONObject(call.arguments)
                    ] as [String: Any]
                ]
            }
            parts = parts.enumerated().map { attachSignature($1, at: $0) }
            return ["role": "model", "parts": parts]

        case .mixed(let text, let calls):
            var parts: [[String: Any]] = [["text": text]]
            for call in calls {
                parts.append([
                    "functionCall": [
                        "name": call.name,
                        "args": try parseJSONObject(call.arguments)
                    ] as [String: Any]
                ])
            }
            parts = parts.enumerated().map { attachSignature($1, at: $0) }
            return ["role": "model", "parts": parts]

        case .toolResult(let toolCallID, let content):
            // Gemini uses `functionResponse.name` (NOT toolCallID — IDs aren't
            // on the wire) to pair with the matching `functionCall`. Look up
            // the actual function name from prior assistant turns; on a miss
            // (orphan tool result), fall back to the toolCallID and warn.
            let functionName: String
            if let resolved = toolNameByCallID[toolCallID] {
                functionName = resolved
            } else {
                logger.warning("Tool result for ID \(toolCallID, privacy: .public) has no matching prior tool call; using ID as function name (Gemini may reject for parallel-call conversations)")
                functionName = toolCallID
            }
            return [
                "role": "user",
                "parts": [
                    [
                        "functionResponse": [
                            "name": functionName,
                            "response": [
                                "content": content
                            ]
                        ] as [String: Any]
                    ]
                ]
            ]
        }
    }

    private func geminiRole(for message: LLMMessage) -> String {
        switch message.role {
        case .user, .tool: return "user"
        case .assistant: return "model"
        // System AND developer messages should both have been extracted into
        // `systemInstruction` before reaching here. Defensive fallback to "user".
        case .system, .developer: return "user"
        }
    }

    // MARK: - Response parsing

    func parseResponse(data: Data) throws -> LLMResponse {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            let preview = String(data: data.prefix(500), encoding: .utf8) ?? "(non-utf8, \(data.count) bytes)"
            logger.error("Response is not a JSON object: \(preview, privacy: .public)")
            throw LLMProviderError.malformedResponse(detail: "not a JSON object: \(preview)")
        }
        guard let candidates = json["candidates"] as? [[String: Any]],
              let candidate = candidates.first
        else {
            let keys = json.keys.sorted().joined(separator: ", ")
            let preview = String(data: data.prefix(500), encoding: .utf8) ?? "(\(data.count) bytes)"
            logger.error("Missing candidates in response. Keys: \(keys, privacy: .public) Body: \(preview, privacy: .public)")
            throw LLMProviderError.malformedResponse(detail: "missing candidates, keys: [\(keys)], body: \(preview)")
        }

        // Gemini returns error finish reasons (e.g. MALFORMED_FUNCTION_CALL) when it fails to
        // produce a valid tool call. Surface these as actionable text so the agent can retry.
        let finishReason = candidate["finishReason"] as? String
        if let finishReason, finishReason != "STOP" && finishReason != "MAX_TOKENS" {
            let finishMessage = candidate["finishMessage"] as? String ?? finishReason
            logger.warning("Gemini finished with \(finishReason, privacy: .public): \(finishMessage, privacy: .public)")
            if candidate["content"] == nil {
                return LLMResponse(
                    text: "[Gemini error: \(finishReason)] \(finishMessage)",
                    toolCalls: []
                )
            }
        }

        // Gemini may return content with no parts (empty response, zero output tokens).
        // Treat as empty text response rather than a parse error.
        guard let content = candidate["content"] as? [String: Any] else {
            let preview = String(data: data.prefix(500), encoding: .utf8) ?? "(\(data.count) bytes)"
            logger.error("Missing content in candidate. finishReason=\(finishReason ?? "nil", privacy: .public) Body: \(preview, privacy: .public)")
            throw LLMProviderError.malformedResponse(detail: "missing content, finishReason=\(finishReason ?? "nil"), body: \(preview)")
        }
        let parts = content["parts"] as? [[String: Any]] ?? []

        var text: String?
        var toolCalls: [LLMToolCall] = []
        // Capture thoughtSignature per original part index. Gemini 2.5 returns
        // these on EVERY response with thinking enabled (default on Pro). They
        // must be echoed back on subsequent turns or the model loses thinking-
        // continuity validation and silently drops tool results. Re-attached
        // to the corresponding outgoing parts in encodeContent.
        var thoughtSignatures: [String: String] = [:]

        for (partIndex, part) in parts.enumerated() {
            if let sig = part["thoughtSignature"] as? String, !sig.isEmpty {
                thoughtSignatures[String(partIndex)] = sig
            }
            if let textContent = part["text"] as? String {
                if let existing = text {
                    text = existing + textContent
                } else {
                    text = textContent
                }
            } else if let functionCall = part["functionCall"] as? [String: Any],
                      let name = functionCall["name"] as? String {
                let args = functionCall["args"]
                let argString: String
                if let argsDict = args {
                    let argsData = try JSONSerialization.data(withJSONObject: argsDict)
                    argString = String(data: argsData, encoding: .utf8) ?? "{}"
                } else {
                    argString = "{}"
                }
                // Gemini has no real tool call IDs. Use UUID for internal routing;
                // functionResponse matching uses the `name` field, not the `id`.
                toolCalls.append(LLMToolCall(id: UUID().uuidString, name: name, arguments: argString))
            }
        }

        // Parse token usage from usageMetadata.
        var tokenUsage: TokenUsage?
        if let usageMeta = json["usageMetadata"] as? [String: Any] {
            let input = usageMeta["promptTokenCount"] as? Int ?? 0
            let output = usageMeta["candidatesTokenCount"] as? Int ?? 0
            let cacheRead = usageMeta["cachedContentTokenCount"] as? Int ?? 0
            let reasoning = usageMeta["thoughtsTokenCount"] as? Int ?? 0
            let rawUsage = TokenUsage.serializeRawUsage(usageMeta)
            tokenUsage = TokenUsage(
                inputTokens: input,
                outputTokens: output,
                reasoningTokens: reasoning,
                cacheReadTokens: cacheRead,
                rawUsage: rawUsage
            )
            if cacheRead > 0 {
                logger.info("Cache: read=\(cacheRead) uncached=\(input - cacheRead)")
            }
            if reasoning > 0 {
                logger.info("Reasoning tokens: \(reasoning)")
            }
        }

        let continuation: ProviderContinuation? = thoughtSignatures.isEmpty
            ? nil
            : ProviderContinuation(geminiThoughtSignatures: thoughtSignatures)

        return LLMResponse(
            text: text?.isEmpty == true ? nil : text,
            toolCalls: toolCalls,
            usage: tokenUsage,
            continuation: continuation
        )
    }

    // MARK: - Message merging

    /// Merges consecutive contents entries that share the same role into a single entry
    /// by concatenating their `parts` arrays. Required because Gemini enforces strict
    /// user/model alternation (e.g. a functionResponse + user text must be one content block).
    private static func mergeConsecutiveSameRole(_ contents: [[String: Any]]) -> [[String: Any]] {
        guard !contents.isEmpty else { return contents }
        var result: [[String: Any]] = []
        for content in contents {
            guard let role = content["role"] as? String else {
                result.append(content)
                continue
            }
            if let lastRole = result.last?["role"] as? String, lastRole == role,
               let existingParts = result[result.count - 1]["parts"] as? [[String: Any]],
               let newParts = content["parts"] as? [[String: Any]] {
                result[result.count - 1]["parts"] = existingParts + newParts
            } else {
                result.append(content)
            }
        }
        return result
    }

    // MARK: - Helpers

    private func parseJSONObject(_ jsonString: String) throws -> Any {
        guard let data = jsonString.data(using: .utf8) else {
            throw LLMProviderError.malformedResponse(detail: "tool arguments not valid UTF-8: \(jsonString.prefix(200))")
        }
        do {
            return try JSONSerialization.jsonObject(with: data)
        } catch {
            throw LLMProviderError.malformedResponse(detail: "tool arguments JSON parse failed: \(error.localizedDescription), input: \(jsonString.prefix(200))")
        }
    }
}
