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
    /// The model's capabilities, for gating knobs whose wrong emission is an HTTP 400.
    private let modelCapabilities: ModelCapabilities
    /// The model's measured thinking-budget ceiling, so a request is clamped to something the
    /// endpoint accepts rather than sent raw.
    private let measuredMaxThinkingBudget: Int?
    private let provider: ModelProvider
    private let readAPIKey: @Sendable () -> String
    private let verboseLogging: Bool
    private let session: URLSession

    public init(
        configuration: ModelConfiguration,
        provider: ModelProvider,
        readAPIKey: @Sendable @escaping () -> String,
        verboseLogging: Bool = false,
        modelCapabilities: ModelCapabilities = ModelCapabilities(),
        measuredMaxThinkingBudget: Int? = nil,
        session: URLSession = llmURLSession
    ) {
        self.configuration = configuration
        self.modelCapabilities = modelCapabilities
        self.measuredMaxThinkingBudget = measuredMaxThinkingBudget
        self.provider = provider
        self.readAPIKey = readAPIKey
        self.verboseLogging = verboseLogging
        self.session = session
    }

    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides = LLMCallOverrides()
    ) async throws -> LLMResponse {
        // Gemini doesn't have an effort enum — depth is controlled via
        // `thinkingConfig.thinkingBudget` (token count), emitted below.
        // `overrides.thinkingEffort` is silently ignored. Consumers routing
        // through Gemini members of a hydra: set thinkingBudget on the
        // underlying LLM config instead of relying on per-call effort.
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

        let body = try buildRequestBody(messages: messages, tools: tools, overrides: overrides)
        // .sortedKeys keeps wire bytes stable across requests for any
        // downstream prefix-caching the provider applies.
        let requestData = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        request.httpBody = requestData

        logger.debug("Request: POST \(url.absoluteString, privacy: .public) model=\(configuration.model, privacy: .public)")
        let requestLog: LLMRequestLogger.RequestLogToken? = verboseLogging
            ? LLMRequestLogger.logRequest(label: "Gemini", url: url, model: configuration.model, body: body, rawData: requestData)
            : nil

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw LLMProviderError.invalidResponse
        }

        if verboseLogging {
            LLMRequestLogger.logResponse(label: "Gemini", statusCode: httpResponse.statusCode, data: data, for: requestLog)
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let responseBody = String(data: data, encoding: .utf8) ?? "unknown"
            logger.error("HTTP \(httpResponse.statusCode, privacy: .public) from \(url.absoluteString, privacy: .public) body=\(responseBody, privacy: .public)")
            throw LLMProviderError.httpError(statusCode: httpResponse.statusCode, body: responseBody, url: url, retryAfter: LLMProviderError.parseRetryAfter(httpResponse.value(forHTTPHeaderField: "Retry-After")))
        }

        return try parseResponse(data: data)
    }

    // MARK: - Request building

    /// Convenience overload for tests that only care about a couple of knobs.
    func buildRequestBody(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice? = nil
    ) throws -> [String: Any] {
        try buildRequestBody(
            messages: messages,
            tools: tools,
            overrides: LLMCallOverrides(toolChoice: toolChoice)
        )
    }

    func buildRequestBody(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides
    ) throws -> [String: Any] {
        let toolChoice = overrides.toolChoice
        let maxOutputTokensOverride = overrides.maxOutputTokens
        let temperatureOverride = overrides.temperature
        let topPOverride = overrides.topP
        let stopSequencesOverride = overrides.stopSequences
        let frequencyPenaltyOverride = overrides.frequencyPenalty
        let presencePenaltyOverride = overrides.presencePenalty
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
        if let temperature = temperatureOverride ?? configuration.temperature {
            generationConfig["temperature"] = temperature
        }
        // Gemini's reasoning depth control, which this file has documented since it was written
        // and never actually sent — a configured `thinkingBudget` was silently dropped. The
        // per-call override wins, and an explicit `reasoningEnabled: false` sends 0, which is
        // Gemini's documented way to turn thinking off.
        let geminiBudget: Int? = {
            switch overrides.reasoningEnabled {
            case false:
                return 0                                    // Gemini's documented "off"
            case true:
                // An explicit ON must beat a configured zero, which would otherwise disable it.
                let requested = overrides.thinkingBudgetTokens ?? configuration.thinkingBudget ?? 0
                return requested > 0 ? requested : ThinkingBudget.minimumTokens
            case nil:
                return overrides.thinkingBudgetTokens ?? configuration.thinkingBudget
            }
        }()
        // Fails CLOSED, unlike Alibaba's `enable_thinking` fallback. That one preserves behaviour
        // this library already had; this field was NEVER emitted, so requiring a known-true costs
        // nothing and avoids sending `thinkingConfig` to the non-thinking Gemini models that
        // reject it — which would break requests that work today.
        if let budget = geminiBudget, budget >= 0,
           modelCapabilities.state(of: .thinkingSupportsTokenBudget) == true {
            // Clamp to the measured ceiling like the other providers; zero passes through, since
            // it is the off switch rather than a budget.
            let sent = budget == 0 ? 0
                : ThinkingBudget.effective(budget, measuredMaximum: measuredMaxThinkingBudget)
            if let sent {
                generationConfig["thinkingConfig"] = ["thinkingBudget": sent] as [String: Any]
            }
        }
        if let topP = topPOverride {
            generationConfig["topP"] = topP
        }
        if let stops = stopSequencesOverride, !stops.isEmpty {
            generationConfig["stopSequences"] = stops
        }
        if let fp = frequencyPenaltyOverride {
            generationConfig["frequencyPenalty"] = fp
        }
        if let pp = presencePenaltyOverride {
            generationConfig["presencePenalty"] = pp
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
            // toolConfig.functionCallingConfig — Gemini's tool-choice analog.
            // Default `AUTO` when caller didn't specify; pinned to `ANY`/`NONE`
            // or `ANY + allowedFunctionNames=[name]` for .specific.
            body["toolConfig"] = [
                "functionCallingConfig": Self.encodeGeminiToolChoice(toolChoice ?? .auto)
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

        // For assistant turns with a saved Gemini parts payload, emit those
        // parts VERBATIM — they preserve every text / functionCall / sig from
        // the original response, in original order. This bypasses the 0.0.24
        // shape-fragility (factory collapsing a multi-part response into one
        // .text/.toolCalls/.mixed content, which misaligned position-keyed
        // signatures). Faithful replay is the goal.
        if message.role == .assistant,
           let parts = message.continuation?.geminiResponseParts {
            // Signing the first functionCall even here protects rotation from
            // an older Gemini that omitted signatures (2.5 treats them as
            // optional) into a strict Gemini 3 model. Real captured
            // signatures are never overwritten.
            return [
                "role": "model",
                "parts": ensuringFirstFunctionCallSigned(parts.map(encodeSavedPart))
            ]
        }

        // Legacy 0.0.24/0.0.25 path: if a saved conversation has the older
        // position-keyed `geminiThoughtSignatures` (no parts), fall back to
        // content-rendering with position-keyed sig attachment. Shape-fragile
        // but the best we can do for legacy data.
        let legacySignatures: [String: String] = legacyGeminiSignatures(message: message)

        func attachLegacySig(_ part: [String: Any], at index: Int) -> [String: Any] {
            guard let sig = legacySignatures[String(index)] else { return part }
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
            if let documents = message.documents, !documents.isEmpty {
                for document in documents {
                    parts.append([
                        "inlineData": [
                            "mimeType": document.mimeType,
                            "data": document.data.base64EncodedString()
                        ]
                    ])
                }
            }
            parts.append(["text": text])
            parts = parts.enumerated().map { attachLegacySig($1, at: $0) }
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
            parts = parts.enumerated().map { attachLegacySig($1, at: $0) }
            return ["role": "model", "parts": ensuringFirstFunctionCallSigned(parts)]

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
            parts = parts.enumerated().map { attachLegacySig($1, at: $0) }
            return ["role": "model", "parts": ensuringFirstFunctionCallSigned(parts)]

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

    /// Google's documented validation-bypass signature for `functionCall`
    /// parts that carry no real captured signature — cross-model history
    /// transfer, manually injected calls, or Gemini 2.5-era captures
    /// recorded before signatures were emitted. Gemini 3 rejects requests
    /// with 400 INVALID_ARGUMENT when the first `functionCall` part of an
    /// assistant step lacks a `thoughtSignature`; Gemini 2.5 treats the
    /// field as optional and ignores it.
    ///
    /// Docs:
    /// - Thought signatures: https://ai.google.dev/gemini-api/docs/thought-signatures
    /// - Validation bypass for injected / cross-model function calls
    ///   ("skip_thought_signature_validator", "context_engineering_is_the_way_to_go"):
    ///   https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures
    static let functionCallDummySignature = "skip_thought_signature_validator"

    /// Returns `parts` with the first `functionCall` part guaranteed to carry
    /// a `thoughtSignature`, attaching the documented validation-bypass dummy
    /// whenever that part is bare — even if other parts in the same step
    /// carry real captured signatures. Real signatures are never overwritten;
    /// turns without function calls are returned unchanged.
    ///
    /// Gemini's rule ("the first `functionCall` part in each step of the
    /// current turn must include its `thought_signature`") and the bypass
    /// values are documented at
    /// https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures
    private func ensuringFirstFunctionCallSigned(_ parts: [[String: Any]]) -> [[String: Any]] {
        guard let index = parts.firstIndex(where: { $0["functionCall"] != nil }) else {
            return parts
        }
        guard parts[index]["thoughtSignature"] == nil else { return parts }
        var signed = parts
        signed[index]["thoughtSignature"] = Self.functionCallDummySignature
        return signed
    }

    /// Serializes a saved `GeminiResponsePart` back to the wire shape. The
    /// args JSON is re-parsed into a Foundation object so the outgoing body
    /// can include it as a nested dict rather than an opaque string.
    private func encodeSavedPart(_ part: GeminiResponsePart) -> [String: Any] {
        var out: [String: Any] = [:]
        if let text = part.text {
            out["text"] = text
        }
        if let fc = part.functionCall {
            let args = (try? JSONSerialization.jsonObject(with: Data(fc.argsJSON.utf8))) ?? [String: Any]()
            out["functionCall"] = [
                "name": fc.name,
                "args": args
            ] as [String: Any]
        }
        if let sig = part.thoughtSignature, !sig.isEmpty {
            out["thoughtSignature"] = sig
        }
        if part.thought == true {
            out["thought"] = true
        }
        return out
    }

    /// Reads the legacy 0.0.24/0.0.25 `geminiThoughtSignatures` field. Isolated
    /// in a helper marked `@available(*, deprecated)` so callers don't surface
    /// the deprecation warning at every legacy fallback site.
    @available(*, deprecated)
    private func legacyGeminiSignatures(message: LLMMessage) -> [String: String] {
        guard message.role == .assistant else { return [:] }
        return message.continuation?.geminiThoughtSignatures ?? [:]
    }

    /// Builds a `ProviderContinuation` that populates BOTH the new
    /// `geminiResponseParts` (primary) AND the legacy `geminiThoughtSignatures`
    /// (for backward compat with code that reads the old field). Isolated in
    /// an `@available(*, deprecated)` helper so the legacy-field assignment
    /// doesn't pepper deprecation warnings across the parser.
    @available(*, deprecated)
    private func makeGeminiContinuation(
        parts: [GeminiResponsePart],
        legacySignatures: [String: String]
    ) -> ProviderContinuation {
        ProviderContinuation(
            geminiResponseParts: parts,
            geminiThoughtSignatures: legacySignatures
        )
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
        // Capture the FULL parts structure (text + functionCall + thoughtSignature
        // + thought flag) so replay can emit byte-faithful parts back to Gemini
        // without depending on per-part-index position keying — which 0.0.24's
        // shape-fragile design depended on. The 0.0.26 redesign stores parts
        // verbatim so `.assistant(from:)` factory shape-collapse (a multi-part
        // response folded into one `.text` / `.toolCalls` / `.mixed` content)
        // doesn't misalign signatures.
        var capturedParts: [GeminiResponsePart] = []

        for part in parts {
            let sig = part["thoughtSignature"] as? String
            let thought = part["thought"] as? Bool
            var partText: String?
            var partFunctionCall: GeminiFunctionCall?

            if let textContent = part["text"] as? String {
                partText = textContent
                if let existing = text {
                    text = existing + textContent
                } else {
                    text = textContent
                }
            }
            if let functionCall = part["functionCall"] as? [String: Any],
               let name = functionCall["name"] as? String {
                let argString: String
                if let args = functionCall["args"] {
                    let argsData = try JSONSerialization.data(withJSONObject: args)
                    argString = String(data: argsData, encoding: .utf8) ?? "{}"
                } else {
                    argString = "{}"
                }
                partFunctionCall = GeminiFunctionCall(name: name, argsJSON: argString)
                // Gemini has no real tool call IDs. Use UUID for internal routing;
                // functionResponse matching uses the `name` field, not the `id`.
                toolCalls.append(LLMToolCall(id: UUID().uuidString, name: name, arguments: argString))
            }

            capturedParts.append(GeminiResponsePart(
                text: partText,
                functionCall: partFunctionCall,
                thoughtSignature: (sig?.isEmpty ?? true) ? nil : sig,
                thought: thought
            ))
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

        // Only attach continuation when at least one part carried a signature —
        // otherwise we'd persist noise on every plain-text response. The
        // legacy 0.0.24/0.0.25 `geminiThoughtSignatures` dict is auto-derived
        // from the captured parts so existing downstream consumers that read
        // the old field still see signatures (deprecated but functional).
        let hasAnySignature = capturedParts.contains { ($0.thoughtSignature?.isEmpty == false) }
        let continuation: ProviderContinuation?
        if hasAnySignature {
            let legacySigs: [String: String] = Dictionary(
                uniqueKeysWithValues: capturedParts.enumerated().compactMap { idx, part in
                    guard let sig = part.thoughtSignature, !sig.isEmpty else { return nil }
                    return (String(idx), sig)
                }
            )
            continuation = makeGeminiContinuation(
                parts: capturedParts,
                legacySignatures: legacySigs
            )
        } else {
            continuation = nil
        }

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

    /// Translates the unified `LLMToolChoice` to Gemini's
    /// `toolConfig.functionCallingConfig` shape. `.specific(name)` becomes
    /// `mode: "ANY"` plus an `allowedFunctionNames` whitelist of one.
    private static func encodeGeminiToolChoice(_ choice: LLMToolChoice) -> [String: Any] {
        switch choice {
        case .auto:
            return ["mode": "AUTO"]
        case .required:
            return ["mode": "ANY"]
        case .textOnly:
            return ["mode": "NONE"]
        case .specific(let name):
            return ["mode": "ANY", "allowedFunctionNames": [name]]
        }
    }
}
