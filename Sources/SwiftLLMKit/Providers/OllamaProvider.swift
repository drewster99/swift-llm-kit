import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "Ollama")

/// LLM provider for Ollama's native API (POST /api/chat, GET /api/tags).
///
/// The `endpoint` in `ModelProvider` must be the base path that `/chat` is appended to,
/// e.g. `http://localhost:11434/api`. Do **not** include a trailing `/chat` in the endpoint itself.
///
/// Differs from the OpenAI-compatible provider in several ways:
/// - Tool arguments in responses are JSON objects, not strings
/// - Tool calls have no `id` — synthetic UUIDs are generated
/// - Tool results omit `tool_call_id`
/// - Images are passed as a base64 array rather than content parts
struct OllamaProvider: LLMProvider {
    private let configuration: ModelConfiguration
    private let provider: ModelProvider
    private let readAPIKey: @Sendable () -> String
    private let verboseLogging: Bool
    private let behaviorFlags: BehaviorFlags
    /// The model's capabilities, so `tool_choice` is gated here as it is on every other route.
    private let modelCapabilities: ModelCapabilities
    private let session: URLSession

    // MARK: - Static regex patterns (compiled once)

    // Patterns are string literals so NSRegularExpression init cannot fail; `try!` is safe.
    private static let xmlBlockRegex = try! NSRegularExpression(
        pattern: #"<function_calls>\s*(.*?)\s*</function_calls>"#,
        options: [.dotMatchesLineSeparators]
    )
    private static let invokeRegex = try! NSRegularExpression(
        pattern: #"<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>"#,
        options: [.dotMatchesLineSeparators]
    )
    private static let paramRegex = try! NSRegularExpression(
        pattern: #"<parameter\s+name="([^"]+)"(?:\s+[^>]*)?>([^<]*)</parameter>"#,
        options: [.dotMatchesLineSeparators]
    )
    private static let xmlStripRegex = try! NSRegularExpression(
        pattern: #"<function_calls>\s*.*?\s*</function_calls>"#,
        options: [.dotMatchesLineSeparators]
    )
    private static let toolCodeBlockRegex = try! NSRegularExpression(
        pattern: #"```tool_code\s*\n(.*?)\n\s*```"#,
        options: [.dotMatchesLineSeparators]
    )
    private static let toolCodeCallRegex = try! NSRegularExpression(
        pattern: #"^(\w+)\((.*)\)$"#,
        options: [.dotMatchesLineSeparators]
    )
    private static let toolCodeStripRegex = try! NSRegularExpression(
        pattern: #"```tool_code\s*\n.*?\n\s*```"#,
        options: [.dotMatchesLineSeparators]
    )

    public init(
        configuration: ModelConfiguration,
        provider: ModelProvider,
        readAPIKey: @Sendable @escaping () -> String,
        verboseLogging: Bool = false,
        behaviorFlags: BehaviorFlags = BehaviorFlags(),
        modelCapabilities: ModelCapabilities = ModelCapabilities(),
        session: URLSession = llmURLSession
    ) {
        self.configuration = configuration
        self.provider = provider
        self.readAPIKey = readAPIKey
        self.verboseLogging = verboseLogging
        self.behaviorFlags = behaviorFlags
        self.modelCapabilities = modelCapabilities
        self.session = session
    }

    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides = LLMCallOverrides()
    ) async throws -> LLMResponse {
        // Ollama doesn't have an effort enum and its chat options don't
        // expose frequency_penalty / presence_penalty. Those fields on
        // `overrides` are silently discarded.
        // `reasoningEnabled` (either layer) is likewise NOT emitted. Ollama's top-level
        // `think` flag is a distinct mechanism with no ReasoningControl case, and Ollama
        // rejects `think: true` on models that don't support thinking — an ungated emit
        // would 400 requests that work today, and gating needs the probe/catalog story a
        // new mechanism case implies (resolver branch, disable-overrides, budget gates).
        // Until that case exists, callers can reach `think` via extraJSONOverrides.

        // Documents are REFUSED, not dropped. Ollama's chat API carries `images` and nothing
        // else — there is no document/PDF field — and `encodeMessage` accordingly has no
        // `documents` branch, as does the merge path in `normalizeMessages`. Until this guard,
        // a caller passing `documents:` got a 200 and an answer about a file the model was
        // never shown: the request succeeded, the evidence vanished, and nothing said so.
        // A refusal the caller can see beats a silent success that answers the wrong question.
        if let offender = messages.first(where: { !($0.documents ?? []).isEmpty }) {
            let count = messages.reduce(0) { $0 + ($1.documents?.count ?? 0) }
            throw LLMProviderError.invalidRequest(detail: """
                Ollama cannot accept document attachments: \(count) document\(count == 1 ? "" : "s") \
                on a \(offender.role.rawValue) message would be discarded. Ollama's chat API \
                supports images only. Send the document as an image, extract its text into the \
                message, or use a provider that accepts documents.
                """)
        }

        let url = provider.endpoint.appendingPathComponent("chat")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let apiKey = readAPIKey()
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        // Always normalize. `normalizeMessages` rewrites tool-role messages into
        // user text and merges consecutive same-role messages. Earlier this was
        // gated on `tools.isEmpty` out of fear it would break tool_call/tool_result
        // pairing — but Ollama's `encodeMessage` doesn't put `tool_call_id` on the
        // wire in the first place, so there's no pairing to break. Without the
        // rewrite, gemma3:27b (whose chat template only knows user/assistant)
        // returns HTTP 400 "Conversation roles must alternate user/assistant/..."
        // every time a tool result lands in Smith's history. Models that DO
        // natively support a tool role still receive the result content, just
        // wrapped as user text with a "[Tool result for <id>]: ..." prefix.
        let finalMessages = Self.normalizeMessages(messages)
        let body = buildRequestBody(messages: Self.extractSystemMessages(finalMessages, allowTrailingSystem: behaviorFlags.supportsTrailingSystemMessage), tools: tools, overrides: overrides)
        // .sortedKeys keeps wire bytes stable across requests for any
        // downstream prefix-caching the model applies.
        let requestData = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        request.httpBody = requestData

        logger.debug("Request: POST \(url.absoluteString, privacy: .public) model=\(configuration.model, privacy: .public)")
        let requestLog: LLMRequestLogger.RequestLogToken? = verboseLogging
            ? LLMRequestLogger.logRequest(label: "Ollama", url: url, model: configuration.model, body: body, rawData: requestData)
            : nil

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw LLMProviderError.invalidResponse
        }

        if verboseLogging {
            LLMRequestLogger.logResponse(label: "Ollama", statusCode: httpResponse.statusCode, data: data, for: requestLog)
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let responseBody = String(data: data, encoding: .utf8) ?? "unknown"
            logger.error("HTTP \(httpResponse.statusCode, privacy: .public) from \(url.absoluteString, privacy: .public) body=\(responseBody, privacy: .public)")
            throw LLMProviderError.httpError(statusCode: httpResponse.statusCode, body: responseBody, url: url, retryAfter: LLMProviderError.parseRetryAfter(httpResponse.value(forHTTPHeaderField: "Retry-After")))
        }

        return try parseResponse(data: data)
    }

    /// Convenience overload for tests that only care about a couple of knobs.
    func buildRequestBody(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice? = nil
    ) -> [String: Any] {
        buildRequestBody(
            messages: messages,
            tools: tools,
            overrides: LLMCallOverrides(toolChoice: toolChoice)
        )
    }

    func buildRequestBody(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides
    ) -> [String: Any] {
        let toolChoice = overrides.toolChoice
        let maxOutputTokensOverride = overrides.maxOutputTokens
        let temperatureOverride = overrides.temperature
        let topPOverride = overrides.topP
        let stopSequencesOverride = overrides.stopSequences
        let effectiveMaxTokens = maxOutputTokensOverride ?? configuration.maxTokens
        var options: [String: Any] = ["num_predict": effectiveMaxTokens]
        // Tell Ollama how large a context window to load the model with. Without `num_ctx`,
        // Ollama ignores the configured context size entirely and loads the model at its
        // Modelfile default (frequently far smaller), so the client's `contextWindowSize`
        // never reaches the server. A large `num_predict` reservation then overflows the real
        // window and Ollama returns "The number of tokens to keep from the initial prompt is
        // greater than the context length". Sending `num_ctx` makes the configured context
        // window actually take effect. NOTE: this raises the model's memory footprint to match
        // the requested window — callers are responsible for configuring a context size their
        // Ollama host can serve.
        if configuration.contextWindowSize > 0 {
            options["num_ctx"] = configuration.contextWindowSize
        }
        if !behaviorFlags.mustNeverSendTemperatureParam,
           let temperature = temperatureOverride ?? configuration.temperature {
            options["temperature"] = temperature
        }
        if let topP = topPOverride {
            options["top_p"] = topP
        }
        if let stops = stopSequencesOverride, !stops.isEmpty {
            options["stop"] = stops
        }
        var body: [String: Any] = [
            "model": configuration.model,
            "stream": false,
            "messages": messages.map(encodeMessage),
            "options": options
        ]

        if !tools.isEmpty {
            body["tools"] = tools.map { tool in
                [
                    "type": "function",
                    "function": [
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters.mapValues(\.rawValue)
                    ] as [String: Any]
                ] as [String: Any]
            }
            // Ollama follows the OpenAI tool_choice shape on the wire. Some
            // models / chat templates respect it; others (gemma3 and other
            // strict-alternation templates) ignore the field. Pass through
            // when set so models that DO honor it get the right control;
            // models that don't are unaffected. No flag gating — emission
            // is benign even when ignored.
            if let toolChoice {
                if modelCapabilities.permitsToolChoice(toolChoice) {
                    body["tool_choice"] = toolChoice.openAIWireValue.rawValue
                }
            }
        }

        // Apply caller-provided overrides last so they win over any defaults
        // this method set. Deep-merges dict-valued keys (e.g. nested `options`).
        if let overrides = configuration.extraJSONOverrides {
            mergeJSONOverrides(&body, with: overrides)
        }

        return body
    }

    /// Test-only helper: returns the exact `messages` array OllamaProvider will
    /// place in the /chat request body for a given conversation + tool list. Runs
    /// the same normalization → encode pipeline as `send`, but without going
    /// through URLSession. Lets role-alternation tests assert on the bytes the
    /// gemma3 chat template ends up parsing.
    func buildEncodedMessagesForTesting(
        messages: [LLMMessage],
        tools: [LLMToolDefinition]
    ) -> [[String: Any]] {
        let normalized = Self.normalizeMessages(messages)
        let withSystemFront = Self.extractSystemMessages(normalized, allowTrailingSystem: behaviorFlags.supportsTrailingSystemMessage)
        return withSystemFront.map(encodeMessage)
    }

    private func encodeMessage(_ message: LLMMessage) -> [String: Any] {
        var result: [String: Any] = ["role": message.role.rawValue]

        switch message.content {
        case .text(let text):
            result["content"] = sanitizeAssistantText(text, role: message.role)
            if let images = message.images, !images.isEmpty {
                // Ollama multimodal: base64 image array alongside text content
                result["images"] = images.map { $0.data.base64EncodedString() }
            }
        case .toolCalls(let calls):
            result["tool_calls"] = calls.map(encodeToolCall)
        case .mixed(let text, let calls):
            result["content"] = sanitizeAssistantText(text, role: message.role)
            result["tool_calls"] = calls.map(encodeToolCall)
        case .toolResult(_, let content):
            // Ollama tool results don't use tool_call_id. Tool results are our output,
            // not the model's, so they can't carry poison and may legitimately contain
            // text that overlaps with chat-template syntax. Pass through unchanged.
            result["role"] = "tool"
            result["content"] = content
        }

        return result
    }

    /// Pre-flight cleanup of assistant text we send back to Ollama. Strips GLM
    /// chat-template control tokens if any are present, scoped to assistant-authored
    /// content because that's the only direction poison flows: the model's previous
    /// turn left tokens in its `content` and replaying them in the next request would
    /// re-poison the conversation. User and tool-result content are passed through.
    /// Gated on the per-model `glmTemplateSalvage` flag — non-GLM models always pass
    /// text through unchanged.
    private func sanitizeAssistantText(_ text: String, role: LLMMessage.Role) -> String {
        guard behaviorFlags.glmTemplateSalvage else { return text }
        guard role == .assistant else { return text }
        guard GLMTemplateSalvage.contentLooksGLMTemplated(text) else { return text }
        return GLMTemplateSalvage.strip(text)
    }

    private func encodeToolCall(_ call: LLMToolCall) -> [String: Any] {
        [
            "function": [
                "name": call.name,
                "arguments": argumentsObject(from: call.arguments)
            ] as [String: Any]
        ]
    }

    /// Converts a JSON string back to a JSON object for Ollama's native format.
    private func argumentsObject(from jsonString: String) -> Any {
        guard let data = jsonString.data(using: .utf8) else {
            return jsonString
        }
        do {
            return try JSONSerialization.jsonObject(with: data)
        } catch {
            logger.warning("Failed to parse tool call arguments as JSON: \(error.localizedDescription, privacy: .public) input=\(jsonString, privacy: .public)")
            return jsonString
        }
    }

    // MARK: - System message extraction

    /// Moves all system messages to the front as a single consolidated system message.
    /// Prevents invalid message ordering (e.g. system after tool) that some backends reject.
    ///
    /// `.developer` is folded into system here — Ollama's chat templates know only
    /// system/user/assistant/tool, and `LLMMessage.Role.developer` documents that
    /// non-OpenAI backends translate it to system. Without this, the developer
    /// turn would survive into `encodeMessage` and emit `"role": "developer"` on
    /// the wire, which Ollama's templates reject or misinterpret.
    private static func extractSystemMessages(_ messages: [LLMMessage], allowTrailingSystem: Bool) -> [LLMMessage] {
        // A trailing `{role: system}` steering turn is left in place — Ollama supports the system
        // role natively and `normalizeMessages` passes it through, so it stays last — on models
        // known to read one. Every other system/developer message is hoisted to a single leading
        // system message, as before.
        let trailingSystemIndex = LLMMessage.trailingSystemTurnIndex(in: messages, allowed: allowTrailingSystem)
        var systemParts: [String] = []
        var nonSystem: [LLMMessage] = []
        for (index, message) in messages.enumerated() {
            if index != trailingSystemIndex,
               (message.role == .system || message.role == .developer),
               case .text(let text) = message.content {
                systemParts.append(text)
            } else {
                nonSystem.append(message)
            }
        }
        guard !systemParts.isEmpty else { return messages }
        var result: [LLMMessage] = [.system(systemParts.joined(separator: "\n\n"))]
        result.append(contentsOf: nonSystem)
        return result
    }

    // MARK: - Message normalization

    /// Ensures the conversation history satisfies strict role alternation rules.
    /// Some model backends reject conversations where roles don't strictly alternate
    /// user/assistant, or don't understand the "tool" role at all.
    ///
    /// This pass:
    /// 1. Merges consecutive user messages.
    /// 2. Converts orphaned tool results (not supported by some backends) into user messages.
    /// 3. Ensures no two assistant messages appear back-to-back.
    private static func normalizeMessages(_ messages: [LLMMessage]) -> [LLMMessage] {
        guard !messages.isEmpty else { return messages }

        var result: [LLMMessage] = []

        for message in messages {
            // System messages pass through; they sit at the start and don't affect alternation.
            if message.role == .system {
                result.append(message)
                continue
            }

            // Convert tool results into user messages so backends that don't understand
            // the "tool" role still receive the information.
            let effectiveMessage: LLMMessage
            if message.role == .tool {
                let text: String
                if case .toolResult(let callID, let content) = message.content {
                    text = "[Tool result for \(callID)]: \(content)"
                } else {
                    text = "[Tool result]"
                }
                effectiveMessage = .user(text)
            } else {
                effectiveMessage = message
            }

            // Merge consecutive same-role messages.
            if let lastIndex = result.indices.last,
               result[lastIndex].role == effectiveMessage.role,
               result[lastIndex].role != .system {
                // Merge text content
                let existingText: String
                switch result[lastIndex].content {
                case .text(let t): existingText = t
                case .toolCalls: existingText = ""
                case .mixed(let t, _): existingText = t
                case .toolResult(_, let t): existingText = t
                }
                let newText: String
                switch effectiveMessage.content {
                case .text(let t): newText = t
                case .toolCalls: newText = ""
                case .mixed(let t, _): newText = t
                case .toolResult(_, let t): newText = t
                }
                let merged = [existingText, newText]
                    .filter { !$0.isEmpty }
                    .joined(separator: "\n\n")
                // Merge produces a text message with possibly-combined images.
                // Use the internal init to bypass the deprecated public init —
                // this is provider-internal normalization, not consumer code.
                let mergedImages = (result[lastIndex].images ?? []) + (effectiveMessage.images ?? [])
                result[lastIndex] = LLMMessage(
                    _role: effectiveMessage.role,
                    _content: .text(merged),
                    _images: mergedImages.isEmpty ? nil : mergedImages
                )
            } else {
                result.append(effectiveMessage)
            }
        }

        return result
    }

    // MARK: - Response parsing

    private func parseResponse(data: Data) throws -> LLMResponse {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            let preview = String(data: data.prefix(500), encoding: .utf8) ?? "(non-utf8, \(data.count) bytes)"
            logger.error("Response is not a JSON object: \(preview, privacy: .public)")
            throw LLMProviderError.malformedResponse(detail: "not a JSON object: \(preview)")
        }
        guard let message = json["message"] as? [String: Any] else {
            let keys = json.keys.sorted().joined(separator: ", ")
            let preview = String(data: data.prefix(500), encoding: .utf8) ?? "(\(data.count) bytes)"
            logger.error("Missing message in response. Keys: \(keys, privacy: .public) Body: \(preview, privacy: .public)")
            throw LLMProviderError.malformedResponse(detail: "missing message, keys: [\(keys)], body: \(preview)")
        }

        // Parse token usage. Ollama returns counts as top-level fields rather than
        // a nested usage object, so synthesize one for the rawUsage snapshot.
        var tokenUsage: TokenUsage?
        let promptEval = json["prompt_eval_count"] as? Int ?? 0
        let eval = json["eval_count"] as? Int ?? 0
        if promptEval > 0 || eval > 0 {
            let synthesized: [String: Any] = [
                "prompt_eval_count": promptEval,
                "eval_count": eval
            ]
            let rawUsage = TokenUsage.serializeRawUsage(synthesized)
            tokenUsage = TokenUsage(
                inputTokens: promptEval,
                outputTokens: eval,
                rawUsage: rawUsage
            )
        }

        let text = message["content"] as? String
        let toolCallsRaw = message["tool_calls"] as? [[String: Any]]

        var toolCalls: [LLMToolCall] = []
        if let rawCalls = toolCallsRaw {
            for raw in rawCalls {
                guard let function = raw["function"] as? [String: Any],
                      let name = function["name"] as? String
                else { continue }

                // Ollama returns arguments as a JSON object, not a string
                let arguments: String
                if let argsObj = function["arguments"] {
                    if let argsString = argsObj as? String {
                        arguments = argsString
                    } else {
                        do {
                            let argsData = try JSONSerialization.data(withJSONObject: argsObj)
                            arguments = String(data: argsData, encoding: .utf8) ?? "{}"
                        } catch {
                            logger.warning("Failed to serialize tool arguments for \(name, privacy: .public): \(error.localizedDescription, privacy: .public)")
                            arguments = "{}"
                        }
                    }
                } else {
                    arguments = "{}"
                }

                // Ollama doesn't return tool call IDs — generate synthetic ones
                toolCalls.append(LLMToolCall(id: UUID().uuidString, name: name, arguments: arguments))
            }
        }

        // Fallback: if no structured tool_calls, check content for text-formatted tool calls.
        // Some models output tool calls as text instead of using Ollama's native tool_calls
        // structure. We support three formats:
        //  1. Anthropic XML: <function_calls><invoke name="...">...</invoke></function_calls>
        //  2. tool_code fences: ```tool_code\nfunction_name(arg: value)\n```
        //  3. GLM chat-template: <tool_call>name\n<arg_key>k</arg_key><arg_value>v</arg_value>\n</tool_call>
        //     Specifically observed when GLM-4 / GLM-5 models are routed through Ollama
        //     (cloud) — Ollama relays the raw model output without translating to its
        //     native tool_calls schema, so we have to recover both names and args from
        //     content text.
        if toolCalls.isEmpty, let content = text {
            var parsedCalls: [LLMToolCall] = []
            var strippedContent = content

            if content.contains("<function_calls>") {
                parsedCalls = Self.parseXMLToolCalls(from: content)
                strippedContent = Self.stripXMLToolCalls(from: content)
            } else if content.contains("```tool_code") {
                parsedCalls = Self.parseToolCodeCalls(from: content)
                strippedContent = Self.stripToolCodeBlocks(from: content)
            } else if behaviorFlags.glmTemplateSalvage,
                      content.contains("<tool_call>") || content.contains("<arg_key>") {
                let salvaged = GLMTemplateSalvage.extractFullCalls(content: content)
                parsedCalls = salvaged.map { call in
                    LLMToolCall(id: UUID().uuidString, name: call.name, arguments: call.arguments)
                }
                strippedContent = GLMTemplateSalvage.strip(content)
            }

            if !parsedCalls.isEmpty {
                let remainingText = strippedContent.trimmingCharacters(in: .whitespacesAndNewlines)
                return LLMResponse(
                    text: remainingText.isEmpty ? nil : remainingText,
                    toolCalls: parsedCalls,
                    usage: tokenUsage
                )
            }
        }

        // GLM template salvage on the structured-tool_calls path: even when Ollama
        // populated the names, GLM's args sometimes stay embedded in the content text
        // as `<arg_key>/<arg_value>` pairs. Gated on `glmTemplateSalvage` so non-GLM
        // models running through Ollama never have their content touched.
        var finalToolCalls = toolCalls
        var finalText = text
        if behaviorFlags.glmTemplateSalvage,
           let raw = text, GLMTemplateSalvage.contentLooksGLMTemplated(raw) {
            if !finalToolCalls.isEmpty {
                finalToolCalls = GLMTemplateSalvage.patchEmptyArgs(finalToolCalls, content: raw)
            }
            let cleaned = GLMTemplateSalvage.strip(raw)
            finalText = cleaned.isEmpty ? nil : cleaned
        }

        return LLMResponse(
            text: finalText?.isEmpty == true ? nil : finalText,
            toolCalls: finalToolCalls,
            usage: tokenUsage
        )
    }

    // MARK: - XML tool call parsing

    /// Parses Anthropic-style XML tool calls from content text.
    ///
    /// Expected format:
    /// ```xml
    /// <function_calls>
    /// <invoke name="tool_name">
    /// <parameter name="param_name">value</parameter>
    /// </invoke>
    /// </function_calls>
    /// ```
    private static func parseXMLToolCalls(from content: String) -> [LLMToolCall] {
        var calls: [LLMToolCall] = []

        let nsContent = content as NSString
        let blockMatches = xmlBlockRegex.matches(
            in: content,
            range: NSRange(location: 0, length: nsContent.length)
        )

        for blockMatch in blockMatches {
            let blockBody = nsContent.substring(with: blockMatch.range(at: 1))
            calls.append(contentsOf: parseInvocations(from: blockBody))
        }

        return calls
    }

    /// Parses `<invoke>` elements within a `<function_calls>` block.
    private static func parseInvocations(from block: String) -> [LLMToolCall] {
        var calls: [LLMToolCall] = []

        let nsBlock = block as NSString
        let invokeMatches = invokeRegex.matches(
            in: block,
            range: NSRange(location: 0, length: nsBlock.length)
        )

        for invokeMatch in invokeMatches {
            let toolName = nsBlock.substring(with: invokeMatch.range(at: 1))
            let paramsBody = nsBlock.substring(with: invokeMatch.range(at: 2))
            let arguments = parseParameters(from: paramsBody)

            let argsJSON: String
            if arguments.isEmpty {
                argsJSON = "{}"
            } else {
                do {
                    let argsData = try JSONSerialization.data(withJSONObject: arguments)
                    argsJSON = String(data: argsData, encoding: .utf8) ?? "{}"
                } catch {
                    logger.warning("Failed to serialize XML tool arguments for \(toolName, privacy: .public): \(error.localizedDescription, privacy: .public)")
                    argsJSON = "{}"
                }
            }

            calls.append(LLMToolCall(id: UUID().uuidString, name: toolName, arguments: argsJSON))
        }

        return calls
    }

    /// Parses `<parameter>` elements into a dictionary.
    private static func parseParameters(from body: String) -> [String: Any] {
        var params: [String: Any] = [:]

        let nsBody = body as NSString
        let paramMatches = paramRegex.matches(
            in: body,
            range: NSRange(location: 0, length: nsBody.length)
        )

        for paramMatch in paramMatches {
            let name = nsBody.substring(with: paramMatch.range(at: 1))
            let value = nsBody.substring(with: paramMatch.range(at: 2))

            // try? justified: probing whether the string is a JSON number/bool; plain strings
            // are the expected fallback, not an error condition.
            if let data = value.data(using: .utf8),
               let jsonValue = try? JSONSerialization.jsonObject(with: data) {
                // Only use JSON interpretation for non-string types to avoid
                // double-interpreting strings that happen to be valid JSON
                if jsonValue is NSNumber {
                    params[name] = jsonValue
                } else {
                    params[name] = value
                }
            } else {
                params[name] = value
            }
        }

        return params
    }

    /// Removes `<function_calls>...</function_calls>` blocks from content text.
    private static func stripXMLToolCalls(from content: String) -> String {
        xmlStripRegex.stringByReplacingMatches(
            in: content,
            range: NSRange(location: 0, length: (content as NSString).length),
            withTemplate: ""
        )
    }

    // MARK: - tool_code fence parsing

    /// Parses tool calls from ` ```tool_code ` fenced code blocks.
    ///
    /// Expected format:
    /// ```
    /// ```tool_code
    /// function_name(arg1: value1, arg2: "string value")
    /// ```
    /// ```
    private static func parseToolCodeCalls(from content: String) -> [LLMToolCall] {
        var calls: [LLMToolCall] = []

        let nsContent = content as NSString
        let matches = toolCodeBlockRegex.matches(
            in: content,
            range: NSRange(location: 0, length: nsContent.length)
        )

        for match in matches {
            let body = nsContent.substring(with: match.range(at: 1))
                .trimmingCharacters(in: .whitespacesAndNewlines)

            let nsBody = body as NSString
            guard let callMatch = toolCodeCallRegex.firstMatch(
                in: body,
                range: NSRange(location: 0, length: nsBody.length)
            ) else { continue }

            let funcName = nsBody.substring(with: callMatch.range(at: 1))
            let argsString = nsBody.substring(with: callMatch.range(at: 2))
                .trimmingCharacters(in: .whitespacesAndNewlines)

            let arguments = parseToolCodeArguments(argsString)
            let argsJSON: String
            if arguments.isEmpty {
                argsJSON = "{}"
            } else {
                do {
                    let argsData = try JSONSerialization.data(withJSONObject: arguments)
                    argsJSON = String(data: argsData, encoding: .utf8) ?? "{}"
                } catch {
                    logger.warning("Failed to serialize tool_code arguments for \(funcName, privacy: .public): \(error.localizedDescription, privacy: .public)")
                    argsJSON = "{}"
                }
            }

            calls.append(LLMToolCall(id: UUID().uuidString, name: funcName, arguments: argsJSON))
        }

        return calls
    }

    /// Parses `key: value` arguments from a function call string like `arg1: "hello", arg2: true`.
    private static func parseToolCodeArguments(_ argsString: String) -> [String: Any] {
        guard !argsString.isEmpty else { return [:] }

        var params: [String: Any] = [:]

        // Split on commas that are not inside quotes.
        // Walk the string tracking quote state to split correctly.
        var parts: [String] = []
        var current = ""
        var inQuote = false
        var escaped = false
        for char in argsString {
            if escaped {
                current.append(char)
                escaped = false
                continue
            }
            if char == "\\" {
                escaped = true
                current.append(char)
                continue
            }
            if char == "\"" {
                inQuote.toggle()
                current.append(char)
            } else if char == "," && !inQuote {
                parts.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            } else {
                current.append(char)
            }
        }
        let last = current.trimmingCharacters(in: .whitespaces)
        if !last.isEmpty { parts.append(last) }

        for part in parts {
            // Split on first ":"
            guard let colonIndex = part.firstIndex(of: ":") else { continue }
            let key = String(part[part.startIndex..<colonIndex]).trimmingCharacters(in: .whitespaces)
            var value = String(part[part.index(after: colonIndex)...]).trimmingCharacters(in: .whitespaces)

            if value.hasPrefix("\"") && value.hasSuffix("\"") && value.count >= 2 {
                // Quoted string — strip quotes
                value = String(value.dropFirst().dropLast())
                params[key] = value
            } else if value == "true" {
                params[key] = true
            } else if value == "false" {
                params[key] = false
            } else if value == "null" || value == "nil" {
                params[key] = NSNull()
            } else if let intVal = Int(value) {
                params[key] = intVal
            } else if let doubleVal = Double(value) {
                params[key] = doubleVal
            } else {
                params[key] = value
            }
        }

        return params
    }

    /// Removes ` ```tool_code ... ``` ` blocks from content text.
    private static func stripToolCodeBlocks(from content: String) -> String {
        toolCodeStripRegex.stringByReplacingMatches(
            in: content,
            range: NSRange(location: 0, length: (content as NSString).length),
            withTemplate: ""
        )
    }

    /// Translates the unified `LLMToolChoice` to Ollama's wire shape, which
    /// mirrors OpenAI's `tool_choice` field.
}
