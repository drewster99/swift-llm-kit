import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "OpenAI")

/// LLM provider for OpenAI-compatible APIs (OpenAI, Ollama, LM Studio, vLLM).
struct OpenAICompatibleProvider: LLMProvider {
    private let configuration: ModelConfiguration
    private let provider: ModelProvider
    private let readAPIKey: @Sendable () -> String
    private let verboseLogging: Bool
    private let parallelToolCalls: Bool
    private let behaviorFlags: BehaviorFlags
    private let session: URLSession
    /// Stable conversation ID for xAI prompt caching. Generated once per provider instance.
    private let conversationID: String

    public init(
        configuration: ModelConfiguration,
        provider: ModelProvider,
        readAPIKey: @Sendable @escaping () -> String,
        verboseLogging: Bool = false,
        parallelToolCalls: Bool = false,
        behaviorFlags: BehaviorFlags = BehaviorFlags(),
        session: URLSession = llmURLSession
    ) {
        self.configuration = configuration
        self.provider = provider
        self.readAPIKey = readAPIKey
        self.verboseLogging = verboseLogging
        self.parallelToolCalls = parallelToolCalls
        self.behaviorFlags = behaviorFlags
        self.session = session
        self.conversationID = UUID().uuidString
    }

    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides = LLMCallOverrides()
    ) async throws -> LLMResponse {
        let url = provider.endpoint.appendingPathComponent("chat/completions")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let apiKey = readAPIKey()
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        if provider.apiType == .xAI {
            request.setValue(conversationID, forHTTPHeaderField: "x-grok-conv-id")
        }
        if provider.apiType == .zAI {
            request.setValue("en-US,en", forHTTPHeaderField: "Accept-Language")
        }
        if provider.apiType == .openRouter {
            // OpenRouter uses these optional headers for app attribution on
            // its public leaderboards and per-app analytics dashboards.
            // https://openrouter.ai/docs/api-reference/overview#headers
            let appName = (Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String)
                ?? (Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String)
                ?? "SwiftLLMKit"
            let bundleID = Bundle.main.bundleIdentifier ?? "com.swiftllmkit.app"
            request.setValue("https://\(bundleID)", forHTTPHeaderField: "HTTP-Referer")
            request.setValue(appName, forHTTPHeaderField: "X-Title")
        }

        let body = buildRequestBody(messages: messages, tools: tools, overrides: overrides)
        // A non-finite Double (NaN/±Inf) reaching JSONSerialization raises an NSException that
        // `try` cannot convert to a Swift error — it aborts the process. Sampling params
        // (temperature/top_p/penalties) and any extra overrides are caller-supplied and unvalidated,
        // so pre-flight with Apple's validity check and surface a normal throw instead.
        guard JSONSerialization.isValidJSONObject(body) else {
            throw LLMProviderError.invalidRequest(
                detail: "request body is not JSON-encodable (likely a non-finite temperature/top_p/penalty)")
        }
        // .sortedKeys keeps wire bytes stable across requests so OpenAI's
        // automatic prefix cache (>=1024-token prefix match) keeps hitting.
        let requestData = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        request.httpBody = requestData

        logger.debug("Request: POST \(url.absoluteString, privacy: .public) model=\(configuration.model, privacy: .public)")
        let requestLog: LLMRequestLogger.RequestLogToken? = verboseLogging
            ? LLMRequestLogger.logRequest(label: "OpenAI", url: url, model: configuration.model, body: body, rawData: requestData)
            : nil

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw LLMProviderError.invalidResponse
        }

        if verboseLogging {
            LLMRequestLogger.logResponse(label: "OpenAI", statusCode: httpResponse.statusCode, data: data, for: requestLog)
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
        let thinkingEffortOverride = overrides.thinkingEffort
        let maxOutputTokensOverride = overrides.maxOutputTokens
        let temperatureOverride = overrides.temperature
        let topPOverride = overrides.topP
        let stopSequencesOverride = overrides.stopSequences
        let frequencyPenaltyOverride = overrides.frequencyPenalty
        let presencePenaltyOverride = overrides.presencePenalty
        // OpenAI API requires system messages at the start of the conversation.
        // Extract any system messages from arbitrary positions (e.g. per-turn task
        // context appended by AgentActor) and consolidate them into a single leading
        // system message, followed by the remaining non-system messages in order.
        //
        // `.developer` is OpenAI's newer role for o-series/GPT-5 — semantically
        // like system but with different precedence. We pass it through as a
        // "developer" role when `BehaviorFlags.supportsDeveloperRole` is set
        // (true for OpenAI proper); otherwise we downgrade to system by
        // folding into systemParts. This keeps OpenAI-compatible-but-not-OpenAI
        // backends (z.ai, Mistral, DeepSeek, …) from getting a role they don't
        // understand.
        var systemParts: [String] = []
        var developerParts: [String] = []
        var nonSystemMessages: [[String: Any]] = []
        // A trailing `{role: system}` steering turn stays at the tail (as role "system") on models
        // known to read one, instead of being folded into the leading system message.
        let trailingSystemIndex = LLMMessage.trailingSystemTurnIndex(
            in: messages, allowed: behaviorFlags.supportsTrailingSystemMessage)
        var trailingSystemText: String?
        for (index, message) in messages.enumerated() {
            if index == trailingSystemIndex, case .text(let text) = message.content {
                trailingSystemText = text
                continue
            }
            if message.role == .system, case .text(let text) = message.content {
                systemParts.append(text)
            } else if message.role == .developer, case .text(let text) = message.content {
                if behaviorFlags.supportsDeveloperRole {
                    developerParts.append(text)
                } else {
                    // Downgrade to system for backends that don't support it.
                    systemParts.append(text)
                }
            } else {
                nonSystemMessages.append(encodeMessage(message))
            }
        }
        var orderedMessages: [[String: Any]] = []
        if !systemParts.isEmpty {
            orderedMessages.append(["role": "system", "content": systemParts.joined(separator: "\n\n")])
        }
        if !developerParts.isEmpty {
            // Developer messages go AFTER system per OpenAI's precedence model.
            orderedMessages.append(["role": "developer", "content": developerParts.joined(separator: "\n\n")])
        }
        orderedMessages.append(contentsOf: nonSystemMessages)
        // The held-out trailing system turn is emitted last, after the final user turn — the shape
        // proven by the trailing-system probe for models that honor it.
        if let trailingSystemText {
            orderedMessages.append(["role": "system", "content": trailingSystemText])
        }

        // OpenAI deprecated `max_tokens` on Chat Completions; GPT-5.x and o-series
        // reject it outright and require `max_completion_tokens`. DeepSeek and the
        // other OpenAI-compatible backends still only accept `max_tokens`. The
        // bundled-defaults JSON marks affected OpenAI models with
        // `useMaxCompletionTokens: true`; users can override per-model.
        let tokenLimitKey = behaviorFlags.useMaxCompletionTokens ? "max_completion_tokens" : "max_tokens"
        var body: [String: Any] = [
            "model": configuration.model,
            tokenLimitKey: maxOutputTokensOverride ?? configuration.maxTokens,
            "messages": orderedMessages
        ]
        // Reasoning models (o-series, GPT-5 family) reject `temperature` outright — omit it
        // entirely when the model is flagged, regardless of override or configured value.
        if !behaviorFlags.mustNeverSendTemperatureParam,
           let temperature = temperatureOverride ?? configuration.temperature {
            body["temperature"] = temperature
        }
        if let topP = topPOverride {
            body["top_p"] = topP
        }
        if let stops = stopSequencesOverride, !stops.isEmpty {
            body["stop"] = stops
        }
        if let fp = frequencyPenaltyOverride {
            body["frequency_penalty"] = fp
        }
        if let pp = presencePenaltyOverride {
            body["presence_penalty"] = pp
        }

        // Alibaba Cloud thinking support (uses different keys than Anthropic).
        if provider.apiType == .alibabaCloud,
           let budget = configuration.thinkingBudget, budget > 0 {
            body["enable_thinking"] = true
            body["thinking_budget"] = ThinkingBudget.effective(budget)
        }

        // OpenAI `reasoning_effort` — depth control for reasoning models
        // (o-series, GPT-5 family). Top-level enum, sibling of `messages`.
        // Gated on the `supportsReasoningEffort` flag because non-reasoning
        // models (GPT-4o, GPT-3.5-turbo, DeepSeek-V*, etc.) reject the
        // field with HTTP 400.
        //
        // Per-call `thinkingEffortOverride` wins over the configuration's
        // value (used by HTTP servers that map `reasoning_effort` from
        // inbound requests). Empty string normalized to nil.
        let effectiveEffort: String? = {
            if let override = thinkingEffortOverride, !override.isEmpty { return override }
            return configuration.thinkingEffort
        }()
        if behaviorFlags.supportsReasoningEffort,
           let effort = effectiveEffort {
            body["reasoning_effort"] = effort
        }

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
            // Sent by default (see the provider factory); a strict endpoint that rejects
            // the field is opted out upstream via `disableParallelToolCalls`, which sets
            // `parallelToolCalls == false` here so the key is omitted entirely.
            if parallelToolCalls {
                body["parallel_tool_calls"] = true
            }

            // tool_choice (OpenAI shape). Only emit when caller set it —
            // otherwise the provider's default (`auto` when tools present)
            // applies. Format depends on case: enum string for auto/required/
            // none, or `{"type": "function", "function": {"name": "..."}}`
            // for a specific tool.
            if let toolChoice {
                body["tool_choice"] = Self.encodeOpenAIToolChoice(toolChoice)
            }
        }

        // OpenRouter passes top-level `cache_control` through to Anthropic upstreams,
        // matching the Anthropic Messages API's automatic-caching shape (covers system
        // + tools + messages prefix). Detect by model lineage — OpenRouter's catalog
        // prefixes Anthropic models with "anthropic/" (e.g. "anthropic/claude-haiku-4.5").
        // Other upstreams ignore unknown top-level keys, but we gate it tightly anyway
        // so we don't send fields a non-Anthropic provider might reject.
        if provider.apiType == .openRouter,
           configuration.model.lowercased().hasPrefix("anthropic/") {
            body["cache_control"] = configuration.extendedCacheTTL
                ? ["type": "ephemeral", "ttl": "1h"] as [String: Any]
                : ["type": "ephemeral"] as [String: Any]
        }

        // Apply caller-provided overrides last so they win over any defaults
        // this method set. Deep-merges dict-valued keys (see mergeJSONOverrides).
        if let overrides = configuration.extraJSONOverrides {
            mergeJSONOverrides(&body, with: overrides)
        }

        return body
    }

    func encodeMessage(_ message: LLMMessage) -> [String: Any] {
        var result: [String: Any] = ["role": message.role.rawValue]

        switch message.content {
        case .text(let text):
            let outgoing = sanitizeAssistantText(text, role: message.role)
            // If there are images and/or documents, encode as a content-parts array (multimodal).
            let images = message.images ?? []
            let documents = message.documents ?? []
            if !images.isEmpty || !documents.isEmpty {
                var parts: [[String: Any]] = images.map { image in
                    var imageURL: [String: Any] = [
                        "url": "data:\(image.mimeType);base64,\(image.data.base64EncodedString())"
                    ]
                    if let detail = image.detail {
                        imageURL["detail"] = detail.rawValue
                    }
                    return [
                        "type": "image_url",
                        "image_url": imageURL
                    ] as [String: Any]
                }
                for document in documents {
                    // OpenAI Chat Completions REQUIRES `filename` alongside `file_data` — omitting it
                    // is a hard 400. Synthesize a fallback when the caller didn't supply one.
                    let file: [String: Any] = [
                        "file_data": "data:\(document.mimeType);base64,\(document.data.base64EncodedString())",
                        "filename": document.filename ?? "document.pdf"
                    ]
                    parts.append(["type": "file", "file": file])
                }
                parts.append(["type": "text", "text": outgoing])
                result["content"] = parts
            } else {
                result["content"] = outgoing
            }
        case .toolCalls(let calls):
            result["tool_calls"] = calls.map { call in
                [
                    "id": call.id,
                    "type": "function",
                    "function": [
                        "name": call.name,
                        "arguments": call.arguments
                    ]
                ] as [String: Any]
            }
        case .mixed(let text, let calls):
            result["content"] = sanitizeAssistantText(text, role: message.role)
            result["tool_calls"] = calls.map { call in
                [
                    "id": call.id,
                    "type": "function",
                    "function": [
                        "name": call.name,
                        "arguments": call.arguments
                    ]
                ] as [String: Any]
            }
        case .toolResult(let toolCallID, let content):
            result["role"] = "tool"
            result["tool_call_id"] = toolCallID
            // Tool results are our output, not the model's — they can't be a poison
            // vector and they may legitimately contain text that overlaps with GLM
            // chat-template syntax (e.g. a file_read of a chat-template file or a
            // grep through model-prompt files). Pass through unchanged.
            result["content"] = content
        }

        // Replay reasoning_content for thinking models that require it (DeepSeek
        // V4 Pro). Gated on the per-model `replayReasoningContent` flag because
        // other reasoning models reject this field on replay (e.g. deepseek-reasoner).
        // Only meaningful on assistant turns — `tool` (the rewritten toolResult role)
        // and `user`/`system` never carry reasoning.
        if behaviorFlags.replayReasoningContent,
           message.role == .assistant,
           let reasoning = message.reasoning,
           !reasoning.isEmpty {
            result["reasoning_content"] = reasoning
        }

        return result
    }

    /// Pre-flight cleanup of assistant text we send back to the provider. Strips GLM
    /// chat-template control tokens if any are present, scoped to assistant-authored
    /// content because that's the only direction the poison flows: the model's
    /// previous turn left tokens in its `content` and replaying them in the next
    /// request would re-poison the conversation. User and tool-result content are
    /// passed through verbatim. Gated on the per-model `glmTemplateSalvage` flag —
    /// non-GLM models always pass text through unchanged.
    private func sanitizeAssistantText(_ text: String, role: LLMMessage.Role) -> String {
        guard behaviorFlags.glmTemplateSalvage else { return text }
        guard role == .assistant else { return text }
        guard GLMTemplateSalvage.contentLooksGLMTemplated(text) else { return text }
        return GLMTemplateSalvage.strip(text)
    }

    /// Splits an OpenAI-compatible `content` field into answer text and any inline reasoning.
    ///
    /// A plain string is the answer, with no inline reasoning. An array is the block form some
    /// reasoning models use: each element carries a `type` — `"text"` blocks are the answer,
    /// `"thinking"`/`"reasoning"` blocks are the chain of thought (their text nested one level
    /// deeper in a `thinking`/`reasoning` array of `{text,type}` pieces). Answer blocks are
    /// concatenated in order; reasoning blocks are joined with newlines. Any block carrying a
    /// top-level `text` that isn't a thinking block counts as answer text, so an unrecognized
    /// block shape degrades to keeping the answer rather than dropping it.
    static func extractContent(_ content: Any?) -> (text: String?, reasoning: String?) {
        if let string = content as? String {
            return (string, nil)
        }
        guard let blocks = content as? [[String: Any]] else {
            return (nil, nil)
        }
        var textParts: [String] = []
        var reasoningParts: [String] = []
        for block in blocks {
            let type = block["type"] as? String
            if type == "thinking" || type == "reasoning" {
                if let inner = (block["thinking"] as? [[String: Any]]) ?? (block["reasoning"] as? [[String: Any]]) {
                    for piece in inner {
                        if let value = piece["text"] as? String { reasoningParts.append(value) }
                    }
                } else if let value = block["text"] as? String {
                    reasoningParts.append(value)
                }
            } else if let value = block["text"] as? String {
                textParts.append(value)
            }
        }
        return (
            text: textParts.isEmpty ? nil : textParts.joined(),
            reasoning: reasoningParts.isEmpty ? nil : reasoningParts.joined(separator: "\n")
        )
    }

    func parseResponse(data: Data) throws -> LLMResponse {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            let preview = String(data: data.prefix(500), encoding: .utf8) ?? "(non-utf8, \(data.count) bytes)"
            logger.error("Response is not a JSON object: \(preview, privacy: .public)")
            throw LLMProviderError.malformedResponse(detail: "not a JSON object: \(preview)")
        }
        guard let choices = json["choices"] as? [[String: Any]],
              let choice = choices.first,
              let message = choice["message"] as? [String: Any]
        else {
            let keys = json.keys.sorted().joined(separator: ", ")
            let preview = String(data: data.prefix(500), encoding: .utf8) ?? "(\(data.count) bytes)"
            logger.error("Missing choices[0].message in response. Keys: \(keys, privacy: .public) Body: \(preview, privacy: .public)")
            throw LLMProviderError.malformedResponse(detail: "missing choices[0].message, keys: [\(keys)], body: \(preview)")
        }

        // Most hosts send `content` as a plain string, but reasoning models on some
        // OpenAI-compatible hosts (Mistral's magistral family) send an ARRAY of typed
        // blocks instead — a `{"type":"text",...}` answer block beside a
        // `{"type":"thinking",...}` chain-of-thought block. A bare `as? String` misses
        // the array form entirely, dropping the answer and leaving callers with an empty
        // response. `extractContent` handles both shapes.
        let (parsedText, inlineReasoning) = Self.extractContent(message["content"])
        var text = parsedText
        // DeepSeek-style hosts emit the reasoning channel as `reasoning_content`;
        // others (ollama.com's glm-5.x, some vLLM builds) use `reasoning`; magistral puts
        // it inline in the content array, surfaced here as `inlineReasoning`.
        let reasoningContent = (message["reasoning_content"] as? String) ?? (message["reasoning"] as? String) ?? inlineReasoning
        let toolCallsRaw = message["tool_calls"] as? [[String: Any]]

        var toolCalls: [LLMToolCall] = []
        if let rawCalls = toolCallsRaw {
            for raw in rawCalls {
                guard let id = raw["id"] as? String,
                      let function = raw["function"] as? [String: Any],
                      let name = function["name"] as? String
                else {
                    logger.warning("Skipping malformed tool_call entry: missing id, function, or name")
                    continue
                }

                let arguments: String
                if let argsObj = function["arguments"] {
                    if let argsString = argsObj as? String {
                        arguments = argsString
                    } else if JSONSerialization.isValidJSONObject(argsObj) {
                        do {
                            let argsData = try JSONSerialization.data(withJSONObject: argsObj)
                            if let argsString = String(data: argsData, encoding: .utf8) {
                                arguments = argsString
                            } else {
                                logger.warning("Tool call arguments for \(name, privacy: .public) produced non-UTF-8 data, defaulting to {}")
                                arguments = "{}"
                            }
                        } catch {
                            logger.warning("Failed to serialize tool_call arguments for \(name, privacy: .public): \(error.localizedDescription, privacy: .public), defaulting to {}")
                            arguments = "{}"
                        }
                    } else {
                        // A top-level non-container (JSON null/number/bool) makes
                        // JSONSerialization.data raise an NSException that `try` cannot catch —
                        // isValidJSONObject gates it out. Default rather than crash.
                        logger.warning("Tool call arguments for \(name, privacy: .public) is not a JSON object (\(String(describing: type(of: argsObj)), privacy: .public)), defaulting to {}")
                        arguments = "{}"
                    }
                } else {
                    arguments = "{}"
                }

                toolCalls.append(LLMToolCall(id: id, name: name, arguments: arguments))
            }
        }

        // GLM-4 / GLM-5 chat-template salvage. Gated on the per-model
        // `glmTemplateSalvage` behavior flag so non-GLM traffic is *never* touched
        // — even if a response somehow contains the markers. Bundled defaults JSON
        // pre-flags known GLM hosts on every routing path (z.ai, OpenRouter, etc.).
        // For an unknown GLM host the user opts in via per-model override.
        if behaviorFlags.glmTemplateSalvage,
           let raw = text, GLMTemplateSalvage.contentLooksGLMTemplated(raw) {
            if !toolCalls.isEmpty {
                toolCalls = GLMTemplateSalvage.patchEmptyArgs(toolCalls, content: raw)
            }
            let cleaned = GLMTemplateSalvage.strip(raw)
            text = cleaned.isEmpty ? nil : cleaned
        }

        // Salvage: reasoning models occasionally emit their ENTIRE answer in the
        // reasoning channel and leave content empty with finish_reason=stop (observed
        // 2026-07-10: glm-5.2 via ollama.com put a complete security verdict in
        // `reasoning`, content "" — the caller saw an empty response and burned
        // retries). An empty text response is always an error path for callers, so
        // when there is no content AND no tool calls, the reasoning text is strictly
        // more useful than nothing.
        if (text?.isEmpty ?? true), toolCalls.isEmpty,
           let reasoningContent,
           !reasoningContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            logger.notice("Empty content with non-empty reasoning — salvaging reasoning as response text (\(reasoningContent.count) chars)")
            text = reasoningContent
        }

        // Parse token usage.
        var tokenUsage: TokenUsage?
        if let usage = json["usage"] as? [String: Any] {
            let input = usage["prompt_tokens"] as? Int ?? 0
            let output = usage["completion_tokens"] as? Int ?? 0
            var cacheRead = 0
            if let details = usage["prompt_tokens_details"] as? [String: Any] {
                cacheRead = details["cached_tokens"] as? Int ?? 0
            }
            var reasoning = 0
            if let details = usage["completion_tokens_details"] as? [String: Any] {
                reasoning = details["reasoning_tokens"] as? Int ?? 0
            }
            let rawUsage = TokenUsage.serializeRawUsage(usage)
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

        return LLMResponse(
            text: text?.isEmpty == true ? nil : text,
            toolCalls: toolCalls,
            reasoning: reasoningContent,
            usage: tokenUsage,
            finishReason: choice["finish_reason"] as? String
        )
    }

    /// Translates the unified `LLMToolChoice` to OpenAI's `tool_choice` wire
    /// shape. The auto/required/none cases are plain enum strings; the
    /// specific case is a nested function-object.
    private static func encodeOpenAIToolChoice(_ choice: LLMToolChoice) -> Any {
        switch choice {
        case .auto:
            return "auto"
        case .required:
            return "required"
        case .textOnly:
            return "none"
        case .specific(let name):
            return [
                "type": "function",
                "function": ["name": name]
            ] as [String: Any]
        }
    }
}
