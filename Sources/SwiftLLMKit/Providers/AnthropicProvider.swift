import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "Anthropic")

/// LLM provider for the Anthropic Messages API.
struct AnthropicProvider: LLMProvider {
    private let configuration: ModelConfiguration
    private let provider: ModelProvider
    private let readAPIKey: @Sendable () -> String
    private let verboseLogging: Bool
    private let session: URLSession
    private let behaviorFlags: BehaviorFlags

    public init(
        configuration: ModelConfiguration,
        provider: ModelProvider,
        readAPIKey: @Sendable @escaping () -> String,
        verboseLogging: Bool = false,
        session: URLSession = llmURLSession,
        behaviorFlags: BehaviorFlags = BehaviorFlags()
    ) {
        self.configuration = configuration
        self.provider = provider
        self.readAPIKey = readAPIKey
        self.verboseLogging = verboseLogging
        self.session = session
        self.behaviorFlags = behaviorFlags
    }

    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides = LLMCallOverrides()
    ) async throws -> LLMResponse {
        // Anthropic doesn't support frequency_penalty / presence_penalty;
        // those fields in `overrides` are silently discarded.

        let base = provider.endpoint.ensureAnthropicV1()
        let url = base.appendingPathComponent("messages")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(readAPIKey(), forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

        let body = try buildRequestBody(messages: messages, tools: tools, overrides: overrides)
        // .sortedKeys keeps wire bytes stable across requests so Anthropic's
        // prompt cache (which is byte-prefix matched) keeps hitting.
        let requestData = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        request.httpBody = requestData

        logger.debug("Request: POST \(url.absoluteString, privacy: .public) model=\(configuration.model, privacy: .public)")
        let requestLog: LLMRequestLogger.RequestLogToken? = verboseLogging
            ? LLMRequestLogger.logRequest(label: "Anthropic", url: url, model: configuration.model, body: body, rawData: requestData)
            : nil

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw LLMProviderError.invalidResponse
        }

        if verboseLogging {
            LLMRequestLogger.logResponse(label: "Anthropic", statusCode: httpResponse.statusCode, data: data, for: requestLog)
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
        toolChoice: LLMToolChoice? = nil,
        maxOutputTokensOverride: Int? = nil
    ) throws -> [String: Any] {
        try buildRequestBody(
            messages: messages,
            tools: tools,
            overrides: LLMCallOverrides(toolChoice: toolChoice, maxOutputTokens: maxOutputTokensOverride)
        )
    }

    func buildRequestBody(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides
    ) throws -> [String: Any] {
        let toolChoice = overrides.toolChoice
        let thinkingEffortOverride = overrides.thinkingEffort
        let maxOutputTokensOverride = overrides.maxOutputTokens
        let temperatureOverride = overrides.temperature
        let topPOverride = overrides.topP
        let stopSequencesOverride = overrides.stopSequences
        // Anthropic requires system prompt separate from messages.
        // Concatenate all system messages so per-turn context doesn't overwrite the base prompt.
        var systemParts: [String] = []
        var conversationMessages: [LLMMessage] = []
        // When the model is known to read a trailing `{role: system}` steering turn, keep that one
        // message in place (emitted as role "system" at the tail below) rather than hoisting it into
        // the top-level `system` field. Every other system/developer message collapses as before.
        let trailingSystemIndex = LLMMessage.trailingSystemTurnIndex(
            in: messages, allowed: behaviorFlags.supportsTrailingSystemMessage)
        var trailingSystemText: String?

        for (index, message) in messages.enumerated() {
            if index == trailingSystemIndex, let text = message.content.textValue {
                trailingSystemText = text
                continue
            }
            // .developer is treated as system for Anthropic (no native support).
            // Both collapse into the top-level system field.
            if (message.role == .system || message.role == .developer),
               let text = message.content.textValue {
                systemParts.append(text)
            } else {
                conversationMessages.append(message)
            }
        }

        let systemPrompt: String? = systemParts.isEmpty ? nil : systemParts.joined(separator: "\n\n")

        // Merge consecutive same-role messages. The Anthropic API requires strict user/assistant
        // alternation. Multiple tool_result messages (role "user") can follow an assistant tool_use,
        // and must be combined into a single user message with all tool_result content blocks.
        var encodedMessages = Self.mergeConsecutiveSameRole(try conversationMessages.map(encodeMessage))
        // Emit the held-out trailing system turn verbatim as role "system", after the final user
        // turn — the shape Anthropic accepts on models that support it (per the trailing-system
        // probe). It follows a user message, so it never merges with the turn before it.
        if let trailingSystemText {
            encodedMessages.append(["role": "system", "content": trailingSystemText])
        }

        // Adaptive thinking (Opus 4.7, 4.8) — `thinking: {type: "adaptive"}`,
        // no budget_tokens. The `thinkingBudget` field is interpreted as a
        // boolean signal on adaptive-thinking models: > 0 means "thinking on,"
        // model picks depth itself (steered via `output_config.effort`).
        let usesAdaptiveThinking = behaviorFlags.requiresAdaptiveThinking
        let thinkingEnabled = (configuration.thinkingBudget ?? 0) > 0

        // When MANUAL extended thinking is enabled, max_tokens must exceed
        // budget_tokens (which is itself floored at 1024 by Anthropic). Clamp
        // the override so a tight per-call cap doesn't produce an API error.
        // Adaptive thinking has no budget concept, so the constraint doesn't
        // apply.
        let effectiveMaxTokens: Int = {
            guard let override = maxOutputTokensOverride else { return configuration.maxTokens }
            if thinkingEnabled, !usesAdaptiveThinking, let budget = configuration.thinkingBudget {
                return max(override, ThinkingBudget.effective(budget) + 1)
            }
            return override
        }()

        var body: [String: Any] = [
            "model": configuration.model,
            "max_tokens": effectiveMaxTokens,
            "messages": encodedMessages,
            // Top-level `cache_control` is Anthropic's "automatic caching" feature:
            // the system applies the cache breakpoint to the last cacheable block
            // and moves it forward as the conversation grows. Supported as a
            // stable feature on the direct Anthropic Messages API (verified against
            // platform.claude.com docs); also passed through by OpenRouter for the
            // Anthropic-upstream path. Used in addition to the explicit per-block
            // breakpoints on `system` and the last tool below — the auto-roll
            // covers conversation history, while the explicit breakpoints lock the
            // stable system + tools prefixes.
            "cache_control": configuration.extendedCacheTTL
                ? ["type": "ephemeral", "ttl": "1h"] as [String: Any]
                : ["type": "ephemeral"] as [String: Any]
        ]

        // Anthropic requires temperature = 1 when extended thinking is enabled
        // (overrides any explicit value or "omit" preference). Otherwise, only
        // send temperature if the caller specified one — nil means omit.
        // Per-call temperatureOverride wins over configuration.temperature
        // (but NOT over the thinking-on constraint, which is API-enforced).
        if thinkingEnabled {
            body["temperature"] = 1.0
        } else if !behaviorFlags.mustNeverSendTemperatureParam,
                  let temperature = temperatureOverride ?? configuration.temperature {
            body["temperature"] = temperature
        }

        // top_p: Anthropic accepts the field but explicitly recommends
        // using temperature OR top_p, not both. We send whatever the caller
        // asked for and let the API enforce its rules. With extended thinking
        // enabled, Anthropic ignores top_p — preserved by emitting nothing.
        if !thinkingEnabled, let topP = topPOverride {
            body["top_p"] = topP
        }

        if let stops = stopSequencesOverride, !stops.isEmpty {
            body["stop_sequences"] = stops
        }

        // Per-block cache_control breakpoint payload, reused for the system block
        // and the last tool definition. Anthropic looks for the longest matching
        // cached prefix on each request, so marking the stable system prompt and
        // the tool schema turns Brown's ~10–15K-token prefix into a cache hit on
        // every turn after the first.
        let cacheControl: [String: Any] = configuration.extendedCacheTTL
            ? ["type": "ephemeral", "ttl": "1h"]
            : ["type": "ephemeral"]

        if let systemPrompt {
            // Encode `system` as a single text content block with cache_control,
            // not the legacy plain-string form. Both are accepted by the Messages
            // API; the array form is what carries the breakpoint.
            body["system"] = [
                [
                    "type": "text",
                    "text": systemPrompt,
                    "cache_control": cacheControl
                ] as [String: Any]
            ]
        }

        if thinkingEnabled {
            if usesAdaptiveThinking {
                // Adaptive thinking — model decides depth, no budget_tokens.
                // Steered via `output_config.effort` below if user set it.
                body["thinking"] = ["type": "adaptive"] as [String: Any]
            } else if let budget = configuration.thinkingBudget {
                body["thinking"] = [
                    "type": "enabled",
                    "budget_tokens": ThinkingBudget.effective(budget)
                ] as [String: Any]
            }
        }

        // Top-level `output_config.effort` (independent of thinking mode). Anthropic
        // emits this on every supported model — Opus 4.5+ accepts it alongside
        // manual thinking; Opus 4.6 / Sonnet 4.6 / Opus 4.7+ accept it alongside
        // adaptive thinking; older models (3.x, 4.0–4.4) will reject the field
        // with HTTP 400. We pass through unconditionally when the user set it
        // — better a clear API error than a silently-dropped knob.
        //
        // Per-call `thinkingEffortOverride` wins over the configuration's value.
        // Empty string normalized to nil (skip emission). Used by HTTP servers
        // that receive `reasoning_effort` per request without rebuilding the
        // provider per call.
        let effectiveEffort: String? = {
            if let override = thinkingEffortOverride, !override.isEmpty { return override }
            return configuration.thinkingEffort
        }()
        if let effort = effectiveEffort {
            body["output_config"] = ["effort": effort] as [String: Any]
        }

        if !tools.isEmpty {
            // Emit each tool, then attach cache_control to the LAST tool only.
            // Anthropic caches the full tools-array prefix at that breakpoint,
            // so a single mark covers every tool definition above it.
            var toolsArray: [[String: Any]] = tools.map { tool in
                [
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters.mapValues(\.rawValue)
                ] as [String: Any]
            }
            if !toolsArray.isEmpty {
                toolsArray[toolsArray.count - 1]["cache_control"] = cacheControl
            }
            body["tools"] = toolsArray

            // tool_choice (Anthropic shape): {"type": "auto"|"any"|"tool"|"none", "name": "..."}
            // Only emit when caller explicitly set it — otherwise Anthropic's
            // default ("auto" when tools are present) applies.
            if let toolChoice {
                body["tool_choice"] = Self.encodeAnthropicToolChoice(toolChoice)
            }
        }

        // Apply caller-provided overrides last so they win over any defaults
        // this method set (e.g. temperature, cache_control, thinking).
        // Deep-merges dict-valued keys so a user overriding one sub-key
        // doesn't wipe sibling structured fields (e.g. the system block's
        // cache_control). Arrays / scalars / strings replace outright.
        if let overrides = configuration.extraJSONOverrides {
            mergeJSONOverrides(&body, with: overrides)
        }

        return body
    }

    /// Maps an LLMMessage role to the Anthropic API role string.
    private func anthropicRole(for message: LLMMessage) -> String {
        switch message.role {
        case .user: return "user"
        case .assistant: return "assistant"
        // System and developer messages should have been extracted into the
        // top-level system field by buildRequestBody. Tool results are
        // encoded as "user" messages per Anthropic API.
        case .system, .developer, .tool: return "user"
        }
    }

    private func encodeMessage(_ message: LLMMessage) throws -> [String: Any] {
        let role = anthropicRole(for: message)

        // For assistant turns, prepend any preserved thinking blocks (with
        // signatures) from `message.continuation`. Anthropic requires these
        // to be replayed unchanged for thinking-continuity during multi-turn
        // / tool-use flows. Only applies to .assistant role since thinking
        // blocks are model output. Other roles ignore the continuation.
        let thinkingBlocks: [[String: Any]] = {
            guard message.role == .assistant,
                  let blocks = message.continuation?.anthropicThinkingBlocks,
                  !blocks.isEmpty else { return [] }
            return blocks.map { block in
                [
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature
                ] as [String: Any]
            }
        }()

        switch message.content {
        case .text(let text):
            // If there are images OR thinking blocks to prepend, encode as
            // a content-blocks array. Empty text blocks are silently skipped
            // — Anthropic rejects `{"type":"text","text":""}` entries.
            if !thinkingBlocks.isEmpty || !(message.images ?? []).isEmpty || !(message.documents ?? []).isEmpty {
                var blocks: [[String: Any]] = thinkingBlocks
                if let images = message.images, !images.isEmpty {
                    blocks.append(contentsOf: images.map { image in
                        [
                            "type": "image",
                            "source": [
                                "type": "base64",
                                "media_type": image.mimeType,
                                "data": image.data.base64EncodedString()
                            ]
                        ] as [String: Any]
                    })
                }
                if let documents = message.documents, !documents.isEmpty {
                    blocks.append(contentsOf: documents.map { document in
                        [
                            "type": "document",
                            "source": [
                                "type": "base64",
                                "media_type": document.mimeType,
                                "data": document.data.base64EncodedString()
                            ]
                        ] as [String: Any]
                    })
                }
                if !text.isEmpty {
                    blocks.append(["type": "text", "text": text])
                }
                return [
                    "role": role,
                    "content": blocks
                ]
            }
            // Plain-text path. Anthropic rejects `"content": ""` with HTTP
            // 400 — substitute a single space so callers passing through
            // `.assistant(from: LLMResponse(text: nil, toolCalls: []))`
            // (the degenerate "model returned nothing" shape) don't crash
            // the request. The model didn't say " ", but the wire needs
            // *some* content; this is the documented Anthropic workaround.
            return [
                "role": role,
                "content": text.isEmpty ? " " : text
            ]
        case .toolCalls(let calls):
            var content: [[String: Any]] = thinkingBlocks
            try content.append(contentsOf: calls.map { call in
                [
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": try Self.parseToolArguments(call.arguments)
                ] as [String: Any]
            })
            return ["role": "assistant", "content": content]
        case .mixed(let text, let calls):
            var content: [[String: Any]] = thinkingBlocks
            // Skip empty text block — Anthropic rejects empty `{"type":"text","text":""}`
            // entries even when other blocks are present.
            if !text.isEmpty {
                content.append(["type": "text", "text": text])
            }
            try content.append(contentsOf: calls.map { call in
                [
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": try Self.parseToolArguments(call.arguments)
                ] as [String: Any]
            })
            return ["role": "assistant", "content": content]
        case .toolResult(let toolCallID, let resultContent):
            return [
                "role": "user",
                "content": [
                    [
                        "type": "tool_result",
                        "tool_use_id": toolCallID,
                        "content": resultContent
                    ]
                ]
            ]
        }
    }

    /// Merges consecutive messages that share the same role into a single message.
    /// The Anthropic API requires strict user/assistant alternation. This handles cases like
    /// multiple tool_result messages (each encoded as role "user") following an assistant tool_use.
    private static func mergeConsecutiveSameRole(_ messages: [[String: Any]]) -> [[String: Any]] {
        guard !messages.isEmpty else { return messages }
        var result: [[String: Any]] = []
        for message in messages {
            guard let role = message["role"] as? String else {
                result.append(message)
                continue
            }
            if let lastRole = result.last?["role"] as? String, lastRole == role {
                // Same role as previous — merge content into the previous message.
                let prevContent = result[result.count - 1]["content"]
                let curContent = message["content"]
                let merged = mergeContent(prevContent, curContent)
                result[result.count - 1]["content"] = merged
            } else {
                result.append(message)
            }
        }
        return result
    }

    /// Merges two Anthropic message content values into a single content-blocks array.
    /// Handles both string content (`"hello"`) and array content (`[{type: "tool_result", ...}]`).
    private static func mergeContent(_ a: Any?, _ b: Any?) -> Any {
        let blocksA = contentToBlocks(a)
        let blocksB = contentToBlocks(b)
        return blocksA + blocksB
    }

    /// Normalizes Anthropic message content to an array of content blocks.
    private static func contentToBlocks(_ content: Any?) -> [[String: Any]] {
        if let blocks = content as? [[String: Any]] {
            return blocks
        }
        if let text = content as? String {
            return [["type": "text", "text": text]]
        }
        return []
    }

    /// Parses a JSON argument string back into a Foundation object for the API request body.
    private static func parseToolArguments(_ jsonString: String) throws -> Any {
        guard let data = jsonString.data(using: .utf8) else {
            throw LLMProviderError.malformedResponse(detail: "tool arguments not valid UTF-8: \(jsonString.prefix(200))")
        }
        do {
            return try JSONSerialization.jsonObject(with: data)
        } catch {
            throw LLMProviderError.malformedResponse(detail: "tool arguments JSON parse failed: \(error.localizedDescription), input: \(jsonString.prefix(200))")
        }
    }

    func parseResponse(data: Data) throws -> LLMResponse {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            let preview = String(data: data.prefix(500), encoding: .utf8) ?? "(non-utf8, \(data.count) bytes)"
            logger.error("Response is not a JSON object: \(preview, privacy: .public)")
            throw LLMProviderError.malformedResponse(detail: "not a JSON object: \(preview)")
        }
        guard let contentBlocks = json["content"] as? [[String: Any]] else {
            let keys = json.keys.sorted().joined(separator: ", ")
            let preview = String(data: data.prefix(500), encoding: .utf8) ?? "(\(data.count) bytes)"
            logger.error("Missing content blocks in response. Keys: \(keys, privacy: .public) Body: \(preview, privacy: .public)")
            throw LLMProviderError.malformedResponse(detail: "missing content blocks, keys: [\(keys)], body: \(preview)")
        }

        // Parse token usage.
        //
        // Normalization note: Anthropic's wire `input_tokens` reports ONLY the
        // uncached portion of the prompt (cache_read and cache_creation are
        // additive). OpenAI and Gemini, by contrast, report `prompt_tokens` /
        // `promptTokenCount` as the FULL prompt with cache fields as subsets.
        // To keep `TokenUsage.inputTokens` semantically consistent across
        // providers ("total prompt input tokens"), we add the cache portions
        // into `inputTokens` here. Consumers can compute cache savings as
        // `cacheReadTokens / inputTokens` uniformly.
        var tokenUsage: TokenUsage?
        if let usage = json["usage"] as? [String: Any] {
            let uncached = usage["input_tokens"] as? Int ?? 0
            let output = usage["output_tokens"] as? Int ?? 0
            let cacheRead = usage["cache_read_input_tokens"] as? Int ?? 0
            let cacheCreation = usage["cache_creation_input_tokens"] as? Int ?? 0
            let totalInput = uncached + cacheRead + cacheCreation
            let rawUsage = TokenUsage.serializeRawUsage(usage)
            tokenUsage = TokenUsage(
                inputTokens: totalInput,
                outputTokens: output,
                cacheReadTokens: cacheRead,
                cacheWriteTokens: cacheCreation,
                rawUsage: rawUsage
            )
            if cacheRead > 0 || cacheCreation > 0 {
                logger.info("Cache: read=\(cacheRead) created=\(cacheCreation) uncached=\(uncached) total=\(totalInput)")
            }
        }

        var text: String?
        var toolCalls: [LLMToolCall] = []
        var thinkingBlocks: [AnthropicThinkingBlock] = []
        var reasoning: String?

        for block in contentBlocks {
            guard let type = block["type"] as? String else { continue }
            switch type {
            case "thinking":
                // Capture BOTH the thinking text (for display/inspection via
                // LLMResponse.reasoning) AND the signature (for multi-turn
                // continuity replay via LLMResponse.continuation). Anthropic
                // requires the signature to be replayed unchanged.
                let thinking = block["thinking"] as? String ?? ""
                let signature = block["signature"] as? String ?? ""
                logger.debug("Thinking block (\(thinking.count) chars, signature \(signature.isEmpty ? "missing" : "present"))")
                if !signature.isEmpty {
                    thinkingBlocks.append(AnthropicThinkingBlock(thinking: thinking, signature: signature))
                }
                // Last thinking block's text feeds reasoning for display.
                if !thinking.isEmpty {
                    reasoning = thinking
                }
            case "text":
                text = block["text"] as? String
            case "tool_use":
                guard let id = block["id"] as? String,
                      let name = block["name"] as? String,
                      let input = block["input"]
                else { continue }
                let argData = try JSONSerialization.data(withJSONObject: input)
                let argString = String(data: argData, encoding: .utf8) ?? "{}"
                toolCalls.append(LLMToolCall(id: id, name: name, arguments: argString))
            default:
                break
            }
        }

        let continuation: ProviderContinuation? = thinkingBlocks.isEmpty
            ? nil
            : ProviderContinuation(anthropicThinkingBlocks: thinkingBlocks)

        return LLMResponse(
            text: text?.isEmpty == true ? nil : text,
            toolCalls: toolCalls,
            reasoning: reasoning,
            usage: tokenUsage,
            continuation: continuation
        )
    }

    /// Translates the unified `LLMToolChoice` to Anthropic's `tool_choice`
    /// wire shape. Anthropic uses "any" for the "require some tool" mode
    /// (not "required" as OpenAI does), and a `name` field on the `"tool"`
    /// case to pin a specific tool.
    private static func encodeAnthropicToolChoice(_ choice: LLMToolChoice) -> [String: Any] {
        switch choice {
        case .auto:
            return ["type": "auto"]
        case .required:
            return ["type": "any"]
        case .textOnly:
            return ["type": "none"]
        case .specific(let name):
            return ["type": "tool", "name": name]
        }
    }
}
