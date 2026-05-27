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
        maxOutputTokensOverride: Int? = nil
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

        let body = buildRequestBody(messages: messages, tools: tools, maxOutputTokensOverride: maxOutputTokensOverride)
        // .sortedKeys keeps wire bytes stable across requests so OpenAI's
        // automatic prefix cache (>=1024-token prefix match) keeps hitting.
        let requestData = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        request.httpBody = requestData

        logger.debug("Request: POST \(url.absoluteString, privacy: .public) model=\(configuration.model, privacy: .public)")
        if verboseLogging {
            LLMRequestLogger.logRequest(label: "OpenAI", url: url, model: configuration.model, body: body, rawData: requestData)
        }

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw LLMProviderError.invalidResponse
        }

        if verboseLogging {
            LLMRequestLogger.logResponse(label: "OpenAI", statusCode: httpResponse.statusCode, data: data)
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let responseBody = String(data: data, encoding: .utf8) ?? "unknown"
            logger.error("HTTP \(httpResponse.statusCode, privacy: .public) from \(url.absoluteString, privacy: .public) body=\(responseBody, privacy: .public)")
            throw LLMProviderError.httpError(statusCode: httpResponse.statusCode, body: responseBody, url: url)
        }

        return try parseResponse(data: data)
    }

    func buildRequestBody(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        maxOutputTokensOverride: Int? = nil
    ) -> [String: Any] {
        // OpenAI API requires system messages at the start of the conversation.
        // Extract any system messages from arbitrary positions (e.g. per-turn task
        // context appended by AgentActor) and consolidate them into a single leading
        // system message, followed by the remaining non-system messages in order.
        var systemParts: [String] = []
        var nonSystemMessages: [[String: Any]] = []
        for message in messages {
            if message.role == .system, case .text(let text) = message.content {
                systemParts.append(text)
            } else {
                nonSystemMessages.append(encodeMessage(message))
            }
        }
        var orderedMessages: [[String: Any]] = []
        if !systemParts.isEmpty {
            orderedMessages.append(["role": "system", "content": systemParts.joined(separator: "\n\n")])
        }
        orderedMessages.append(contentsOf: nonSystemMessages)

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
        if !configuration.useDefaultTemperature {
            body["temperature"] = configuration.temperature
        }

        // Alibaba Cloud thinking support (uses different keys than Anthropic).
        if provider.apiType == .alibabaCloud,
           let budget = configuration.thinkingBudget, budget > 0 {
            body["enable_thinking"] = true
            body["thinking_budget"] = max(budget, 1024)
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
            if parallelToolCalls {
                body["parallel_tool_calls"] = true
            }
            // Alibaba Cloud defaults parallel_tool_calls to false; enable explicitly.
            if provider.apiType == .alibabaCloud {
                body["parallel_tool_calls"] = true
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
            // If there are images, encode as content parts array (multimodal)
            if let images = message.images, !images.isEmpty {
                var parts: [[String: Any]] = images.map { image in
                    [
                        "type": "image_url",
                        "image_url": [
                            "url": "data:\(image.mimeType);base64,\(image.data.base64EncodedString())"
                        ]
                    ] as [String: Any]
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

        var text = message["content"] as? String
        let reasoningContent = message["reasoning_content"] as? String
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
                    } else {
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
            usage: tokenUsage
        )
    }

}
