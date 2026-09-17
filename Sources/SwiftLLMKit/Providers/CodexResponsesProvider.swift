import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "CodexResponses")

/// Talks to the Codex backend's Responses endpoint using a ChatGPT-subscription credential.
///
/// **Why this exists as a separate provider.** A ChatGPT OAuth token cannot reach the OpenAI
/// platform API at all — it is scope-gated (`Missing scopes: model.request` on
/// `/v1/chat/completions`, `api.responses.write` on `/v1/responses`), so there is no header or
/// base-URL arrangement that lets `OpenAICompatibleProvider` carry it. The only endpoint the token
/// opens is `chatgpt.com/backend-api/codex/responses`, which speaks the **Responses** shape:
/// `instructions` + `input` items rather than `messages`, `function_call` / `function_call_output`
/// rather than `tool_calls` / `role: tool`, and SSE event names of its own.
///
/// **Stateless.** `store: false` and the full history on every turn, which is what the endpoint
/// expects and what the rest of this library already does.
///
/// The translation and parsing below are `static` and pure so they can be tested against recorded
/// fixtures without a network or a credential — the request path is the only part that needs either.
struct CodexResponsesProvider: LLMProvider {
    private let configuration: ModelConfiguration
    private let provider: ModelProvider
    private let auth: CodexAuthCoordinator
    private let verboseLogging: Bool
    private let session: URLSession
    private let clientVersion: String
    private let modelMaxOutputTokens: Int?

    /// The Codex backend rejects `/models` without a `client_version`, and may police it on
    /// `/responses` too. Hand-maintained and undocumented; `1.0.0` is accepted as of 2026-09-16
    /// even though the shipped CLI reports 0.154.0.
    static let defaultClientVersion = "1.0.0"

    init(
        configuration: ModelConfiguration,
        provider: ModelProvider,
        auth: CodexAuthCoordinator = CodexAuthCoordinator(),
        verboseLogging: Bool = false,
        clientVersion: String = CodexResponsesProvider.defaultClientVersion,
        modelMaxOutputTokens: Int? = nil,
        session: URLSession = llmURLSession
    ) {
        self.configuration = configuration
        self.provider = provider
        self.auth = auth
        self.verboseLogging = verboseLogging
        self.clientVersion = clientVersion
        self.modelMaxOutputTokens = modelMaxOutputTokens
        self.session = session
    }

    // MARK: - Sending

    func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides = LLMCallOverrides()
    ) async throws -> LLMResponse {
        let tokens = try await auth.validTokens()

        var components = URLComponents(
            url: provider.endpoint.appendingPathComponent("responses"), resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "client_version", value: clientVersion)]
        guard let url = components?.url else {
            throw LLMProviderError.invalidRequest(
                detail: "Could not form a Codex responses URL from \(provider.endpoint)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue("Bearer \(tokens.accessToken)", forHTTPHeaderField: "Authorization")
        // Routes the request to the ChatGPT account whose allowance it is spent from. Account-linked
        // — never log this value.
        request.setValue(tokens.accountId, forHTTPHeaderField: "chatgpt-account-id")
        request.setValue("responses=v1", forHTTPHeaderField: "OpenAI-Beta")
        request.setValue("codex_cli_rs/\(clientVersion)", forHTTPHeaderField: "User-Agent")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")

        let body = Self.buildRequestBody(
            model: configuration.model,
            messages: messages,
            tools: tools,
            overrides: overrides,
            maxOutputTokens: modelMaxOutputTokens)
        guard JSONSerialization.isValidJSONObject(body) else {
            throw LLMProviderError.invalidRequest(detail: "Codex request body is not valid JSON")
        }
        let rawBody = try JSONSerialization.data(withJSONObject: body)
        request.httpBody = rawBody

        let token = LLMRequestLogger.logRequest(
            label: "Codex", url: url, model: configuration.model, body: body, rawData: rawBody)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw LLMProviderError.invalidResponse }
        let text = String(data: data, encoding: .utf8) ?? ""
        LLMRequestLogger.logResponse(
            label: "Codex", statusCode: http.statusCode, data: data, for: token)

        guard http.statusCode == 200 else {
            throw LLMProviderError.httpError(
                statusCode: http.statusCode, body: text, url: url,
                retryAfter: LLMProviderError.parseRetryAfter(http.value(forHTTPHeaderField: "Retry-After")))
        }
        // Recorded on EVERY successful call, not just when something goes wrong: these headers are
        // how the window can be shown filling rather than discovered at the ceiling.
        let window = CodexUsageWindow.from(headers: Self.limitHeaders(http))
        CodexUsageMonitor.record(window)
        if verboseLogging, let percent = window.primaryPercentUsed {
            logger.debug("Codex window \(percent, privacy: .public)% used (\(window.limitName ?? "unnamed", privacy: .public))")
        }
        return try Self.parseStream(text)
    }

    /// The `x-codex-*` window headers, which arrive on ORDINARY successful responses — so consumption
    /// can be shown filling rather than discovered by hitting the ceiling.
    static func limitHeaders(_ response: HTTPURLResponse) -> [String: String] {
        var out: [String: String] = [:]
        for (key, value) in response.allHeaderFields {
            guard let key = key as? String, key.lowercased().hasPrefix("x-codex-"),
                  let value = value as? String else { continue }
            out[key.lowercased()] = value
        }
        return out
    }

    // MARK: - Request translation

    static func buildRequestBody(
        model: String,
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides,
        maxOutputTokens: Int? = nil
    ) -> [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "input": buildInput(messages),
            // Stateless: the endpoint keeps nothing between turns, and we send the whole history.
            "store": false,
            "stream": true
        ]
        if let instructions = buildInstructions(messages) { body["instructions"] = instructions }
        if !tools.isEmpty {
            body["tools"] = encodeTools(tools)
            body["tool_choice"] = encodeToolChoice(overrides.toolChoice)
        }
        if let effort = overrides.reasoningEffort {
            // `summary: auto` is what makes the model emit reasoning_summary deltas; without it a
            // reasoning turn streams nothing until the answer lands.
            body["reasoning"] = ["effort": effort, "summary": "auto"]
        }
        if let maxOutputTokens, maxOutputTokens > 0 { body["max_output_tokens"] = maxOutputTokens }
        return body
    }

    /// System and developer messages, folded into the one `instructions` string the Responses shape
    /// provides. Order is preserved; there is no separate developer channel here.
    ///
    /// Note that NO identity prefix is prepended. The "You are Codex, based on GPT-5…" string is
    /// widely reported as a hard OAuth gate; a request without it returns 200 (verified 2026-09-16),
    /// and prepending it would put a second, conflicting identity in front of every role prompt.
    static func buildInstructions(_ messages: [LLMMessage]) -> String? {
        let parts: [String] = messages.compactMap { message in
            guard message.role == .system || message.role == .developer else { return nil }
            guard case .text(let text) = message.content else { return nil }
            return text
        }
        return parts.isEmpty ? nil : parts.joined(separator: "\n\n")
    }

    /// Conversation history as Responses `input` items.
    ///
    /// Tool traffic is the part that differs most from chat/completions: an assistant tool call is a
    /// TOP-LEVEL `function_call` item rather than a field on a message, and its result is a
    /// top-level `function_call_output` correlated by `call_id` rather than a `role: tool` message.
    static func buildInput(_ messages: [LLMMessage]) -> [[String: Any]] {
        var items: [[String: Any]] = []
        for message in messages {
            switch (message.role, message.content) {
            case (.system, _), (.developer, _):
                continue    // folded into `instructions`

            case (_, .text(let text)):
                items.append(textItem(role: message.role, text: text))

            case (_, .toolCalls(let calls)):
                items.append(contentsOf: calls.map(functionCallItem))

            case (_, .mixed(let text, let calls)):
                // The assistant's prose precedes the calls it made, preserving turn order.
                if !text.isEmpty { items.append(textItem(role: message.role, text: text)) }
                items.append(contentsOf: calls.map(functionCallItem))

            case (_, .toolResult(let callID, let content)):
                items.append([
                    "type": "function_call_output",
                    "call_id": callID,
                    "output": content
                ])
            }
        }
        return items
    }

    private static func textItem(role: LLMMessage.Role, text: String) -> [String: Any] {
        // The content-part type is role-dependent: assistant text is `output_text`, everything the
        // caller supplies is `input_text`. Sending the wrong one is a 400.
        let partType = role == .assistant ? "output_text" : "input_text"
        return [
            "type": "message",
            "role": role == .assistant ? "assistant" : "user",
            "content": [["type": partType, "text": text]]
        ]
    }

    private static func functionCallItem(_ call: LLMToolCall) -> [String: Any] {
        [
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            // Arguments travel as a JSON STRING, exactly as the model emitted them.
            "arguments": call.arguments
        ]
    }

    /// Tools are flat here — `{type, name, description, parameters}` — not nested under a
    /// `function` key the way chat/completions nests them.
    static func encodeTools(_ tools: [LLMToolDefinition]) -> [[String: Any]] {
        tools.map { tool in
            var encoded: [String: Any] = [
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.mapValues(\.rawValue)
            ]
            if let strict = tool.strict { encoded["strict"] = strict }
            return encoded
        }
    }

    static func encodeToolChoice(_ choice: LLMToolChoice?) -> Any {
        switch choice {
        case .none, .some(.auto): return "auto"
        case .some(.required): return "required"
        case .some(.textOnly): return "none"
        case .some(.specific(let name)): return ["type": "function", "name": name]
        }
    }

    // MARK: - Response parsing

    /// Reassembles one `LLMResponse` from a complete SSE stream.
    ///
    /// Text and tool-call arguments both arrive as deltas keyed by `item_id`, so both are
    /// accumulated per item and joined at the end. Reasoning summaries are collected separately into
    /// `reasoning` — they are display material, not part of the answer.
    static func parseStream(_ sse: String) throws -> LLMResponse {
        var textByItem: [String: String] = [:]
        var textOrder: [String] = []
        var argumentsByItem: [String: String] = [:]
        var callByItem: [String: (id: String, name: String)] = [:]
        var callOrder: [String] = []
        var reasoning = ""
        var usage: TokenUsage?
        var finishReason: String?
        var failure: String?

        for line in sse.split(separator: "\n", omittingEmptySubsequences: true) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("data:") else { continue }
            let payload = trimmed.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
            guard payload != "[DONE]", let data = payload.data(using: .utf8),
                  let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let type = event["type"] as? String else { continue }

            switch type {
            case "response.output_item.added":
                // Keyed by the ITEM's own id, which is what the delta events carry as `item_id`.
                // A real stream sends BOTH `item_id` and `output_index` on deltas, so keying this
                // side by `output_index` and the other by `item_id` silently produced two different
                // keys — the arguments never joined their call and every tool call arrived as `{}`.
                guard let item = event["item"] as? [String: Any],
                      let itemID = (item["id"] as? String) ?? event["output_index"].map({ "\($0)" })
                else { continue }
                if item["type"] as? String == "function_call" {
                    let name = item["name"] as? String ?? ""
                    // `call_id` is what a later `function_call_output` must quote; `id` is the
                    // stream's own handle for the item and is NOT interchangeable.
                    let callID = item["call_id"] as? String ?? item["id"] as? String ?? itemID
                    callByItem[itemID] = (id: callID, name: name)
                    callOrder.append(itemID)
                }

            case "response.output_text.delta":
                guard let itemID = Self.itemKey(event), let delta = event["delta"] as? String
                else { continue }
                if textByItem[itemID] == nil { textOrder.append(itemID) }
                textByItem[itemID, default: ""] += delta

            case "response.function_call_arguments.delta":
                guard let itemID = Self.itemKey(event), let delta = event["delta"] as? String
                else { continue }
                argumentsByItem[itemID, default: ""] += delta

            case "response.function_call_arguments.done":
                // The terminal event carries the whole argument string. Preferred over the
                // accumulation when present — a dropped delta would otherwise yield invalid JSON.
                guard let itemID = Self.itemKey(event), let arguments = event["arguments"] as? String
                else { continue }
                argumentsByItem[itemID] = arguments

            case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
                reasoning += event["delta"] as? String ?? ""

            case "response.completed":
                finishReason = "completed"
                if let response = event["response"] as? [String: Any] {
                    usage = Self.parseUsage(response["usage"] as? [String: Any])
                }

            case "response.incomplete":
                let response = event["response"] as? [String: Any]
                // Verbatim, per the finishReason convention: each provider spells this its own way
                // and `hitOutputTokenLimit` owns the translation table.
                finishReason = ((response?["incomplete_details"] as? [String: Any])?["reason"] as? String)
                    ?? "incomplete"
                usage = Self.parseUsage(response?["usage"] as? [String: Any]) ?? usage

            case "response.failed":
                finishReason = "failed"
                let response = event["response"] as? [String: Any]
                failure = (response?["error"] as? [String: Any])?["message"] as? String ?? "unknown error"

            default:
                continue
            }
        }

        if let failure {
            throw LLMProviderError.malformedResponse(detail: "Codex stream failed: \(failure)")
        }

        let text = textOrder.compactMap { textByItem[$0] }.joined()
        let toolCalls: [LLMToolCall] = callOrder.compactMap { itemID in
            guard let call = callByItem[itemID] else { return nil }
            return LLMToolCall(
                id: call.id, name: call.name,
                // An argument-less call is `{}`, not an empty string — the latter is not valid JSON
                // and every consumer of `arguments` parses it.
                arguments: argumentsByItem[itemID].flatMap { $0.isEmpty ? nil : $0 } ?? "{}")
        }
        return LLMResponse(
            text: text.isEmpty ? nil : text,
            toolCalls: toolCalls,
            reasoning: reasoning.isEmpty ? nil : reasoning,
            usage: usage,
            finishReason: finishReason)
    }

    /// Events key their item by `item_id`; a few carry only `output_index`. Either identifies the
    /// item within one stream, which is all the accumulation needs.
    private static func itemKey(_ event: [String: Any]) -> String? {
        if let itemID = event["item_id"] as? String { return itemID }
        if let index = event["output_index"] { return "\(index)" }
        return nil
    }

    static func parseUsage(_ usage: [String: Any]?) -> TokenUsage? {
        guard let usage else { return nil }
        let inputDetails = usage["input_tokens_details"] as? [String: Any]
        let outputDetails = usage["output_tokens_details"] as? [String: Any]
        return TokenUsage(
            inputTokens: usage["input_tokens"] as? Int ?? 0,
            outputTokens: usage["output_tokens"] as? Int ?? 0,
            reasoningTokens: outputDetails?["reasoning_tokens"] as? Int ?? 0,
            cacheReadTokens: inputDetails?["cached_tokens"] as? Int ?? 0,
            cacheWriteTokens: inputDetails?["cache_write_tokens"] as? Int ?? 0)
    }
}
