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
    /// Not `private`: a test asserts that every provider the factory builds holds the SAME
    /// coordinator, which is the entire point of the default below. The struct is internal, so
    /// widening this does not widen the package's API.
    let auth: CodexAuthCoordinator
    private let verboseLogging: Bool
    private let session: URLSession
    private let clientVersion: String

    /// The Codex backend rejects `/models` without a `client_version`, and may police it on
    /// `/responses` too. Hand-maintained and undocumented; `1.0.0` is accepted as of 2026-09-16
    /// even though the shipped CLI reports 0.154.0.
    static let defaultClientVersion = "1.0.0"

    /// - Parameter auth: Defaults to the process-wide coordinator for the `codex` CLI's credential
    ///   file. **That default IS the single-flight protection** — one provider is built per agent
    ///   role, and a per-provider coordinator lets five roles refresh at once and race each other
    ///   writing `auth.json`. Pass one only from a test, with its own store and transport.
    init(
        configuration: ModelConfiguration,
        provider: ModelProvider,
        auth: CodexAuthCoordinator = .sharedForDefaultStore,
        verboseLogging: Bool = false,
        clientVersion: String = CodexResponsesProvider.defaultClientVersion,
        session: URLSession = llmURLSession
    ) {
        self.configuration = configuration
        self.provider = provider
        self.auth = auth
        self.verboseLogging = verboseLogging
        self.clientVersion = clientVersion
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
            overrides: overrides)
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
        overrides: LLMCallOverrides
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
        // NO output-token cap is sent, and this builder deliberately takes NO cap argument —
        // there is nothing half-wired here for a later reader to "finish".
        //
        // The endpoint rejects the field outright:
        //     400 {"detail":"Unsupported parameter: max_output_tokens"}
        // (verified against the live backend 2026-09-17). So neither the user's configured limit
        // nor `LLMCallOverrides.maxOutputTokens` — documented as "honored by every provider" —
        // can be honored here, and the parameters that used to carry them were removed rather
        // than left accepted-and-ignored, because the alternative is a 400 on every single call.
        //
        // Worth stating plainly because the code before that LOOKED like it sent a cap and only
        // worked by accident: it read the model's catalog ceiling, no Codex model publishes one,
        // so the field was never emitted. Making it honor the configuration — the obviously
        // correct change — broke every request. `neverSendsAnOutputCap` is the regression guard.
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
                // Only function_call items need registering: text accumulates from its own deltas.
                guard let item = event["item"] as? [String: Any],
                      item["type"] as? String == "function_call" else { continue }
                // Keyed by the ITEM's own id — the SAME identity the delta events carry as
                // `item_id`, derived by the SAME function. Keying this side by `output_index` and
                // the other by `item_id` is what silently produced two different keys once before:
                // the arguments never joined their call and every tool call arrived as `{}`.
                guard let itemID = Self.itemKey(event) else {
                    logger.error("""
                        Codex stream: a function_call item carries no id; dropping the call rather \
                        than filing it under its output_index, which is a different identity
                        """)
                    continue
                }
                // `call_id` is what a later `function_call_output` must quote; `id` is the
                // stream's own handle for the item and is NOT interchangeable.
                let callID = item["call_id"] as? String ?? itemID
                // `callOrder` is the emission order and `callByItem` the contents; appending
                // unconditionally would let a repeated `added` for one item emit the call twice.
                if callByItem[itemID] == nil { callOrder.append(itemID) }
                callByItem[itemID] = (id: callID, name: item["name"] as? String ?? "")

            case "response.output_item.done":
                // The terminal item is SELF-CONTAINED — id, `call_id`, `name` and the COMPLETE
                // `arguments` string in one event (verified against live captures 2026-09-17). So
                // it needs no join at all, and is preferred over the accumulation for exactly the
                // reason `function_call_arguments.done` already is: deltas that never arrived — or
                // never matched their call — otherwise leave `{}`, which is the precise symptom
                // the 0.0.187 keying defect produced on every call.
                //
                // Every field falls back to what `added` already recorded rather than to a
                // default. A terminal item that omits `call_id` must not downgrade a good one to
                // the item handle: that id is what correlates the result, so preserving beats
                // re-deriving — the same rule the model-override sheets follow.
                guard let item = event["item"] as? [String: Any],
                      item["type"] as? String == "function_call",
                      let itemID = Self.itemKey(event) else { continue }
                if callByItem[itemID] == nil { callOrder.append(itemID) }
                callByItem[itemID] = (
                    id: item["call_id"] as? String ?? callByItem[itemID]?.id ?? itemID,
                    name: item["name"] as? String ?? callByItem[itemID]?.name ?? "")
                if let arguments = item["arguments"] as? String, !arguments.isEmpty {
                    argumentsByItem[itemID] = arguments
                }

            case "response.output_text.delta":
                guard let delta = event["delta"] as? String else { continue }
                // Text is the one case position is allowed to group, and the reason is that there
                // is nothing here to cross: a text bucket is only accumulated and ordered, never
                // matched against an item some OTHER event registered. Dropping the delta instead
                // would silently shorten the answer with nothing downstream able to notice. The
                // prefix keeps the positional space visibly disjoint from the id space.
                let itemID = Self.itemKey(event)
                    ?? (event["output_index"] as? Int).map { "position:\($0)" }
                    ?? "position:unknown"
                if textByItem[itemID] == nil { textOrder.append(itemID) }
                textByItem[itemID, default: ""] += delta

            case "response.function_call_arguments.delta":
                // The payload is checked FIRST so a delta-less event is never reported as an
                // unkeyed one: two different defects must not share one log line.
                guard let delta = event["delta"] as? String else { continue }
                guard let itemID = Self.itemKey(event) else {
                    Self.logUnkeyedEvent(type, event)
                    continue
                }
                argumentsByItem[itemID, default: ""] += delta

            case "response.function_call_arguments.done":
                // The terminal event carries the whole argument string. Preferred over the
                // accumulation when present — a dropped delta would otherwise yield invalid JSON.
                guard let arguments = event["arguments"] as? String else { continue }
                guard let itemID = Self.itemKey(event) else {
                    Self.logUnkeyedEvent(type, event)
                    continue
                }
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

        // Arguments that never met a call. Silent before: the shipped defect produced exactly this
        // state on EVERY call and the parser reported a healthy response with `{}` arguments.
        for orphan in argumentsByItem.keys.sorted() where callByItem[orphan] == nil {
            logger.error("""
                Codex stream: argument deltas for item \(orphan, privacy: .public) never matched a \
                function_call item — the call's arguments were dropped
                """)
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

    /// The item an event is about, by the item's OWN id — the one identity both event shapes carry:
    /// deltas spell it `item_id`, `response.output_item.added`/`.done` nest it as `item.id`.
    ///
    /// `output_index` is deliberately NOT a fallback tier. It is the item's POSITION, a DIFFERENT
    /// identity, and the two event shapes prefer different fields — so a positional tier lets one
    /// side key by id while the other keys by position, and the arguments never join their call.
    /// That is the defect that shipped in 0.0.187 with every tool call arriving `{}`; one shared
    /// function does not fix it, because a shared function still takes different branches on
    /// different event shapes. Removing the tier is what makes the crossing unrepresentable.
    ///
    /// `item_id` is checked FIRST deliberately: if an item event ever carried both spellings with
    /// different values, the deltas would use `item_id`, so preferring it keeps the sides agreeing.
    ///
    /// When the id is absent the honest answer is "unknown item", and the caller drops the event
    /// loudly rather than filing it where it may meet the other half by luck.
    private static func itemKey(_ event: [String: Any]) -> String? {
        if let itemID = event["item_id"] as? String { return itemID }
        if let item = event["item"] as? [String: Any], let itemID = item["id"] as? String {
            return itemID
        }
        return nil
    }

    /// An event that could not name its item, recorded rather than swallowed. Item ids are
    /// per-stream handles, not account-linked — unlike `chatgpt-account-id`, which is never logged.
    private static func logUnkeyedEvent(_ type: String, _ event: [String: Any]) {
        let position = (event["output_index"] as? Int).map(String.init) ?? "none"
        logger.error("""
            Codex stream: \(type, privacy: .public) carries no item id \
            (output_index \(position, privacy: .public)) — dropped rather than keyed by position
            """)
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
