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
    /// Per-model request-forming knobs, resolved from the catalog at construction — the same path
    /// the OpenAI-compatible provider reads them by, never re-derived from `apiType` in here.
    private let behaviorFlags: BehaviorFlags
    /// The model's reasoning-effort support. Read fail-OPEN here (sent unless KNOWN unsupported),
    /// unlike the chat/completions provider: every model this endpoint serves is a reasoning model
    /// reached through the Responses API, so an unrecorded ladder is far more likely a model the
    /// listing has not been seeded for than a model that rejects the field.
    private let reasoningEffortSupport: EffortSupport?
    /// Capabilities gating the knobs whose wrong emission is an HTTP 400 (structured output,
    /// tool_choice options).
    private let modelCapabilities: ModelCapabilities
    /// The `prompt_cache_key` every request from this instance carries — one per provider, and a
    /// provider is built per agent role, so one per conversation. It is the routing hint for the
    /// endpoint's prefix cache: requests sharing a key land on the shard holding their prefix.
    /// Measured 2026-09-19 with a ~5k-token identical prefix: with the key the SECOND call
    /// reported 4,864 cached tokens; without it, only the third. (A ~1.2k prefix cached nothing
    /// either way — the threshold is higher than the nominal 1,024 counts here.)
    private let promptCacheKey: String

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
        session: URLSession = llmURLSession,
        behaviorFlags: BehaviorFlags = BehaviorFlags(),
        reasoningEffortSupport: EffortSupport? = nil,
        modelCapabilities: ModelCapabilities = ModelCapabilities(),
        promptCacheKey: String = UUID().uuidString
    ) {
        self.configuration = configuration
        self.provider = provider
        self.auth = auth
        self.verboseLogging = verboseLogging
        self.clientVersion = clientVersion
        self.session = session
        self.behaviorFlags = behaviorFlags
        self.reasoningEffortSupport = reasoningEffortSupport
        self.modelCapabilities = modelCapabilities
        self.promptCacheKey = promptCacheKey
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
            configuration: configuration,
            messages: messages,
            tools: tools,
            overrides: overrides,
            behaviorFlags: behaviorFlags,
            reasoningEffortSupport: reasoningEffortSupport,
            modelCapabilities: modelCapabilities,
            promptCacheKey: promptCacheKey)
        // A non-finite Double (a caller-supplied temperature) reaching JSONSerialization raises an
        // NSException that `try` cannot convert — pre-flight it into a normal throw.
        guard JSONSerialization.isValidJSONObject(body) else {
            throw LLMProviderError.invalidRequest(detail: "Codex request body is not valid JSON")
        }
        // .sortedKeys keeps the wire bytes stable across turns, the precondition for the
        // endpoint's prefix cache to hit at all.
        let rawBody = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
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

    /// Convenience for tests that only care about the message translation and one or two knobs:
    /// no catalog data, so every gate sits at its "nothing known" default.
    static func buildRequestBody(
        model: String,
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides
    ) -> [String: Any] {
        buildRequestBody(
            configuration: ModelConfiguration(
                name: "codex:\(model)", providerID: BuiltInProviders.ID.codexChatGPT, modelID: model,
                temperature: nil),
            messages: messages, tools: tools, overrides: overrides)
    }

    /// The complete request body. `static` and pure so it can be asserted against without a
    /// network or a credential.
    ///
    /// What is emitted, and why each gate is the way it is:
    /// - `temperature` / `top_p` — sent when asked for, unless `mustNeverSendTemperatureParam`.
    ///   The endpoint rejects both (verified live 2026-09-19); they are sent anyway so a probe
    ///   measures that rejection and the flag is DERIVED from it, exactly as for every
    ///   OpenAI-compatible model. Dropping them silently recorded "accepts temperature" for a
    ///   model that 400s on it.
    /// - `reasoning.effort` — the per-call override, else an explicit reasoning-off as `none`,
    ///   else the configuration's level; withheld only when the ladder is KNOWN unsupported.
    /// - `text.format` — structured output, fails CLOSED on the mode's capability like
    ///   chat/completions (`response_format`), since an unsupported format is a 400.
    /// - `tool_choice` — only when the caller set one, and only when the option is not KNOWN
    ///   rejected (`permitsToolChoice`, fail-open), the same rule as chat/completions.
    /// - `parallel_tool_calls: false` — only under `disableParallelToolCalls`; the endpoint's own
    ///   default is `true`, so nothing is sent otherwise.
    /// - `include: reasoning.encrypted_content` — always, so a stateless conversation can carry
    ///   its reasoning forward (see ``CodexReasoningItem``).
    /// - `prompt_cache_key` — the instance's key, when given, so a conversation's turns share a
    ///   prefix-cache shard (see the stored property).
    /// - `extraJSONOverrides` — merged LAST and unconditionally, the probe-only escape hatch past
    ///   every gate above. Before this the Codex builder ignored them, so every forced probe
    ///   (effort ladders, structured output, tool_choice options) sent a bare request and graded
    ///   the ordinary success as support.
    /// - NO output-token cap, ever — see `neverSendsAnOutputCap`.
    static func buildRequestBody(
        configuration: ModelConfiguration,
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides,
        behaviorFlags: BehaviorFlags = BehaviorFlags(),
        reasoningEffortSupport: EffortSupport? = nil,
        modelCapabilities: ModelCapabilities = ModelCapabilities(),
        promptCacheKey: String? = nil
    ) -> [String: Any] {
        var body: [String: Any] = [
            "model": configuration.model,
            "input": buildInput(messages, behaviorFlags: behaviorFlags),
            // Stateless: the endpoint keeps nothing between turns, and we send the whole history.
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"]
        ]
        if let instructions = buildInstructions(messages, behaviorFlags: behaviorFlags) {
            body["instructions"] = instructions
        }
        if let promptCacheKey, !promptCacheKey.isEmpty {
            body["prompt_cache_key"] = promptCacheKey
        }
        if !behaviorFlags.mustNeverSendTemperatureParam,
           let temperature = overrides.temperature ?? configuration.temperature {
            body["temperature"] = temperature
        }
        if let topP = overrides.topP {
            body["top_p"] = topP
        }
        if !tools.isEmpty {
            body["tools"] = encodeTools(tools)
            if let choice = overrides.toolChoice, modelCapabilities.permitsToolChoice(choice) {
                body["tool_choice"] = encodeToolChoice(choice)
            }
            if behaviorFlags.disableParallelToolCalls {
                body["parallel_tool_calls"] = false
            }
        }
        if let effort = effectiveReasoningEffort(
            configuration: configuration, overrides: overrides, support: reasoningEffortSupport,
            capabilities: modelCapabilities) {
            // `summary: auto` is what makes the model emit reasoning_summary deltas; without it a
            // reasoning turn streams nothing until the answer lands.
            body["reasoning"] = ["effort": effort, "summary": "auto"]
        }
        if let format = overrides.responseFormat,
           modelCapabilities.state(of: format.requiredCapability) == true {
            body["text"] = ["format": format.responsesWireValue.mapValues(\.rawValue)]
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
        if let extra = configuration.extraJSONOverrides {
            mergeJSONOverrides(&body, with: extra)
        }
        return body
    }

    /// The `reasoning.effort` to send, if any. The per-call override outranks an explicit
    /// reasoning-off, which outranks the configured depth — the same layering the
    /// OpenAI-compatible provider applies to `reasoning_effort`. An explicit off has exactly one
    /// wire form on an effort-only model, `none`, and is stated only when
    /// ``ReasoningControl/effortOffFormPermitted(support:capabilities:)`` allows it — the measured
    /// off-switch first, the declared ladder's silence second.
    static func effectiveReasoningEffort(
        configuration: ModelConfiguration,
        overrides: LLMCallOverrides,
        support: EffortSupport?,
        capabilities: ModelCapabilities = ModelCapabilities()
    ) -> String? {
        guard support?.isSupported != false else { return nil }
        if let override = overrides.reasoningEffort, !override.isEmpty { return override }
        if (overrides.reasoningEnabled ?? configuration.reasoningEnabled) == false,
           ReasoningControl.effortOffFormPermitted(support: support, capabilities: capabilities) {
            return "none"
        }
        if let configured = configuration.reasoningEffort, !configured.isEmpty { return configured }
        return nil
    }

    /// System messages — and developer messages, unless the model is flagged as reading a
    /// `developer` role — folded into the one `instructions` string the Responses shape provides.
    /// Order is preserved.
    ///
    /// Folding is not a shortcut: the endpoint rejects a `{role: system}` input item outright
    /// (`400 {"detail":"System messages are not allowed"}`, verified live 2026-09-19), so
    /// `instructions` is the ONLY place a system turn can go — including a trailing steering
    /// turn, which chat/completions would keep at the tail under `supportsTrailingSystemMessage`.
    ///
    /// Note that NO identity prefix is prepended. The "You are Codex, based on GPT-5…" string is
    /// widely reported as a hard OAuth gate; a request without it returns 200 (verified 2026-09-16),
    /// and prepending it would put a second, conflicting identity in front of every role prompt.
    static func buildInstructions(
        _ messages: [LLMMessage], behaviorFlags: BehaviorFlags = BehaviorFlags()
    ) -> String? {
        let parts: [String] = messages.compactMap { message in
            guard foldsIntoInstructions(message, behaviorFlags: behaviorFlags) else { return nil }
            guard case .text(let text) = message.content else { return nil }
            return text
        }
        return parts.isEmpty ? nil : parts.joined(separator: "\n\n")
    }

    /// ONE predicate for "this message travels as `instructions`", shared by the instructions
    /// builder and the input builder so a message can never be both folded AND emitted, or neither.
    private static func foldsIntoInstructions(_ message: LLMMessage, behaviorFlags: BehaviorFlags) -> Bool {
        switch message.role {
        case .system: return true
        case .developer: return !behaviorFlags.supportsDeveloperRole
        case .user, .assistant, .tool: return false
        }
    }

    /// Conversation history as Responses `input` items.
    ///
    /// Tool traffic is the part that differs most from chat/completions: an assistant tool call is a
    /// TOP-LEVEL `function_call` item rather than a field on a message, and its result is a
    /// top-level `function_call_output` correlated by `call_id` rather than a `role: tool` message.
    ///
    /// An assistant turn is preceded by the `reasoning` items the endpoint emitted for it, replayed
    /// verbatim from `message.continuation` — the Responses analogue of Anthropic's signed thinking
    /// blocks. A user turn carries its images and documents as `input_image` / `input_file` parts
    /// beside the text; before 2026-09-19 they were dropped on the floor, and the model — quite
    /// correctly — answered "I don't see an image attached".
    static func buildInput(
        _ messages: [LLMMessage], behaviorFlags: BehaviorFlags = BehaviorFlags()
    ) -> [[String: Any]] {
        var items: [[String: Any]] = []
        for message in messages {
            if foldsIntoInstructions(message, behaviorFlags: behaviorFlags) { continue }
            if message.role == .assistant,
               let reasoning = message.continuation?.codexReasoningItems, !reasoning.isEmpty {
                items.append(contentsOf: reasoning.map(reasoningItem))
            }
            switch message.content {
            case .text(let text):
                items.append(messageItem(role: message.role, text: text,
                                         images: message.images ?? [],
                                         documents: message.documents ?? []))

            case .toolCalls(let calls):
                items.append(contentsOf: calls.map(functionCallItem))

            case .mixed(let text, let calls):
                // The assistant's prose precedes the calls it made, preserving turn order.
                if !text.isEmpty { items.append(messageItem(role: message.role, text: text)) }
                items.append(contentsOf: calls.map(functionCallItem))

            case .toolResult(let callID, let content):
                items.append([
                    "type": "function_call_output",
                    "call_id": callID,
                    "output": content
                ])
            }
        }
        return items
    }

    private static func messageItem(
        role: LLMMessage.Role, text: String,
        images: [LLMImageContent] = [], documents: [LLMDocumentContent] = []
    ) -> [String: Any] {
        // The content-part type is role-dependent: assistant text is `output_text`, everything the
        // caller supplies is `input_text`. Sending the wrong one is a 400.
        let isAssistant = role == .assistant
        var parts: [[String: Any]] = images.map { image in
            var part: [String: Any] = [
                "type": "input_image",
                "image_url": "data:\(image.mimeType);base64,\(image.data.base64EncodedString())"
            ]
            if let detail = image.detail { part["detail"] = detail.rawValue }
            return part
        }
        for document in documents {
            // `filename` is required beside `file_data` here as on chat/completions; synthesize
            // one when the caller didn't supply it rather than 400.
            parts.append([
                "type": "input_file",
                "filename": document.filename ?? "document.pdf",
                "file_data": "data:\(document.mimeType);base64,\(document.data.base64EncodedString())"
            ])
        }
        parts.append(["type": isAssistant ? "output_text" : "input_text", "text": text])
        return [
            "type": "message",
            "role": wireRole(role),
            "content": parts
        ]
    }

    /// The Responses role string. `.developer` only reaches here when the flag says the model reads
    /// it (otherwise it was folded); `.tool` never does (tool results are their own item type).
    private static func wireRole(_ role: LLMMessage.Role) -> String {
        switch role {
        case .assistant: return "assistant"
        case .developer: return "developer"
        case .user, .system, .tool: return "user"
        }
    }

    private static func reasoningItem(_ item: CodexReasoningItem) -> [String: Any] {
        [
            "type": "reasoning",
            "id": item.id,
            "summary": item.summary.map { ["type": "summary_text", "text": $0] as [String: Any] },
            "encrypted_content": item.encryptedContent
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
    /// `function` key the way chat/completions nests them. `strict` is a first-class field of a
    /// Responses function tool (accepted live 2026-09-19), so it is not capability-gated the way
    /// chat/completions endpoints of unknown strictness require.
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

    /// The Responses `tool_choice` shape — ONE source, shared with the probe that forces the raw
    /// field, so what is measured is what is shipped.
    static func encodeToolChoice(_ choice: LLMToolChoice) -> Any {
        choice.responsesWireValue.rawValue
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
        var reasoningItems: [CodexReasoningItem] = []
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
                      let itemID = Self.itemKey(event) else { continue }
                if item["type"] as? String == "reasoning" {
                    // The sealed chain of thought for this turn, carried forward as continuation
                    // (see `CodexReasoningItem`). Only a non-empty payload is worth replaying.
                    if let sealed = item["encrypted_content"] as? String, !sealed.isEmpty {
                        let summary = (item["summary"] as? [[String: Any]] ?? [])
                            .compactMap { $0["text"] as? String }
                        reasoningItems.append(
                            CodexReasoningItem(id: itemID, encryptedContent: sealed, summary: summary))
                    }
                    continue
                }
                guard item["type"] as? String == "function_call" else { continue }
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
            continuation: reasoningItems.isEmpty
                ? nil : ProviderContinuation(codexReasoningItems: reasoningItems),
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
            cacheWriteTokens: inputDetails?["cache_write_tokens"] as? Int ?? 0,
            rawUsage: TokenUsage.serializeRawUsage(usage))
    }
}
