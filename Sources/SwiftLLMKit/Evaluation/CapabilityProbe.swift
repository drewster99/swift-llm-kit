import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "CapabilityProbe")

/// Determines what a model can actually do by asking it, rather than by believing a catalog.
///
/// Every capability we hold is a *claim*: LiteLLM says `gemini-2.5-flash-image` supports function
/// calling, and says nothing whatsoever about most of the catalog. Claims are not evidence, and
/// the two are not distinguishable from the outside — so this probe hands a model a real tool and
/// watches what comes back.
///
/// **It cannot consult the catalog, by construction.** It is handed an `any LLMProvider`, which
/// carries no capability data — there is no `capabilities` to read even by mistake. Build the
/// provider with `LLMKitManager.makeProvider(configuration:provider:)`, which does no catalog
/// clamping, and the guarantee holds through the type system rather than through discipline.
///
/// Every call is logged by the provider layer via `LLMRequestLogger` when verbose logging is on.
public enum CapabilityProbe {

    // MARK: - Tool-call probe

    /// What a tool-call probe established. Deliberately more than a Bool: "we asked and it
    /// refused" and "we never got an answer" are different facts, and collapsing a timeout into
    /// "unsupported" would write a false negative into data we intend to trust.
    public enum ToolCallVerdict: String, Sendable, Codable {
        /// Called the tool AND returned the identifier it fetched. The whole loop works — which
        /// is what an agent actually needs, not merely that a tool call was emitted.
        case roundTripCompleted
        /// Emitted a well-formed call but never returned the identifier. Tool calling works;
        /// something about handling the result does not.
        case toolCallOnly
        /// Emitted a well-formed call, but the tool-RESULT request never got an answer (transport
        /// failure / rate limit). Tool calling is proven; the round-trip is simply UNRESOLVED — not
        /// a failure. Distinct from `toolCallOnly` so a network hiccup isn't written as "can't use
        /// tool results".
        case roundTripInconclusive
        /// Answered in prose without calling the tool, despite being told to and (where
        /// supported) being forced to. Evidence of no usable tool calling.
        case noToolCall
        /// The endpoint refused the request outright — the strongest evidence of "cannot".
        case rejected
        /// No answer: transport failure, timeout, missing credentials. Establishes NOTHING.
        case inconclusive
    }

    /// The outcome of one model's probe, with enough context to audit the verdict later.
    public struct ToolCallResult: Sendable, Codable {
        public let providerID: String
        public let modelID: String
        public let verdict: ToolCallVerdict
        /// Whether the model was *forced* to call (`tool_choice: required`). False means the
        /// endpoint rejected that parameter and the probe retried with the choice left free — so
        /// a `noToolCall` verdict here is weaker evidence: it was invited, not compelled.
        public let toolChoiceForced: Bool
        /// The identifier the tool minted, and whether the model echoed it back.
        public let expectedIdentifier: String
        public let returnedText: String?
        /// Names of tools the model called. Should be exactly `[probeToolName]`.
        public let calledTools: [String]
        public let errorDescription: String?
        public let duration: TimeInterval
        /// Whether the probe's FIRST request returned an HTTP 200 at all — regardless of what it
        /// contained. Any 200 from a chat/completions request proves the endpoint serves chat for
        /// this model, which lets the sweep resolve a deferred chat verdict from the tool battery
        /// instead of spending a dedicated replay call. Optional: nil on records from before the
        /// field existed ("not recorded"), never assumed.
        public let sawSuccessfulResponse: Bool?

        public init(
            providerID: String, modelID: String, verdict: ToolCallVerdict, toolChoiceForced: Bool,
            expectedIdentifier: String, returnedText: String?, calledTools: [String],
            errorDescription: String?, duration: TimeInterval, sawSuccessfulResponse: Bool? = nil
        ) {
            self.providerID = providerID
            self.modelID = modelID
            self.verdict = verdict
            self.toolChoiceForced = toolChoiceForced
            self.expectedIdentifier = expectedIdentifier
            self.returnedText = returnedText
            self.calledTools = calledTools
            self.errorDescription = errorDescription
            self.duration = duration
            self.sawSuccessfulResponse = sawSuccessfulResponse
        }

        /// Whether this run proved tool calling. Only the two positive verdicts count; note that
        /// `inconclusive` maps to `nil`, not `false`.
        public var toolUse: Bool? {
            switch verdict {
            case .roundTripCompleted, .toolCallOnly, .roundTripInconclusive: return true
            case .noToolCall, .rejected: return false
            case .inconclusive: return nil
            }
        }
    }

    /// The single tool offered during a probe.
    public static let probeToolName = "get_test_identifier"

    /// Builds the probe's tool. No parameters: anything the model must *construct* adds a way to
    /// fail that has nothing to do with whether it can call a tool at all.
    public static func makeProbeTool() -> LLMToolDefinition {
        LLMToolDefinition(
            name: probeToolName,
            description: "Returns the test identifier string. Call this to obtain the identifier.",
            parameters: [
                "type": .string("object"),
                "properties": .dictionary([:]),
                "required": .array([])
            ]
        )
    }

    /// A random 9-character alphanumeric identifier the tool will mint.
    ///
    /// Random per run and per model so the round-trip can only be completed by actually calling
    /// the tool this time — a fixed string could be echoed from the prompt, or parroted from
    /// training data, and would look identical to success.
    public static func makeIdentifier() -> String {
        let alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  // no O/0/I/1 — unambiguous in logs
        return String((0..<9).map { _ in alphabet.randomElement() ?? "X" })
    }

    /// Probes whether a model can call a tool and use the result.
    ///
    /// The model is told plainly that this is a test, which tool to call, and what to do with the
    /// answer — the probe is measuring capability, not instruction-inference, so there is nothing
    /// to be gained by being subtle.
    ///
    /// - Parameters:
    ///   - llm: a provider already bound to the model under test. It exposes no capability data,
    ///     which is what keeps catalog claims out of a measurement of the truth.
    ///   - providerID/modelID: recorded on the result; not consulted.
    public static func probeToolCalling(
        llm: any LLMProvider,
        providerID: String,
        modelID: String,
        calls: ProbeCallCounter? = nil
    ) async -> ToolCallResult {
        let started = Date()
        let identifier = makeIdentifier()
        let tool = makeProbeTool()

        let messages: [LLMMessage] = [
            .system("""
            You are being tested to confirm that you can make a tool call. This is a capability \
            test, not a conversation. Call the tool `\(probeToolName)` to fetch the test \
            identifier, then reply with that identifier and nothing else.
            """),
            .user("""
            We are testing to confirm tool support. Use the tool `\(probeToolName)` to fetch the \
            test identifier, then return the identifier it gives you. Reply with only the \
            identifier.
            """)
        ]

        func fail(_ verdict: ToolCallVerdict, _ error: String?, forced: Bool,
                  sawSuccess: Bool = false) -> ToolCallResult {
            ToolCallResult(
                providerID: providerID, modelID: modelID, verdict: verdict,
                toolChoiceForced: forced, expectedIdentifier: identifier,
                returnedText: nil, calledTools: [], errorDescription: error,
                duration: Date().timeIntervalSince(started),
                sawSuccessfulResponse: sawSuccess
            )
        }

        // Force a tool call where the endpoint allows it. Whether it does is itself a claim we
        // won't take on faith, so: try forced, and fall back to a free choice only if the
        // endpoint rejects the parameter. A model that calls the tool unforced is just as much
        // proof; a model that doesn't is weaker evidence, which `toolChoiceForced` records.
        var forced = true
        var first: LLMResponse
        do {
            calls?.increment()
            first = try await llm.send(messages: messages, tools: [tool],
                                       overrides: LLMCallOverrides(toolChoice: .required))
        } catch {
            let detail = Self.rejectionDetail(error)
            // A body that SAYS "tools is not supported in this model" is the endpoint answering
            // the question, whatever status it rides on (OpenAI ships it as 404) — but only when
            // it isn't really saying the model is gone or this account is denied.
            if !Self.textIndicatesModelGone(detail), !Self.textIndicatesAccessDenied(detail),
               Self.textIndicatesToolsUnsupported(detail) {
                return fail(.rejected, detail, forced: true)
            }
            switch Self.classifyFailure(error) {
            case .noAnswer:
                logger.error("Probe \(modelID, privacy: .public): no answer — \(error.localizedDescription, privacy: .public)")
                return fail(.inconclusive, detail, forced: true)
            case .refusedTools, .refusedOurRequest:
                // Either way, retry with the choice free. A refusal naming tools might still only
                // be about tool_choice; a refusal about something else might be cured by dropping
                // the parameter it disliked. Only the retry can tell, and guessing here is how a
                // false negative gets written.
                forced = false
                do {
                    calls?.increment()
                    first = try await llm.send(messages: messages, tools: [tool], overrides: LLMCallOverrides())
                } catch {
                    let retryDetail = Self.rejectionDetail(error)
                    if !Self.textIndicatesModelGone(retryDetail), !Self.textIndicatesAccessDenied(retryDetail),
                       Self.textIndicatesToolsUnsupported(retryDetail) {
                        return fail(.rejected, retryDetail, forced: false)
                    }
                    switch Self.classifyFailure(error) {
                    case .refusedTools:
                        return fail(.rejected, retryDetail, forced: false)
                    case .refusedOurRequest, .noAnswer:
                        // It refused, but not over tools — so we learned nothing about tools.
                        return fail(.inconclusive, retryDetail, forced: false)
                    }
                }
            }
        }

        guard first.toolCalls.isEmpty == false else {
            // A 200 with no tool call is only evidence of declining when the model actually
            // ANSWERED. A truncated generation (finish_reason "length") or an empty response —
            // typically a reasoning model that burned the whole token budget thinking
            // (gpt-5-nano: 512/512 tokens as reasoning, empty content, graded "noToolCall"
            // while its dated sibling completed the round trip seconds later) — proves nothing.
            let answered = !(first.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            let truncated = first.finishReason == "length"
            if truncated || !answered {
                let reasoningNote = (first.usage?.reasoningTokens).map { " (\($0) reasoning tokens)" } ?? ""
                return fail(.inconclusive,
                            "no tool call, but the response was \(truncated ? "truncated at the token budget" : "empty")\(reasoningNote) — not evidence of declining",
                            forced: forced, sawSuccess: true)
            }
            return ToolCallResult(
                providerID: providerID, modelID: modelID, verdict: .noToolCall,
                toolChoiceForced: forced, expectedIdentifier: identifier,
                returnedText: first.text, calledTools: [], errorDescription: nil,
                duration: Date().timeIntervalSince(started),
                sawSuccessfulResponse: true
            )
        }

        // It called. Answer with the identifier and see whether it can use a tool result — the
        // half an agent actually depends on. EVERY tool_call_id gets a result: models are free
        // to answer a forced call with N parallel calls (gpt-4.1-nano sent 2-3), and OpenAI
        // 400-rejects a follow-up that leaves any of them unanswered — a harness-made error the
        // old single-result follow-up recorded as "did not return the identifier".
        let followUp: [LLMMessage] = messages
            + [.assistant(from: first)]
            + first.toolCalls.map { LLMMessage.toolResult(identifier, callID: $0.id) }

        let second: LLMResponse
        do {
            calls?.increment()
            second = try await llm.send(messages: followUp, tools: [tool], overrides: LLMCallOverrides())
        } catch {
            // The call itself is already proven. A transport failure / rate limit on the tool-result
            // request leaves the round-trip UNRESOLVED (roundTripInconclusive), not failed — only a
            // coherent refusal of the result submission is evidence the round-trip doesn't work.
            let verdict: ToolCallVerdict = Self.classifyFailure(error) == .noAnswer ? .roundTripInconclusive : .toolCallOnly
            return ToolCallResult(
                providerID: providerID, modelID: modelID, verdict: verdict,
                toolChoiceForced: forced, expectedIdentifier: identifier,
                returnedText: nil, calledTools: first.toolCalls.map(\.name),
                errorDescription: error.localizedDescription,
                duration: Date().timeIntervalSince(started),
                sawSuccessfulResponse: true
            )
        }

        let echoed = (second.text ?? "").contains(identifier)
        return ToolCallResult(
            providerID: providerID, modelID: modelID,
            verdict: echoed ? .roundTripCompleted : .toolCallOnly,
            toolChoiceForced: forced, expectedIdentifier: identifier,
            returnedText: second.text, calledTools: first.toolCalls.map(\.name),
            errorDescription: nil, duration: Date().timeIntervalSince(started),
            sawSuccessfulResponse: true
        )
    }

    /// Why a call failed, to the extent it can be told from the outside.
    public enum FailureKind: Sendable, Equatable {
        /// The endpoint refused in terms that name tools. Evidence the model can't call them.
        case refusedTools
        /// The endpoint refused, but over something else — a parameter it dislikes, auth, a bad
        /// model name. Says nothing about tools.
        case refusedOurRequest
        /// We never got an answer: timeout, rate limit, server fault.
        case noAnswer
    }

    /// Classifies a failure, biased hard toward admitting ignorance.
    ///
    /// The two mistakes are not symmetric. Recording `inconclusive` for a model that genuinely
    /// can't call tools costs a re-probe. Recording `rejected` for a model that can is a false
    /// negative written into data we intend to trust, and nothing downstream can tell it from a
    /// real measurement. So a refusal is only read as a capability answer when the endpoint's own
    /// words implicate tools; otherwise it's assumed to be our fault.
    ///
    /// This is not hypothetical. On its first live run this probe sent `temperature: 0` to
    /// claude-fable-5, which answered `400 "temperature is deprecated for this model"`, and the
    /// old any-4xx-is-a-rejection rule recorded "claude-fable-5 cannot call tools" — a flat lie
    /// about a flagship model, produced entirely by our own request.
    ///
    /// Several 4xx codes are `noAnswer`, not capability answers, because none of them is the model
    /// saying "I can't do that" (only a 400 is):
    /// - **429** rate limit — the endpoint is busy, not the model incapable.
    /// - **404** not found — a missing model or wrong endpoint (retired / Responses-only).
    /// - **401 / 403** auth — an expired or unauthorized key. Grading these as a capability `false`
    ///   would disable a perfectly capable model because we couldn't reach it. (An access-denied
    ///   403 that names the model is still surfaced separately via ``textIndicatesAccessDenied``,
    ///   which the caller scans on the resulting inconclusive finding.)
    static func classifyFailure(_ error: any Error) -> FailureKind {
        let noAnswerCodes: Set<Int> = [401, 403, 404, 429]
        guard let providerError = error as? LLMProviderError,
              case .httpError(let statusCode, let body, _, _) = providerError,
              (400..<500).contains(statusCode), !noAnswerCodes.contains(statusCode) else {
            return .noAnswer
        }
        let lowered = body.lowercased()
        // Refusals of OUR OWN request knobs must never read as tool refusals, no matter what
        // other words the body contains. `parallel_tool_calls` and `tool_choice` are parameters
        // the client sends; a body rejecting them says nothing about the model's tool support —
        // yet both contain the substring "tool", which is how eight o-series models were
        // recorded toolCalling=false in the 2026-07-18 audit. Likewise the endpoint-combination
        // family whose text AFFIRMS tool support while naming a remedy ("Function tools with
        // reasoning_effort are not supported ... use /v1/responses or set reasoning_effort to
        // 'none'"): an error that says tools work elsewhere is not a tools refusal.
        let ourRequestMarkers = ["parallel_tool_calls", "tool_choice", "reasoning_effort", "v1/responses"]
        if ourRequestMarkers.contains(where: { lowered.contains($0) }) {
            return .refusedOurRequest
        }
        // "tool"/"tools" as whole words only (underscores are word characters, so parameter
        // names like tools[0] still match while parallel_tool_calls cannot).
        let mentionsTools = lowered.range(of: #"\btools?\b"#, options: .regularExpression) != nil
            || ["function_call", "function call", "functions"].contains { lowered.contains($0) }
        return mentionsTools ? .refusedTools : .refusedOurRequest
    }

    /// Whether an error body says the MODEL itself is gone — retired, removed, never existed — as
    /// opposed to rejecting something about our request. Gemini keeps listing `gemini-2.0-flash-lite`
    /// in `/models` long after every call 404s with "is no longer available"; a delisted-but-listed
    /// model must not be read as "rejected the attachment" or "isn't a chat model."
    ///
    /// This is a text signal on top of the status code because the code alone can't tell "model
    /// retired" (404) from "wrong image format" (also 4xx). The phrases are provider-agnostic and
    /// specific enough not to fire on a normal capability refusal.
    public static func textIndicatesModelGone(_ text: String) -> Bool {
        let lowered = text.lowercased()
        // A METHOD/VERSION limitation means the model EXISTS but doesn't serve this call — Gemini
        // answers a chat request to an embedding/imagen/veo/audio model with 404 "models/X is not
        // found for API version v1beta, or is not supported for generateContent". The bare "is not
        // found" / "not_found" substrings below would otherwise read that as GONE and stamp a live
        // model isAvailable=false. It is "not a chat model", handled by the chat=false path, not a
        // retirement — so never treat a method-limitation body as gone.
        if lowered.contains("not supported for generatecontent") { return false }
        // "No longer available TO NEW USERS / FOR NEW PROJECTS / to new API keys" is account-scoped,
        // not global retirement — existing users still call the model. It is access-denial for THIS
        // key (see textIndicatesAccessDenied); stamping it globally gone would mislead any account
        // that already has access. Match "available to/for new" rather than one exact phrase so a
        // wording variant can't fall through to the "no longer available" gone match below. Genuine
        // retirement ("is no longer available. Use a newer model") has no "available to/for new".
        if lowered.contains("available to new") || lowered.contains("available for new") { return false }
        return ["no longer available", "is not found", "does not exist", "model not found",
                "not_found", "has been deprecated and is no longer",
                // Ollama Cloud retires models with HTTP 410 "X was retired at <date>" — a gone
                // signal (the model existed and is a chat model, it's just removed), which without
                // this fell through to chat=established(false), i.e. "not a chat model". Match the
                // retirement phrasing, not bare "retired" (a parameter-deprecation note could say
                // "retired" without the model being gone).
                "was retired", "retired at",
                // Alibaba Cloud: a model listed in /models but not enabled for this workspace/region
                // returns "Model is not supported in current workspace service". Treated as
                // unavailable (per product decision) rather than access-denied.
                "not supported in current workspace", "not supported in this workspace",
                // Mistral serves its gone/not-routable answer as HTTP 400 "Invalid model: <id>"
                // (type invalid_model) — without a text rule, a coherent 400 like that graded
                // the chat probe established(false) instead of unavailable. The colon guards
                // against unrelated phrases like "invalid model configuration".
                "invalid model:", "unknown model"]
            .contains { lowered.contains($0) }
    }

    /// Whether an error body affirmatively states the model is NOT a chat model — OpenAI's
    /// completion-era and audio models answer chat/completions with "This is not a chat model
    /// and thus not supported in the v1/chat/completions endpoint" behind an HTTP 404. The
    /// status alone reads as "unreachable" and the definitive capability statement was lost:
    /// no verdict, no record, and the model re-probed on every sweep. The phrase is a statement
    /// about the MODEL (the endpoint recognized it), so it cannot fire on the wrong-URL or
    /// retired-model 404s the status exclusion exists to guard.
    public static func textIndicatesNotAChatModel(_ text: String) -> Bool {
        let lowered = text.lowercased()
        // "not a chat model" — OpenAI's completion-era 404s. "not supported for generatecontent" —
        // Gemini's method-limitation 404 for a non-chat model (embeddings, imagen, veo, audio):
        // the model exists but doesn't serve the chat method, which is chat=false, not gone.
        return lowered.contains("not a chat model")
            || lowered.contains("not supported for generatecontent")
    }

    /// Whether an error body affirmatively states the model does not support tools. OpenAI's
    /// search-preview/search-api models answer a tools-bearing request with "tools is not
    /// supported in this model" behind an HTTP 404 — a first-party capability statement the
    /// status-code gate discarded, leaving toolCalling permanently inconclusive.
    public static func textIndicatesToolsUnsupported(_ text: String) -> Bool {
        let lowered = text.lowercased()
        return ["tools is not supported in this model", "tools are not supported in this model"]
            .contains { lowered.contains($0) }
    }

    /// Whether an error body says the ACCOUNT may not call this model — the model exists and is
    /// current, but this key hasn't been granted access (Alibaba Cloud's `Model.AccessDenied`,
    /// returned until you enable the model in their dashboard). Distinct from "model gone": the
    /// remedy is a permission change, not a model swap, so a caller may want to surface it
    /// differently. Like ``textIndicatesModelGone(_:)``, a text signal because the status code
    /// (403) is shared with unrelated auth failures.
    public static func textIndicatesAccessDenied(_ text: String) -> Bool {
        let lowered = text.lowercased()
        return ["access denied", "accessdenied", "model access", "not authorized to access",
                "does not have access",
                // Gemini restricts whole model generations to existing users over time: "This model
                // models/X is no longer available to new users" (also "for new projects" / "to new
                // API keys"). The model is live for grandfathered keys, so this is account-scoped
                // access denial, not global retirement. Robust to wording variants, and "newer
                // model" in a genuine-retirement message never contains "available to/for new".
                "available to new", "available for new",
                // Mistral gates preview "Labs" models behind an org setting: HTTP 403 "Model X is a
                // Labs model. To use Labs models, an admin must enable them ..." (type
                // labs_not_enabled). Same category as Alibaba's Model.AccessDenied — the model exists
                // and this account simply hasn't been granted it — so surface it as access-denied and
                // RETAIN the record, rather than letting the 403 fall through to noAnswer → chat
                // inconclusive → pruned (the model then vanished from the catalog with no explanation).
                // Match both the type code (stable) and the human phrase (survives a JSON reshape).
                "labs_not_enabled", "admin must enable"]
            .contains { lowered.contains($0) }
    }

    /// The endpoint's own words about why it refused, kept for the report — a 400 saying
    /// "tools are not supported for this model" and one saying "unknown parameter
    /// 'tool_choice'" are entirely different findings, and the status code alone loses that.
    static func rejectionDetail(_ error: any Error) -> String {
        guard let providerError = error as? LLMProviderError,
              case .httpError(let statusCode, let body, _, _) = providerError else {
            return error.localizedDescription
        }
        return "HTTP \(statusCode): \(body.prefix(500))"
    }
}
