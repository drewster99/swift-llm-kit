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

        /// Whether this run proved tool calling. Only the two positive verdicts count; note that
        /// `inconclusive` maps to `nil`, not `false`.
        public var toolUse: Bool? {
            switch verdict {
            case .roundTripCompleted, .toolCallOnly: return true
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

        func fail(_ verdict: ToolCallVerdict, _ error: String?, forced: Bool) -> ToolCallResult {
            ToolCallResult(
                providerID: providerID, modelID: modelID, verdict: verdict,
                toolChoiceForced: forced, expectedIdentifier: identifier,
                returnedText: nil, calledTools: [], errorDescription: error,
                duration: Date().timeIntervalSince(started)
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
            switch Self.classifyFailure(error) {
            case .noAnswer:
                logger.error("Probe \(modelID, privacy: .public): no answer — \(error.localizedDescription, privacy: .public)")
                return fail(.inconclusive, Self.rejectionDetail(error), forced: true)
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
                    switch Self.classifyFailure(error) {
                    case .refusedTools:
                        return fail(.rejected, Self.rejectionDetail(error), forced: false)
                    case .refusedOurRequest, .noAnswer:
                        // It refused, but not over tools — so we learned nothing about tools.
                        return fail(.inconclusive, Self.rejectionDetail(error), forced: false)
                    }
                }
            }
        }

        guard let call = first.toolCalls.first(where: { $0.name == probeToolName }) ?? first.toolCalls.first else {
            return ToolCallResult(
                providerID: providerID, modelID: modelID, verdict: .noToolCall,
                toolChoiceForced: forced, expectedIdentifier: identifier,
                returnedText: first.text, calledTools: [], errorDescription: nil,
                duration: Date().timeIntervalSince(started)
            )
        }

        // It called. Answer with the identifier and see whether it can use a tool result — the
        // half an agent actually depends on.
        let followUp: [LLMMessage] = messages
            + [.assistant(from: first)]
            + [.toolResult(identifier, callID: call.id)]

        let second: LLMResponse
        do {
            calls?.increment()
            second = try await llm.send(messages: followUp, tools: [tool], overrides: LLMCallOverrides())
        } catch {
            // The call itself is already proven; only the round-trip is unresolved.
            return ToolCallResult(
                providerID: providerID, modelID: modelID, verdict: .toolCallOnly,
                toolChoiceForced: forced, expectedIdentifier: identifier,
                returnedText: nil, calledTools: first.toolCalls.map(\.name),
                errorDescription: error.localizedDescription,
                duration: Date().timeIntervalSince(started)
            )
        }

        let echoed = (second.text ?? "").contains(identifier)
        return ToolCallResult(
            providerID: providerID, modelID: modelID,
            verdict: echoed ? .roundTripCompleted : .toolCallOnly,
            toolChoiceForced: forced, expectedIdentifier: identifier,
            returnedText: second.text, calledTools: first.toolCalls.map(\.name),
            errorDescription: nil, duration: Date().timeIntervalSince(started)
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
    /// 429 is `noAnswer` too: rate limiting means the endpoint is busy, not that the model is
    /// incapable. 404 likewise: a "not found" is a missing model or wrong endpoint (gone /
    /// Responses-only), never the model answering "I can't do that" — that's a 400. Grading a 404
    /// as a capability `false` would fabricate a "no" for a model that simply isn't reachable here.
    static func classifyFailure(_ error: any Error) -> FailureKind {
        guard let providerError = error as? LLMProviderError,
              case .httpError(let statusCode, let body, _, _) = providerError,
              (400..<500).contains(statusCode), statusCode != 429, statusCode != 404 else {
            return .noAnswer
        }
        let lowered = body.lowercased()
        let mentionsTools = ["tool", "function_call", "function call", "functions"]
            .contains { lowered.contains($0) }
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
        return ["no longer available", "is not found", "does not exist", "model not found",
                "not_found", "has been deprecated and is no longer",
                // Alibaba Cloud: a model listed in /models but not enabled for this workspace/region
                // returns "Model is not supported in current workspace service". Treated as
                // unavailable (per product decision) rather than access-denied.
                "not supported in current workspace", "not supported in this workspace"]
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
                "does not have access"]
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
