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
        modelID: String
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
            first = try await llm.send(messages: messages, tools: [tool],
                                       overrides: LLMCallOverrides(toolChoice: .required))
        } catch {
            guard Self.looksLikeRejection(error) else {
                logger.error("Probe \(modelID, privacy: .public): transport failure — \(error.localizedDescription, privacy: .public)")
                return fail(.inconclusive, Self.rejectionDetail(error), forced: true)
            }
            // Rejected while forcing. Retry unforced to tell "won't accept tool_choice" apart
            // from "won't accept tools at all" — only the latter is a capability answer.
            forced = false
            do {
                first = try await llm.send(messages: messages, tools: [tool], overrides: LLMCallOverrides())
            } catch {
                let rejected = Self.looksLikeRejection(error)
                return fail(rejected ? .rejected : .inconclusive, Self.rejectionDetail(error), forced: false)
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

    /// Whether an error is the endpoint saying "no" rather than the network failing.
    ///
    /// This decides between recording `rejected` — a capability fact — and `inconclusive`, which
    /// records nothing, so it must never read a timeout as a refusal. A 4xx means the server read
    /// the request and declined it; anything else means we failed to ask.
    ///
    /// 429 is excluded deliberately: rate limiting says the endpoint is busy, not that the model
    /// can't do this. Filing it as `rejected` would write "no tool calling" for a model we simply
    /// asked too quickly — the exact false negative this probe exists to avoid.
    static func looksLikeRejection(_ error: any Error) -> Bool {
        guard let providerError = error as? LLMProviderError,
              case .httpError(let statusCode, _, _, _) = providerError else { return false }
        return (400..<500).contains(statusCode) && statusCode != 429
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
