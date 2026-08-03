import Foundation
import Testing
@testable import SwiftLLMKit

/// Tests for the 0.0.28 release: OpenAI `reasoning_effort` emission.
///
/// Extends 0.0.27's `ModelConfiguration.reasoningEffort` to flow through
/// the OpenAI-compatible wire path. OpenAI o-series (o1, o3, o4-mini) and
/// GPT-5 family accept a top-level `reasoning_effort: <enum>` field. Non-
/// reasoning models (GPT-4o, GPT-3.5, DeepSeek-V*, etc.) reject it, so
/// emission is gated on `BehaviorFlags.supportsReasoningEffort`. Bundled
/// registry sets the flag for known reasoning model IDs.
@Suite("0.0.28: OpenAI reasoning_effort")
struct V0_0_28_Tests {

    private static let dummyKey: @Sendable () -> String = { "" }

    private func openAI(
        reasoningEffort: String? = nil,
        flags: BehaviorFlags = BehaviorFlags(),
        reasoningEffortSupport: EffortSupport? = nil
    ) throws -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            configuration: ModelConfiguration(
                name: "t",
                providerID: "p",
                modelID: "m",
                reasoningEffort: reasoningEffort
            ),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .openAICompatible,
                endpoint: try #require(URL(string: "https://api.openai.com/v1"))
            ),
            readAPIKey: Self.dummyKey,
            behaviorFlags: flags,
            reasoningEffortSupport: reasoningEffortSupport
        )
    }

    // MARK: - Emission gating

    @Test("reasoning_effort emitted when the model supports it AND a value is set")
    func reasoningEffortEmittedWhenSupportedAndSet() throws {
        let flags = BehaviorFlags()
        let provider = try openAI(reasoningEffort: "high", flags: flags,
                                  reasoningEffortSupport: .supportedLevelsUnknown)
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["reasoning_effort"] as? String == "high")
    }

    @Test("reasoning_effort NOT emitted when support is unknown (fails closed)")
    func reasoningEffortSkippedOnNonReasoningModel() throws {
        // Default flags: supportsReasoningEffort = false. Even with reasoningEffort
        // set, must NOT emit the field — non-reasoning models 400 if it's present.
        let provider = try openAI(reasoningEffort: "high")
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["reasoning_effort"] == nil)
    }

    @Test("reasoning_effort NOT emitted when reasoningEffort is nil")
    func reasoningEffortSkippedWhenEffortNil() throws {
        let flags = BehaviorFlags()
        let provider = try openAI(reasoningEffort: nil, flags: flags)
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["reasoning_effort"] == nil)
    }

    @Test("reasoning_effort field name and position — top-level, snake_case")
    func reasoningEffortFieldShape() throws {
        let flags = BehaviorFlags()
        let provider = try openAI(reasoningEffort: "medium", flags: flags,
                                  reasoningEffortSupport: .supportedLevelsUnknown)
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        // Must be at top level, not nested under another key.
        #expect(body["reasoning_effort"] != nil, "must be at top level of body")
        #expect(body["reasoningEffort"] == nil, "must be snake_case, not camelCase")
        // No `reasoning` wrapper object — OpenAI uses a flat enum field.
        #expect(body["reasoning"] == nil)
    }

    // MARK: - Effort enum values

    @Test("all valid OpenAI effort values pass through verbatim",
          arguments: ["minimal", "low", "medium", "high"])
    func validEffortValuesPassThrough(effort: String) throws {
        let flags = BehaviorFlags()
        let provider = try openAI(reasoningEffort: effort, flags: flags,
                                  reasoningEffortSupport: .supportedLevelsUnknown)
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["reasoning_effort"] as? String == effort)
    }

    @Test("Anthropic-only effort values (xhigh, max) still pass through")
    func anthropicOnlyEffortValuesPassThrough() throws {
        // swift-llm-kit doesn't validate effort strings — it just passes them
        // through. The user takes responsibility for picking a value the
        // model supports. (If they send xhigh to GPT-5, OpenAI rejects.)
        let flags = BehaviorFlags()
        for value in ["xhigh", "max"] {
            let provider = try openAI(reasoningEffort: value, flags: flags,
                                      reasoningEffortSupport: .supportedLevelsUnknown)
            let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
            #expect(body["reasoning_effort"] as? String == value)
        }
    }

    // MARK: - Codable + override

    @Test("EffortSupport round-trips all three states, keeping them distinguishable on disk")
    func effortSupportCodableRoundTrip() throws {
        // The predecessor stored a bare [String]; `[]` could not be told from "unknown", which is
        // what let a measured "no level works" read as "nobody asked".
        for value in [EffortSupport.unsupported, .supportedLevelsUnknown, .levels(["low", "high"])] {
            let data = try JSONEncoder().encode(value)
            #expect(try JSONDecoder().decode(EffortSupport.self, from: data) == value)
        }
    }

    @Test("An empty ladder normalizes to .unsupported, so the illegal state can't be built")
    func emptyLadderNormalizes() {
        #expect(EffortSupport(levels: []) == .unsupported)
        #expect(EffortSupport(levels: ["max", "low"]) == .levels(["low", "max"]))   // rank-ordered
    }

    @Test("rejects() fails safe: an unknown ladder never rejects")
    func rejectsFailsSafe() {
        #expect(EffortSupport.supportedLevelsUnknown.rejects("anything") == false)
        #expect(EffortSupport.unsupported.rejects("low") == true)
        #expect(EffortSupport.levels(["low"]).rejects("high") == true)
        #expect(EffortSupport.levels(["low"]).rejects("low") == false)
    }

    @Test("requiresAdaptiveThinking and reasoning-effort support are independent")
    func adaptiveAndEffortFlagsAreIndependent() throws {
        // Both flags can be set on the same BehaviorFlags value without
        // interaction. AnthropicProvider only reads requiresAdaptiveThinking;
        // OpenAICompatibleProvider only reads supportsReasoningEffort.
        // Setting both produces no cross-provider weirdness — each provider
        // emits its own wire shape from its own flag + the shared
        // reasoningEffort config field.
        let flags = BehaviorFlags(requiresAdaptiveThinking: true)
        #expect(flags.requiresAdaptiveThinking == true)

        // OpenAI provider with reasoning effort supported: still emits reasoning_effort
        // (not output_config.effort — that's Anthropic's field).
        let provider = try openAI(reasoningEffort: "high", flags: flags,
                                  reasoningEffortSupport: .supportedLevelsUnknown)
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["reasoning_effort"] as? String == "high")
        #expect(body["output_config"] == nil,
                "OpenAI provider must not emit Anthropic's output_config")
    }
}
