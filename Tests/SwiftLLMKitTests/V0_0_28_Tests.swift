import Foundation
import Testing
@testable import SwiftLLMKit

/// Tests for the 0.0.28 release: OpenAI `reasoning_effort` emission.
///
/// Extends 0.0.27's `ModelConfiguration.thinkingEffort` to flow through
/// the OpenAI-compatible wire path. OpenAI o-series (o1, o3, o4-mini) and
/// GPT-5 family accept a top-level `reasoning_effort: <enum>` field. Non-
/// reasoning models (GPT-4o, GPT-3.5, DeepSeek-V*, etc.) reject it, so
/// emission is gated on `BehaviorFlags.supportsReasoningEffort`. Bundled
/// registry sets the flag for known reasoning model IDs.
@Suite("0.0.28: OpenAI reasoning_effort")
struct V0_0_28_Tests {

    private static let dummyKey: @Sendable () -> String = { "" }

    private func openAI(
        thinkingEffort: String? = nil,
        flags: BehaviorFlags = BehaviorFlags()
    ) throws -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            configuration: ModelConfiguration(
                name: "t",
                providerID: "p",
                modelID: "m",
                thinkingEffort: thinkingEffort
            ),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .openAICompatible,
                endpoint: try #require(URL(string: "https://api.openai.com/v1"))
            ),
            readAPIKey: Self.dummyKey,
            behaviorFlags: flags
        )
    }

    // MARK: - Emission gating

    @Test("reasoning_effort emitted when flag is true AND thinkingEffort is set")
    func reasoningEffortEmittedWhenSupportedAndSet() throws {
        let flags = BehaviorFlags(supportsReasoningEffort: true)
        let provider = try openAI(thinkingEffort: "high", flags: flags)
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["reasoning_effort"] as? String == "high")
    }

    @Test("reasoning_effort NOT emitted when flag is false (non-reasoning model)")
    func reasoningEffortSkippedOnNonReasoningModel() throws {
        // Default flags: supportsReasoningEffort = false. Even with thinkingEffort
        // set, must NOT emit the field — non-reasoning models 400 if it's present.
        let provider = try openAI(thinkingEffort: "high")
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["reasoning_effort"] == nil)
    }

    @Test("reasoning_effort NOT emitted when thinkingEffort is nil")
    func reasoningEffortSkippedWhenEffortNil() throws {
        let flags = BehaviorFlags(supportsReasoningEffort: true)
        let provider = try openAI(thinkingEffort: nil, flags: flags)
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["reasoning_effort"] == nil)
    }

    @Test("reasoning_effort field name and position — top-level, snake_case")
    func reasoningEffortFieldShape() throws {
        let flags = BehaviorFlags(supportsReasoningEffort: true)
        let provider = try openAI(thinkingEffort: "medium", flags: flags)
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
        let flags = BehaviorFlags(supportsReasoningEffort: true)
        let provider = try openAI(thinkingEffort: effort, flags: flags)
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["reasoning_effort"] as? String == effort)
    }

    @Test("Anthropic-only effort values (xhigh, max) still pass through")
    func anthropicOnlyEffortValuesPassThrough() throws {
        // swift-llm-kit doesn't validate effort strings — it just passes them
        // through. The user takes responsibility for picking a value the
        // model supports. (If they send xhigh to GPT-5, OpenAI rejects.)
        let flags = BehaviorFlags(supportsReasoningEffort: true)
        for value in ["xhigh", "max"] {
            let provider = try openAI(thinkingEffort: value, flags: flags)
            let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
            #expect(body["reasoning_effort"] as? String == value)
        }
    }

    // MARK: - Codable + override

    @Test("BehaviorFlags Codable round-trips supportsReasoningEffort")
    func flagsCodableRoundTrip() throws {
        let flags = BehaviorFlags(supportsReasoningEffort: true)
        let data = try JSONEncoder().encode(flags)
        let decoded = try JSONDecoder().decode(BehaviorFlags.self, from: data)
        #expect(decoded.supportsReasoningEffort == true)
        // Default (false) must not be serialized.
        let defaultFlags = BehaviorFlags()
        let defaultData = try JSONEncoder().encode(defaultFlags)
        let json = String(data: defaultData, encoding: .utf8) ?? ""
        #expect(!json.contains("supportsReasoningEffort"))
    }

    @Test("BehaviorFlagsOverride patches supportsReasoningEffort")
    func overrideAppliesFlag() throws {
        var flags = BehaviorFlags()
        let patch = BehaviorFlagsOverride(supportsReasoningEffort: true)
        patch.apply(to: &flags, forceReplace: false)
        #expect(flags.supportsReasoningEffort == true)
    }

    @Test("displayLabels includes reasoning_effort when flag is true")
    func displayLabelsIncludesReasoningEffort() {
        let flags = BehaviorFlags(supportsReasoningEffort: true)
        #expect(flags.displayLabels.contains("reasoning_effort"))
    }

    @Test("requiresAdaptiveThinking and supportsReasoningEffort are independent")
    func adaptiveAndEffortFlagsAreIndependent() throws {
        // Both flags can be set on the same BehaviorFlags value without
        // interaction. AnthropicProvider only reads requiresAdaptiveThinking;
        // OpenAICompatibleProvider only reads supportsReasoningEffort.
        // Setting both produces no cross-provider weirdness — each provider
        // emits its own wire shape from its own flag + the shared
        // thinkingEffort config field.
        let flags = BehaviorFlags(
            requiresAdaptiveThinking: true,
            supportsReasoningEffort: true
        )
        #expect(flags.requiresAdaptiveThinking == true)
        #expect(flags.supportsReasoningEffort == true)

        // OpenAI provider with both flags: still emits reasoning_effort
        // (not output_config.effort — that's Anthropic's field).
        let provider = try openAI(thinkingEffort: "high", flags: flags)
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["reasoning_effort"] as? String == "high")
        #expect(body["output_config"] == nil,
                "OpenAI provider must not emit Anthropic's output_config")
    }
}
