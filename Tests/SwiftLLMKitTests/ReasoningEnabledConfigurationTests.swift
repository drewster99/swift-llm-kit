import Testing
import Foundation
@testable import SwiftLLMKit

/// `ModelConfiguration.reasoningEnabled` — the persistent thinking switch (2026-08-06). The
/// per-call `LLMCallOverrides.reasoningEnabled` already worked; these pin the CONFIGURATION
/// layer beneath it: config decides when no per-call override speaks, and the per-call value
/// still wins when both are set.
@Suite("Configuration-level reasoningEnabled")
struct ReasoningEnabledConfigurationTests {

    private func anthropic(reasoningEnabled: Bool?, thinkingBudget: Int?) throws -> AnthropicProvider {
        AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m",
                                              thinkingBudget: thinkingBudget,
                                              reasoningEnabled: reasoningEnabled),
            provider: ModelProvider(id: "p", name: "p", apiType: .anthropic,
                                    endpoint: try #require(URL(string: "https://api.anthropic.com"))),
            readAPIKey: { "" })
    }

    @Test("Anthropic: configured ON with a budget emits the thinking block with no per-call override")
    func anthropicConfiguredOn() throws {
        let body = try anthropic(reasoningEnabled: true, thinkingBudget: 4096)
            .buildRequestBody(messages: [.user("hi")], tools: [], overrides: LLMCallOverrides())
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["budget_tokens"] as? Int == 4096)
    }

    @Test("Anthropic: configured OFF beats a configured budget")
    func anthropicConfiguredOff() throws {
        let body = try anthropic(reasoningEnabled: false, thinkingBudget: 4096)
            .buildRequestBody(messages: [.user("hi")], tools: [], overrides: LLMCallOverrides())
        #expect(body["thinking"] == nil)
    }

    @Test("Anthropic: the per-call override still wins over the configuration")
    func perCallWinsOverConfiguration() throws {
        let body = try anthropic(reasoningEnabled: false, thinkingBudget: 4096)
            .buildRequestBody(messages: [.user("hi")], tools: [],
                              overrides: LLMCallOverrides(reasoningEnabled: true))
        #expect(body["thinking"] != nil, "per-call ON must beat configured OFF")
    }

    @Test("Gemini: configured OFF sends the documented thinkingBudget: 0 off-form")
    func geminiConfiguredOff() throws {
        let provider = GeminiProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m",
                                              reasoningEnabled: false),
            provider: ModelProvider(id: "p", name: "p", apiType: .gemini,
                                    endpoint: try #require(URL(string: "https://g.test/v1beta"))),
            readAPIKey: { "" },
            modelCapabilities: ModelCapabilities([.thinkingSupportsTokenBudget]))
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        let config = try #require(body["generationConfig"] as? [String: Any])
        let thinking = try #require(config["thinkingConfig"] as? [String: Any])
        #expect(thinking["thinkingBudget"] as? Int == 0)
    }

    // MARK: Override plumbing

    @Test("The override resolves reasoningEnabled into the configuration and counts as non-empty")
    func overrideCarriesReasoningEnabled() {
        var override = ModelConfigurationOverride()
        #expect(override.isEmpty)
        override.reasoningEnabled = false
        #expect(!override.isEmpty)
        let info = ModelInfo(providerID: "p", modelID: "m", displayName: "M")
        #expect(override.resolved(against: info).reasoningEnabled == false)
    }

    @Test("Forcing a direction the model was measured unable to switch warns; unknown warns nothing")
    func directionWarnings() {
        var info = ModelInfo(providerID: "p", modelID: "m", displayName: "M")
        info.capabilities = ModelCapabilities(states: [.reasoningCanBeEnabled: false])
        let on = ModelConfigurationOverride(reasoningEnabled: true)
        #expect(on.warnings(against: info).map(\.field) == [.reasoningEnabled])
        // The OFF direction was never measured -> no warning for it.
        let off = ModelConfigurationOverride(reasoningEnabled: false)
        #expect(off.warnings(against: info).isEmpty)
    }

    @Test("A budget outside the measured range warns, inside does not")
    func budgetBoundWarnings() {
        var info = ModelInfo(providerID: "p", modelID: "m", displayName: "M")
        info.maxThinkingBudgetTokens = 30_000
        info.minThinkingBudgetTokens = 1024
        #expect(ModelConfigurationOverride(thinkingBudget: 60_000).warnings(against: info)
            .map(\.field) == [.thinkingBudget])
        #expect(ModelConfigurationOverride(thinkingBudget: 512).warnings(against: info)
            .map(\.field) == [.thinkingBudget])
        #expect(ModelConfigurationOverride(thinkingBudget: 2048).warnings(against: info).isEmpty)
    }

    // MARK: Planned-state resolver — the display companion of the emission switches.

    @Test("Planned state: unrecorded mechanism and unsupported are their own states")
    func plannedStateUnknownAndUnsupported() {
        #expect(ReasoningControl.plannedThinkingState(
            control: nil, capabilities: ModelCapabilities(), reasoningEnabled: true,
            thinkingBudget: nil, reasoningEffort: nil, reasoningEffortSupport: nil).label == "unknown")
        #expect(ReasoningControl.plannedThinkingState(
            control: .unsupported, capabilities: ModelCapabilities(), reasoningEnabled: true,
            thinkingBudget: 4096, reasoningEffort: nil, reasoningEffortSupport: nil) == .unsupported)
    }

    @Test("Planned state mirrors Anthropic emission: budget on, off, and on-without-budget")
    func plannedStateAnthropic() {
        func state(_ enabled: Bool?, _ budget: Int?) -> PlannedThinkingState {
            ReasoningControl.plannedThinkingState(
                control: .anthropicThinking, capabilities: ModelCapabilities(),
                reasoningEnabled: enabled, thinkingBudget: budget,
                reasoningEffort: nil, reasoningEffortSupport: nil)
        }
        #expect(state(nil, 4096).label == "on")
        #expect(state(false, 4096).label == "off")
        #expect(state(nil, nil).label == "off")
        // ON with no budget seeds the minimum at emission — so the resolver says ON, seeded.
        #expect(state(true, nil).label == "on")
        // A budget capability measured FALSE withholds the block (emission fails open on unknown,
        // closed only on a measured false) — the resolver mirrors it.
        #expect(ReasoningControl.plannedThinkingState(
            control: .anthropicThinking,
            capabilities: ModelCapabilities(states: [.thinkingSupportsTokenBudget: false]),
            reasoningEnabled: true, thinkingBudget: 4096,
            reasoningEffort: nil, reasoningEffortSupport: nil).label == "unknown")
    }

    @Test("Anthropic: configured ON with no budget seeds the minimum, matching per-call ON")
    func anthropicConfiguredOnSeedsMinimum() throws {
        let body = try anthropic(reasoningEnabled: true, thinkingBudget: nil)
            .buildRequestBody(messages: [.user("hi")], tools: [], overrides: LLMCallOverrides())
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["budget_tokens"] as? Int == ThinkingBudget.minimumTokens)
    }

    @Test("Planned state: a bare budget on a thinkingBlock model is sent typeless, not silent")
    func plannedStateThinkingBlockBareBudget() {
        let state = ReasoningControl.plannedThinkingState(
            control: .thinkingBlock,
            capabilities: ModelCapabilities([.thinkingSupportsTokenBudget]),
            reasoningEnabled: nil, thinkingBudget: 2048,
            reasoningEffort: nil, reasoningEffortSupport: nil)
        #expect(state.label == "unknown")
        #expect(state.detail.contains("budget_tokens"))
    }

    @Test("Planned state gates directions on the measured switchability, like emission")
    func plannedStateDirectionGates() {
        let canBoth = ModelCapabilities(states: [.reasoningCanBeEnabled: true,
                                                 .reasoningCanBeDisabled: true])
        let cannotOn = ModelCapabilities(states: [.reasoningCanBeEnabled: false])
        func state(_ enabled: Bool?, caps: ModelCapabilities) -> PlannedThinkingState {
            ReasoningControl.plannedThinkingState(
                control: .thinkingBlock, capabilities: caps, reasoningEnabled: enabled,
                thinkingBudget: nil, reasoningEffort: nil, reasoningEffortSupport: nil)
        }
        #expect(state(true, caps: canBoth).label == "on")
        #expect(state(false, caps: canBoth).label == "off")
        #expect(state(true, caps: cannotOn).label == "unknown")
        #expect(state(nil, caps: canBoth).label == "unknown")
    }

    @Test("Planned state: effort-only models turn off via reasoning_effort none, ladder-gated")
    func plannedStateEffortOnly() {
        func state(effortSupport: EffortSupport?) -> PlannedThinkingState {
            ReasoningControl.plannedThinkingState(
                control: .reasoningEffortOnly, capabilities: ModelCapabilities(),
                reasoningEnabled: false, thinkingBudget: nil,
                reasoningEffort: nil, reasoningEffortSupport: effortSupport)
        }
        // A NIL support means `reasoning_effort` is not measured-supported: emission fails
        // closed and sends nothing, so the honest answer is unknown — not "off".
        #expect(state(effortSupport: nil).label == "unknown")
        #expect(state(effortSupport: EffortSupport(levels: ["low", "high"])).label == "unknown")
        #expect(state(effortSupport: EffortSupport(levels: ["none", "high"])).label == "off")
        // And with no off request it is always on.
        #expect(ReasoningControl.plannedThinkingState(
            control: .reasoningEffortOnly, capabilities: ModelCapabilities(),
            reasoningEnabled: nil, thinkingBudget: nil,
            reasoningEffort: "high", reasoningEffortSupport: nil).label == "on")
    }
}
