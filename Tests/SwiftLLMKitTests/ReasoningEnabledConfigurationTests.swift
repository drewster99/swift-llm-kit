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

    @Test("The thinking-switch warning fires exactly when the resolver says nothing reaches the wire")
    func directionWarnings() {
        // thinkingBlock: emission fails closed per direction, so an unmeasured or
        // measured-false direction warns.
        var block = ModelInfo(providerID: "p", modelID: "m", displayName: "M")
        block.reasoningControl = .thinkingBlock
        block.capabilities = ModelCapabilities(states: [.reasoningCanBeEnabled: false])
        #expect(ModelConfigurationOverride(reasoningEnabled: true).warnings(against: block)
            .map(\.field) == [.reasoningEnabled])
        #expect(ModelConfigurationOverride(reasoningEnabled: false).warnings(against: block)
            .map(\.field) == [.reasoningEnabled])   // off unmeasured -> nothing sent -> warns

        // An UNRECORDED mechanism warns nothing: the legacy fallbacks honor the switch.
        let unrecorded = ModelInfo(providerID: "p", modelID: "m", displayName: "M")
        #expect(ModelConfigurationOverride(reasoningEnabled: true).warnings(against: unrecorded).isEmpty)

        // Anthropic's budgeted mechanism emits regardless of the direction capabilities —
        // the old capability-keyed warning fired falsely here.
        var anthropic = ModelInfo(providerID: "p", modelID: "m", displayName: "M")
        anthropic.reasoningControl = .anthropicThinking
        anthropic.capabilities = ModelCapabilities(states: [.reasoningCanBeEnabled: false,
                                                            .reasoningCanBeDisabled: false])
        #expect(ModelConfigurationOverride(thinkingBudget: 4096, reasoningEnabled: true)
            .warnings(against: anthropic).isEmpty)
        #expect(ModelConfigurationOverride(reasoningEnabled: false).warnings(against: anthropic).isEmpty)

        // Effort-only: off with a ladder that rejects "none" really sends nothing -> warns.
        var effortOnly = ModelInfo(providerID: "p", modelID: "m", displayName: "M")
        effortOnly.reasoningControl = .reasoningEffortOnly
        effortOnly.reasoningEffort = EffortSupport(levels: ["low", "high"])
        #expect(ModelConfigurationOverride(reasoningEnabled: false).warnings(against: effortOnly)
            .map(\.field) == [.reasoningEnabled])

        // No reasoning control at all: any explicit setting warns.
        var unsupported = ModelInfo(providerID: "p", modelID: "m", displayName: "M")
        unsupported.reasoningControl = .unsupported
        #expect(ModelConfigurationOverride(reasoningEnabled: true).warnings(against: unsupported)
            .map(\.field) == [.reasoningEnabled])
    }

    @Test("Per-call OFF outranks a configured reasoning effort; same-layer stays effort-wins")
    func perCallOffBeatsConfiguredEffort() throws {
        func provider(configOff: Bool, effort: String?) throws -> OpenAICompatibleProvider {
            OpenAICompatibleProvider(
                configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m",
                                                  reasoningEnabled: configOff ? false : nil,
                                                  reasoningEffort: effort),
                provider: ModelProvider(id: "p", name: "p", apiType: .openAICompatible,
                                        endpoint: try #require(URL(string: "https://x.test/v1"))),
                readAPIKey: { "" },
                reasoningEffortSupport: EffortSupport(levels: ["none", "high"]),
                reasoningControl: .reasoningEffortOnly)
        }
        // Per-call off wins over configured depth.
        let perCall = try provider(configOff: false, effort: "high").buildRequestBody(
            messages: [.user("hi")], tools: [],
            overrides: LLMCallOverrides(reasoningEnabled: false))
        #expect(perCall["reasoning_effort"] as? String == "none")
        // Same-layer conflict deliberately stays effort-wins.
        let sameLayer = try provider(configOff: true, effort: "high").buildRequestBody(
            messages: [.user("hi")], tools: [], overrides: LLMCallOverrides())
        #expect(sameLayer["reasoning_effort"] as? String == "high")
    }

    @Test("Anthropic: explicit ON beats an explicit legacy 0 budget and pins temperature")
    func anthropicOnBeatsZeroBudget() throws {
        let on = try anthropic(reasoningEnabled: true, thinkingBudget: 0)
            .buildRequestBody(messages: [.user("hi")], tools: [], overrides: LLMCallOverrides())
        let thinking = try #require(on["thinking"] as? [String: Any])
        #expect(thinking["budget_tokens"] as? Int == ThinkingBudget.minimumTokens)
        #expect(on["temperature"] as? Double == 1.0)
        // Switch nil + 0 budget stays the legacy no-emission shape, temperature untouched.
        let legacy = try anthropic(reasoningEnabled: nil, thinkingBudget: 0)
            .buildRequestBody(messages: [.user("hi")], tools: [], overrides: LLMCallOverrides())
        #expect(legacy["thinking"] == nil)
        #expect(legacy["temperature"] as? Double != 1.0)
    }

    @Test("thinkingBlock: no budget beside an explicit off; nil-switch bare budget still sends")
    func thinkingBlockBudgetRespectsOff() throws {
        func provider(off: Bool?, budget: Int?, capabilities: ModelCapabilities) throws -> OpenAICompatibleProvider {
            OpenAICompatibleProvider(
                configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m",
                                                  thinkingBudget: budget,
                                                  reasoningEnabled: off == true ? false : nil),
                provider: ModelProvider(id: "p", name: "p", apiType: .openAICompatible,
                                        endpoint: try #require(URL(string: "https://x.test/v1"))),
                readAPIKey: { "" },
                reasoningControl: .thinkingBlock,
                modelCapabilities: capabilities)
        }
        let caps = ModelCapabilities(states: [.reasoningCanBeDisabled: true,
                                              .thinkingSupportsTokenBudget: true])
        let off = try provider(off: true, budget: 2048, capabilities: caps)
            .buildRequestBody(messages: [.user("hi")], tools: [], overrides: LLMCallOverrides())
        let offThinking = try #require(off["thinking"] as? [String: Any])
        #expect(offThinking["type"] as? String == "disabled")
        #expect(offThinking["budget_tokens"] == nil)
        // Deliberate: nil switch + budget + measured support sends the typeless budget.
        let bare = try provider(off: nil, budget: 2048,
                                capabilities: ModelCapabilities([.thinkingSupportsTokenBudget]))
            .buildRequestBody(messages: [.user("hi")], tools: [], overrides: LLMCallOverrides())
        let bareThinking = try #require(bare["thinking"] as? [String: Any])
        #expect(bareThinking["budget_tokens"] as? Int != nil)
        #expect(bareThinking["type"] == nil)
    }

    @Test("An inverted measured budget range yields exactly one warning, the ceiling's")
    func invertedBudgetRangeSingleWarning() {
        var info = ModelInfo(providerID: "p", modelID: "m", displayName: "M")
        info.maxThinkingBudgetTokens = 512
        info.minThinkingBudgetTokens = 1024
        let warnings = ModelConfigurationOverride(thinkingBudget: 800).warnings(against: info)
        #expect(warnings.map(\.id) == ["thinkingBudget"])
        #expect(warnings.first?.message.contains("maximum") == true)
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
        #expect(ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
            control: nil, apiType: nil, capabilities: ModelCapabilities(), reasoningEnabled: true,
            thinkingBudget: nil, reasoningEffort: nil, reasoningEffortSupport: nil)).label == "unknown")
        #expect(ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
            control: .unsupported, apiType: nil, capabilities: ModelCapabilities(), reasoningEnabled: true,
            thinkingBudget: 4096, reasoningEffort: nil, reasoningEffortSupport: nil)) == .unsupported)
    }

    @Test("Planned state mirrors Anthropic emission: budget on, off, and on-without-budget")
    func plannedStateAnthropic() {
        func state(_ enabled: Bool?, _ budget: Int?) -> PlannedThinkingState {
            ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
                control: .anthropicThinking, apiType: nil, capabilities: ModelCapabilities(),
                reasoningEnabled: enabled, thinkingBudget: budget,
                reasoningEffort: nil, reasoningEffortSupport: nil))
        }
        #expect(state(nil, 4096).label == "on")
        #expect(state(false, 4096).label == "off")
        #expect(state(nil, nil).label == "off")
        // ON with no budget seeds the minimum at emission — so the resolver says ON, seeded.
        #expect(state(true, nil).label == "on")
        // A budget capability measured FALSE withholds the block (emission fails open on unknown,
        // closed only on a measured false) — the resolver mirrors it.
        #expect(ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
            control: .anthropicThinking, apiType: nil,
            capabilities: ModelCapabilities(states: [.thinkingSupportsTokenBudget: false]),
            reasoningEnabled: true, thinkingBudget: 4096,
            reasoningEffort: nil, reasoningEffortSupport: nil)).label == "unknown")
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
        let state = ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
            control: .thinkingBlock, apiType: nil,
            capabilities: ModelCapabilities([.thinkingSupportsTokenBudget]),
            reasoningEnabled: nil, thinkingBudget: 2048,
            reasoningEffort: nil, reasoningEffortSupport: nil))
        #expect(state.label == "unknown")
        #expect(state.detail.contains("budget_tokens"))
    }

    @Test("Planned state gates directions on the measured switchability, like emission")
    func plannedStateDirectionGates() {
        let canBoth = ModelCapabilities(states: [.reasoningCanBeEnabled: true,
                                                 .reasoningCanBeDisabled: true])
        let cannotOn = ModelCapabilities(states: [.reasoningCanBeEnabled: false])
        func state(_ enabled: Bool?, caps: ModelCapabilities) -> PlannedThinkingState {
            ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
                control: .thinkingBlock, apiType: nil, capabilities: caps, reasoningEnabled: enabled,
                thinkingBudget: nil, reasoningEffort: nil, reasoningEffortSupport: nil))
        }
        #expect(state(true, caps: canBoth).label == "on")
        #expect(state(false, caps: canBoth).label == "off")
        #expect(state(true, caps: cannotOn).label == "unknown")
        #expect(state(nil, caps: canBoth).label == "unknown")
    }

    @Test("Planned state: effort-only models turn off via reasoning_effort none, ladder-gated")
    func plannedStateEffortOnly() {
        func state(effortSupport: EffortSupport?) -> PlannedThinkingState {
            ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
                control: .reasoningEffortOnly, apiType: nil, capabilities: ModelCapabilities(),
                reasoningEnabled: false, thinkingBudget: nil,
                reasoningEffort: nil, reasoningEffortSupport: effortSupport))
        }
        // A NIL support means `reasoning_effort` is not measured-supported: emission fails
        // closed and sends nothing, so the honest answer is unknown — not "off".
        #expect(state(effortSupport: nil).label == "unknown")
        #expect(state(effortSupport: EffortSupport(levels: ["low", "high"])).label == "unknown")
        #expect(state(effortSupport: EffortSupport(levels: ["none", "high"])).label == "off")
        // And with no off request it is always on.
        #expect(ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
            control: .reasoningEffortOnly, apiType: nil, capabilities: ModelCapabilities(),
            reasoningEnabled: nil, thinkingBudget: nil,
            reasoningEffort: "high", reasoningEffortSupport: nil)).label == "on")
    }

    @Test("Legacy fallbacks: an unrecorded mechanism resolves per apiType the way emission does")
    func legacyFallbacksMirrorEmission() {
        func state(apiType: ProviderAPIType?, enabled: Bool?, budget: Int?,
                   caps: ModelCapabilities = ModelCapabilities()) -> PlannedThinkingState {
            ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
                control: nil, apiType: apiType, capabilities: caps,
                reasoningEnabled: enabled, thinkingBudget: budget,
                reasoningEffort: nil, reasoningEffortSupport: nil))
        }
        // Anthropic's manual path is control-independent: budget → on, off → off, ON+nil → seeded.
        #expect(state(apiType: .anthropic, enabled: nil, budget: 4096).label == "on")
        #expect(state(apiType: .anthropic, enabled: false, budget: 4096).label == "off")
        #expect(state(apiType: .anthropic, enabled: true, budget: nil).detail.contains("seeded"))
        // Alibaba's fallback is EXEMPT from the capability gates, exactly as emission is.
        let alibabaOn = state(apiType: .alibabaCloud, enabled: true, budget: 2048)
        #expect(alibabaOn.label == "on")
        #expect(alibabaOn.detail.contains("thinking_budget"))
        // A RECORDED enableThinkingFlag with empty capabilities stays strict — the exemption
        // is legacy-only.
        #expect(ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
            control: .enableThinkingFlag, apiType: .alibabaCloud,
            capabilities: ModelCapabilities(), reasoningEnabled: true, thinkingBudget: 2048,
            reasoningEffort: nil, reasoningEffortSupport: nil)).label == "unknown")
        // Gemini fallback keeps its fail-closed budget gate.
        #expect(state(apiType: .gemini, enabled: false, budget: nil,
                      caps: ModelCapabilities([.thinkingSupportsTokenBudget])).label == "off")
        #expect(state(apiType: .gemini, enabled: false, budget: nil).label == "unknown")
        // No fallback for plain OpenAI-compatible: honest unknown.
        #expect(state(apiType: .openAICompatible, enabled: true, budget: 2048)
            .detail.contains("unrecorded"))
    }

    @Test("thinkingBlock ON-detail names the budget only when the wire will carry it")
    func thinkingBlockOnDetailMirrorsBudgetGate() {
        func detail(caps: ModelCapabilities) -> String {
            ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
                control: .thinkingBlock, apiType: nil, capabilities: caps,
                reasoningEnabled: true, thinkingBudget: 2048,
                reasoningEffort: nil, reasoningEffortSupport: nil)).detail
        }
        #expect(!detail(caps: ModelCapabilities(states: [.reasoningCanBeEnabled: true]))
            .contains("budget"))
        #expect(detail(caps: ModelCapabilities(states: [.reasoningCanBeEnabled: true,
                                                        .thinkingSupportsTokenBudget: true]))
            .contains("budget"))
    }

    @Test("Resolver: explicit ON with a legacy 0 budget reports the seeded minimum, like emission")
    func resolverOnWithZeroBudgetReportsSeeded() {
        let state = ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
            control: .anthropicThinking, apiType: nil, capabilities: ModelCapabilities(),
            reasoningEnabled: true, thinkingBudget: 0,
            reasoningEffort: nil, reasoningEffortSupport: nil))
        #expect(state.label == "on")
        #expect(state.detail.contains("seeded"))
    }

    @Test("overrideSeed: floored at the documented minimum, clamped to the measured ceiling")
    func overrideSeedTable() {
        #expect(ThinkingBudget.overrideSeed(measuredMinimum: nil, measuredMaximum: nil) == 1024)
        #expect(ThinkingBudget.overrideSeed(measuredMinimum: 4096, measuredMaximum: nil) == 4096)
        #expect(ThinkingBudget.overrideSeed(measuredMinimum: 512, measuredMaximum: nil) == 1024)
        #expect(ThinkingBudget.overrideSeed(measuredMinimum: nil, measuredMaximum: 600) == 600)
        #expect(ThinkingBudget.overrideSeed(measuredMinimum: 4096, measuredMaximum: 2048) == 2048)
        #expect(ThinkingBudget.overrideSeed(measuredMinimum: nil, measuredMaximum: 0) == 0)
    }

    @Test("effectiveReasoningControl: recorded control wins; the adaptive flag fills its absence")
    func effectiveReasoningControlCoalesce() {
        var info = ModelInfo(providerID: "p", modelID: "m", displayName: "M")
        #expect(info.effectiveReasoningControl == nil)
        info.behaviorFlags.requiresAdaptiveThinking = true
        #expect(info.effectiveReasoningControl == .anthropicAdaptiveThinking)
        info.reasoningControl = .thinkingBlock
        #expect(info.effectiveReasoningControl == .thinkingBlock)
    }
}

