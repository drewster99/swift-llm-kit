import Foundation

/// HOW a model's reasoning is switched on and off — the wire mechanism, as per-model data.
///
/// ## Why this is data and not a `switch` on `apiType`
///
/// The request builders decided this by branching on the provider's API type
/// (`provider.apiType == .alibabaCloud` for `enable_thinking`, and so on). That works only while
/// one apiType means one mechanism, and it stopped being true the moment Moonshot and DeepSeek
/// arrived: both are `openAICompatible`, alongside OpenAI, and all three want different keys —
/// `thinking{type, keep}`, `thinking` plus `reasoning_effort`, and `reasoning_effort` alone. A
/// per-apiType branch cannot tell them apart, so every new backend became another special case in
/// the body builder.
///
/// ## Why an enum rather than booleans
///
/// The mechanisms are mutually exclusive: a model does not take both `thinking{type}` and
/// `enable_thinking`. As separate flags, `usesThinkingBlock && usesEnableThinkingFlag` would be
/// representable and would describe a model that cannot exist. One value, one mechanism.
///
/// Whether reasoning may be turned on or off at all is a SEPARATE question, carried by the
/// `reasoningEnableable` / `reasoningDisableable` capabilities — Kimi documents models that support
/// only one direction, so those are two independent facts rather than part of the mechanism.
///
/// ## `nil` means unknown, and unknown is not "off"
///
/// A `nil` control is "no source has said", and the providers fall back to their existing
/// behaviour. Emitting nothing on unknown would silently disable reasoning for every model whose
/// mechanism has not yet been recorded — a regression dressed up as caution.
public enum ReasoningControl: String, Sendable, Equatable, Hashable, Codable, CaseIterable {
    /// No reasoning knob: the model either never reasons or exposes no control over it.
    ///
    /// Deliberately NOT spelled `none`: this type is almost always used as `ReasoningControl?`, and
    /// there `.none` binds to `Optional.none` — so "the model has no reasoning control" and "no
    /// source has said" would be written identically and mean opposite things.
    case unsupported
    /// Depth only, via top-level `reasoning_effort`. No on/off switch (OpenAI o-series, GPT-5).
    case reasoningEffortOnly
    /// A top-level `thinking` object taking `{"type": "enabled" | "disabled"}`, optionally with
    /// `keep`. Moonshot's Kimi models and some DeepSeek models. Often pairs with `reasoning_effort`
    /// for depth, which is why depth lives in ``EffortSupport`` rather than here.
    case thinkingBlock
    /// A boolean `enable_thinking` alongside a separate `thinking_budget` (Alibaba Cloud DashScope).
    case enableThinkingFlag
    /// Anthropic's `thinking` object: `{"type": "adaptive"}` or `{"type": "enabled",
    /// "budget_tokens": N}`. Depth is steered by the separate general-effort parameter.
    case anthropicThinking
    /// Anthropic's ADAPTIVE thinking: `{"type": "adaptive"}`, with depth steered by the separate
    /// `output_config.effort` rather than a token budget.
    ///
    /// A distinct mechanism, not a variant of ``anthropicThinking``: the newest Claude models refuse
    /// `{"type": "enabled"}` outright — *"thinking.type.enabled is not supported for this model. Use
    /// thinking.type.adaptive and output_config.effort"* — so a probe that only knows the budgeted
    /// form records them as unable to reason at all. Observed 2026-08-05 on opus-4-7, opus-4-8,
    /// opus-5, sonnet-5 and fable-5. It carries NO token budget, which is why it needs its own case
    /// rather than a flag: the budget probes must skip it.
    case anthropicAdaptiveThinking
    /// Gemini's `generationConfig.thinkingConfig.thinkingBudget`.
    case geminiThinkingConfig

    /// Whether this mechanism carries a reasoning budget expressed IN TOKENS.
    ///
    /// Exhaustive so a mechanism added later cannot silently default to either answer. It gates the
    /// budget range probes, which measure nothing on an endpoint that has no such parameter: it
    /// ignores the field, accepts every value, and the searches converge on nonsense — a 1-token
    /// floor and a fabricated ceiling, as happened to 72 models on 2026-08-04.
    ///
    /// ``reasoningEffortOnly`` is the case that matters here. OpenAI's reasoning models genuinely
    /// reason, so "does it reason?" is the wrong question to gate a TOKEN budget on; depth there is
    /// a named level, not a token count, and asking for one measures the endpoint's tolerance for
    /// unknown keys.
    public var carriesTokenBudget: Bool {
        switch self {
        case .thinkingBlock, .enableThinkingFlag, .anthropicThinking, .geminiThinkingConfig: return true
        // Adaptive takes its depth from `output_config.effort`, not a token count.
        case .anthropicAdaptiveThinking, .reasoningEffortOnly, .unsupported: return false
        }
    }

    /// How this mechanism's OFF request reads on the wire, for evidence strings.
    ///
    /// So a verdict names the field that was actually sent. The audit trail said
    /// `thinking.type=disabled` for all five mechanisms, describing a field that never went out for
    /// four of them.
    public var disableWireDescription: String {
        switch self {
        case .thinkingBlock, .anthropicThinking, .anthropicAdaptiveThinking: return "thinking.type=disabled"
        case .reasoningEffortOnly:               return "reasoning_effort=none"
        case .enableThinkingFlag:                return "enable_thinking=false"
        case .geminiThinkingConfig:              return "thinkingBudget=0"
        case .unsupported:                       return "no reasoning switch"
        }
    }

    /// The raw body fields that turn reasoning OFF for this mechanism, or nil when there is nothing
    /// to turn off.
    ///
    /// Callers that need reasoning disabled for an unrelated measurement — the tool-choice probe,
    /// which Moonshot and DeepSeek refuse while thinking is on — must send THIS rather than assume a
    /// shape. Assuming `thinking: {type: disabled}` everywhere is what made an OpenAI endpoint
    /// answer "Unrecognized request argument supplied: thinking" and lose the finding entirely.
    public var reasoningDisableOverrides: [String: AnyCodable]? {
        switch self {
        case .thinkingBlock, .anthropicThinking, .anthropicAdaptiveThinking:
            return ["thinking": .dictionary(["type": .string("disabled")])]
        case .reasoningEffortOnly:
            return ["reasoning_effort": .string("none")]
        case .enableThinkingFlag:
            return ["enable_thinking": .bool(false)]
        case .geminiThinkingConfig:
            return ["generationConfig": .dictionary(["thinkingConfig":
                        .dictionary(["thinkingBudget": .int(0)])])]
        case .unsupported:
            return nil
        }
    }

    /// The raw body fields that put a reasoning budget of `budget` on the wire for this mechanism,
    /// or nil when the mechanism has no token budget.
    ///
    /// Exists so the budget probes can FORCE the field instead of going through production emission.
    /// They were the only probes in the library that didn't, and both consequences were live:
    /// production gates the field on `thinkingSupportsTokenBudget`, which nothing establishes — so
    /// for `.thinkingBlock` models no budget reached the wire at all and every value was "accepted",
    /// converging on the reservation arithmetic (kimi-k3 recorded 1,047,552 = its context window
    /// minus the reservation). And `ThinkingBudget.effective` floors the value, so a probe asking
    /// for 1023 sent 1024, the bisection walked down to a fabricated floor of 1, and that 1 was
    /// written to the catalog where it LOWERED the production floor — making the next run measure
    /// something different again. A probe whose input is filtered by the thing it is measuring
    /// cannot measure it.
    ///
    /// `pairedMaxTokens` is emitted only where the budget is drawn from the output allowance; a
    /// separate allowance needs no pairing and forcing one would cap the reply instead.
    public func budgetForcingOverrides(budget: Int, pairedMaxTokens: Int) -> [String: AnyCodable]? {
        switch self {
        case .anthropicThinking, .thinkingBlock:
            return ["thinking": .dictionary(["type": .string("enabled"),
                                             "budget_tokens": .int(budget)]),
                    "max_tokens": .int(pairedMaxTokens)]
        case .anthropicAdaptiveThinking:
            return nil
        case .enableThinkingFlag:
            return ["enable_thinking": .bool(true), "thinking_budget": .int(budget)]
        case .geminiThinkingConfig:
            return ["generationConfig": .dictionary(["thinkingConfig":
                        .dictionary(["thinkingBudget": .int(budget)])])]
        case .reasoningEffortOnly, .unsupported:
            return nil
        }
    }

    /// Label for the per-model editor picker.
    public var editorTitle: String {
        switch self {
        case .unsupported: return "No reasoning control"
        case .reasoningEffortOnly: return "`reasoning_effort` only"
        case .thinkingBlock: return "`thinking` block (type/keep)"
        case .enableThinkingFlag: return "`enable_thinking` + budget"
        case .anthropicThinking: return "Anthropic `thinking` + budget"
        case .anthropicAdaptiveThinking: return "Anthropic `thinking` (adaptive)"
        case .geminiThinkingConfig: return "Gemini `thinkingConfig`"
        }
    }

    /// One-line explanation for the editor, naming the wire shape so the choice is checkable.
    public var editorDescription: String {
        switch self {
        case .unsupported:
            return "The model exposes no way to turn reasoning on or off."
        case .reasoningEffortOnly:
            return "Depth only, via top-level `reasoning_effort`. No on/off switch."
        case .thinkingBlock:
            return "Top-level `thinking: {type: enabled|disabled}`, optionally with `keep`."
        case .enableThinkingFlag:
            return "Top-level `enable_thinking: true` with a separate `thinking_budget`."
        case .anthropicThinking:
            return "Anthropic `thinking: {type: enabled, budget_tokens: N}`."
        case .anthropicAdaptiveThinking:
            return "Anthropic `thinking: {type: adaptive}`; depth comes from `output_config.effort`."
        case .geminiThinkingConfig:
            return "Gemini `generationConfig.thinkingConfig.thinkingBudget`."
        }
    }
}
