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
    /// Gemini's `generationConfig.thinkingConfig.thinkingBudget`.
    case geminiThinkingConfig

    /// Whether the mechanism carries an explicit token budget rather than only a named level.
    ///
    /// Advisory: the authoritative per-model answer is the `thinkingBudgetTokens` capability, since
    /// a model may expose the mechanism while rejecting a budget. Used for defaults and UI hints,
    /// never as the emission gate.
    public var typicallyAcceptsTokenBudget: Bool {
        switch self {
        case .anthropicThinking, .enableThinkingFlag, .geminiThinkingConfig: return true
        case .unsupported, .reasoningEffortOnly, .thinkingBlock: return false
        }
    }

    /// Label for the per-model editor picker.
    public var editorTitle: String {
        switch self {
        case .unsupported: return "No reasoning control"
        case .reasoningEffortOnly: return "`reasoning_effort` only"
        case .thinkingBlock: return "`thinking` block (type/keep)"
        case .enableThinkingFlag: return "`enable_thinking` + budget"
        case .anthropicThinking: return "Anthropic `thinking`"
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
            return "Anthropic `thinking: {type: adaptive}` or `{type: enabled, budget_tokens: N}`."
        case .geminiThinkingConfig:
            return "Gemini `generationConfig.thinkingConfig.thinkingBudget`."
        }
    }
}
