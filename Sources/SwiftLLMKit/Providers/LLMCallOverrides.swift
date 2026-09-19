import Foundation

/// Per-call overrides applied on top of an `LLMProvider`'s underlying
/// `ModelConfiguration`. Every field is optional; a nil field means "use the
/// provider's natural default for this call."
///
/// Replaces the long parameter list that `LLMProvider.send` carried through
/// 0.0.35 (toolChoice + 6 overrides). New per-call knobs go here, not as
/// new parameters on `send`.
///
/// Field semantics across providers:
/// - `toolChoice`: each provider translates to its native shape (Anthropic
///   `tool_choice` object, OpenAI string-or-object, Gemini
///   `functionCallingConfig.mode`).
/// - `thinkingEffort`: honored by Anthropic (adaptive thinking models) and
///   OpenAI-compatible (when `behaviorFlags.supportsReasoningEffort` is set);
///   ignored by Gemini and Ollama.
/// - `maxOutputTokens`: honored by every provider EXCEPT Codex. Anthropic
///   clamps the value to `max(override, budget_tokens + 1)` when MANUAL
///   extended thinking is on (API constraint).
/// - `temperature`: honored by every provider except Codex. Anthropic forces
///   1.0 when extended thinking is on (overrides this field).
/// - `topP`: honored by every provider except Codex. Anthropic ignores when
///   extended thinking is on.
/// - `stopSequences`: honored by every provider except Codex, each in its
///   native field name (Anthropic `stop_sequences`, OpenAI `stop`, Gemini
///   `stopSequences`, Ollama `stop`).
/// - `frequencyPenalty` / `presencePenalty`: honored by OpenAI-compatible
///   and Gemini; ignored by Anthropic, Ollama and Codex (no native field).
///
/// **Codex (`.codexChatGPT`) honors `toolChoice`, `reasoningEffort`, `reasoningEnabled`,
/// `responseFormat`, `temperature` and `topP`, and drops `maxOutputTokens`, `stopSequences`
/// and the penalties.** The endpoint answers `400 {"detail":"Unsupported parameter: …"}` for
/// `max_output_tokens`, `temperature` and `top_p` (all verified live 2026-09-19), and the
/// Responses shape has no stop or penalty fields at all. Temperature is nonetheless emitted when
/// asked for — gated on `mustNeverSendTemperatureParam` exactly as the OpenAI-compatible provider
/// does — so that a probe measures the endpoint's real answer and derives that flag from it,
/// instead of a silently dropped parameter recording as "accepted". The cap is the one knob with
/// no plumbing at all: `CodexResponsesProvider` takes no cap parameter, so there is nothing to
/// "finish".
public struct LLMCallOverrides: Sendable, Equatable {
    public var toolChoice: LLMToolChoice?
    /// Per-call GENERAL effort (Anthropic `output_config.effort`).
    public var effort: String?
    /// Per-call REASONING effort (`reasoning_effort`).
    public var reasoningEffort: String?
    /// Turn reasoning on or off for this call. `nil` leaves the model's default alone.
    ///
    /// The wire shape depends on the model's ``ReasoningControl``; the provider translates. Gated
    /// on `reasoningEnableable` / `reasoningDisableable`, since Kimi documents models that support
    /// only one direction.
    public var reasoningEnabled: Bool?
    /// Per-call reasoning token budget. Gated on the `thinkingBudgetTokens` capability.
    public var thinkingBudgetTokens: Int?
    /// Retain reasoning content across turns (`thinking.keep`). Gated on `thinkingKeepAll`.
    public var keepThinking: Bool?
    /// Request structured output. Gated on the matching capability — see
    /// ``LLMResponseFormat/requiredCapability``.
    public var responseFormat: LLMResponseFormat?
    public var maxOutputTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var stopSequences: [String]?
    public var frequencyPenalty: Double?
    public var presencePenalty: Double?

    public init(
        toolChoice: LLMToolChoice? = nil,
        effort: String? = nil,
        reasoningEffort: String? = nil,
        reasoningEnabled: Bool? = nil,
        thinkingBudgetTokens: Int? = nil,
        keepThinking: Bool? = nil,
        responseFormat: LLMResponseFormat? = nil,
        maxOutputTokens: Int? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        stopSequences: [String]? = nil,
        frequencyPenalty: Double? = nil,
        presencePenalty: Double? = nil
    ) {
        self.toolChoice = toolChoice
        self.effort = effort
        self.reasoningEffort = reasoningEffort
        self.reasoningEnabled = reasoningEnabled
        self.thinkingBudgetTokens = thinkingBudgetTokens
        self.keepThinking = keepThinking
        self.responseFormat = responseFormat
        self.maxOutputTokens = maxOutputTokens
        self.temperature = temperature
        self.topP = topP
        self.stopSequences = stopSequences
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
    }
}
