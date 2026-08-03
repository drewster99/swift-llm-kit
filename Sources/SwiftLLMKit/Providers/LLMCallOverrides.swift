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
/// - `maxOutputTokens`: honored by every provider. Anthropic clamps the value
///   to `max(override, budget_tokens + 1)` when MANUAL extended thinking is
///   on (API constraint).
/// - `temperature`: honored by every provider. Anthropic forces 1.0 when
///   extended thinking is on (overrides this field).
/// - `topP`: honored by every provider. Anthropic ignores when extended
///   thinking is on.
/// - `stopSequences`: honored by every provider, each in its native field
///   name (Anthropic `stop_sequences`, OpenAI `stop`, Gemini
///   `stopSequences`, Ollama `stop`).
/// - `frequencyPenalty` / `presencePenalty`: honored by OpenAI-compatible
///   and Gemini; ignored by Anthropic and Ollama (no native field).
public struct LLMCallOverrides: Sendable, Equatable {
    public var toolChoice: LLMToolChoice?
    /// Per-call GENERAL effort (Anthropic `output_config.effort`).
    public var effort: String?
    /// Per-call REASONING effort (`reasoning_effort`).
    public var reasoningEffort: String?
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
        self.maxOutputTokens = maxOutputTokens
        self.temperature = temperature
        self.topP = topP
        self.stopSequences = stopSequences
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
    }
}
