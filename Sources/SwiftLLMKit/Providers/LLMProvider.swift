import Foundation

/// Abstraction over any LLM backend that supports tool calling.
public protocol LLMProvider: Sendable {
    /// Sends a conversation to the LLM and returns the response.
    /// - Parameters:
    ///   - messages: The conversation history.
    ///   - tools: Available tool definitions the model may invoke.
    ///   - toolChoice: Optional per-call control over how the model picks tools.
    ///     `nil` (default) omits the field — each provider's natural default
    ///     applies (typically `auto`). See `LLMToolChoice` for cross-provider
    ///     semantics. Meaningful only when `tools` is non-empty.
    ///   - thinkingEffortOverride: Optional per-call override of the
    ///     configuration's `thinkingEffort`. Useful for HTTP servers that
    ///     receive `reasoning_effort` per request and don't want to rebuild
    ///     the provider per call. `nil` means "use the configuration's value
    ///     (which itself may be nil)." Empty string treated as nil.
    ///   - maxOutputTokensOverride: When non-nil, overrides the configuration's
    ///     `maxOutputTokens` for this single call. Useful for callers that need
    ///     a tight per-call cap (e.g. a security gatekeeper that produces only
    ///     short verdicts) without mutating the shared configuration. Each
    ///     provider is responsible for clamping the value to its own minimums
    ///     where the API requires it (e.g. extended-thinking budget floors).
    ///   - temperatureOverride: Optional per-call override of the
    ///     configuration's `temperature`. Useful for HTTP servers that receive
    ///     `temperature` per request and don't want to rebuild the provider.
    ///     `nil` means "use the configuration's value (which itself may be
    ///     nil)." Provider-specific constraints still apply (e.g. Anthropic
    ///     forces temperature=1.0 when extended thinking is enabled, ignoring
    ///     this override).
    ///   - topPOverride: Optional per-call value for nucleus sampling.
    ///     `nil` means "don't emit the field" — provider's default applies.
    ///     `ModelConfiguration` has no `topP` field today, so this parameter
    ///     is the ONLY way to send it. Provider-specific constraints apply
    ///     (e.g. Anthropic ignores top_p when extended thinking is enabled).
    ///   - stopSequencesOverride: Optional list of strings that cause the
    ///     model to stop generation when encountered. `nil` or `[]` means
    ///     "don't emit the field." Each provider has its own wire shape:
    ///     - Anthropic: `stop_sequences: [...]`
    ///     - OpenAI-compatible: `stop: [...]`
    ///     - Gemini: `generationConfig.stopSequences: [...]`
    ///     - Ollama: `options.stop: [...]`
    ///   - frequencyPenaltyOverride: Optional per-call frequency penalty
    ///     (OpenAI's `frequency_penalty`, typically -2.0…2.0). `nil` omits
    ///     the field. Honored by OpenAI-compatible and Gemini (as
    ///     `frequencyPenalty`); ignored by Anthropic (no native field) and
    ///     Ollama (no native field via standard chat options).
    ///   - presencePenaltyOverride: Optional per-call presence penalty
    ///     (OpenAI's `presence_penalty`, typically -2.0…2.0). `nil` omits
    ///     the field. Honored by OpenAI-compatible and Gemini (as
    ///     `presencePenalty`); ignored by Anthropic and Ollama.
    /// - Returns: The model's response (text, tool calls, or both).
    func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice?,
        thinkingEffortOverride: String?,
        maxOutputTokensOverride: Int?,
        temperatureOverride: Double?,
        topPOverride: Double?,
        stopSequencesOverride: [String]?,
        frequencyPenaltyOverride: Double?,
        presencePenaltyOverride: Double?
    ) async throws -> LLMResponse
}

extension LLMProvider {
    /// Convenience overload — original two-argument call site (no toolChoice,
    /// no effort, no maxOutput override). Existing callers from before 0.0.30
    /// continue to compile without source changes.
    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition]
    ) async throws -> LLMResponse {
        try await send(messages: messages, tools: tools, toolChoice: nil, thinkingEffortOverride: nil, maxOutputTokensOverride: nil, temperatureOverride: nil, topPOverride: nil, stopSequencesOverride: nil, frequencyPenaltyOverride: nil, presencePenaltyOverride: nil)
    }

    /// Convenience overload — maxOutput override only.
    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        maxOutputTokensOverride: Int?
    ) async throws -> LLMResponse {
        try await send(messages: messages, tools: tools, toolChoice: nil, thinkingEffortOverride: nil, maxOutputTokensOverride: maxOutputTokensOverride, temperatureOverride: nil, topPOverride: nil, stopSequencesOverride: nil, frequencyPenaltyOverride: nil, presencePenaltyOverride: nil)
    }

    /// Convenience overload — toolChoice + maxOutput override (0.0.30 shape).
    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice?,
        maxOutputTokensOverride: Int?
    ) async throws -> LLMResponse {
        try await send(messages: messages, tools: tools, toolChoice: toolChoice, thinkingEffortOverride: nil, maxOutputTokensOverride: maxOutputTokensOverride, temperatureOverride: nil, topPOverride: nil, stopSequencesOverride: nil, frequencyPenaltyOverride: nil, presencePenaltyOverride: nil)
    }

    /// Convenience overload — 0.0.31 shape.
    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice?,
        thinkingEffortOverride: String?,
        maxOutputTokensOverride: Int?
    ) async throws -> LLMResponse {
        try await send(messages: messages, tools: tools, toolChoice: toolChoice, thinkingEffortOverride: thinkingEffortOverride, maxOutputTokensOverride: maxOutputTokensOverride, temperatureOverride: nil, topPOverride: nil, stopSequencesOverride: nil, frequencyPenaltyOverride: nil, presencePenaltyOverride: nil)
    }

    /// Convenience overload — 0.0.32 shape.
    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice?,
        thinkingEffortOverride: String?,
        maxOutputTokensOverride: Int?,
        temperatureOverride: Double?
    ) async throws -> LLMResponse {
        try await send(messages: messages, tools: tools, toolChoice: toolChoice, thinkingEffortOverride: thinkingEffortOverride, maxOutputTokensOverride: maxOutputTokensOverride, temperatureOverride: temperatureOverride, topPOverride: nil, stopSequencesOverride: nil, frequencyPenaltyOverride: nil, presencePenaltyOverride: nil)
    }

    /// Convenience overload — 0.0.33 shape.
    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice?,
        thinkingEffortOverride: String?,
        maxOutputTokensOverride: Int?,
        temperatureOverride: Double?,
        topPOverride: Double?
    ) async throws -> LLMResponse {
        try await send(messages: messages, tools: tools, toolChoice: toolChoice, thinkingEffortOverride: thinkingEffortOverride, maxOutputTokensOverride: maxOutputTokensOverride, temperatureOverride: temperatureOverride, topPOverride: topPOverride, stopSequencesOverride: nil, frequencyPenaltyOverride: nil, presencePenaltyOverride: nil)
    }

    /// Convenience overload — 0.0.34 shape (without penalties).
    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice?,
        thinkingEffortOverride: String?,
        maxOutputTokensOverride: Int?,
        temperatureOverride: Double?,
        topPOverride: Double?,
        stopSequencesOverride: [String]?
    ) async throws -> LLMResponse {
        try await send(messages: messages, tools: tools, toolChoice: toolChoice, thinkingEffortOverride: thinkingEffortOverride, maxOutputTokensOverride: maxOutputTokensOverride, temperatureOverride: temperatureOverride, topPOverride: topPOverride, stopSequencesOverride: stopSequencesOverride, frequencyPenaltyOverride: nil, presencePenaltyOverride: nil)
    }
}

/// A URLSession configured with generous timeouts for LLM API calls.
/// Local models (Ollama) can take minutes to generate complex responses;
/// the default 60-second URLSession timeout causes spurious failures.
///
/// - `timeoutIntervalForRequest` (10 min): max gap between data packets.
/// - `timeoutIntervalForResource` (15 min): hard cap on total request duration.
///   Without this, the default is 7 days — dead connections after sleep/wake can hang indefinitely.
public let llmURLSession: URLSession = {
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = 600
    config.timeoutIntervalForResource = 900
    return URLSession(configuration: config)
}()
