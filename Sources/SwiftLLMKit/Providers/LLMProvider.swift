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
    ///   - maxOutputTokensOverride: When non-nil, overrides the configuration's
    ///     `maxOutputTokens` for this single call. Useful for callers that need
    ///     a tight per-call cap (e.g. a security gatekeeper that produces only
    ///     short verdicts) without mutating the shared configuration. Each
    ///     provider is responsible for clamping the value to its own minimums
    ///     where the API requires it (e.g. extended-thinking budget floors).
    /// - Returns: The model's response (text, tool calls, or both).
    func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        toolChoice: LLMToolChoice?,
        maxOutputTokensOverride: Int?
    ) async throws -> LLMResponse
}

extension LLMProvider {
    /// Convenience overload — original two-argument call site (no toolChoice,
    /// no maxOutput override). Existing callers from before 0.0.30 continue
    /// to compile without source changes.
    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition]
    ) async throws -> LLMResponse {
        try await send(messages: messages, tools: tools, toolChoice: nil, maxOutputTokensOverride: nil)
    }

    /// Convenience overload — maxOutput override but no toolChoice. Preserves
    /// the 0.0.x three-argument signature that callers may have adopted.
    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        maxOutputTokensOverride: Int?
    ) async throws -> LLMResponse {
        try await send(messages: messages, tools: tools, toolChoice: nil, maxOutputTokensOverride: maxOutputTokensOverride)
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
