import Foundation

/// Abstraction over any LLM backend that supports tool calling.
public protocol LLMProvider: Sendable {
    /// Sends a conversation to the LLM and returns the response.
    /// - Parameters:
    ///   - messages: The conversation history.
    ///   - tools: Available tool definitions the model may invoke.
    ///   - overrides: Per-call knobs that override the provider's
    ///     configured defaults for this single call. Pass
    ///     `LLMCallOverrides()` for "use defaults only".
    /// - Returns: The model's response (text, tool calls, or both).
    func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition],
        overrides: LLMCallOverrides
    ) async throws -> LLMResponse
}

extension LLMProvider {
    /// Convenience for the most common case: send with the provider's
    /// configured defaults, no per-call overrides.
    public func send(
        messages: [LLMMessage],
        tools: [LLMToolDefinition]
    ) async throws -> LLMResponse {
        try await send(messages: messages, tools: tools, overrides: LLMCallOverrides())
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

/// Session for CAPABILITY PROBES only — never production traffic. Probe calls are tiny by
/// construction (512-token caps, one-line prompts), so anything past a minute is a stalled
/// endpoint, not a long generation: the 2026-07-18 sweep's slowest legitimate call was ~15s
/// while one gpt-5.4-nano server error held a connection for 301s. 60s is ~4-6x the observed
/// legitimate ceiling; a timeout surfaces as a transport error, which every probe already
/// grades as inconclusive — a timeout can never fabricate a verdict.
public let probeURLSession: URLSession = {
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = 60
    config.timeoutIntervalForResource = 90
    return URLSession(configuration: config)
}()
