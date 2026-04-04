import Foundation

/// Token usage statistics returned by the provider after an LLM call.
public struct TokenUsage: Sendable, Codable {
    /// Number of tokens in the input (prompt + context).
    public let inputTokens: Int
    /// Number of tokens generated in the output.
    public let outputTokens: Int
    /// Anthropic: tokens served from prompt cache (cheaper than uncached input). 0 for other providers.
    public let cacheReadTokens: Int
    /// Anthropic: tokens written to prompt cache this request. 0 for other providers.
    public let cacheWriteTokens: Int

    public init(inputTokens: Int, outputTokens: Int, cacheReadTokens: Int = 0, cacheWriteTokens: Int = 0) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.cacheReadTokens = cacheReadTokens
        self.cacheWriteTokens = cacheWriteTokens
    }
}

/// The response from an LLM call.
public struct LLMResponse: Sendable {
    /// The text content of the response, if any.
    public let text: String?
    /// Tool calls requested by the model, if any.
    public let toolCalls: [LLMToolCall]
    /// Reasoning/thinking content from models that support it (e.g., DeepSeek-R1, o1).
    /// Not part of the visible response — used for inspector display only.
    public let reasoning: String?
    /// Token usage statistics from the provider, if available.
    public let usage: TokenUsage?

    public init(text: String? = nil, toolCalls: [LLMToolCall] = [], reasoning: String? = nil, usage: TokenUsage? = nil) {
        self.text = text
        self.toolCalls = toolCalls
        self.reasoning = reasoning
        self.usage = usage
    }
}
