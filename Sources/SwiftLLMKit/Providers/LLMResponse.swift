import Foundation

/// Token usage statistics returned by the provider after an LLM call.
public struct TokenUsage: Sendable {
    /// Number of tokens in the input (prompt + context).
    public let inputTokens: Int
    /// Number of tokens generated in the output.
    public let outputTokens: Int

    public init(inputTokens: Int, outputTokens: Int) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
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
