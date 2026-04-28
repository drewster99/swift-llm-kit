import Foundation

/// Token usage statistics returned by the provider after an LLM call.
public struct TokenUsage: Sendable, Codable, Equatable {
    /// Number of tokens in the input (prompt + context).
    public let inputTokens: Int
    /// Number of tokens generated in the output.
    public let outputTokens: Int
    /// Anthropic: tokens served from prompt cache (cheaper than uncached input). 0 for other providers.
    public let cacheReadTokens: Int
    /// Anthropic: tokens written to prompt cache this request. 0 for other providers.
    public let cacheWriteTokens: Int
    /// Raw JSON of the provider's full usage object, preserved verbatim so callers
    /// can recover fields not yet parsed (reasoning tokens, vendor-specific cache
    /// fields, etc.) without needing access to the original API response. The
    /// concrete shape varies by provider:
    ///   - Anthropic: the value of the `usage` object
    ///   - Gemini: the value of `usageMetadata`
    ///   - OpenAI-compatible: the value of `usage`
    ///   - Ollama: a synthesized object containing `prompt_eval_count` and `eval_count`
    /// Nil only for providers that didn't return any usage data.
    public let rawUsage: String?

    public init(
        inputTokens: Int,
        outputTokens: Int,
        cacheReadTokens: Int = 0,
        cacheWriteTokens: Int = 0,
        rawUsage: String? = nil
    ) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.cacheReadTokens = cacheReadTokens
        self.cacheWriteTokens = cacheWriteTokens
        self.rawUsage = rawUsage
    }

    /// Serializes a provider's raw usage dictionary to a compact JSON string
    /// suitable for storing in ``TokenUsage/rawUsage``. Returns nil if the
    /// dictionary isn't JSON-encodable (which shouldn't happen for usage
    /// objects in practice). Sorted keys keep the output stable for diffing.
    /// Used internally by provider adapters when constructing TokenUsage.
    static func serializeRawUsage(_ usage: [String: Any]) -> String? {
        // try? justified: rawUsage is best-effort archival data, not critical
        // path. The isValidJSONObject guard above already eliminates the only
        // realistic failure mode. If serialization still fails for an exotic
        // reason (invalid UTF-8 in a string value, etc.), nil is a valid
        // fallback — TokenUsage.rawUsage is optional and callers handle nil.
        guard JSONSerialization.isValidJSONObject(usage),
              let data = try? JSONSerialization.data(withJSONObject: usage, options: [.sortedKeys]),
              let string = String(data: data, encoding: .utf8) else {
            return nil
        }
        return string
    }
}

/// The response from an LLM call.
public struct LLMResponse: Sendable, Equatable {
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
