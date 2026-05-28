import Foundation

/// Provider-specific "thinking continuation" blobs that must be preserved
/// across multi-turn conversations to keep extended-thinking / reasoning
/// continuity intact.
///
/// Different providers encode their continuation differently:
/// - **Anthropic** returns one or more `thinking` content blocks containing
///   `thinking` text plus an opaque `signature`. To continue a multi-turn
///   conversation (especially tool-use loops) the client MUST replay those
///   blocks unchanged at the start of the assistant turn — Anthropic's docs
///   say: "you must pass thinking blocks back to the API, and you must
///   include the complete unmodified block back to the API."
/// - **Gemini 2.5** returns a `thoughtSignature` blob on each response part
///   (sibling to `functionCall` / `text`). To continue a multi-turn
///   conversation the signature for the corresponding original part must be
///   re-attached when replaying the assistant turn. Without it, Gemini's
///   strict-validation path discards tool results silently.
/// - **OpenAI** (and DeepSeek/etc.) handle reasoning continuity differently:
///   OpenAI's own o-series keep reasoning state server-side and don't need
///   client replay; DeepSeek-R1 needs `reasoning_content` replayed and that's
///   already plumbed via `LLMMessage.reasoning` (no continuation needed).
///
/// All fields are optional. Each provider parses only what it knows and
/// ignores other providers' blobs — a hydra-style rotation that puts a
/// previous provider's continuation in the conversation is safe: the new
/// provider just doesn't look at it.
public struct ProviderContinuation: Sendable, Codable, Equatable {
    /// Anthropic `thinking` content blocks in original order. Replayed verbatim
    /// at the start of the assistant turn during multi-turn / tool-use flows.
    public let anthropicThinkingBlocks: [AnthropicThinkingBlock]?

    /// Gemini per-part thought signatures, keyed by the part index in the
    /// original response. Replayed as sibling fields on the corresponding
    /// outgoing parts. (Keys are stringified Ints to keep the value Codable
    /// without a custom encoder; valid keys are "0", "1", "2", ...)
    public let geminiThoughtSignatures: [String: String]?

    public init(
        anthropicThinkingBlocks: [AnthropicThinkingBlock]? = nil,
        geminiThoughtSignatures: [String: String]? = nil
    ) {
        self.anthropicThinkingBlocks = anthropicThinkingBlocks
        self.geminiThoughtSignatures = geminiThoughtSignatures
    }

    /// True if every field is nil — no continuation data carried.
    public var isEmpty: Bool {
        anthropicThinkingBlocks == nil && geminiThoughtSignatures == nil
    }

    /// Backward-compatible decoder: legacy JSON without these keys decodes
    /// cleanly with all fields nil.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.anthropicThinkingBlocks = try c.decodeIfPresent([AnthropicThinkingBlock].self, forKey: .anthropicThinkingBlocks)
        self.geminiThoughtSignatures = try c.decodeIfPresent([String: String].self, forKey: .geminiThoughtSignatures)
    }

    private enum CodingKeys: String, CodingKey {
        case anthropicThinkingBlocks, geminiThoughtSignatures
    }
}

/// One Anthropic `thinking` content block from a response. The `signature`
/// must be replayed unchanged when continuing the conversation. The `thinking`
/// text content can be empty for "omitted-display" thinking; what matters
/// for continuity is the signature.
public struct AnthropicThinkingBlock: Sendable, Codable, Equatable {
    public let thinking: String
    public let signature: String

    public init(thinking: String, signature: String) {
        self.thinking = thinking
        self.signature = signature
    }
}
