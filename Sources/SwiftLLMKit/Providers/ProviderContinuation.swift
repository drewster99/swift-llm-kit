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

    /// Full Gemini response parts in original order, preserving every part's
    /// content (text / functionCall) AND its thoughtSignature in one
    /// structurally-faithful payload. Replayed verbatim by `GeminiProvider`
    /// when constructing the next request — bypasses content-shape collapse
    /// (e.g. the factory turning `[thoughtPart, fcA, fcB]` into
    /// `.toolCalls([fcA, fcB])`, which would mis-align position-keyed sigs).
    ///
    /// Non-Gemini providers ignore this field. The parts are stored verbatim
    /// from the parse-side; the encoder applies tool-name-by-call-id lookup
    /// only to `functionResponse` parts (which come from the agent loop's
    /// tool execution), never to `functionCall` parts (which came verbatim
    /// from the model).
    public let geminiResponseParts: [GeminiResponsePart]?

    /// Legacy 0.0.24–0.0.25 storage for Gemini thought signatures keyed by
    /// part-array index. Shape-fragile across `.assistant(from:)` factory
    /// collapses — superseded by `geminiResponseParts` in 0.0.26. Retained
    /// for backward-compatible decoding of saved conversations.
    @available(*, deprecated, message: "Superseded by geminiResponseParts in 0.0.26. Position-keyed signatures misalign when the factory collapses a multi-part response into a single .text/.toolCalls/.mixed content. Reads from disk for legacy saved conversations still work; new code path captures and replays the full parts structure.")
    public let geminiThoughtSignatures: [String: String]?

    public init(
        anthropicThinkingBlocks: [AnthropicThinkingBlock]? = nil,
        geminiResponseParts: [GeminiResponsePart]? = nil,
        geminiThoughtSignatures: [String: String]? = nil
    ) {
        self.anthropicThinkingBlocks = anthropicThinkingBlocks
        self.geminiResponseParts = geminiResponseParts
        self.geminiThoughtSignatures = geminiThoughtSignatures
    }

    /// True if every field is nil — no continuation data carried.
    public var isEmpty: Bool {
        anthropicThinkingBlocks == nil
            && geminiResponseParts == nil
            && geminiThoughtSignatures == nil
    }

    /// Backward-compatible decoder: legacy JSON without these keys decodes
    /// cleanly with all fields nil. 0.0.24–0.0.25 saved files with the old
    /// `geminiThoughtSignatures` field still load — the field is kept for
    /// read-side compatibility even though new captures go via
    /// `geminiResponseParts`.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.anthropicThinkingBlocks = try c.decodeIfPresent([AnthropicThinkingBlock].self, forKey: .anthropicThinkingBlocks)
        self.geminiResponseParts = try c.decodeIfPresent([GeminiResponsePart].self, forKey: .geminiResponseParts)
        self.geminiThoughtSignatures = try c.decodeIfPresent([String: String].self, forKey: .geminiThoughtSignatures)
    }

    private enum CodingKeys: String, CodingKey {
        case anthropicThinkingBlocks, geminiResponseParts, geminiThoughtSignatures
    }
}

/// One part from a Gemini response. Preserves text content, functionCall,
/// and the part's thoughtSignature in a single structurally-faithful
/// payload. The Gemini encoder emits saved parts verbatim on replay.
public struct GeminiResponsePart: Sendable, Codable, Equatable {
    /// Text content of this part. `nil` for pure-functionCall parts.
    public let text: String?
    /// Function call emitted by the model. `nil` for pure-text parts.
    public let functionCall: GeminiFunctionCall?
    /// Opaque signature binding this part to the model's reasoning. Must
    /// be replayed verbatim on the corresponding outgoing part or Gemini's
    /// strict validation drops tool results downstream.
    public let thoughtSignature: String?
    /// True if this part was marked as "thought" content (only emitted by
    /// Gemini when `thinkingConfig.includeThoughts: true` is requested —
    /// off by default in our configuration).
    public let thought: Bool?

    public init(
        text: String? = nil,
        functionCall: GeminiFunctionCall? = nil,
        thoughtSignature: String? = nil,
        thought: Bool? = nil
    ) {
        self.text = text
        self.functionCall = functionCall
        self.thoughtSignature = thoughtSignature
        self.thought = thought
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.text = try c.decodeIfPresent(String.self, forKey: .text)
        self.functionCall = try c.decodeIfPresent(GeminiFunctionCall.self, forKey: .functionCall)
        self.thoughtSignature = try c.decodeIfPresent(String.self, forKey: .thoughtSignature)
        self.thought = try c.decodeIfPresent(Bool.self, forKey: .thought)
    }

    private enum CodingKeys: String, CodingKey {
        case text, functionCall, thoughtSignature, thought
    }
}

/// Gemini functionCall payload: name + raw JSON arguments. Args are stored
/// as a raw JSON string for byte-stable replay (matches what we get from
/// the wire and what we hand back to it).
public struct GeminiFunctionCall: Sendable, Codable, Equatable {
    public let name: String
    /// Raw JSON-encoded argument object captured at parse time (`"{}"` for
    /// no args). The string itself is not guaranteed byte-identical to the
    /// wire bytes Gemini originally emitted — `JSONSerialization` decides
    /// key ordering at capture. But the *outgoing* body that swift-llm-kit
    /// emits IS deterministic across replays (via `.sortedKeys` on the
    /// outer body), which is what prompt-prefix caching downstream needs.
    public let argsJSON: String

    public init(name: String, argsJSON: String) {
        self.name = name
        self.argsJSON = argsJSON
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
