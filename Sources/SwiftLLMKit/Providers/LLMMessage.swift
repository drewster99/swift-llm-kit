import Foundation

/// Image data for multimodal LLM messages.
public struct LLMImageContent: Sendable, Equatable {
    public var data: Data
    public var mimeType: String

    public init(data: Data, mimeType: String) {
        self.data = data
        self.mimeType = mimeType
    }
}

/// A single message in an LLM conversation.
public struct LLMMessage: Codable, Sendable, Equatable {
    public enum Role: String, Codable, Sendable {
        case system
        case user
        case assistant
        case tool
        /// OpenAI's `developer` role (introduced for o-series/GPT-5). Semantically
        /// like `system` but with different precedence per OpenAI's reasoning-model
        /// docs. Providers that don't natively support it (Anthropic, Gemini, most
        /// OpenAI-compatible backends) translate it to `system`. Check
        /// `BehaviorFlags.supportsDeveloperRole` before sending.
        case developer
    }

    public enum Content: Codable, Sendable, Equatable {
        case text(String)
        case toolCalls([LLMToolCall])
        /// Assistant returned both reasoning text and tool calls in one response.
        case mixed(text: String, toolCalls: [LLMToolCall])
        case toolResult(toolCallID: String, content: String)

        private enum CodingKeys: String, CodingKey {
            case type, text, toolCalls, toolCallID, content
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .text(let value):
                try container.encode("text", forKey: .type)
                try container.encode(value, forKey: .text)
            case .toolCalls(let calls):
                try container.encode("toolCalls", forKey: .type)
                try container.encode(calls, forKey: .toolCalls)
            case .mixed(let value, let calls):
                try container.encode("mixed", forKey: .type)
                try container.encode(value, forKey: .text)
                try container.encode(calls, forKey: .toolCalls)
            case .toolResult(let id, let content):
                try container.encode("toolResult", forKey: .type)
                try container.encode(id, forKey: .toolCallID)
                try container.encode(content, forKey: .content)
            }
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type)
            switch type {
            case "text":
                let value = try container.decode(String.self, forKey: .text)
                self = .text(value)
            case "toolCalls":
                let calls = try container.decode([LLMToolCall].self, forKey: .toolCalls)
                self = .toolCalls(calls)
            case "mixed":
                let value = try container.decode(String.self, forKey: .text)
                let calls = try container.decode([LLMToolCall].self, forKey: .toolCalls)
                self = .mixed(text: value, toolCalls: calls)
            case "toolResult":
                let id = try container.decode(String.self, forKey: .toolCallID)
                let content = try container.decode(String.self, forKey: .content)
                self = .toolResult(toolCallID: id, content: content)
            default:
                throw DecodingError.dataCorruptedError(
                    forKey: .type,
                    in: container,
                    debugDescription: "Unknown content type: \(type)"
                )
            }
        }

        /// Convenience accessor for the text value, if this is a text content.
        public var textValue: String? {
            switch self {
            case .text(let value): return value
            case .mixed(let value, _): return value
            default: return nil
            }
        }
    }

    public var role: Role
    public var content: Content
    /// Image/media data for multimodal messages. Not included in Codable.
    public var images: [LLMImageContent]?
    /// Reasoning / thinking content the model produced for this turn, if any.
    /// Captured from `reasoning_content` (OpenAI-compatible) or `thinking`
    /// blocks (Anthropic). Whether it's emitted back on the next request is
    /// per-provider: gated on `BehaviorFlags.replayReasoningContent` for
    /// OpenAI-compatible providers; Anthropic uses `continuation` for the
    /// signed thinking blocks (this `reasoning` field is the text display
    /// only — for replay use `continuation`).
    public var reasoning: String?
    /// Provider-specific continuation data carried across multi-turn
    /// conversations. Currently used by Anthropic (thinking blocks with
    /// signatures) and Gemini (thoughtSignatures per part). Build a message
    /// from a real LLMResponse via `.assistant(from: response)` to preserve
    /// this automatically. See ``ProviderContinuation`` for details.
    public var continuation: ProviderContinuation?

    private enum CodingKeys: String, CodingKey {
        case role, content, reasoning, continuation
    }

    /// General initializer. **Prefer the static factory methods** for the
    /// common cases — `.user(_:)`, `.system(_:)`, `.assistant(from:)`,
    /// `.toolResult(_:callID:)`, etc. This initializer is for advanced cases
    /// (images, custom content shapes) and tests. Constructing an assistant
    /// turn from an `LLMResponse` here will silently drop the response's
    /// `reasoning` and `continuation` unless you remember to pass both —
    /// multi-turn thinking and tool-use loops may break.
    @available(*, deprecated, message: "Prefer .user(_:), .system(_:), .assistant(from:), .toolResult(_:callID:), .developer(_:). This init is for tests / advanced cases (images, custom content). When recording an assistant turn from an LLMResponse, manual construction silently drops `reasoning` and `continuation` — multi-turn thinking and tool-use loops may break.")
    public init(role: Role, content: Content, images: [LLMImageContent]? = nil, reasoning: String? = nil, continuation: ProviderContinuation? = nil) {
        self.role = role
        self.content = content
        self.images = images
        self.reasoning = reasoning
        self.continuation = continuation
    }

    /// Convenience initializer for simple text messages.
    /// **Deprecated** — prefer the role-specific factories (`.user(_:)`,
    /// `.system(_:)`, etc.) which document intent and don't risk losing
    /// reasoning / continuation when used for assistant turns.
    @available(*, deprecated, message: "Prefer .user(_:), .system(_:), .assistant(from:), .toolResult(_:callID:). For assistant turns built from a real LLMResponse, this convenience init silently loses `continuation` — multi-turn thinking may break.")
    public init(role: Role, text: String, images: [LLMImageContent]? = nil, reasoning: String? = nil) {
        self.role = role
        self.content = .text(text)
        self.images = images
        self.reasoning = reasoning
        self.continuation = nil
    }

    /// Internal initializer used by the static factories — bypasses the
    /// deprecation warnings on the public inits.
    init(
        _role: Role,
        _content: Content,
        _images: [LLMImageContent]? = nil,
        _reasoning: String? = nil,
        _continuation: ProviderContinuation? = nil
    ) {
        self.role = _role
        self.content = _content
        self.images = _images
        self.reasoning = _reasoning
        self.continuation = _continuation
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encode(content, forKey: .content)
        if let reasoning {
            try container.encode(reasoning, forKey: .reasoning)
        }
        if let continuation, !continuation.isEmpty {
            try container.encode(continuation, forKey: .continuation)
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(Role.self, forKey: .role)
        content = try container.decode(Content.self, forKey: .content)
        images = nil
        reasoning = try container.decodeIfPresent(String.self, forKey: .reasoning)
        continuation = try container.decodeIfPresent(ProviderContinuation.self, forKey: .continuation)
    }

    // MARK: - Static factories
    //
    // Role-tagged factories that read better at call sites and make the
    // assistant-from-response path impossible to misuse. Prefer these to the
    // deprecated initializers.

    /// User text message. For multimodal (text + images), use the deprecated
    /// general init with `.text` content + `images:`.
    public static func user(_ text: String) -> LLMMessage {
        LLMMessage(_role: .user, _content: .text(text))
    }

    /// User multimodal message — text plus one or more images.
    public static func user(_ text: String, images: [LLMImageContent]) -> LLMMessage {
        LLMMessage(_role: .user, _content: .text(text), _images: images)
    }

    /// System prompt.
    public static func system(_ text: String) -> LLMMessage {
        LLMMessage(_role: .system, _content: .text(text))
    }

    /// OpenAI's `developer` role (introduced for o-series/GPT-5). Providers
    /// that don't support it translate to `system` at the wire layer; check
    /// `BehaviorFlags.supportsDeveloperRole` to know which is which.
    public static func developer(_ text: String) -> LLMMessage {
        LLMMessage(_role: .developer, _content: .text(text))
    }

    /// **Build the assistant turn from a real LLMResponse — preserves
    /// text, tool calls, reasoning, and provider continuation data in one
    /// shot.** This is the misuse-resistant path: use it whenever you're
    /// recording an actual model response in the conversation history.
    /// Manual construction via the deprecated init or `.assistant(text:)`
    /// drops continuation, which breaks multi-turn thinking on Anthropic
    /// (with thinkingBudget set) and Gemini 2.5 (thinking on by default).
    public static func assistant(from response: LLMResponse) -> LLMMessage {
        let content: Content
        if response.toolCalls.isEmpty {
            content = .text(response.text ?? "")
        } else if let text = response.text, !text.isEmpty {
            content = .mixed(text: text, toolCalls: response.toolCalls)
        } else {
            content = .toolCalls(response.toolCalls)
        }
        return LLMMessage(
            _role: .assistant,
            _content: content,
            _reasoning: response.reasoning,
            _continuation: response.continuation
        )
    }

    /// Synthetic assistant text message — for tests and replay only.
    ///
    /// **For real responses, use `.assistant(from: response)`.** This factory
    /// cannot carry provider continuation data (thinking signatures), and you
    /// must pass `reasoning:` manually. Multi-turn thinking and tool-use
    /// loops will lose context if used to record an actual LLM response.
    @available(*, deprecated, message: "Synthetic — for tests / replay only. For real responses use .assistant(from:) which preserves reasoning + continuation. May lose reasoning context across turns.")
    public static func assistant(text: String, reasoning: String? = nil) -> LLMMessage {
        LLMMessage(_role: .assistant, _content: .text(text), _reasoning: reasoning)
    }

    /// Synthetic assistant message with tool calls (and optional text +
    /// reasoning) — for tests and saved-conversation reload.
    ///
    /// **For real responses, use `.assistant(from: response)`.** This factory
    /// cannot carry provider continuation data. Multi-turn thinking and
    /// tool-use loops will lose context if used to record an actual LLM
    /// response.
    @available(*, deprecated, message: "Synthetic — for tests / saved-conversation reload only. For real responses use .assistant(from:) which preserves reasoning + continuation. May lose reasoning context across turns.")
    public static func assistant(
        toolCalls: [LLMToolCall],
        text: String? = nil,
        reasoning: String? = nil
    ) -> LLMMessage {
        let content: Content
        if let text, !text.isEmpty {
            content = .mixed(text: text, toolCalls: toolCalls)
        } else {
            content = .toolCalls(toolCalls)
        }
        return LLMMessage(_role: .assistant, _content: content, _reasoning: reasoning)
    }

    /// Tool-result message linking back to the assistant call by ID.
    public static func toolResult(_ content: String, callID: String) -> LLMMessage {
        LLMMessage(_role: .tool, _content: .toolResult(toolCallID: callID, content: content))
    }

    /// Rough character count for context window estimation (~3 chars per token).
    public var estimatedCharacterCount: Int {
        // Each image is roughly 1000 tokens worth
        let imageChars = (images?.count ?? 0) * 3000
        // Reasoning is sent over the wire when `BehaviorFlags.replayReasoningContent`
        // is set, so it counts against the model's input budget. Always include it
        // in the estimate — the cost of overshooting on an unflagged model (no
        // reasoning emitted, slight over-estimate) is much less than the cost of
        // under-estimating on a flagged one (silent prune underfire).
        let reasoningChars = reasoning?.count ?? 0
        switch content {
        case .text(let s):
            return s.count + imageChars + reasoningChars
        case .toolCalls(let calls):
            return calls.reduce(0) { $0 + $1.name.count + $1.arguments.count + 20 } + imageChars + reasoningChars
        case .mixed(let text, let calls):
            return text.count + calls.reduce(0) { $0 + $1.name.count + $1.arguments.count + 20 } + imageChars + reasoningChars
        case .toolResult(let toolCallID, let content):
            return toolCallID.count + content.count + 20 + imageChars + reasoningChars
        }
    }
}
