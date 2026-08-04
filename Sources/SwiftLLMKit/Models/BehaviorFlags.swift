import Foundation

/// One runtime behavior knob. The set of cases IS the set of typed flags on ``BehaviorFlags`` —
/// adding a flag means adding a case here, a matching stored `var` on ``BehaviorFlags`` (so
/// providers can read it directly), and a line in each subscript below. Editors and label lists
/// iterate `allCases`, so a new flag surfaces everywhere automatically.
public enum BehaviorFlag: String, CaseIterable, Sendable, Hashable {
    case glmTemplateSalvage
    case useMaxCompletionTokens
    case disableParallelToolCalls
    case replayReasoningContent
    case supportsDeveloperRole
    case requiresAdaptiveThinking
    case mustNeverSendTemperatureParam
    case supportsTrailingSystemMessage

    /// Short human-readable label, used by ``BehaviorFlags/displayLabels``.
    public var label: String {
        switch self {
        case .glmTemplateSalvage:            return "GLM salvage"
        case .useMaxCompletionTokens:        return "max_completion_tokens"
        case .disableParallelToolCalls:      return "no parallel tools"
        case .replayReasoningContent:        return "replay reasoning"
        case .supportsDeveloperRole:         return "developer role"
        case .requiresAdaptiveThinking:      return "adaptive thinking"
        case .mustNeverSendTemperatureParam: return "no temperature"
        case .supportsTrailingSystemMessage: return "trailing system"
        }
    }

    /// Full title for the per-model behavior-flags editor. Exhaustive so a new case can't ship without one.
    public var editorTitle: String {
        switch self {
        case .glmTemplateSalvage:            return "GLM template salvage"
        case .useMaxCompletionTokens:        return "Use `max_completion_tokens`"
        case .disableParallelToolCalls:      return "Disable parallel tool calls"
        case .replayReasoningContent:        return "Replay reasoning content"
        case .supportsDeveloperRole:         return "Supports `developer` role"
        case .requiresAdaptiveThinking:      return "Requires adaptive thinking"
        case .mustNeverSendTemperatureParam: return "Never send temperature"
        case .supportsTrailingSystemMessage: return "Supports trailing system message"
        }
    }

    /// One-line help for the per-model behavior-flags editor. Exhaustive by design.
    public var editorDescription: String {
        switch self {
        case .glmTemplateSalvage:
            return "Recover tool-call args from `<arg_key>/<arg_value>` blocks the model leaks into `content`, and strip GLM chat-template control tokens. Required for GLM-4 / GLM-5 on most adapters; harmless on non-GLM responses."
        case .useMaxCompletionTokens:
            return "Send `max_completion_tokens` instead of `max_tokens` on chat completions. Required for OpenAI GPT-5 / o-series; rejected by DeepSeek and most other OpenAI-compatible backends."
        case .disableParallelToolCalls:
            return "Omit `parallel_tool_calls` from the request. It's sent as `true` by default; turn this ON only for a strict endpoint that rejects the unknown field with HTTP 400."
        case .replayReasoningContent:
            return "Replay each assistant message's `reasoning_content` on later requests. Required by some thinking models that 400 without it (e.g. DeepSeek V4 Pro); rejected by others (e.g. deepseek-reasoner), so opt-in per model."
        case .supportsDeveloperRole:
            return "Model accepts OpenAI's `developer` message role (o-series / GPT-5). When off, `developer` messages are downgraded to `system` — the safe default for non-OpenAI backends."
        case .requiresAdaptiveThinking:
            return "Anthropic models that require `thinking: {type: \"adaptive\"}` and reject the legacy manual budget format with HTTP 400 (Claude Opus 4.7 / 4.8). When on, the configured thinking budget is ignored."
        case .mustNeverSendTemperatureParam:
            return "Model rejects the `temperature` parameter entirely — any value (even 0) fails with HTTP 400 (OpenAI o-series / GPT-5 reasoning family). When on, temperature is omitted from every request."
        case .supportsTrailingSystemMessage:
            return "Model reads a `{role: system}` steering turn placed as the LAST message, immediately after the final user turn. When on, the provider leaves such a trailing system message in place instead of hoisting it into the top-level system prompt. Probe-established (Anthropic gates it per model, and OpenAI-compatible / Ollama backends vary); force only to correct a stale verdict."
        }
    }
}

/// Per-(provider+model) behavior flags consumed by providers at request time.
///
/// Flags here are *runtime behavior* knobs, not catalog metadata — they shape
/// how a provider builds its request or interprets its response for a specific
/// model. They live alongside `ModelInfo.capabilities` (which describes what a
/// model *can* do) but answer a different question: "what should this client
/// do differently when talking to this model?"
///
/// **Default = no-op.** Every flag defaults to `false` / `nil`. A model with no
/// flags set behaves exactly like a model from before this struct existed —
/// no quirks-handling, no salvage, no special request shapes.
///
/// **Resolution layering** (highest priority wins, same as `ModelMetadataOverride`):
/// user override → provider API → app-bundled defaults → all-false
///
/// Bundled defaults populate flags for the models we know about (GLM hosts,
/// OpenAI GPT-5/o-series, Mistral, etc.). User overrides force-replace.
public struct BehaviorFlags: Codable, Sendable, Equatable {
    /// Run GLM chat-template salvage on this model's responses.
    ///
    /// When true, recover tool-call args from `<tool_call>...<arg_key>K</arg_key>
    /// <arg_value>V</arg_value>...</tool_call>` blocks the model leaks into
    /// `content` (because some adapters don't fully translate GLM's chat-template
    /// form), and strip `<|...|>` / `<tool_call>` / `<tool_response>` markers from
    /// assistant text so they don't poison the next request.
    ///
    /// When false (default), no GLM-specific handling runs even if those markers
    /// happen to appear in `content` — leakage tolerance is opt-in per-model.
    public var glmTemplateSalvage: Bool = false

    /// Send `max_completion_tokens` instead of `max_tokens` on chat completions.
    ///
    /// Required by OpenAI's GPT-5 / o-series; rejected by every other provider.
    /// Used to be a hardcoded `provider.id == BuiltInProviders.ID.openai` check
    /// in `OpenAICompatibleProvider`; now bundled defaults flag the OpenAI models
    /// and the provider just reads this.
    public var useMaxCompletionTokens: Bool = false

    /// Opt a provider/model OUT of `parallel_tool_calls: true`.
    ///
    /// The client sends `parallel_tool_calls: true` on every OpenAI-compatible
    /// request by default: the agent loop pairs multi-call turns safely (matched
    /// tool_call/result counts, even on truncation), and most endpoints default
    /// the param to `true` anyway, so an unconditional send is a no-op there.
    /// Set this only for the rare endpoint that STRICTLY validates the request
    /// body and rejects the unknown `parallel_tool_calls` field with HTTP 400 —
    /// then the client omits it and lets that endpoint's own default apply.
    /// Default false (send it). This is the per-provider opt-out.
    public var disableParallelToolCalls: Bool = false

    /// Replay each assistant message's `reasoning_content` on subsequent
    /// chat-completions requests.
    ///
    /// Required by some "thinking" / reasoning models that fail with HTTP 400
    /// "The `reasoning_content` in the thinking mode must be passed back to the
    /// API." (observed on DeepSeek V4 Pro). The provider captures
    /// `reasoning_content` from the response and stores it on the assistant
    /// `LLMMessage`; when this flag is true, the encoder emits it back as a
    /// `reasoning_content` field on the outgoing message.
    ///
    /// Opt-in per model because the same field is *rejected* by other thinking
    /// models (e.g. `deepseek-reasoner`'s docs explicitly say to discard
    /// `reasoning_content` before the next round). When false, reasoning is
    /// captured but never sent — safe default.
    public var replayReasoningContent: Bool = false

    /// Whether the model accepts OpenAI's `developer` message role. OpenAI's
    /// own o-series / GPT-5 do; most other OpenAI-compatible backends (z.ai,
    /// Mistral, DeepSeek, etc.) don't and will reject or ignore it.
    /// `OpenAICompatibleProvider` downgrades `developer` messages to `system`
    /// when this flag is false. Default false — opt-in per model via bundled
    /// metadata or user override.
    public var supportsDeveloperRole: Bool = false

    /// Anthropic models that REQUIRE adaptive thinking
    /// (`thinking: {type: "adaptive"}`) and REJECT the legacy manual format
    /// (`thinking: {type: "enabled", budget_tokens: N}`) with HTTP 400.
    /// Applies to Claude Opus 4.7, 4.8, and Mythos Preview. When true,
    /// `AnthropicProvider` emits `{type: "adaptive"}` regardless of
    /// `ModelConfiguration.thinkingBudget` (the budget field is silently
    /// ignored — Anthropic doesn't accept it alongside adaptive). Default
    /// false — older models (Opus 4.5 and earlier, Sonnet 4.6 and earlier
    /// where manual still works) keep the legacy path.
    public var requiresAdaptiveThinking: Bool = false

    // The `supportsReasoningEffort` flag that stood here is RETIRED. What it recorded — "this
    // model accepts a top-level `reasoning_effort`" — is now `ModelInfo.reasoningEffort`, an
    // `EffortSupport` that also carries WHICH values are legal. The 18 bundled entries that set
    // this flag were migrated to `.supportedLevelsUnknown`, which is exactly what it meant.

    /// Models that reject the `temperature` request parameter entirely — sending ANY value
    /// (including 0) fails with HTTP 400. OpenAI's o-series (o1, o3, o4-mini) and the GPT-5
    /// reasoning family accept only their fixed internal default. When true, every provider
    /// OMITS temperature from the request — both the per-call override AND the configured
    /// value are dropped. Default false: temperature is sent as usual. Named for the action
    /// it forces so the meaning and the default (false = send normally) are both unambiguous.
    public var mustNeverSendTemperatureParam: Bool = false

    /// Model reads a `{role: system}` steering turn placed as the LAST message, immediately after
    /// the final user turn. When true, providers that normally hoist system messages out of the
    /// conversation (Anthropic's top-level `system`, OpenAI-compatible's `messages[0]`, Ollama's
    /// leading system) instead leave exactly that one trailing system message in place, encoded as
    /// role `system`. Established by ``ModelProfile/trailingSystemMessage``; nothing publishes it,
    /// so it is probe-only. Default false — the trailing turn is hoisted like every other system
    /// message, the behavior from before this flag existed.
    public var supportsTrailingSystemMessage: Bool = false

    /// Free-form key/value bag for one-off provider tweaks that haven't earned
    /// a typed field yet. Promoted fields must always have a Codable default
    /// so older persisted state still decodes cleanly.
    ///
    /// Example future use: `extras["responseFormat"] = "json_object"` for a
    /// model that needs a specific response format to behave.
    public var extras: [String: String] = [:]

    public init(
        glmTemplateSalvage: Bool = false,
        useMaxCompletionTokens: Bool = false,
        disableParallelToolCalls: Bool = false,
        replayReasoningContent: Bool = false,
        supportsDeveloperRole: Bool = false,
        requiresAdaptiveThinking: Bool = false,
        mustNeverSendTemperatureParam: Bool = false,
        supportsTrailingSystemMessage: Bool = false,
        extras: [String: String] = [:]
    ) {
        self.glmTemplateSalvage = glmTemplateSalvage
        self.useMaxCompletionTokens = useMaxCompletionTokens
        self.disableParallelToolCalls = disableParallelToolCalls
        self.replayReasoningContent = replayReasoningContent
        self.supportsDeveloperRole = supportsDeveloperRole
        self.requiresAdaptiveThinking = requiresAdaptiveThinking
        self.mustNeverSendTemperatureParam = mustNeverSendTemperatureParam
        self.supportsTrailingSystemMessage = supportsTrailingSystemMessage
        self.extras = extras
    }

    /// Read/write a flag by its ``BehaviorFlag`` case. The typed `var`s remain the storage (and the
    /// direct read path providers use); this is the enum-keyed view editors and iteration use.
    public subscript(flag: BehaviorFlag) -> Bool {
        get {
            switch flag {
            case .glmTemplateSalvage:            return glmTemplateSalvage
            case .useMaxCompletionTokens:        return useMaxCompletionTokens
            case .disableParallelToolCalls:      return disableParallelToolCalls
            case .replayReasoningContent:        return replayReasoningContent
            case .supportsDeveloperRole:         return supportsDeveloperRole
            case .requiresAdaptiveThinking:      return requiresAdaptiveThinking
            case .mustNeverSendTemperatureParam: return mustNeverSendTemperatureParam
            case .supportsTrailingSystemMessage: return supportsTrailingSystemMessage
            }
        }
        set {
            switch flag {
            case .glmTemplateSalvage:            glmTemplateSalvage = newValue
            case .useMaxCompletionTokens:        useMaxCompletionTokens = newValue
            case .disableParallelToolCalls:      disableParallelToolCalls = newValue
            case .replayReasoningContent:        replayReasoningContent = newValue
            case .supportsDeveloperRole:         supportsDeveloperRole = newValue
            case .requiresAdaptiveThinking:      requiresAdaptiveThinking = newValue
            case .mustNeverSendTemperatureParam: mustNeverSendTemperatureParam = newValue
            case .supportsTrailingSystemMessage: supportsTrailingSystemMessage = newValue
            }
        }
    }

    /// True when every flag is at its default value. Useful to short-circuit
    /// equality checks and to skip persisting empty entries.
    public var isAllDefault: Bool {
        BehaviorFlag.allCases.allSatisfy { !self[$0] } && extras.isEmpty
    }

    /// Short human-readable labels for each non-default flag — for display in the Settings UI.
    /// Order is the ``BehaviorFlag`` case order (a stable left-to-right scan). `extras` keys are
    /// appended last with a `*` prefix.
    public var displayLabels: [String] {
        var out = BehaviorFlag.allCases.filter { self[$0] }.map(\.label)
        for key in extras.keys.sorted() {
            out.append("*\(key)")
        }
        return out
    }

    // MARK: - Codable (backward-compatible)

    private enum CodingKeys: String, CodingKey {
        case glmTemplateSalvage, useMaxCompletionTokens, disableParallelToolCalls, replayReasoningContent, supportsDeveloperRole, requiresAdaptiveThinking, mustNeverSendTemperatureParam, supportsTrailingSystemMessage, extras
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        glmTemplateSalvage = try c.decodeIfPresent(Bool.self, forKey: .glmTemplateSalvage) ?? false
        useMaxCompletionTokens = try c.decodeIfPresent(Bool.self, forKey: .useMaxCompletionTokens) ?? false
        disableParallelToolCalls = try c.decodeIfPresent(Bool.self, forKey: .disableParallelToolCalls) ?? false
        replayReasoningContent = try c.decodeIfPresent(Bool.self, forKey: .replayReasoningContent) ?? false
        supportsDeveloperRole = try c.decodeIfPresent(Bool.self, forKey: .supportsDeveloperRole) ?? false
        requiresAdaptiveThinking = try c.decodeIfPresent(Bool.self, forKey: .requiresAdaptiveThinking) ?? false
        mustNeverSendTemperatureParam = try c.decodeIfPresent(Bool.self, forKey: .mustNeverSendTemperatureParam) ?? false
        supportsTrailingSystemMessage = try c.decodeIfPresent(Bool.self, forKey: .supportsTrailingSystemMessage) ?? false
        extras = try c.decodeIfPresent([String: String].self, forKey: .extras) ?? [:]
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        // Skip encoding default-valued fields so on-disk JSON stays compact and
        // round-trips cleanly through older clients that don't know these keys.
        if glmTemplateSalvage { try c.encode(glmTemplateSalvage, forKey: .glmTemplateSalvage) }
        if useMaxCompletionTokens { try c.encode(useMaxCompletionTokens, forKey: .useMaxCompletionTokens) }
        if disableParallelToolCalls { try c.encode(disableParallelToolCalls, forKey: .disableParallelToolCalls) }
        if replayReasoningContent { try c.encode(replayReasoningContent, forKey: .replayReasoningContent) }
        if supportsDeveloperRole { try c.encode(supportsDeveloperRole, forKey: .supportsDeveloperRole) }
        if requiresAdaptiveThinking { try c.encode(requiresAdaptiveThinking, forKey: .requiresAdaptiveThinking) }
        if mustNeverSendTemperatureParam { try c.encode(mustNeverSendTemperatureParam, forKey: .mustNeverSendTemperatureParam) }
        if supportsTrailingSystemMessage { try c.encode(supportsTrailingSystemMessage, forKey: .supportsTrailingSystemMessage) }
        if !extras.isEmpty { try c.encode(extras, forKey: .extras) }
    }
}

/// Optional patch over a `BehaviorFlags` value. Mirrors `ModelCapabilitiesOverride`
/// — non-nil fields replace the corresponding flag during the layered enrichment
/// pipeline; nil fields leave the flag alone.
///
/// In `forceReplace: false` mode (gap-fill / bundled / LiteLLM equivalent), a flag
/// only upgrades from `false` → `true`; the override can't downgrade. In
/// `forceReplace: true` mode (user overrides), any non-nil value replaces — so
/// users can explicitly disable a flag that bundled defaults turned on.
public struct BehaviorFlagsOverride: Codable, Sendable, Equatable {
    public var glmTemplateSalvage: Bool?
    public var useMaxCompletionTokens: Bool?
    public var disableParallelToolCalls: Bool?
    public var replayReasoningContent: Bool?
    public var supportsDeveloperRole: Bool?
    public var requiresAdaptiveThinking: Bool?
    public var mustNeverSendTemperatureParam: Bool?
    public var supportsTrailingSystemMessage: Bool?
    public var extras: [String: String]?

    public init(
        glmTemplateSalvage: Bool? = nil,
        useMaxCompletionTokens: Bool? = nil,
        disableParallelToolCalls: Bool? = nil,
        replayReasoningContent: Bool? = nil,
        supportsDeveloperRole: Bool? = nil,
        requiresAdaptiveThinking: Bool? = nil,
        mustNeverSendTemperatureParam: Bool? = nil,
        supportsTrailingSystemMessage: Bool? = nil,
        extras: [String: String]? = nil
    ) {
        self.glmTemplateSalvage = glmTemplateSalvage
        self.useMaxCompletionTokens = useMaxCompletionTokens
        self.disableParallelToolCalls = disableParallelToolCalls
        self.replayReasoningContent = replayReasoningContent
        self.supportsDeveloperRole = supportsDeveloperRole
        self.requiresAdaptiveThinking = requiresAdaptiveThinking
        self.mustNeverSendTemperatureParam = mustNeverSendTemperatureParam
        self.supportsTrailingSystemMessage = supportsTrailingSystemMessage
        self.extras = extras
    }

    /// Read/write an override field by its ``BehaviorFlag`` case. `nil` = no override (inherit).
    public subscript(flag: BehaviorFlag) -> Bool? {
        get {
            switch flag {
            case .glmTemplateSalvage:            return glmTemplateSalvage
            case .useMaxCompletionTokens:        return useMaxCompletionTokens
            case .disableParallelToolCalls:      return disableParallelToolCalls
            case .replayReasoningContent:        return replayReasoningContent
            case .supportsDeveloperRole:         return supportsDeveloperRole
            case .requiresAdaptiveThinking:      return requiresAdaptiveThinking
            case .mustNeverSendTemperatureParam: return mustNeverSendTemperatureParam
            case .supportsTrailingSystemMessage: return supportsTrailingSystemMessage
            }
        }
        set {
            switch flag {
            case .glmTemplateSalvage:            glmTemplateSalvage = newValue
            case .useMaxCompletionTokens:        useMaxCompletionTokens = newValue
            case .disableParallelToolCalls:      disableParallelToolCalls = newValue
            case .replayReasoningContent:        replayReasoningContent = newValue
            case .supportsDeveloperRole:         supportsDeveloperRole = newValue
            case .requiresAdaptiveThinking:      requiresAdaptiveThinking = newValue
            case .mustNeverSendTemperatureParam: mustNeverSendTemperatureParam = newValue
            case .supportsTrailingSystemMessage: supportsTrailingSystemMessage = newValue
            }
        }
    }

    public func apply(to flags: inout BehaviorFlags, forceReplace: Bool) {
        for flag in BehaviorFlag.allCases {
            if let v = self[flag], (forceReplace || v) { flags[flag] = v }
        }
        if let extrasPatch = extras {
            if forceReplace {
                flags.extras = extrasPatch
            } else {
                // Gap-fill: only fill keys not already set
                for (k, v) in extrasPatch where flags.extras[k] == nil {
                    flags.extras[k] = v
                }
            }
        }
    }

    /// True when every override field is nil. Lets callers skip applying empty patches.
    public var isEmpty: Bool {
        BehaviorFlag.allCases.allSatisfy { self[$0] == nil } && (extras?.isEmpty ?? true)
    }
}
