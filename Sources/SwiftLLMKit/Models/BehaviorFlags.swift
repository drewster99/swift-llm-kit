import Foundation

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

    /// Send `parallel_tool_calls: true` regardless of the model's catalog
    /// `capabilities.parallelToolCalls` flag.
    ///
    /// Used by Mistral models (which support parallel calls but where LiteLLM
    /// metadata doesn't flag the capability). Bundled defaults set this for the
    /// affected models so the hardcoded apiType check can retire.
    public var forceParallelToolCalls: Bool = false

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

    /// OpenAI-compatible models that accept a top-level `reasoning_effort`
    /// field for reasoning depth control. Set on the OpenAI o-series (o1,
    /// o1-preview, o1-mini, o3, o3-mini, o4-mini) and the GPT-5 family
    /// (gpt-5, gpt-5-mini, gpt-5-pro). Non-reasoning models (GPT-4o,
    /// GPT-3.5-turbo, etc.) reject the field with HTTP 400, so emission
    /// is gated. When true AND `ModelConfiguration.thinkingEffort` is
    /// set, `OpenAICompatibleProvider` emits `reasoning_effort: <value>`
    /// at the top level of the request body. Default false.
    public var supportsReasoningEffort: Bool = false

    /// Models that reject the `temperature` request parameter entirely — sending ANY value
    /// (including 0) fails with HTTP 400. OpenAI's o-series (o1, o3, o4-mini) and the GPT-5
    /// reasoning family accept only their fixed internal default. When true, every provider
    /// OMITS temperature from the request — both the per-call override AND the configured
    /// value are dropped. Default false: temperature is sent as usual. Named for the action
    /// it forces so the meaning and the default (false = send normally) are both unambiguous.
    public var mustNeverSendTemperatureParam: Bool = false

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
        forceParallelToolCalls: Bool = false,
        replayReasoningContent: Bool = false,
        supportsDeveloperRole: Bool = false,
        requiresAdaptiveThinking: Bool = false,
        supportsReasoningEffort: Bool = false,
        mustNeverSendTemperatureParam: Bool = false,
        extras: [String: String] = [:]
    ) {
        self.glmTemplateSalvage = glmTemplateSalvage
        self.useMaxCompletionTokens = useMaxCompletionTokens
        self.forceParallelToolCalls = forceParallelToolCalls
        self.replayReasoningContent = replayReasoningContent
        self.supportsDeveloperRole = supportsDeveloperRole
        self.requiresAdaptiveThinking = requiresAdaptiveThinking
        self.supportsReasoningEffort = supportsReasoningEffort
        self.mustNeverSendTemperatureParam = mustNeverSendTemperatureParam
        self.extras = extras
    }

    /// True when every flag is at its default value. Useful to short-circuit
    /// equality checks and to skip persisting empty entries.
    public var isAllDefault: Bool {
        !glmTemplateSalvage
            && !useMaxCompletionTokens
            && !forceParallelToolCalls
            && !replayReasoningContent
            && !supportsDeveloperRole
            && !requiresAdaptiveThinking
            && !supportsReasoningEffort
            && !mustNeverSendTemperatureParam
            && extras.isEmpty
    }

    /// Short human-readable labels for each non-default flag — for display in
    /// the Settings UI. Order is stable and reflects how a reader would scan
    /// the row left-to-right. `extras` keys are appended last with a `*` prefix.
    public var displayLabels: [String] {
        var out: [String] = []
        if glmTemplateSalvage { out.append("GLM salvage") }
        if useMaxCompletionTokens { out.append("max_completion_tokens") }
        if forceParallelToolCalls { out.append("parallel tools") }
        if replayReasoningContent { out.append("replay reasoning") }
        if supportsDeveloperRole { out.append("developer role") }
        if requiresAdaptiveThinking { out.append("adaptive thinking") }
        if supportsReasoningEffort { out.append("reasoning_effort") }
        if mustNeverSendTemperatureParam { out.append("no temperature") }
        for key in extras.keys.sorted() {
            out.append("*\(key)")
        }
        return out
    }

    // MARK: - Codable (backward-compatible)

    private enum CodingKeys: String, CodingKey {
        case glmTemplateSalvage, useMaxCompletionTokens, forceParallelToolCalls, replayReasoningContent, supportsDeveloperRole, requiresAdaptiveThinking, supportsReasoningEffort, mustNeverSendTemperatureParam, extras
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        glmTemplateSalvage = try c.decodeIfPresent(Bool.self, forKey: .glmTemplateSalvage) ?? false
        useMaxCompletionTokens = try c.decodeIfPresent(Bool.self, forKey: .useMaxCompletionTokens) ?? false
        forceParallelToolCalls = try c.decodeIfPresent(Bool.self, forKey: .forceParallelToolCalls) ?? false
        replayReasoningContent = try c.decodeIfPresent(Bool.self, forKey: .replayReasoningContent) ?? false
        supportsDeveloperRole = try c.decodeIfPresent(Bool.self, forKey: .supportsDeveloperRole) ?? false
        requiresAdaptiveThinking = try c.decodeIfPresent(Bool.self, forKey: .requiresAdaptiveThinking) ?? false
        supportsReasoningEffort = try c.decodeIfPresent(Bool.self, forKey: .supportsReasoningEffort) ?? false
        mustNeverSendTemperatureParam = try c.decodeIfPresent(Bool.self, forKey: .mustNeverSendTemperatureParam) ?? false
        extras = try c.decodeIfPresent([String: String].self, forKey: .extras) ?? [:]
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        // Skip encoding default-valued fields so on-disk JSON stays compact and
        // round-trips cleanly through older clients that don't know these keys.
        if glmTemplateSalvage { try c.encode(glmTemplateSalvage, forKey: .glmTemplateSalvage) }
        if useMaxCompletionTokens { try c.encode(useMaxCompletionTokens, forKey: .useMaxCompletionTokens) }
        if forceParallelToolCalls { try c.encode(forceParallelToolCalls, forKey: .forceParallelToolCalls) }
        if replayReasoningContent { try c.encode(replayReasoningContent, forKey: .replayReasoningContent) }
        if supportsDeveloperRole { try c.encode(supportsDeveloperRole, forKey: .supportsDeveloperRole) }
        if requiresAdaptiveThinking { try c.encode(requiresAdaptiveThinking, forKey: .requiresAdaptiveThinking) }
        if supportsReasoningEffort { try c.encode(supportsReasoningEffort, forKey: .supportsReasoningEffort) }
        if mustNeverSendTemperatureParam { try c.encode(mustNeverSendTemperatureParam, forKey: .mustNeverSendTemperatureParam) }
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
    public var forceParallelToolCalls: Bool?
    public var replayReasoningContent: Bool?
    public var supportsDeveloperRole: Bool?
    public var requiresAdaptiveThinking: Bool?
    public var supportsReasoningEffort: Bool?
    public var mustNeverSendTemperatureParam: Bool?
    public var extras: [String: String]?

    public init(
        glmTemplateSalvage: Bool? = nil,
        useMaxCompletionTokens: Bool? = nil,
        forceParallelToolCalls: Bool? = nil,
        replayReasoningContent: Bool? = nil,
        supportsDeveloperRole: Bool? = nil,
        requiresAdaptiveThinking: Bool? = nil,
        supportsReasoningEffort: Bool? = nil,
        mustNeverSendTemperatureParam: Bool? = nil,
        extras: [String: String]? = nil
    ) {
        self.glmTemplateSalvage = glmTemplateSalvage
        self.useMaxCompletionTokens = useMaxCompletionTokens
        self.forceParallelToolCalls = forceParallelToolCalls
        self.replayReasoningContent = replayReasoningContent
        self.supportsDeveloperRole = supportsDeveloperRole
        self.requiresAdaptiveThinking = requiresAdaptiveThinking
        self.supportsReasoningEffort = supportsReasoningEffort
        self.mustNeverSendTemperatureParam = mustNeverSendTemperatureParam
        self.extras = extras
    }

    public func apply(to flags: inout BehaviorFlags, forceReplace: Bool) {
        if let v = glmTemplateSalvage, (forceReplace || v) { flags.glmTemplateSalvage = v }
        if let v = useMaxCompletionTokens, (forceReplace || v) { flags.useMaxCompletionTokens = v }
        if let v = forceParallelToolCalls, (forceReplace || v) { flags.forceParallelToolCalls = v }
        if let v = replayReasoningContent, (forceReplace || v) { flags.replayReasoningContent = v }
        if let v = supportsDeveloperRole, (forceReplace || v) { flags.supportsDeveloperRole = v }
        if let v = requiresAdaptiveThinking, (forceReplace || v) { flags.requiresAdaptiveThinking = v }
        if let v = supportsReasoningEffort, (forceReplace || v) { flags.supportsReasoningEffort = v }
        if let v = mustNeverSendTemperatureParam, (forceReplace || v) { flags.mustNeverSendTemperatureParam = v }
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
        glmTemplateSalvage == nil
            && useMaxCompletionTokens == nil
            && forceParallelToolCalls == nil
            && replayReasoningContent == nil
            && supportsDeveloperRole == nil
            && requiresAdaptiveThinking == nil
            && supportsReasoningEffort == nil
            && mustNeverSendTemperatureParam == nil
            && (extras?.isEmpty ?? true)
    }
}
