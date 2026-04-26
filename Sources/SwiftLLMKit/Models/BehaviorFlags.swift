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
        extras: [String: String] = [:]
    ) {
        self.glmTemplateSalvage = glmTemplateSalvage
        self.useMaxCompletionTokens = useMaxCompletionTokens
        self.forceParallelToolCalls = forceParallelToolCalls
        self.extras = extras
    }

    /// True when every flag is at its default value. Useful to short-circuit
    /// equality checks and to skip persisting empty entries.
    public var isAllDefault: Bool {
        !glmTemplateSalvage
            && !useMaxCompletionTokens
            && !forceParallelToolCalls
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
        for key in extras.keys.sorted() {
            out.append("*\(key)")
        }
        return out
    }

    // MARK: - Codable (backward-compatible)

    private enum CodingKeys: String, CodingKey {
        case glmTemplateSalvage, useMaxCompletionTokens, forceParallelToolCalls, extras
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        glmTemplateSalvage = try c.decodeIfPresent(Bool.self, forKey: .glmTemplateSalvage) ?? false
        useMaxCompletionTokens = try c.decodeIfPresent(Bool.self, forKey: .useMaxCompletionTokens) ?? false
        forceParallelToolCalls = try c.decodeIfPresent(Bool.self, forKey: .forceParallelToolCalls) ?? false
        extras = try c.decodeIfPresent([String: String].self, forKey: .extras) ?? [:]
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        // Skip encoding default-valued fields so on-disk JSON stays compact and
        // round-trips cleanly through older clients that don't know these keys.
        if glmTemplateSalvage { try c.encode(glmTemplateSalvage, forKey: .glmTemplateSalvage) }
        if useMaxCompletionTokens { try c.encode(useMaxCompletionTokens, forKey: .useMaxCompletionTokens) }
        if forceParallelToolCalls { try c.encode(forceParallelToolCalls, forKey: .forceParallelToolCalls) }
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
    public var extras: [String: String]?

    public init(
        glmTemplateSalvage: Bool? = nil,
        useMaxCompletionTokens: Bool? = nil,
        forceParallelToolCalls: Bool? = nil,
        extras: [String: String]? = nil
    ) {
        self.glmTemplateSalvage = glmTemplateSalvage
        self.useMaxCompletionTokens = useMaxCompletionTokens
        self.forceParallelToolCalls = forceParallelToolCalls
        self.extras = extras
    }

    public func apply(to flags: inout BehaviorFlags, forceReplace: Bool) {
        if let v = glmTemplateSalvage, (forceReplace || v) { flags.glmTemplateSalvage = v }
        if let v = useMaxCompletionTokens, (forceReplace || v) { flags.useMaxCompletionTokens = v }
        if let v = forceParallelToolCalls, (forceReplace || v) { flags.forceParallelToolCalls = v }
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
            && (extras?.isEmpty ?? true)
    }
}
