import Foundation

/// What ONE source states about a model — and nothing more.
///
/// Every field is optional, and `nil` always means "this source did not say", never "no". This is
/// the load-bearing distinction the flattened pipeline could not express: `ModelInfo.capabilities`
/// is non-optional `Bool`s, so a decoder that had nothing to say wrote `false`, and "we have no
/// idea" became indistinguishable from "the vendor said it can't". Layered composition only works
/// if a source can stay silent — so each of the five layers (authoritative `/models` decode,
/// empirical probe data, downloaded overrides, LiteLLM enrichment, user overrides) produces one of
/// these, and ``ModelFactsMerger`` composes them per field.
///
/// Capabilities and behavior flags reuse the existing per-flag-optional patch types
/// (``ModelCapabilitiesOverride``, ``BehaviorFlagsOverride``) rather than duplicating their field
/// lists — those types already are the tri-state shape this record needs.
///
/// Composite values (``pricing``, ``samplingDefaults``, ``benchmarks``) are single fields on
/// purpose: they merge whole-value-or-nothing, because a pricing structure stitched together from
/// two sources' halves (a vendor's tiers plus a third party's cache rates, of different vintages)
/// describes a price that never existed anywhere.
public struct ModelFacts: Codable, Sendable, Equatable {
    public var displayName: String?
    public var createdAt: Date?
    public var maxInputTokens: Int?
    public var maxOutputTokens: Int?
    /// Per-capability tri-state. The container itself is non-optional so field keypaths compose
    /// (`\ModelFacts.capabilities.toolUse`); an all-nil container is simply a silent source.
    public var capabilities: ModelCapabilitiesOverride
    public var sizeLabel: String?
    public var quantizationLabel: String?
    /// Atomic: merges whole-value-or-nothing.
    public var pricing: ModelPricing?
    public var supportsChatCompletions: Bool?
    public var mode: String?
    /// `nil` = source said nothing about effort. An explicit `[]` is a statement: "no effort levels".
    public var validEffortLevels: [String]?
    /// Per-flag tri-state, same container reasoning as `capabilities`.
    public var behaviorFlags: BehaviorFlagsOverride
    public var deprecatedOn: Date?
    public var deprecationReplacement: String?
    public var maxTemperature: Double?
    public var modelDescription: String?
    /// Atomic: merges whole-value-or-nothing.
    public var samplingDefaults: SamplingDefaults?
    public var isFree: Bool?
    /// Atomic: merges whole-value-or-nothing.
    public var benchmarks: ModelBenchmarks?
    public var huggingFaceID: String?
    /// Presentation, not fact: any layer may hide a model from pickers (user declutter, or a
    /// downloaded override hiding a known-broken entry). Hiding is never deletion — the records
    /// underneath survive, and un-hiding is removing one override field.
    public var hidden: Bool?

    public init() {
        self.capabilities = ModelCapabilitiesOverride()
        self.behaviorFlags = BehaviorFlagsOverride()
    }

    /// Whether this source stated nothing at all.
    public var isSilent: Bool {
        ModelFactsFieldTable.fields.allSatisfy { !$0.isSet(self) }
    }

    // MARK: - Materialization

    /// Collapses merged facts into the app-facing ``ModelInfo``, applying the historical defaults
    /// for everything still unknown (capabilities `nil` → `false`, `supportsChatCompletions` `nil`
    /// → `true`, effort levels `nil` → `[]`). Materialization is the ONLY place those defaults are
    /// applied — inside the layers, unknown stays `nil` so lower layers can still speak.
    public func materialize(providerID: String, modelID: String) -> ModelInfo {
        var caps = ModelCapabilities()
        capabilities.apply(to: &caps, forceReplace: true)
        var flags = BehaviorFlags()
        behaviorFlags.apply(to: &flags, forceReplace: true)
        return ModelInfo(
            providerID: providerID,
            modelID: modelID,
            displayName: displayName ?? "",
            createdAt: createdAt,
            maxInputTokens: maxInputTokens,
            maxOutputTokens: maxOutputTokens,
            capabilities: caps,
            sizeLabel: sizeLabel,
            quantizationLabel: quantizationLabel,
            pricing: pricing,
            supportsChatCompletions: supportsChatCompletions ?? true,
            mode: mode,
            validEffortLevels: validEffortLevels ?? [],
            behaviorFlags: flags,
            deprecatedOn: deprecatedOn,
            deprecationReplacement: deprecationReplacement,
            maxTemperature: maxTemperature,
            modelDescription: modelDescription,
            samplingDefaults: samplingDefaults,
            isFree: isFree,
            benchmarks: benchmarks,
            huggingFaceID: huggingFaceID,
            hidden: hidden
        )
    }
}

/// One model as a source describes it: the key it goes by plus the facts stated about it.
/// Decoders emit these; the merge is keyed on `modelID` within a provider.
public struct DecodedModelFacts: Sendable, Equatable {
    public let modelID: String
    public var facts: ModelFacts

    public init(modelID: String, facts: ModelFacts) {
        self.modelID = modelID
        self.facts = facts
    }
}

extension ModelMetadataOverride {
    /// The override expressed as a per-source facts record, so the bundled and user layers speak
    /// the same language as every other layer. Lossless for everything the override type carries;
    /// fields the override type predates (samplingDefaults, benchmarks, …) simply stay silent.
    public var asFacts: ModelFacts {
        var facts = ModelFacts()
        facts.displayName = displayName
        facts.maxInputTokens = maxInputTokens
        facts.maxOutputTokens = maxOutputTokens
        if let capabilities { facts.capabilities = capabilities }
        facts.pricing = pricing
        facts.supportsChatCompletions = supportsChatCompletions
        if let behaviorFlags { facts.behaviorFlags = behaviorFlags }
        facts.hidden = hidden
        return facts
    }
}
