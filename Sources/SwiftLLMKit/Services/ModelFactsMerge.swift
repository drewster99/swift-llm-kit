import Foundation

/// The five sources a merged model can draw from, in the order they are consulted.
/// The names are the layer's ROLE, not its transport: "downloadedOverrides" is app-curated fix
/// data whether it arrived bundled in the app or (later) from a server.
public enum MetadataLayer: String, Codable, Sendable, CaseIterable {
    /// The provider's own `/models` payload — the vendor describing its own model.
    case authoritative
    /// Probe results: what we established by calling the model. (Empty until the probe store lands.)
    case empirical
    /// App-curated overrides. FORCE semantics: they exist to fix wrong vendor claims and wrong
    /// probe results, so their non-nil fields overwrite. Every entry carries evidence of why.
    case downloadedOverrides
    /// LiteLLM — "possibly correct" third-party claims. Gap-fill only, never overwrites.
    case enrichment
    /// The user's own overrides. FORCE semantics: the user always wins.
    case userOverrides
}

/// One mergeable field of ``ModelFacts``: its name, and type-erased accessors the merge loop,
/// provenance report, and silence check all share.
///
/// This table is the single source of truth for "what fields exist" — the merge, provenance,
/// disagreement detection, and the reflection test all iterate it, so a field added to
/// `ModelFacts` but not registered here fails a test instead of silently not merging.
public struct ModelFactsField: Sendable {
    public let name: String
    public let isSet: @Sendable (ModelFacts) -> Bool
    public let copy: @Sendable (ModelFacts, inout ModelFacts) -> Void
    /// Whether two sources state DIFFERENT values for this field (both must be set).
    public let differs: @Sendable (ModelFacts, ModelFacts) -> Bool
    /// Human-readable value for the inspector; nil when unset.
    public let describe: @Sendable (ModelFacts) -> String?

    static func field<T: Equatable & Sendable>(
        _ name: String, _ keyPath: WritableKeyPath<ModelFacts, T?> & Sendable
    ) -> ModelFactsField {
        ModelFactsField(
            name: name,
            isSet: { $0[keyPath: keyPath] != nil },
            copy: { source, target in target[keyPath: keyPath] = source[keyPath: keyPath] },
            differs: { a, b in
                guard let va = a[keyPath: keyPath], let vb = b[keyPath: keyPath] else { return false }
                return va != vb
            },
            describe: { $0[keyPath: keyPath].map { "\($0)" } }
        )
    }
}

/// The canonical field list. Capabilities and behavior flags are registered per-flag (each is an
/// independent fact); composites (pricing, samplingDefaults, benchmarks) are single atomic entries.
public enum ModelFactsFieldTable {
    public static let fields: [ModelFactsField] = scalars + capabilityFlags + behaviorFlagFields

    private static let scalars: [ModelFactsField] = [
        .field("displayName", \ModelFacts.displayName),
        .field("createdAt", \ModelFacts.createdAt),
        .field("maxInputTokens", \ModelFacts.maxInputTokens),
        .field("maxOutputTokens", \ModelFacts.maxOutputTokens),
        .field("sizeLabel", \ModelFacts.sizeLabel),
        .field("quantizationLabel", \ModelFacts.quantizationLabel),
        .field("pricing", \ModelFacts.pricing),                 // atomic composite
        .field("mode", \ModelFacts.mode),
        .field("generalEffort", \ModelFacts.generalEffort),
        .field("reasoningEffort", \ModelFacts.reasoningEffort),
        .field("reasoningControl", \ModelFacts.reasoningControl),
        .field("thinkingBudgetAccounting", \ModelFacts.thinkingBudgetAccounting),
        .field("maxThinkingBudgetTokens", \ModelFacts.maxThinkingBudgetTokens),
        .field("deprecatedOn", \ModelFacts.deprecatedOn),
        .field("deprecationReplacement", \ModelFacts.deprecationReplacement),
        .field("maxTemperature", \ModelFacts.maxTemperature),
        .field("modelDescription", \ModelFacts.modelDescription),
        .field("samplingDefaults", \ModelFacts.samplingDefaults), // atomic composite
        .field("isFree", \ModelFacts.isFree),
        .field("benchmarks", \ModelFacts.benchmarks),             // atomic composite
        .field("huggingFaceID", \ModelFacts.huggingFaceID),
        .field("hidden", \ModelFacts.hidden),
        .field("isAvailable", \ModelFacts.isAvailable),
        .field("outputBoundedByContext", \ModelFacts.outputBoundedByContext),
        .field("isAccessDenied", \ModelFacts.isAccessDenied),
    ]

    private static let capabilityFlags: [ModelFactsField] = [
        .field("capabilities.chat", \ModelFacts.capabilities.chat),
        .field("capabilities.batch", \ModelFacts.capabilities.batch),
        .field("capabilities.toolUse", \ModelFacts.capabilities.toolUse),
        .field("capabilities.vision", \ModelFacts.capabilities.vision),
        .field("capabilities.reasoning", \ModelFacts.capabilities.reasoning),
        .field("capabilities.codeExecution", \ModelFacts.capabilities.codeExecution),
        .field("capabilities.promptCaching", \ModelFacts.capabilities.promptCaching),
        .field("capabilities.computerUse", \ModelFacts.capabilities.computerUse),
        .field("capabilities.audioInput", \ModelFacts.capabilities.audioInput),
        .field("capabilities.audioOutput", \ModelFacts.capabilities.audioOutput),
        .field("capabilities.videoInput", \ModelFacts.capabilities.videoInput),
        .field("capabilities.responseSchema", \ModelFacts.capabilities.responseSchema),
        .field("capabilities.parallelToolCalls", \ModelFacts.capabilities.parallelToolCalls),
        .field("capabilities.pdfInput", \ModelFacts.capabilities.pdfInput),
        .field("capabilities.webSearch", \ModelFacts.capabilities.webSearch),
        .field("capabilities.systemMessages", \ModelFacts.capabilities.systemMessages),
        .field("capabilities.assistantPrefill", \ModelFacts.capabilities.assistantPrefill),
        .field("capabilities.toolChoice", \ModelFacts.capabilities.toolChoice),
        .field("capabilities.reasoningEnableable", \ModelFacts.capabilities.reasoningEnableable),
        .field("capabilities.reasoningDisableable", \ModelFacts.capabilities.reasoningDisableable),
        .field("capabilities.thinkingKeepAll", \ModelFacts.capabilities.thinkingKeepAll),
        .field("capabilities.thinkingBudgetTokens", \ModelFacts.capabilities.thinkingBudgetTokens),
        .field("capabilities.structuredOutputJSONObject", \ModelFacts.capabilities.structuredOutputJSONObject),
        .field("capabilities.toolChoiceRequired", \ModelFacts.capabilities.toolChoiceRequired),
        .field("capabilities.toolChoiceNone", \ModelFacts.capabilities.toolChoiceNone),
        .field("capabilities.toolChoiceSpecificFunction", \ModelFacts.capabilities.toolChoiceSpecificFunction),
        .field("capabilities.strictToolDefinitions", \ModelFacts.capabilities.strictToolDefinitions),
        .field("capabilities.toolResultRoundTrip", \ModelFacts.capabilities.toolResultRoundTrip),
    ]

    private static let behaviorFlagFields: [ModelFactsField] = [
        .field("behaviorFlags.glmTemplateSalvage", \ModelFacts.behaviorFlags.glmTemplateSalvage),
        .field("behaviorFlags.useMaxCompletionTokens", \ModelFacts.behaviorFlags.useMaxCompletionTokens),
        .field("behaviorFlags.disableParallelToolCalls", \ModelFacts.behaviorFlags.disableParallelToolCalls),
        .field("behaviorFlags.replayReasoningContent", \ModelFacts.behaviorFlags.replayReasoningContent),
        .field("behaviorFlags.supportsDeveloperRole", \ModelFacts.behaviorFlags.supportsDeveloperRole),
        .field("behaviorFlags.requiresAdaptiveThinking", \ModelFacts.behaviorFlags.requiresAdaptiveThinking),
        .field("behaviorFlags.mustNeverSendTemperatureParam", \ModelFacts.behaviorFlags.mustNeverSendTemperatureParam),
        .field("behaviorFlags.supportsTrailingSystemMessage", \ModelFacts.behaviorFlags.supportsTrailingSystemMessage),
        .field("behaviorFlags.extras", \ModelFacts.behaviorFlags.extras),
    ]
}

/// Where a lower layer stated a value the merge did NOT take — kept visible, per the rule that a
/// probe (or anyone) contradicting the winner is a finding, not noise.
public struct MetadataDisagreement: Sendable, Equatable {
    public let field: String
    public let winningLayer: MetadataLayer
    public let winningValue: String
    public let dissentingLayer: MetadataLayer
    public let dissentingValue: String
}

/// The full composition for one model: the merged result plus everything needed to explain it —
/// each layer's own record, which layer won each field, and where layers disagreed.
public struct MergedModelComposition: Sendable {
    public let merged: ModelFacts
    /// Each contributing layer's record, as supplied (silent layers omitted).
    public let layers: [MetadataLayer: ModelFacts]
    /// Per field name: the layer whose value the merged result carries. Absent = no layer spoke.
    public let provenance: [String: MetadataLayer]
    public let disagreements: [MetadataDisagreement]
}

/// The deterministic five-layer merge. Pure — same inputs, same output — so it is trivially
/// testable and can be recomputed whenever any layer changes.
public enum ModelFactsMerger {

    /// Merge semantics per layer, in application order:
    ///   authoritative (base) → empirical (gap-fill) → downloadedOverrides (FORCE)
    ///   → enrichment (gap-fill) → userOverrides (FORCE)
    ///
    /// Gap-fill writes only fields still unknown; force overwrites any field it states. Forcing
    /// makes a field non-nil, which is what shields it from later gap-fill layers (a downloaded
    /// override automatically blocks LiteLLM for the fields it fixes).
    public static func merge(
        authoritative: ModelFacts,
        empirical: ModelFacts = ModelFacts(),
        downloadedOverrides: ModelFacts = ModelFacts(),
        enrichment: ModelFacts = ModelFacts(),
        userOverrides: ModelFacts = ModelFacts()
    ) -> MergedModelComposition {
        var merged = ModelFacts()
        var provenance: [String: MetadataLayer] = [:]

        let ordered: [(MetadataLayer, ModelFacts, MergeSemantics)] = [
            (.authoritative, authoritative, .gapFill),   // base = gap-fill into an empty record
            (.empirical, empirical, .gapFill),
            (.downloadedOverrides, downloadedOverrides, .force),
            (.enrichment, enrichment, .gapFill),
            (.userOverrides, userOverrides, .force),
        ]

        for (layer, source, semantics) in ordered {
            for field in ModelFactsFieldTable.fields where field.isSet(source) {
                switch semantics {
                case .gapFill:
                    guard !field.isSet(merged) else { continue }
                case .force:
                    break
                }
                field.copy(source, &merged)
                provenance[field.name] = layer
            }
        }

        // Disagreements: any layer that stated a value differing from the final winner.
        var disagreements: [MetadataDisagreement] = []
        let layerRecords = ordered.map { ($0.0, $0.1) }
        for field in ModelFactsFieldTable.fields {
            guard let winner = provenance[field.name],
                  let winningValue = field.describe(merged) else { continue }
            for (layer, source) in layerRecords where layer != winner {
                guard field.isSet(source), field.differs(source, merged),
                      let dissenting = field.describe(source) else { continue }
                disagreements.append(MetadataDisagreement(
                    field: field.name,
                    winningLayer: winner, winningValue: winningValue,
                    dissentingLayer: layer, dissentingValue: dissenting
                ))
            }
        }

        var layers: [MetadataLayer: ModelFacts] = [:]
        for (layer, source) in layerRecords where !source.isSilent {
            layers[layer] = source
        }

        return MergedModelComposition(
            merged: merged, layers: layers,
            provenance: provenance, disagreements: disagreements
        )
    }

    private enum MergeSemantics { case gapFill, force }
}

extension ModelFacts {
    /// Force-copies every field the other record states onto this one (later wins). Used to fold
    /// the downloaded-overrides key axes (provider-wide → apiType-scoped → providerID-scoped,
    /// least specific first) into the single layer record the merge consumes.
    public mutating func overlay(_ other: ModelFacts) {
        for field in ModelFactsFieldTable.fields where field.isSet(other) {
            field.copy(other, &self)
        }
    }
}

extension LiteLLMEntry {
    /// LiteLLM's claims as an enrichment-layer record. POSITIVES ONLY for capabilities: LiteLLM's
    /// schema marks supported features `true` and simply omits the rest, so a `false` in its data
    /// means "didn't say", never "can't" — exactly the false-means-unknown trap the facts model
    /// exists to avoid. Chat-completions support is the one negative it genuinely states (derived
    /// from `mode`), so only the negative is recorded.
    public var asFacts: ModelFacts {
        var facts = ModelFacts()
        facts.maxInputTokens = maxInputTokens
        facts.maxOutputTokens = maxOutputTokens
        facts.pricing = pricing
        facts.mode = mode
        if supportsToolUse { facts.capabilities.toolUse = true }
        if supportsVision { facts.capabilities.vision = true }
        if supportsReasoning { facts.capabilities.reasoning = true }
        if supportsPromptCaching { facts.capabilities.promptCaching = true }
        if supportsComputerUse { facts.capabilities.computerUse = true }
        if supportsAudioInput { facts.capabilities.audioInput = true }
        if supportsAudioOutput { facts.capabilities.audioOutput = true }
        if supportsVideoInput { facts.capabilities.videoInput = true }
        if supportsResponseSchema { facts.capabilities.responseSchema = true }
        if supportsParallelToolCalls { facts.capabilities.parallelToolCalls = true }
        if supportsPdfInput { facts.capabilities.pdfInput = true }
        if supportsWebSearch { facts.capabilities.webSearch = true }
        if supportsSystemMessages { facts.capabilities.systemMessages = true }
        if supportsAssistantPrefill { facts.capabilities.assistantPrefill = true }
        if supportsToolChoice { facts.capabilities.toolChoice = true }
        // Chat is the one capability LiteLLM can state NEGATIVELY: mode != chat is a definitive
        // "not a chat model" (embedding/moderation/…), so record the known false.
        if !supportsChatCompletions { facts.capabilities.chat = false }
        return facts
    }
}
