import Foundation

/// Optional overrides for individual capability flags.
/// Non-nil values replace the corresponding flag on ``ModelCapabilities``;
/// `nil` values leave the existing flag unchanged.
public struct ModelCapabilitiesOverride: Codable, Sendable, Equatable {
    /// Chat-completions support. Lives here (not as a standalone override field) so chat is a
    /// capability at every layer — its resolved value is ``ModelInfo/supportsChatCompletions``.
    public var chat: Bool?
    /// Batch-only variant marker (see ``ModelCapability/batch``).
    public var batch: Bool?
    public var toolUse: Bool?
    public var vision: Bool?
    public var reasoning: Bool?
    public var codeExecution: Bool?
    public var promptCaching: Bool?
    public var computerUse: Bool?
    public var audioInput: Bool?
    public var audioOutput: Bool?
    public var videoInput: Bool?
    public var structuredOutputSupportsJSONSchema: Bool?
    public var parallelToolCalls: Bool?
    public var pdfInput: Bool?
    public var webSearch: Bool?
    public var systemMessages: Bool?
    public var assistantPrefill: Bool?
    public var toolChoiceSupported: Bool?
    public var reasoningCanBeEnabled: Bool?
    public var reasoningCanBeDisabled: Bool?
    public var thinkingSupportsKeepAll: Bool?
    public var thinkingSupportsTokenBudget: Bool?
    public var structuredOutputSupportsJSONObject: Bool?
    public var toolChoiceSupportsValueRequired: Bool?
    public var toolChoiceSupportsValueNone: Bool?
    public var toolChoiceSupportsNamedFunction: Bool?
    public var toolDefinitionsSupportStrict: Bool?
    public var toolResultRoundTrip: Bool?

    public init(
        chat: Bool? = nil,
        batch: Bool? = nil,
        toolUse: Bool? = nil,
        vision: Bool? = nil,
        reasoning: Bool? = nil,
        codeExecution: Bool? = nil,
        promptCaching: Bool? = nil,
        computerUse: Bool? = nil,
        audioInput: Bool? = nil,
        audioOutput: Bool? = nil,
        videoInput: Bool? = nil,
        structuredOutputSupportsJSONSchema: Bool? = nil,
        parallelToolCalls: Bool? = nil,
        pdfInput: Bool? = nil,
        webSearch: Bool? = nil,
        systemMessages: Bool? = nil,
        assistantPrefill: Bool? = nil,
        toolChoiceSupported: Bool? = nil,
        reasoningCanBeEnabled: Bool? = nil,
        reasoningCanBeDisabled: Bool? = nil,
        thinkingSupportsKeepAll: Bool? = nil,
        thinkingSupportsTokenBudget: Bool? = nil,
        structuredOutputSupportsJSONObject: Bool? = nil,
        toolChoiceSupportsValueRequired: Bool? = nil,
        toolChoiceSupportsValueNone: Bool? = nil,
        toolChoiceSupportsNamedFunction: Bool? = nil,
        toolDefinitionsSupportStrict: Bool? = nil,
        toolResultRoundTrip: Bool? = nil
    ) {
        self.chat = chat
        self.batch = batch
        self.toolUse = toolUse
        self.vision = vision
        self.reasoning = reasoning
        self.codeExecution = codeExecution
        self.promptCaching = promptCaching
        self.computerUse = computerUse
        self.audioInput = audioInput
        self.audioOutput = audioOutput
        self.videoInput = videoInput
        self.structuredOutputSupportsJSONSchema = structuredOutputSupportsJSONSchema
        self.parallelToolCalls = parallelToolCalls
        self.pdfInput = pdfInput
        self.webSearch = webSearch
        self.systemMessages = systemMessages
        self.assistantPrefill = assistantPrefill
        self.toolChoiceSupported = toolChoiceSupported
        self.reasoningCanBeEnabled = reasoningCanBeEnabled
        self.reasoningCanBeDisabled = reasoningCanBeDisabled
        self.thinkingSupportsKeepAll = thinkingSupportsKeepAll
        self.thinkingSupportsTokenBudget = thinkingSupportsTokenBudget
        self.structuredOutputSupportsJSONObject = structuredOutputSupportsJSONObject
        self.toolChoiceSupportsValueRequired = toolChoiceSupportsValueRequired
        self.toolChoiceSupportsValueNone = toolChoiceSupportsValueNone
        self.toolChoiceSupportsNamedFunction = toolChoiceSupportsNamedFunction
        self.toolDefinitionsSupportStrict = toolDefinitionsSupportStrict
        self.toolResultRoundTrip = toolResultRoundTrip
    }

    /// Applies this override to a ``ModelCapabilities`` value.
    ///
    /// - Parameters:
    ///   - capabilities: The capabilities to modify in-place.
    ///   - forceReplace: When `true`, non-nil values always replace existing values.
    ///     When `false`, non-nil values only upgrade `false` → `true` (OR semantics).
    public func apply(to capabilities: inout ModelCapabilities, forceReplace: Bool) {
        if let v = chat, (forceReplace || v) { capabilities[.chat] = v }
        if let v = batch, (forceReplace || v) { capabilities[.batch] = v }
        if let v = toolUse, (forceReplace || v) { capabilities.toolUse = v }
        if let v = vision, (forceReplace || v) { capabilities.vision = v }
        if let v = reasoning, (forceReplace || v) { capabilities.reasoning = v }
        if let v = codeExecution, (forceReplace || v) { capabilities.codeExecution = v }
        if let v = promptCaching, (forceReplace || v) { capabilities.promptCaching = v }
        if let v = computerUse, (forceReplace || v) { capabilities.computerUse = v }
        if let v = audioInput, (forceReplace || v) { capabilities.audioInput = v }
        if let v = audioOutput, (forceReplace || v) { capabilities.audioOutput = v }
        if let v = videoInput, (forceReplace || v) { capabilities.videoInput = v }
        if let v = structuredOutputSupportsJSONSchema, (forceReplace || v) { capabilities.structuredOutputSupportsJSONSchema = v }
        if let v = parallelToolCalls, (forceReplace || v) { capabilities.parallelToolCalls = v }
        if let v = pdfInput, (forceReplace || v) { capabilities.pdfInput = v }
        if let v = webSearch, (forceReplace || v) { capabilities.webSearch = v }
        if let v = systemMessages, (forceReplace || v) { capabilities.systemMessages = v }
        if let v = assistantPrefill, (forceReplace || v) { capabilities.assistantPrefill = v }
        if let v = toolChoiceSupported, (forceReplace || v) { capabilities.toolChoiceSupported = v }
        if let v = reasoningCanBeEnabled, (forceReplace || v) { capabilities.reasoningCanBeEnabled = v }
        if let v = reasoningCanBeDisabled, (forceReplace || v) { capabilities.reasoningCanBeDisabled = v }
        if let v = thinkingSupportsKeepAll, (forceReplace || v) { capabilities.thinkingSupportsKeepAll = v }
        if let v = thinkingSupportsTokenBudget, (forceReplace || v) { capabilities.thinkingSupportsTokenBudget = v }
        if let v = structuredOutputSupportsJSONObject, (forceReplace || v) { capabilities.structuredOutputSupportsJSONObject = v }
        if let v = toolChoiceSupportsValueRequired, (forceReplace || v) { capabilities.toolChoiceSupportsValueRequired = v }
        if let v = toolChoiceSupportsValueNone, (forceReplace || v) { capabilities.toolChoiceSupportsValueNone = v }
        if let v = toolChoiceSupportsNamedFunction, (forceReplace || v) { capabilities.toolChoiceSupportsNamedFunction = v }
        if let v = toolDefinitionsSupportStrict, (forceReplace || v) { capabilities.toolDefinitionsSupportStrict = v }
        if let v = toolResultRoundTrip, (forceReplace || v) { capabilities.toolResultRoundTrip = v }
    }

    /// Enum-keyed access to a capability's override value (`nil` = no override). The exhaustive
    /// switch makes a newly-added ``ModelCapability`` a COMPILE error here until it's wired, which is
    /// what lets `CapabilitiesEditorSheet` drive its rows from `ModelCapability.allCases` safely.
    public subscript(_ capability: ModelCapability) -> Bool? {
        get {
            switch capability {
            case .chat: return chat
            case .batch: return batch
            case .toolUse: return toolUse
            case .vision: return vision
            case .reasoning: return reasoning
            case .codeExecution: return codeExecution
            case .promptCaching: return promptCaching
            case .computerUse: return computerUse
            case .audioInput: return audioInput
            case .audioOutput: return audioOutput
            case .videoInput: return videoInput
            case .structuredOutputSupportsJSONSchema: return structuredOutputSupportsJSONSchema
            case .parallelToolCalls: return parallelToolCalls
            case .pdfInput: return pdfInput
            case .webSearch: return webSearch
            case .systemMessages: return systemMessages
            case .assistantPrefill: return assistantPrefill
            case .toolChoiceSupported: return toolChoiceSupported
            case .reasoningCanBeEnabled: return reasoningCanBeEnabled
            case .reasoningCanBeDisabled: return reasoningCanBeDisabled
            case .thinkingSupportsKeepAll: return thinkingSupportsKeepAll
            case .thinkingSupportsTokenBudget: return thinkingSupportsTokenBudget
            case .structuredOutputSupportsJSONObject: return structuredOutputSupportsJSONObject
            case .toolChoiceSupportsValueRequired: return toolChoiceSupportsValueRequired
            case .toolChoiceSupportsValueNone: return toolChoiceSupportsValueNone
            case .toolChoiceSupportsNamedFunction: return toolChoiceSupportsNamedFunction
            case .toolDefinitionsSupportStrict: return toolDefinitionsSupportStrict
            case .toolResultRoundTrip: return toolResultRoundTrip
            }
        }
        set {
            switch capability {
            case .chat: chat = newValue
            case .batch: batch = newValue
            case .toolUse: toolUse = newValue
            case .vision: vision = newValue
            case .reasoning: reasoning = newValue
            case .codeExecution: codeExecution = newValue
            case .promptCaching: promptCaching = newValue
            case .computerUse: computerUse = newValue
            case .audioInput: audioInput = newValue
            case .audioOutput: audioOutput = newValue
            case .videoInput: videoInput = newValue
            case .structuredOutputSupportsJSONSchema: structuredOutputSupportsJSONSchema = newValue
            case .parallelToolCalls: parallelToolCalls = newValue
            case .pdfInput: pdfInput = newValue
            case .webSearch: webSearch = newValue
            case .systemMessages: systemMessages = newValue
            case .assistantPrefill: assistantPrefill = newValue
            case .toolChoiceSupported: toolChoiceSupported = newValue
            case .reasoningCanBeEnabled: reasoningCanBeEnabled = newValue
            case .reasoningCanBeDisabled: reasoningCanBeDisabled = newValue
            case .thinkingSupportsKeepAll: thinkingSupportsKeepAll = newValue
            case .thinkingSupportsTokenBudget: thinkingSupportsTokenBudget = newValue
            case .structuredOutputSupportsJSONObject: structuredOutputSupportsJSONObject = newValue
            case .toolChoiceSupportsValueRequired: toolChoiceSupportsValueRequired = newValue
            case .toolChoiceSupportsValueNone: toolChoiceSupportsValueNone = newValue
            case .toolChoiceSupportsNamedFunction: toolChoiceSupportsNamedFunction = newValue
            case .toolDefinitionsSupportStrict: toolDefinitionsSupportStrict = newValue
            case .toolResultRoundTrip: toolResultRoundTrip = newValue
            }
        }
    }
}

/// User or app-bundled overrides for model metadata.
///
/// Every field is optional. Non-nil values override the corresponding field
/// from lower-priority data sources during the enrichment pipeline.
///
/// Priority order (highest wins):
/// 1. User-provided overrides (`forceReplace: true`)
/// 2. Provider API data (the base `ModelInfo`)
/// 3. App-bundled overrides (`forceReplace: false` — gap-fill only)
/// 4. LiteLLM data (`forceReplace: false` — gap-fill only)
public struct ModelMetadataOverride: Codable, Sendable, Equatable {
    /// Override display name.
    public var displayName: String?
    /// Override maximum input tokens.
    public var maxInputTokens: Int?
    /// Override maximum output tokens.
    public var maxOutputTokens: Int?
    /// Override the parameter-size label (e.g. "397B"). Curation for providers that don't report
    /// it — Ollama Cloud's /api/tags omits parameter_size for cloud models.
    public var sizeLabel: String?
    /// Override individual capabilities. Per-flag, not all-or-nothing.
    public var capabilities: ModelCapabilitiesOverride?
    /// Override pricing data. Replaces the entire pricing structure when set.
    public var pricing: ModelPricing?
    /// Override chat-completions support. A convenience SHIM over ``capabilities`` `.chat` — chat is
    /// a capability now, so this is not a separate stored field; reading and writing route through
    /// the capabilities container, keeping one source of truth.
    public var supportsChatCompletions: Bool? {
        get { capabilities?[.chat] }
        set {
            if newValue != nil && capabilities == nil { capabilities = ModelCapabilitiesOverride() }
            capabilities?[.chat] = newValue
        }
    }
    /// Override per-(provider+model) runtime behavior flags. Per-flag, not all-or-nothing.
    public var behaviorFlags: BehaviorFlagsOverride?
    /// Override general-effort support (Anthropic `output_config.effort`).
    ///
    /// Overridable because a wrong effort record is otherwise uncorrectable: the vendor may publish
    /// nothing, and the empirical layer can only speak after a full ladder probe. Without a slot
    /// here, neither a user nor a downloaded fix could repair it.
    public var generalEffort: EffortSupport?
    /// Override reasoning-effort support (`reasoning_effort`). Same reasoning as ``generalEffort``.
    public var reasoningEffort: EffortSupport?
    /// Override the reasoning control mechanism. Overridable because no vendor publishes it today —
    /// without a slot here the picker in the model editor would have nowhere to persist.
    public var reasoningControl: ReasoningControl?
    /// Override the thinking-budget accounting. Overridable because it is unverified for most
    /// providers and a wrong value silently truncates replies rather than erroring.
    public var thinkingBudgetAccounting: ThinkingBudgetAccounting?
    /// Override the measured maximum thinking budget.
    public var maxThinkingBudgetTokens: Int?
    /// Manual override for the smallest accepted reasoning token budget.
    public var minThinkingBudgetTokens: Int?
    /// Hide this model from pickers. Hiding is presentation, never deletion — every record
    /// underneath survives, and un-hiding is removing this one field.
    public var hidden: Bool?
    /// Override the empirical availability verdict (e.g. clear a stale probed `false`).
    public var isAvailable: Bool?
    /// Override the empirical access-denial verdict.
    public var isAccessDenied: Bool?

    public init(
        displayName: String? = nil,
        maxInputTokens: Int? = nil,
        maxOutputTokens: Int? = nil,
        sizeLabel: String? = nil,
        capabilities: ModelCapabilitiesOverride? = nil,
        pricing: ModelPricing? = nil,
        supportsChatCompletions: Bool? = nil,
        behaviorFlags: BehaviorFlagsOverride? = nil,
        generalEffort: EffortSupport? = nil,
        reasoningEffort: EffortSupport? = nil,
        reasoningControl: ReasoningControl? = nil,
        thinkingBudgetAccounting: ThinkingBudgetAccounting? = nil,
        maxThinkingBudgetTokens: Int? = nil,
        minThinkingBudgetTokens: Int? = nil,
        hidden: Bool? = nil,
        isAvailable: Bool? = nil,
        isAccessDenied: Bool? = nil
    ) {
        self.displayName = displayName
        self.maxInputTokens = maxInputTokens
        self.maxOutputTokens = maxOutputTokens
        self.sizeLabel = sizeLabel
        self.capabilities = capabilities
        self.pricing = pricing
        self.behaviorFlags = behaviorFlags
        self.generalEffort = generalEffort
        self.reasoningEffort = reasoningEffort
        self.reasoningControl = reasoningControl
        self.thinkingBudgetAccounting = thinkingBudgetAccounting
        self.maxThinkingBudgetTokens = maxThinkingBudgetTokens
        self.minThinkingBudgetTokens = minThinkingBudgetTokens
        self.hidden = hidden
        self.isAvailable = isAvailable
        self.isAccessDenied = isAccessDenied
        // Route the convenience param into capabilities.chat, but only when explicitly given, so a
        // nil here never clobbers a chat value supplied through the `capabilities` argument. (Done
        // last: the computed setter touches `self`, which requires every stored property set first.)
        if let supportsChatCompletions { self.supportsChatCompletions = supportsChatCompletions }
    }

    /// Applies this override to a ``ModelInfo`` value.
    ///
    /// NOT the production merge path — the live pipeline composes overrides through ``asFacts`` and
    /// ``ModelFactsMerger`` (per-field, priority-based), never this method. Retained for direct
    /// application and tests. Its `forceReplace: false` gap-fill is 2-state ("upgrade false→true");
    /// under the tri-state model it does not read the target's known/unknown state, so use the
    /// facts merger for anything that must respect tri-state semantics.
    ///
    /// - Parameters:
    ///   - model: The model info to modify in-place.
    ///   - forceReplace: When `true`, non-nil override fields replace existing
    ///     non-nil values (user override semantics). When `false`, override fields
    ///     only fill nil/empty gaps (gap-fill semantics for bundled/LiteLLM data).
    public func apply(to model: inout ModelInfo, forceReplace: Bool) {
        if let v = displayName, (forceReplace || model.displayName.isEmpty || model.displayName == model.modelID) {
            model.displayName = v
        }
        if let v = maxInputTokens, (forceReplace || model.maxInputTokens == nil) {
            model.maxInputTokens = v
        }
        if let v = maxOutputTokens, (forceReplace || model.maxOutputTokens == nil) {
            model.maxOutputTokens = v
        }
        if let v = sizeLabel, (forceReplace || model.sizeLabel == nil) {
            model.sizeLabel = v
        }
        if let v = pricing, (forceReplace || model.pricing == nil) {
            model.pricing = v
        }
        // Chat is a capability now — it rides inside `capabilities` and is applied by the override
        // below alongside every other flag; no separate chat handling.
        if let capOverride = capabilities {
            capOverride.apply(to: &model.capabilities, forceReplace: forceReplace)
        }
        if let flagsOverride = behaviorFlags {
            flagsOverride.apply(to: &model.behaviorFlags, forceReplace: forceReplace)
        }
        if let v = hidden, (forceReplace || model.hidden == nil) {
            model.hidden = v
        }
        if let v = isAvailable, (forceReplace || model.isAvailable == nil) {
            model.isAvailable = v
        }
        if let v = isAccessDenied, (forceReplace || model.isAccessDenied == nil) {
            model.isAccessDenied = v
        }
        // The five fields added with the effort/reasoning work. Omitting them here left this path
        // silently dropping them, exactly as the missing CodingKeys did on the persistence path.
        if let v = generalEffort, (forceReplace || model.generalEffort == nil) { model.generalEffort = v }
        if let v = reasoningEffort, (forceReplace || model.reasoningEffort == nil) { model.reasoningEffort = v }
        if let v = reasoningControl, (forceReplace || model.reasoningControl == nil) { model.reasoningControl = v }
        if let v = thinkingBudgetAccounting, (forceReplace || model.thinkingBudgetAccounting == nil) {
            model.thinkingBudgetAccounting = v
        }
        if let v = maxThinkingBudgetTokens, (forceReplace || model.maxThinkingBudgetTokens == nil) {
            model.maxThinkingBudgetTokens = v
        }
        if let v = minThinkingBudgetTokens, (forceReplace || model.minThinkingBudgetTokens == nil) {
            model.minThinkingBudgetTokens = v
        }
    }
}

// MARK: - Codable (chat migration)
//
// Custom because `supportsChatCompletions` is now a computed shim over `capabilities.chat`, not a
// stored field: synthesized Codable would neither encode it nor migrate an OLD override that still
// carries the standalone `supportsChatCompletions` key. On read that legacy key is folded into
// `capabilities.chat` (new-format capabilities win); on write nothing separate is emitted.
extension ModelMetadataOverride {
    private enum CodingKeys: String, CodingKey {
        case displayName, maxInputTokens, maxOutputTokens, sizeLabel, capabilities, pricing
        case supportsChatCompletions
        case behaviorFlags, hidden, isAvailable, isAccessDenied
        case generalEffort, reasoningEffort, reasoningControl
        case thinkingBudgetAccounting, maxThinkingBudgetTokens, minThinkingBudgetTokens
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        displayName = try c.decodeIfPresent(String.self, forKey: .displayName)
        maxInputTokens = try c.decodeIfPresent(Int.self, forKey: .maxInputTokens)
        maxOutputTokens = try c.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        sizeLabel = try c.decodeIfPresent(String.self, forKey: .sizeLabel)
        capabilities = try c.decodeIfPresent(ModelCapabilitiesOverride.self, forKey: .capabilities)
        pricing = try c.decodeIfPresent(ModelPricing.self, forKey: .pricing)
        behaviorFlags = try c.decodeIfPresent(BehaviorFlagsOverride.self, forKey: .behaviorFlags)
        hidden = try c.decodeIfPresent(Bool.self, forKey: .hidden)
        isAvailable = try c.decodeIfPresent(Bool.self, forKey: .isAvailable)
        isAccessDenied = try c.decodeIfPresent(Bool.self, forKey: .isAccessDenied)
        generalEffort = try c.decodeIfPresent(EffortSupport.self, forKey: .generalEffort)
        reasoningEffort = try c.decodeIfPresent(EffortSupport.self, forKey: .reasoningEffort)
        reasoningControl = try c.decodeIfPresent(ReasoningControl.self, forKey: .reasoningControl)
        thinkingBudgetAccounting = try c.decodeIfPresent(ThinkingBudgetAccounting.self, forKey: .thinkingBudgetAccounting)
        maxThinkingBudgetTokens = try c.decodeIfPresent(Int.self, forKey: .maxThinkingBudgetTokens)
        minThinkingBudgetTokens = try c.decodeIfPresent(Int.self, forKey: .minThinkingBudgetTokens)
        // Migrate the legacy standalone chat flag into capabilities.chat — new-format capabilities win.
        if capabilities?[.chat] == nil,
           let legacyChat = try c.decodeIfPresent(Bool.self, forKey: .supportsChatCompletions) {
            if capabilities == nil { capabilities = ModelCapabilitiesOverride() }
            capabilities?[.chat] = legacyChat
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(displayName, forKey: .displayName)
        try c.encodeIfPresent(maxInputTokens, forKey: .maxInputTokens)
        try c.encodeIfPresent(maxOutputTokens, forKey: .maxOutputTokens)
        try c.encodeIfPresent(sizeLabel, forKey: .sizeLabel)
        try c.encodeIfPresent(capabilities, forKey: .capabilities)   // chat rides inside here
        try c.encodeIfPresent(pricing, forKey: .pricing)
        try c.encodeIfPresent(behaviorFlags, forKey: .behaviorFlags)
        try c.encodeIfPresent(hidden, forKey: .hidden)
        try c.encodeIfPresent(isAvailable, forKey: .isAvailable)
        try c.encodeIfPresent(isAccessDenied, forKey: .isAccessDenied)
        try c.encodeIfPresent(generalEffort, forKey: .generalEffort)
        try c.encodeIfPresent(reasoningEffort, forKey: .reasoningEffort)
        try c.encodeIfPresent(reasoningControl, forKey: .reasoningControl)
        try c.encodeIfPresent(thinkingBudgetAccounting, forKey: .thinkingBudgetAccounting)
        try c.encodeIfPresent(maxThinkingBudgetTokens, forKey: .maxThinkingBudgetTokens)
        try c.encodeIfPresent(minThinkingBudgetTokens, forKey: .minThinkingBudgetTokens)
    }
}
