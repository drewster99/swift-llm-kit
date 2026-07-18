import Foundation

/// Optional overrides for individual capability flags.
/// Non-nil values replace the corresponding flag on ``ModelCapabilities``;
/// `nil` values leave the existing flag unchanged.
public struct ModelCapabilitiesOverride: Codable, Sendable, Equatable {
    public var toolUse: Bool?
    public var vision: Bool?
    public var reasoning: Bool?
    public var codeExecution: Bool?
    public var promptCaching: Bool?
    public var computerUse: Bool?
    public var audioInput: Bool?
    public var audioOutput: Bool?
    public var videoInput: Bool?
    public var responseSchema: Bool?
    public var parallelToolCalls: Bool?
    public var pdfInput: Bool?
    public var webSearch: Bool?
    public var systemMessages: Bool?
    public var assistantPrefill: Bool?
    public var toolChoice: Bool?

    public init(
        toolUse: Bool? = nil,
        vision: Bool? = nil,
        reasoning: Bool? = nil,
        codeExecution: Bool? = nil,
        promptCaching: Bool? = nil,
        computerUse: Bool? = nil,
        audioInput: Bool? = nil,
        audioOutput: Bool? = nil,
        videoInput: Bool? = nil,
        responseSchema: Bool? = nil,
        parallelToolCalls: Bool? = nil,
        pdfInput: Bool? = nil,
        webSearch: Bool? = nil,
        systemMessages: Bool? = nil,
        assistantPrefill: Bool? = nil,
        toolChoice: Bool? = nil
    ) {
        self.toolUse = toolUse
        self.vision = vision
        self.reasoning = reasoning
        self.codeExecution = codeExecution
        self.promptCaching = promptCaching
        self.computerUse = computerUse
        self.audioInput = audioInput
        self.audioOutput = audioOutput
        self.videoInput = videoInput
        self.responseSchema = responseSchema
        self.parallelToolCalls = parallelToolCalls
        self.pdfInput = pdfInput
        self.webSearch = webSearch
        self.systemMessages = systemMessages
        self.assistantPrefill = assistantPrefill
        self.toolChoice = toolChoice
    }

    /// Applies this override to a ``ModelCapabilities`` value.
    ///
    /// - Parameters:
    ///   - capabilities: The capabilities to modify in-place.
    ///   - forceReplace: When `true`, non-nil values always replace existing values.
    ///     When `false`, non-nil values only upgrade `false` → `true` (OR semantics).
    public func apply(to capabilities: inout ModelCapabilities, forceReplace: Bool) {
        if let v = toolUse, (forceReplace || v) { capabilities.toolUse = v }
        if let v = vision, (forceReplace || v) { capabilities.vision = v }
        if let v = reasoning, (forceReplace || v) { capabilities.reasoning = v }
        if let v = codeExecution, (forceReplace || v) { capabilities.codeExecution = v }
        if let v = promptCaching, (forceReplace || v) { capabilities.promptCaching = v }
        if let v = computerUse, (forceReplace || v) { capabilities.computerUse = v }
        if let v = audioInput, (forceReplace || v) { capabilities.audioInput = v }
        if let v = audioOutput, (forceReplace || v) { capabilities.audioOutput = v }
        if let v = videoInput, (forceReplace || v) { capabilities.videoInput = v }
        if let v = responseSchema, (forceReplace || v) { capabilities.responseSchema = v }
        if let v = parallelToolCalls, (forceReplace || v) { capabilities.parallelToolCalls = v }
        if let v = pdfInput, (forceReplace || v) { capabilities.pdfInput = v }
        if let v = webSearch, (forceReplace || v) { capabilities.webSearch = v }
        if let v = systemMessages, (forceReplace || v) { capabilities.systemMessages = v }
        if let v = assistantPrefill, (forceReplace || v) { capabilities.assistantPrefill = v }
        if let v = toolChoice, (forceReplace || v) { capabilities.toolChoice = v }
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
    /// Override individual capabilities. Per-flag, not all-or-nothing.
    public var capabilities: ModelCapabilitiesOverride?
    /// Override pricing data. Replaces the entire pricing structure when set.
    public var pricing: ModelPricing?
    /// Override chat completions support.
    public var supportsChatCompletions: Bool?
    /// Override per-(provider+model) runtime behavior flags. Per-flag, not all-or-nothing.
    public var behaviorFlags: BehaviorFlagsOverride?
    /// Hide this model from pickers. Hiding is presentation, never deletion — every record
    /// underneath survives, and un-hiding is removing this one field.
    public var hidden: Bool?

    public init(
        displayName: String? = nil,
        maxInputTokens: Int? = nil,
        maxOutputTokens: Int? = nil,
        capabilities: ModelCapabilitiesOverride? = nil,
        pricing: ModelPricing? = nil,
        supportsChatCompletions: Bool? = nil,
        behaviorFlags: BehaviorFlagsOverride? = nil,
        hidden: Bool? = nil
    ) {
        self.displayName = displayName
        self.maxInputTokens = maxInputTokens
        self.maxOutputTokens = maxOutputTokens
        self.capabilities = capabilities
        self.pricing = pricing
        self.supportsChatCompletions = supportsChatCompletions
        self.behaviorFlags = behaviorFlags
        self.hidden = hidden
    }

    /// Applies this override to a ``ModelInfo`` value.
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
        if let v = pricing, (forceReplace || model.pricing == nil) {
            model.pricing = v
        }
        if let v = supportsChatCompletions {
            if forceReplace {
                model.supportsChatCompletions = v
            } else if !v {
                // Gap-fill: only downgrade (mark as unsupported), never upgrade
                model.supportsChatCompletions = false
            }
        }
        if let capOverride = capabilities {
            capOverride.apply(to: &model.capabilities, forceReplace: forceReplace)
        }
        if let flagsOverride = behaviorFlags {
            flagsOverride.apply(to: &model.behaviorFlags, forceReplace: forceReplace)
        }
        if let v = hidden, (forceReplace || model.hidden == nil) {
            model.hidden = v
        }
    }
}
