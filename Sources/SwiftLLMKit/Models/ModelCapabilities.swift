import Foundation

/// One feature a model can support. The set of cases IS the set of capability flags — adding a
/// capability means adding a case here and a matching computed accessor on ``ModelCapabilities``.
public enum ModelCapability: String, CaseIterable, Sendable, Codable, Hashable {
    case toolUse
    case vision
    case reasoning
    case codeExecution
    case promptCaching
    case computerUse
    case audioInput
    case audioOutput
    case videoInput
    case responseSchema
    case parallelToolCalls
    case pdfInput
    case webSearch
    case systemMessages
    case assistantPrefill
    case toolChoice
    /// Empirical-only: the model called a tool AND consumed the tool result (returned the probe's
    /// identifier). The half an agent actually depends on; no vendor publishes it.
    case toolResultRoundTrip

    /// Short human-readable label, used by ``ModelCapabilities/enabledLabels``.
    public var label: String {
        switch self {
        case .toolUse:             return "Tools"
        case .vision:              return "Vision"
        case .reasoning:           return "Reasoning"
        case .codeExecution:       return "Code Exec"
        case .promptCaching:       return "Caching"
        case .computerUse:         return "Computer Use"
        case .audioInput:          return "Audio In"
        case .audioOutput:         return "Audio Out"
        case .videoInput:          return "Video In"
        case .responseSchema:      return "Schema"
        case .parallelToolCalls:   return "Parallel Tools"
        case .pdfInput:            return "PDF"
        case .webSearch:           return "Web Search"
        case .systemMessages:      return "System Msgs"
        case .assistantPrefill:    return "Prefill"
        case .toolChoice:          return "Tool Choice"
        case .toolResultRoundTrip: return "Tool Round-Trip"
        }
    }
}

/// Feature flags describing what a model supports.
///
/// Backed by a `Set<ModelCapability>`: a capability is "on" iff its case is present. Every historical
/// `public var …: Bool` survives as a computed get/set over the set, so callers and the on-disk
/// Codable JSON are unchanged — presence in the set is the single source of truth.
public struct ModelCapabilities: Sendable, Equatable {
    private var capabilities: Set<ModelCapability>

    /// The one place presence is read/written, so the boolean accessors stay uniform. No lock: this
    /// is a value type — each copy owns its own set (copy-on-write), so there is no shared mutable
    /// state to guard.
    private func has(_ capability: ModelCapability) -> Bool { capabilities.contains(capability) }
    private mutating func set(_ capability: ModelCapability, _ on: Bool) {
        if on { capabilities.insert(capability) } else { capabilities.remove(capability) }
    }

    public var toolUse: Bool { get { has(.toolUse) } set { set(.toolUse, newValue) } }
    public var vision: Bool { get { has(.vision) } set { set(.vision, newValue) } }
    public var reasoning: Bool { get { has(.reasoning) } set { set(.reasoning, newValue) } }
    public var codeExecution: Bool { get { has(.codeExecution) } set { set(.codeExecution, newValue) } }
    public var promptCaching: Bool { get { has(.promptCaching) } set { set(.promptCaching, newValue) } }
    public var computerUse: Bool { get { has(.computerUse) } set { set(.computerUse, newValue) } }
    public var audioInput: Bool { get { has(.audioInput) } set { set(.audioInput, newValue) } }
    public var audioOutput: Bool { get { has(.audioOutput) } set { set(.audioOutput, newValue) } }
    public var videoInput: Bool { get { has(.videoInput) } set { set(.videoInput, newValue) } }
    public var responseSchema: Bool { get { has(.responseSchema) } set { set(.responseSchema, newValue) } }
    public var parallelToolCalls: Bool { get { has(.parallelToolCalls) } set { set(.parallelToolCalls, newValue) } }
    public var pdfInput: Bool { get { has(.pdfInput) } set { set(.pdfInput, newValue) } }
    public var webSearch: Bool { get { has(.webSearch) } set { set(.webSearch, newValue) } }
    public var systemMessages: Bool { get { has(.systemMessages) } set { set(.systemMessages, newValue) } }
    public var assistantPrefill: Bool { get { has(.assistantPrefill) } set { set(.assistantPrefill, newValue) } }
    public var toolChoice: Bool { get { has(.toolChoice) } set { set(.toolChoice, newValue) } }
    public var toolResultRoundTrip: Bool { get { has(.toolResultRoundTrip) } set { set(.toolResultRoundTrip, newValue) } }

    /// Whether the given capability is present. Lets `availableModels`'s filter read the set directly.
    public func contains(_ capability: ModelCapability) -> Bool { has(capability) }

    /// The enabled capabilities as a set — the underlying representation, exposed read-only.
    public var asSet: Set<ModelCapability> { capabilities }

    /// Builds directly from a capability set.
    public init(_ capabilities: Set<ModelCapability> = []) {
        self.capabilities = capabilities
    }

    /// Boolean-argument initializer, preserved verbatim so every existing call site compiles unchanged.
    public init(
        toolUse: Bool = false,
        vision: Bool = false,
        reasoning: Bool = false,
        codeExecution: Bool = false,
        promptCaching: Bool = false,
        computerUse: Bool = false,
        audioInput: Bool = false,
        audioOutput: Bool = false,
        videoInput: Bool = false,
        responseSchema: Bool = false,
        parallelToolCalls: Bool = false,
        pdfInput: Bool = false,
        webSearch: Bool = false,
        systemMessages: Bool = false,
        assistantPrefill: Bool = false,
        toolChoice: Bool = false,
        toolResultRoundTrip: Bool = false
    ) {
        var set = Set<ModelCapability>()
        if toolUse { set.insert(.toolUse) }
        if vision { set.insert(.vision) }
        if reasoning { set.insert(.reasoning) }
        if codeExecution { set.insert(.codeExecution) }
        if promptCaching { set.insert(.promptCaching) }
        if computerUse { set.insert(.computerUse) }
        if audioInput { set.insert(.audioInput) }
        if audioOutput { set.insert(.audioOutput) }
        if videoInput { set.insert(.videoInput) }
        if responseSchema { set.insert(.responseSchema) }
        if parallelToolCalls { set.insert(.parallelToolCalls) }
        if pdfInput { set.insert(.pdfInput) }
        if webSearch { set.insert(.webSearch) }
        if systemMessages { set.insert(.systemMessages) }
        if assistantPrefill { set.insert(.assistantPrefill) }
        if toolChoice { set.insert(.toolChoice) }
        if toolResultRoundTrip { set.insert(.toolResultRoundTrip) }
        capabilities = set
    }

    /// Human-readable labels for capabilities that are enabled, in the canonical case order.
    public var enabledLabels: [String] {
        ModelCapability.allCases.filter { capabilities.contains($0) }.map(\.label)
    }
}

// MARK: - Codable (backward-compatible)
//
// The wire format is UNCHANGED — a flat object of per-capability Bools (`{"toolUse": true, …}`), so
// every persisted catalog still decodes. The synthesized Codable would encode the private `Set`,
// breaking that, so both directions are hand-written to map set ⇄ booleans.

extension ModelCapabilities: Codable {
    private enum CodingKeys: String, CodingKey {
        case toolUse, vision, reasoning, codeExecution, promptCaching
        case computerUse, audioInput, audioOutput, videoInput
        case responseSchema, parallelToolCalls
        case pdfInput, webSearch, systemMessages, assistantPrefill, toolChoice
        case toolResultRoundTrip
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func flag(_ key: CodingKeys) throws -> Bool { try c.decodeIfPresent(Bool.self, forKey: key) ?? false }
        self.init(
            toolUse: try flag(.toolUse),
            vision: try flag(.vision),
            reasoning: try flag(.reasoning),
            codeExecution: try flag(.codeExecution),
            promptCaching: try flag(.promptCaching),
            computerUse: try flag(.computerUse),
            audioInput: try flag(.audioInput),
            audioOutput: try flag(.audioOutput),
            videoInput: try flag(.videoInput),
            responseSchema: try flag(.responseSchema),
            parallelToolCalls: try flag(.parallelToolCalls),
            pdfInput: try flag(.pdfInput),
            webSearch: try flag(.webSearch),
            systemMessages: try flag(.systemMessages),
            assistantPrefill: try flag(.assistantPrefill),
            toolChoice: try flag(.toolChoice),
            toolResultRoundTrip: try flag(.toolResultRoundTrip)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(toolUse, forKey: .toolUse)
        try c.encode(vision, forKey: .vision)
        try c.encode(reasoning, forKey: .reasoning)
        try c.encode(codeExecution, forKey: .codeExecution)
        try c.encode(promptCaching, forKey: .promptCaching)
        try c.encode(computerUse, forKey: .computerUse)
        try c.encode(audioInput, forKey: .audioInput)
        try c.encode(audioOutput, forKey: .audioOutput)
        try c.encode(videoInput, forKey: .videoInput)
        try c.encode(responseSchema, forKey: .responseSchema)
        try c.encode(parallelToolCalls, forKey: .parallelToolCalls)
        try c.encode(pdfInput, forKey: .pdfInput)
        try c.encode(webSearch, forKey: .webSearch)
        try c.encode(systemMessages, forKey: .systemMessages)
        try c.encode(assistantPrefill, forKey: .assistantPrefill)
        try c.encode(toolChoice, forKey: .toolChoice)
        try c.encode(toolResultRoundTrip, forKey: .toolResultRoundTrip)
    }
}
