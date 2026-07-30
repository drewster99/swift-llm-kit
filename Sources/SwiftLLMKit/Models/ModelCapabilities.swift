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

// A capability is usable as a coding KEY (its rawValue), so ``ModelCapabilities`` can serialize as a
// flat object of `capabilityName -> Bool`. Distinct from its `Codable` conformance, which is how a
// capability serializes as a VALUE (e.g. inside a `Set<ModelCapability>`).
extension ModelCapability: CodingKey {
    public var stringValue: String { rawValue }
    public init?(stringValue: String) { self.init(rawValue: stringValue) }
    public var intValue: Int? { nil }
    public init?(intValue: Int) { nil }
}

/// Tri-state feature flags describing what a model supports.
///
/// Backed by a `[ModelCapability: Bool]`: a capability PRESENT in the map is **known** (true or
/// false); ABSENT is **unknown** (never measured / provider silent). This distinction is the whole
/// point — a required-capability filter can then reject only an *explicit* false and leave unprobed
/// models visible (see ``satisfies(requiredCapabilities:mustNotBePresent:includedAvailabilityStates:)``).
///
/// The historical `var …: Bool` accessors are preserved as a 2-state VIEW where unknown reads as
/// `false`, so every existing reader behaves exactly as before. Reach the tri-state truth through the
/// ``subscript(_:)`` / ``state(of:)`` / ``isKnown(_:)`` API.
public struct ModelCapabilities: Sendable, Equatable {
    /// Known states only. Absent key ⇒ unknown. The single source of truth.
    private var states: [ModelCapability: Bool]

    // MARK: Tri-state access

    /// The known state of a capability, or `nil` when unknown. Assigning `nil` marks it unknown.
    public subscript(_ capability: ModelCapability) -> Bool? {
        get { states[capability] }
        set { states[capability] = newValue }
    }

    /// The known state of a capability, or `nil` when unknown.
    public func state(of capability: ModelCapability) -> Bool? { states[capability] }

    /// Whether the capability has any known state (true or false) as opposed to being unmeasured.
    public func isKnown(_ capability: ModelCapability) -> Bool { states[capability] != nil }

    // MARK: 2-state boolean view (unknown reads as false)

    /// The one place presence is read/written for the boolean view. Reading collapses unknown to
    /// `false`; WRITING records an EXPLICIT true/false (a deliberate statement, never "unknown").
    /// No lock: this is a value type — each copy owns its own map (copy-on-write).
    private func has(_ capability: ModelCapability) -> Bool { states[capability] == true }
    private mutating func set(_ capability: ModelCapability, _ on: Bool) { states[capability] = on }

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

    /// Whether the capability is KNOWN-TRUE. Unknown and known-false both read `false`.
    public func contains(_ capability: ModelCapability) -> Bool { has(capability) }

    /// The known-true capabilities as a set (the 2-state view of "on"). Unknown and false are absent.
    public var asSet: Set<ModelCapability> { Set(states.filter { $0.value }.keys) }

    // MARK: Initializers

    /// Every capability in the set is known-TRUE; all others are unknown.
    public init(_ trueCapabilities: Set<ModelCapability> = []) {
        states = Dictionary(uniqueKeysWithValues: trueCapabilities.map { ($0, true) })
    }

    /// Explicit tri-state: exactly the given known states; anything omitted is unknown.
    public init(states: [ModelCapability: Bool]) {
        self.states = states
    }

    /// Boolean-argument initializer, preserved for source compatibility. A passed `true` records a
    /// KNOWN-TRUE; a `false` (including the defaults) records NOTHING — it stays unknown. This is the
    /// convenience form for "these are on"; state an explicit false via ``subscript(_:)`` or the
    /// setters. (Under the old 2-state type, false and unknown were the same value, so no caller
    /// meant "known-false" by a defaulted argument.)
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
        var trues = Set<ModelCapability>()
        if toolUse { trues.insert(.toolUse) }
        if vision { trues.insert(.vision) }
        if reasoning { trues.insert(.reasoning) }
        if codeExecution { trues.insert(.codeExecution) }
        if promptCaching { trues.insert(.promptCaching) }
        if computerUse { trues.insert(.computerUse) }
        if audioInput { trues.insert(.audioInput) }
        if audioOutput { trues.insert(.audioOutput) }
        if videoInput { trues.insert(.videoInput) }
        if responseSchema { trues.insert(.responseSchema) }
        if parallelToolCalls { trues.insert(.parallelToolCalls) }
        if pdfInput { trues.insert(.pdfInput) }
        if webSearch { trues.insert(.webSearch) }
        if systemMessages { trues.insert(.systemMessages) }
        if assistantPrefill { trues.insert(.assistantPrefill) }
        if toolChoice { trues.insert(.toolChoice) }
        if toolResultRoundTrip { trues.insert(.toolResultRoundTrip) }
        states = Dictionary(uniqueKeysWithValues: trues.map { ($0, true) })
    }

    /// Human-readable labels for capabilities that are known-TRUE, in canonical case order.
    public var enabledLabels: [String] {
        ModelCapability.allCases.filter { states[$0] == true }.map(\.label)
    }
}

// MARK: - Codable
//
// Serializes as a flat object of only the KNOWN capabilities (`{"toolUse": true, "vision": false}`);
// unknown capabilities are OMITTED, which is how "unknown" survives a round-trip. Reading is
// backward-compatible: an old full 17-Bool object decodes as all-known — acceptable because that
// data was already flattened (unknown⇒false) at the source before tri-state existed, and the next
// catalog refresh repopulates true tri-state from the override/probe layers.

extension ModelCapabilities: Codable {
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: ModelCapability.self)
        var s = [ModelCapability: Bool]()
        for capability in ModelCapability.allCases {
            if let value = try c.decodeIfPresent(Bool.self, forKey: capability) {
                s[capability] = value
            }
        }
        states = s
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: ModelCapability.self)
        // Canonical order so the encoded object is stable across runs.
        for capability in ModelCapability.allCases {
            if let value = states[capability] {
                try c.encode(value, forKey: capability)
            }
        }
    }
}
