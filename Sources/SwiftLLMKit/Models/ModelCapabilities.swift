import Foundation

/// One feature a model can support. The set of cases IS the set of capability flags — adding a
/// capability means adding a case here and a matching computed accessor on ``ModelCapabilities``.
public enum ModelCapability: String, CaseIterable, Sendable, Codable, Hashable {
    /// The model answers on `/v1/chat/completions` — the baseline every agent role needs. Read like
    /// every other capability (no special default; unknown is not "supported"); see
    /// ``ModelInfo/supportsChatCompletions``, a view over this capability. Consumers that must not
    /// penalize an unmeasured model gate on a KNOWN non-chat (`state(of: .chat) == false`).
    case chat
    /// The model is a BATCH-only variant (async batch-submission API, e.g. OpenRouter's `:batch`
    /// slug), so it cannot back an interactive agent. Present so a role can forbid it via
    /// `mustNotBePresent: [.batch]`; it is never a thing to REQUIRE.
    case batch
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
    case reasoningEnableable
    case reasoningDisableable
    case thinkingKeepAll
    case thinkingBudgetTokens
    case structuredOutputJSONObject
    case toolChoiceRequired
    case toolChoiceNone
    case toolChoiceSpecificFunction
    case strictToolDefinitions

    /// Short human-readable label, used by ``ModelCapabilities/enabledLabels``.
    public var label: String {
        switch self {
        case .chat:                return "Chat"
        case .batch:               return "Batch-only"
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
        case .reasoningEnableable: return "Reasoning can be turned ON"
        case .reasoningDisableable: return "Reasoning can be turned OFF"
        case .thinkingKeepAll: return "Supports `thinking.keep`"
        case .thinkingBudgetTokens: return "Accepts a thinking token budget"
        case .structuredOutputJSONObject: return "Structured output: `json_object`"
        case .toolChoiceRequired: return "Tool choice: `required`"
        case .toolChoiceNone: return "Tool choice: `none`"
        case .toolChoiceSpecificFunction: return "Tool choice: specific function"
        case .strictToolDefinitions: return "Supports `strict` tool definitions"
        case .toolResultRoundTrip: return "Tool Round-Trip"
        }
    }

    /// Full title for the per-model capability editor. Exhaustive so a new case can't ship without one.
    public var editorTitle: String {
        switch self {
        case .chat:                return "Chat completions"
        case .batch:               return "Batch-only model"
        case .toolUse:             return "Tool use"
        case .vision:              return "Vision (image input)"
        case .reasoning:           return "Reasoning / thinking"
        case .codeExecution:       return "Code execution"
        case .promptCaching:       return "Prompt caching"
        case .computerUse:         return "Computer use"
        case .audioInput:          return "Audio input"
        case .audioOutput:         return "Audio output"
        case .videoInput:          return "Video input"
        case .responseSchema:      return "Response schema"
        case .parallelToolCalls:   return "Parallel tool calls"
        case .pdfInput:            return "PDF input"
        case .webSearch:           return "Web search"
        case .systemMessages:      return "System messages"
        case .assistantPrefill:    return "Assistant prefill"
        case .toolChoice:          return "Tool choice"
        case .reasoningEnableable: return "reasoning on"
        case .reasoningDisableable: return "reasoning off"
        case .thinkingKeepAll: return "thinking keep"
        case .thinkingBudgetTokens: return "thinking budget"
        case .structuredOutputJSONObject: return "json_object"
        case .toolChoiceRequired: return "tool_choice req"
        case .toolChoiceNone: return "tool_choice none"
        case .toolChoiceSpecificFunction: return "tool_choice fn"
        case .strictToolDefinitions: return "strict tools"
        case .toolResultRoundTrip: return "Tool result round-trip"
        }
    }

    /// One-line help for the per-model capability editor. Exhaustive by design.
    public var editorDescription: String {
        switch self {
        case .chat:                return "Model serves the chat-completions surface Agent Smith talks to. Off means it's responses-/embeddings-only, and assigning it to an agent fails with HTTP 404."
        case .batch:               return "This is a batch-only variant (async batch-submission API, e.g. OpenRouter's `:batch`). It can't run an interactive agent, so it's excluded from role pickers."
        case .toolUse:             return "Model can call tools. Frequently mis-reported as off for cloud/self-hosted models that do support it."
        case .vision:              return "Model can accept images in the prompt. Off means a pasted image is rejected (HTTP 400)."
        case .reasoning:           return "Model supports extended reasoning (thinking budget / effort)."
        case .codeExecution:       return "Model has a built-in code-execution tool."
        case .promptCaching:       return "Provider supports prompt/context caching for this model."
        case .computerUse:         return "Model supports computer-use / GUI-control tooling."
        case .audioInput:          return "Model can accept audio as input."
        case .audioOutput:         return "Model can produce audio output."
        case .videoInput:          return "Model can accept video as input."
        case .responseSchema:      return "Model supports structured-output / JSON-schema responses."
        case .parallelToolCalls:   return "Model can emit multiple tool calls in one turn."
        case .pdfInput:            return "Model can accept PDF documents as input."
        case .webSearch:           return "Model has a built-in web-search tool."
        case .systemMessages:      return "Model accepts a system message (some backends fold it into the first user turn)."
        case .assistantPrefill:    return "Model supports prefilling the start of the assistant's reply."
        case .toolChoice:          return "Model honors an explicit `tool_choice` selection."
        case .reasoningEnableable: return "Model accepts an explicit request to enable reasoning. Some reasoning models only ever reason and reject being switched on."
        case .reasoningDisableable: return "Model accepts an explicit request to disable reasoning. Thinking-only models reject this."
        case .thinkingKeepAll: return "Model accepts `thinking.keep: \"all\"` to retain reasoning content across turns. Applies ONLY to the `thinking` block mechanism (Moonshot/DeepSeek); meaningless on any other reasoning control."
        case .thinkingBudgetTokens: return "Model accepts an explicit reasoning token budget rather than only a named effort level."
        case .structuredOutputJSONObject: return "Model accepts `response_format: {type: json_object}` and returns syntactically valid JSON."
        case .toolChoiceRequired: return "Model accepts `tool_choice: required` (Anthropic `any`), forcing some tool call."
        case .toolChoiceNone: return "Model accepts `tool_choice: none`, forbidding tool calls while tools are present."
        case .toolChoiceSpecificFunction: return "Model accepts `tool_choice` naming one function, forcing that tool."
        case .strictToolDefinitions: return "Model accepts `strict: true` on a function definition for guaranteed schema adherence."
        case .toolResultRoundTrip: return "Model consumes tool RESULTS, not just emits calls — the half an agent depends on. Probe-established; force only to correct a stale verdict."
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
    public var reasoningEnableable: Bool { get { has(.reasoningEnableable) } set { set(.reasoningEnableable, newValue) } }
    public var reasoningDisableable: Bool { get { has(.reasoningDisableable) } set { set(.reasoningDisableable, newValue) } }
    public var thinkingKeepAll: Bool { get { has(.thinkingKeepAll) } set { set(.thinkingKeepAll, newValue) } }
    public var thinkingBudgetTokens: Bool { get { has(.thinkingBudgetTokens) } set { set(.thinkingBudgetTokens, newValue) } }
    public var structuredOutputJSONObject: Bool { get { has(.structuredOutputJSONObject) } set { set(.structuredOutputJSONObject, newValue) } }
    public var toolChoiceRequired: Bool { get { has(.toolChoiceRequired) } set { set(.toolChoiceRequired, newValue) } }
    public var toolChoiceNone: Bool { get { has(.toolChoiceNone) } set { set(.toolChoiceNone, newValue) } }
    public var toolChoiceSpecificFunction: Bool { get { has(.toolChoiceSpecificFunction) } set { set(.toolChoiceSpecificFunction, newValue) } }
    public var strictToolDefinitions: Bool { get { has(.strictToolDefinitions) } set { set(.strictToolDefinitions, newValue) } }

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
    /// `.chat` is deliberately omitted — it's the baseline every usable model has, so surfacing it as
    /// a chip is noise (the capability is still queryable via ``state(of:)``).
    public var enabledLabels: [String] {
        ModelCapability.allCases.filter { $0 != .chat && states[$0] == true }.map(\.label)
    }

    /// A copy with every KNOWN-false capability demoted to unknown, except those in `keeping`.
    /// Used when loading a LEGACY 2-state catalog where a stored `false` ambiguously meant
    /// "unknown, collapsed to false"; demoting restores fail-open filtering until the next refresh.
    public func demotingKnownFalse(keeping: Set<ModelCapability>) -> ModelCapabilities {
        var copy = self
        for capability in ModelCapability.allCases where !keeping.contains(capability) && copy.states[capability] == false {
            copy.states[capability] = nil
        }
        return copy
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

public extension ModelCapabilities {
    /// Whether a `tool_choice` carrying this option would actually be emitted.
    ///
    /// Two facts, because the decoders write two: ``ModelCapability/toolChoice`` is "this endpoint
    /// accepts the PARAMETER at all" (LiteLLM's `supportsToolChoice`, OpenRouter's
    /// `supported_parameters`), and the per-option capability is that option's own veto.
    ///
    /// Fails OPEN on unknown — `tool_choice` predates this gating, and silently dropping a
    /// caller's explicit choice on every model no source has described would be a behaviour
    /// change rather than a safety measure.
    ///
    /// Shared with `CapabilityProbe`, which must not claim it FORCED a tool call through a field
    /// the provider then suppressed. A second copy of this rule is how the probe's record and the
    /// wire drift apart.
    func permitsToolChoice(_ choice: LLMToolChoice) -> Bool {
        state(of: .toolChoice) != false && state(of: choice.requiredCapability) != false
    }
}
