import Foundation

/// What a model accepts for ONE effort construct — the parameter's existence and its legal values
/// in a single value, so the two can never contradict each other.
///
/// ## Two constructs, not one
///
/// "Effort" is two different things wearing one name, and conflating them is what this type exists
/// to end:
///
/// - **General effort** — Anthropic's top-level `output_config.effort`. It steers how much work the
///   model does overall and applies **even when reasoning is disabled**. It is not a thinking knob.
/// - **Reasoning effort** — OpenAI's / Moonshot's `reasoning_effort`. It exists only for reasoning
///   models and is rejected outright by models without them.
///
/// A model may support either, both, or neither, with different ladders for each, so they are
/// carried as two independent fields (``ModelFacts/generalEffort``, ``ModelFacts/reasoningEffort``).
///
/// ## Why one value instead of a ladder plus a flag
///
/// The predecessor stored a `[String]` ladder beside a `supportsReasoningEffort` boolean. Four
/// states were needed and four were representable — but so were contradictions, and one of them
/// was live: a probe that attempted the complete ladder and had every level rejected wrote an
/// empty ladder, while a forced override set the flag true. The provider then emitted
/// `reasoning_effort` on every request (HTTP 400), and pre-flight validation could not catch it
/// because it keyed on `!ladder.isEmpty`. The strongest evidence in the system — a complete probe
/// where nothing was accepted — was discarded by all three readers.
///
/// The four states are genuinely needed, so they are enumerated instead:
///
/// | | meaning |
/// |---|---|
/// | `nil` (the field is optional) | no source has said anything |
/// | ``unsupported`` | the parameter is rejected, or no level works |
/// | ``supportedLevelsUnknown`` | the parameter is accepted; which values, we don't know |
/// | ``levels(_:)`` | the parameter is accepted, and exactly these values are legal |
///
/// `supportedLevelsUnknown` is not a placeholder — it is the majority state for OpenAI, which
/// publishes no ladder at all, so its models are hand-authored as "the parameter works" with the
/// values left genuinely unknown.
public enum EffortSupport: Sendable, Equatable, Hashable {
    /// The parameter is rejected, or every level was tried and none worked. Do not send it.
    case unsupported
    /// The parameter is accepted. Which values are legal is not known, so nothing may be rejected
    /// on the basis of this record.
    case supportedLevelsUnknown
    /// The parameter is accepted and exactly these values are legal, ordered shallow → deep by
    /// ``EffortRank``. Never empty — an empty ladder is ``unsupported``, which ``init(levels:)``
    /// enforces so the illegal state cannot be constructed.
    case levels([String])

    /// Builds the state a stated ladder implies, normalizing the empty ladder to ``unsupported``.
    ///
    /// Callers decoding a vendor payload should use this rather than `.levels(_:)` directly: a
    /// vendor that says "supported, and here are zero levels" has said the parameter is useless,
    /// which is ``unsupported`` — and letting `.levels([])` exist would reintroduce the very
    /// contradiction this type removes.
    public init(levels: [String]) {
        let ordered = levels.sorted { EffortRank.rank(of: $0) < EffortRank.rank(of: $1) }
        self = ordered.isEmpty ? .unsupported : .levels(ordered)
    }

    /// Whether the parameter may be sent at all. `false` only for ``unsupported``.
    public var isSupported: Bool {
        self != .unsupported
    }

    /// The legal values when they are known, otherwise `nil`.
    ///
    /// Deliberately `nil` rather than `[]` for both ``unsupported`` and ``supportedLevelsUnknown`` —
    /// callers filtering a picker must not confuse "no levels exist" with "we don't know which",
    /// and neither may be rendered as an empty menu the user can't act on.
    public var knownLevels: [String]? {
        if case .levels(let levels) = self { return levels }
        return nil
    }

    /// Whether a specific value is known to be rejected.
    ///
    /// Fails SAFE: `false` whenever the ladder is unknown, so validation never rejects a value on
    /// an absence of evidence. Only a known ladder that omits the value, or an outright
    /// ``unsupported``, returns `true`.
    public func rejects(_ level: String) -> Bool {
        switch self {
        case .unsupported: return true
        case .supportedLevelsUnknown: return false
        case .levels(let levels): return !levels.contains(level)
        }
    }
}

// MARK: - Codable

/// Encoded as a tagged object rather than a bare array so the three states stay distinguishable on
/// disk. A bare `[]` was exactly the ambiguity that made the predecessor unreadable.
extension EffortSupport: Codable {
    private enum CodingKeys: String, CodingKey { case state, levels }
    private enum State: String, Codable { case unsupported, supportedLevelsUnknown, levels }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(State.self, forKey: .state) {
        case .unsupported: self = .unsupported
        case .supportedLevelsUnknown: self = .supportedLevelsUnknown
        case .levels: self = EffortSupport(levels: try container.decode([String].self, forKey: .levels))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .unsupported:
            try container.encode(State.unsupported, forKey: .state)
        case .supportedLevelsUnknown:
            try container.encode(State.supportedLevelsUnknown, forKey: .state)
        case .levels(let levels):
            try container.encode(State.levels, forKey: .state)
            try container.encode(levels, forKey: .levels)
        }
    }
}

// MARK: - Display

public extension EffortSupport {
    /// One-line summary for editors and inspectors.
    var editorSummary: String {
        switch self {
        case .unsupported: return "Not supported"
        case .supportedLevelsUnknown: return "Supported, levels unknown"
        case .levels(let levels): return levels.joined(separator: ", ")
        }
    }
}
