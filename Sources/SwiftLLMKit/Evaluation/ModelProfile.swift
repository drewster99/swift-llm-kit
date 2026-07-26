import Foundation

/// What we have *established* about a model by calling it, as opposed to what a catalog claims.
///
/// Deliberately separate from ``ModelInfo``. `ModelInfo` is the merged view the app runs on —
/// provider payload, LiteLLM, bundled overrides and user overrides layered together, with no way
/// to tell which layer said what. A profile is narrower and stricter: every field records how we
/// know it, and "we couldn't find out" is a first-class answer rather than a `false`.
///
/// Keeping them apart is what makes the profile usable as evidence. If probe results were written
/// straight into `ModelInfo` they would be indistinguishable from LiteLLM's claims the moment they
/// landed, and the whole point is that those are different kinds of thing.
public struct ModelProfile: Sendable, Codable, Equatable {
    public let providerID: String
    public let modelID: String
    public let probedAt: Date

    /// Identity as the provider reports it — decoded, never probed.
    public var displayName: String?
    public var createdAt: Date?

    /// Per-token pricing as the provider's own `/models` payload published it — decoded, never
    /// probed (there is nothing to probe). `nil` when the provider doesn't state prices (OpenAI,
    /// Anthropic, z.ai omit them; xAI, OpenRouter, HuggingFace publish them). Carried here so a
    /// written profile is a complete record, not so the probe measures cost.
    public var pricing: ModelPricing?
    /// When the provider marked this model for deprecation, if it did (Mistral publishes a date).
    /// `nil` means unmarked — never a guarantee of currency. A future date is "still usable, going
    /// away then."
    public var deprecatedOn: Date?

    /// The highest `temperature` the provider says this model accepts, when it publishes one (only
    /// Gemini does: `maxTemperature`, typically 2). A real request-validation ceiling — sending
    /// above it is rejected — not a default. `nil` when unstated.
    public var maxTemperature: Double?

    /// The default sampling parameters the provider publishes (Gemini, Mistral, OpenRouter).
    /// Decoded-only reference metadata; `nil` when unstated.
    public var samplingDefaults: SamplingDefaults?
    /// Whether the provider serves this model free (HuggingFace per-provider `is_free`). Decoded-only.
    public var isFree: Bool?
    /// Third-party benchmark scores (OpenRouter's `benchmarks`). Decoded-only reference metadata.
    public var benchmarks: ModelBenchmarks?

    /// Whether the model is actually reachable, as opposed to merely listed. Seeded `decoded(true)`
    /// ("present in /models") and flipped to `established(false)` when a live call reports it gone
    /// (a 404 "no longer available"). Kept separate from ``deprecatedOn`` on purpose: a model can be
    /// scheduled-for-deprecation yet fully usable, or delisted-but-still-listed yet already dead.
    public var isAvailable: ProbeFinding<Bool>

    /// Whether the account is allowed to call this model. `established(false)` when a live call is
    /// refused for access reasons (Alibaba Cloud's `Model.AccessDenied` — the model exists and is
    /// current, but the dashboard hasn't enabled it for this key). Separate from ``isAvailable``:
    /// "you can't call it" and "it no longer exists" are different facts a caller may treat
    /// differently.
    public var isAccessDenied: ProbeFinding<Bool>

    /// The model's context window. Decoded-only today (Anthropic's `max_input_tokens`, Gemini's
    /// `inputTokenLimit`): probing it means actually sending a window's worth of tokens, which at
    /// 1M-context prices is deferred until everything else is finished and proven.
    public var maxContextTokens: ProbeFinding<Int>

    /// Does the chat endpoint accept this model and answer coherently.
    public var chat: ProbeFinding<Bool>
    /// Does it emit a well-formed tool call. The one hard requirement no vendor publishes
    /// (Anthropic's capabilities block has no tool key; OpenAI's `/models` has four fields).
    public var toolCalling: ProbeFinding<Bool>
    /// Does it take a tool RESULT back and use it — the half an agent actually depends on.
    public var toolResultRoundTrip: ProbeFinding<Bool>
    /// Does it accept an image AND read it. Accepting is not reading: a model that ignores the
    /// image and guesses looks identical from the outside, so this is only `true` when it
    /// returned a code it could not have known otherwise.
    public var vision: ProbeFinding<Bool>
    /// As `vision`, for a PDF document part.
    public var pdfInput: ProbeFinding<Bool>
    /// Whether the endpoint tolerates a `temperature` parameter at all. Not a capability anyone
    /// publishes, and a live hazard: claude-fable-5 answers it with HTTP 400, which made every
    /// request from this app fail before we learned it the hard way.
    public var acceptsTemperature: ProbeFinding<Bool>
    /// The model's real output ceiling, read out of the endpoint's own rejection.
    public var maxOutputTokens: ProbeFinding<Int>
    /// Set (established) only when the endpoint has NO independent output cap — it bounds
    /// max_tokens solely by context length (gpt-4's "maximum context length is 8192"). The value
    /// is that context length. Optional so records written before this field decode as nil. When
    /// established, `maxOutputTokens` stays inconclusive (there is no output cap to state) and
    /// this drives the context-based validation/clamp instead.
    public var maxOutputBoundedByContext: ProbeFinding<Int>?
    /// Whether the endpoint accepts a `{"role": "system"}` turn at the TAIL of `messages` — after
    /// the last user turn, as a steering nudge immediately before generation — and actually acts on
    /// it. Anthropic gates this per model and answers a no with `role 'system' is not supported on
    /// this model`; nothing in its `/models` payload states it, so it can only be established by
    /// asking. Optional so records written before this field decode as nil rather than failing.
    ///
    /// `established(true)` requires more than a 200: the turn carries a nonce the model cannot
    /// otherwise know, so an echo proves the content was *read*, not merely that the role was
    /// tolerated. A 200 without the echo stays inconclusive — "accepted but ignored" and "the model
    /// rambled" are not distinguishable from one call, and neither is a measured no.
    public var trailingSystemTurn: ProbeFinding<Bool>?

    /// Per named effort level: accepted or rejected. Keys are the levels attempted.
    public var effortLevels: [String: ProbeFinding<Bool>]

    /// Total wall clock and call count, so the cost of a full run is visible rather than guessed.
    public var callCount: Int
    public var duration: TimeInterval

    public init(
        providerID: String,
        modelID: String,
        probedAt: Date = Date(),
        displayName: String? = nil,
        createdAt: Date? = nil,
        pricing: ModelPricing? = nil,
        deprecatedOn: Date? = nil,
        maxTemperature: Double? = nil,
        samplingDefaults: SamplingDefaults? = nil,
        isFree: Bool? = nil,
        benchmarks: ModelBenchmarks? = nil,
        isAvailable: ProbeFinding<Bool> = .notAttempted,
        isAccessDenied: ProbeFinding<Bool> = .notAttempted,
        maxContextTokens: ProbeFinding<Int> = .notAttempted,
        chat: ProbeFinding<Bool> = .notAttempted,
        toolCalling: ProbeFinding<Bool> = .notAttempted,
        toolResultRoundTrip: ProbeFinding<Bool> = .notAttempted,
        vision: ProbeFinding<Bool> = .notAttempted,
        pdfInput: ProbeFinding<Bool> = .notAttempted,
        acceptsTemperature: ProbeFinding<Bool> = .notAttempted,
        maxOutputTokens: ProbeFinding<Int> = .notAttempted,
        maxOutputBoundedByContext: ProbeFinding<Int>? = nil,
        trailingSystemTurn: ProbeFinding<Bool>? = nil,
        effortLevels: [String: ProbeFinding<Bool>] = [:],
        callCount: Int = 0,
        duration: TimeInterval = 0
    ) {
        self.providerID = providerID
        self.modelID = modelID
        self.probedAt = probedAt
        self.displayName = displayName
        self.createdAt = createdAt
        self.pricing = pricing
        self.deprecatedOn = deprecatedOn
        self.maxTemperature = maxTemperature
        self.samplingDefaults = samplingDefaults
        self.isFree = isFree
        self.benchmarks = benchmarks
        self.isAvailable = isAvailable
        self.isAccessDenied = isAccessDenied
        self.maxContextTokens = maxContextTokens
        self.chat = chat
        self.toolCalling = toolCalling
        self.toolResultRoundTrip = toolResultRoundTrip
        self.vision = vision
        self.pdfInput = pdfInput
        self.acceptsTemperature = acceptsTemperature
        self.maxOutputTokens = maxOutputTokens
        self.maxOutputBoundedByContext = maxOutputBoundedByContext
        self.trailingSystemTurn = trailingSystemTurn
        self.effortLevels = effortLevels
        self.callCount = callCount
        self.duration = duration
    }

    /// The effort levels this model accepted, ordered shallow → deep. Levels we never tried, or
    /// tried inconclusively, are absent rather than assumed unsupported.
    public var establishedEffortLevels: [String] {
        effortLevels
            .filter { $0.value.value == true }
            .map(\.key)
            .sorted { EffortRank.rank(of: $0) < EffortRank.rank(of: $1) }
    }
}

// MARK: - Call counting

/// Counts the API calls a probe run actually spends, so the cost of a sweep is measured rather
/// than estimated. A reference type because the probes that increment it are separate async calls;
/// optional at every probe so an individual probe can be run standalone without one.
public final class ProbeCallCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    public init() {}
    public func increment() { lock.lock(); count += 1; lock.unlock() }
    public var value: Int { lock.lock(); defer { lock.unlock() }; return count }
}

// MARK: - Findings

/// One established (or not-established) fact, with its provenance.
///
/// Two distinctions earn this type. First: **"we asked and it said no" and "we never got an
/// answer" are different facts**, and a plain `Bool` cannot tell them apart — collapsing a
/// timeout into `false` writes a fabricated measurement into data we intend to trust. So `value`
/// is `nil` unless `status == .established`.
///
/// Second: **how we know matters as much as what we know.** A fact decoded from the vendor's own
/// `/models` payload and a fact established by calling the model are both trustworthy, but they
/// age differently (decoded facts refresh with the payload; probed facts go stale) and they
/// disagree differently (a probe contradicting a payload is a finding, not noise). `source`
/// records which one this is, so a profile can be a complete model record — the decoded facts
/// carried in for free, probes spent only on the gaps — without flattening the two into one.
public struct ProbeFinding<Value: Codable & Sendable & Equatable>: Sendable, Codable, Equatable {
    public enum Status: String, Sendable, Codable {
        /// We know. `value` is trustworthy.
        case established
        /// We asked and learned nothing — timeout, rate limit, server fault, or a refusal that
        /// was about our own request rather than the thing under test.
        case inconclusive
        /// Not run: skipped, or made moot by an earlier finding.
        case notAttempted
    }

    /// How the fact was established.
    public enum Source: String, Sendable, Codable {
        /// By calling the model and grading what came back.
        case probed
        /// Read from the provider's own `/models` payload — the vendor describing its own model,
        /// believed as given per the governing rule (decode what's published; probe the rest).
        case decoded
    }

    public var status: Status
    /// Non-nil only when `status == .established`.
    public var value: Value?
    /// What convinced us — ideally the endpoint's own words. A 400 saying "tools are not
    /// supported" and one saying "unknown parameter 'temperature'" are entirely different
    /// findings, and the status code alone loses that.
    public var evidence: String?
    public var source: Source?
    public var duration: TimeInterval?

    public static var notAttempted: Self { .init(status: .notAttempted, value: nil) }

    public static func established(_ value: Value, _ evidence: String? = nil, duration: TimeInterval? = nil) -> Self {
        .init(status: .established, value: value, evidence: evidence, source: .probed, duration: duration)
    }

    /// A fact read out of the provider's `/models` payload rather than established by a call.
    public static func decoded(_ value: Value, _ evidence: String? = nil) -> Self {
        .init(status: .established, value: value, evidence: evidence, source: .decoded)
    }

    public static func inconclusive(_ reason: String, duration: TimeInterval? = nil) -> Self {
        .init(status: .inconclusive, value: nil, evidence: reason, source: .probed, duration: duration)
    }
}

// MARK: - Effort ranking

/// Orders named effort levels, because the names do not sort and the vendors do not tell us.
///
/// Anthropic ships both `max` and `xhigh` and nothing in the strings says which is deeper —
/// alphabetically `max` sorts first, which is wrong. The order below is from Anthropic's effort
/// documentation, whose table runs max → xhigh → high → medium → low, and matches OpenAI's
/// listing for gpt-5.6 (`none, low, medium, high, xhigh, max`). Both vendors agree, so one table
/// serves both; only *membership* is per-model.
///
/// Spaced by 100 to leave room for levels vendors haven't invented yet. Nothing persists a rank —
/// it is derived from the name at sort time — so renumbering later costs nothing.
///
/// A caution learned the hard way: support is NOT nested by depth. Sonnet 4.6 accepts `max` but
/// not `xhigh`, because `xhigh` is a newer, specialised level ("long-horizon work"), not one notch
/// below max. Do not infer ordering from which models support what.
public enum EffortRank {
    public static let table: [String: Int] = [
        "none": -100,    // OpenAI only — doubles as the reasoning off-switch. Anthropic has no equivalent.
        "minimal": 0,    // OpenAI only.
        "low": 100,
        "medium": 200,
        "high": 300,     // Anthropic's default; identical to omitting the parameter.
        "xhigh": 400,
        "max": 500
    ]

    /// Unknown names sort last, so a level a vendor adds tomorrow appears at the deep end rather
    /// than silently ranking as shallowest.
    public static func rank(of name: String) -> Int {
        table[name.lowercased()] ?? Int.max
    }

    /// Every level we know how to ask for, shallow → deep. The probe walks this; a model's actual
    /// set is whatever it accepts.
    public static var allKnown: [String] {
        table.keys.sorted { rank(of: $0) < rank(of: $1) }
    }
}
