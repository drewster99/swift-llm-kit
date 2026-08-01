import Foundation
import CryptoKit

/// A user-defined configuration pairing a provider + model with inference settings.
public struct ModelConfiguration: Codable, Identifiable, Sendable, Equatable {
    /// Unique identifier for this configuration. Mutable so callers can clone an
    /// existing configuration via `var copy = original; copy.id = UUID()` without
    /// having to enumerate every other field by hand.
    public var id: UUID
    /// User-defined name, e.g. "Claude Heavy", "Local Fast".
    public var name: String
    /// References `ModelProvider.id`.
    public var providerID: String
    /// Raw model ID from the provider (used in API calls).
    public var modelID: String
    /// Sampling temperature. **`nil` means "omit the field entirely"** so the
    /// provider falls back to the model's default. Required for some models
    /// (e.g. Claude Opus 4.7, GPT-5) that reject explicit `temperature`.
    /// Previously a non-optional `Double` paired with `useDefaultTemperature`
    /// — that two-flag design is gone; setting `temperature = nil` is the
    /// single source of truth for "use default".
    public var temperature: Double?
    /// Maximum tokens to generate per response.
    public var maxOutputTokens: Int
    /// Total context window budget in tokens (for conversation pruning).
    public var maxContextTokens: Int
    /// Extended thinking token budget. Relevant for `.anthropic` and `.alibabaCloud` providers.
    ///
    /// On Anthropic models with `BehaviorFlags.requiresAdaptiveThinking` set
    /// (Opus 4.7, 4.8), this budget is no longer used as a token count — those
    /// models only accept adaptive thinking which lets the model choose depth.
    /// The field is interpreted as a boolean signal: budget > 0 means "enable
    /// adaptive thinking", budget == 0 / nil means "thinking off." Use
    /// `thinkingEffort` to control depth on adaptive-thinking models.
    public var thinkingBudget: Int?
    /// Anthropic adaptive-thinking effort hint. Valid values: `"low"`,
    /// `"medium"`, `"high"`, `"xhigh"`, `"max"` (xhigh only on Opus 4.7, 4.8).
    /// `nil` (default) omits the field and lets Anthropic default to `"high"`.
    ///
    /// Emitted on Anthropic requests as a top-level `output_config: {effort: <value>}`
    /// field, independent of the thinking mode. Older Anthropic models that don't
    /// support the effort parameter will reject the request — set only on
    /// supported models (Opus 4.5+, Sonnet 4.6+, Opus 4.6+, Opus 4.7+, Opus 4.8).
    public var thinkingEffort: String?
    /// Use 1-hour prompt cache TTL instead of the default 5-minute ephemeral cache.
    /// Only relevant for `.anthropic` providers. Cached tokens cost 2x the base input price.
    public var extendedCacheTTL: Bool
    /// Whether to request streaming responses.
    public var streaming: Bool
    /// Set during validation — `false` if the config references a missing provider/model.
    public var isValid: Bool
    /// Human-readable reason the configuration is invalid, if any.
    public var validationError: String?
    /// Free-form key/value pairs merged into the outbound provider request body
    /// at top level (overriding defaults built by the provider). Useful for
    /// reaching provider-specific knobs the typed API doesn't model yet
    /// (e.g. Anthropic `thinking`, OpenAI `reasoning_effort`, Gemini
    /// `safetySettings`, structured-output schemas).
    public var extraJSONOverrides: [String: AnyCodable]?

    public init(
        id: UUID = UUID(),
        name: String,
        providerID: String,
        modelID: String,
        temperature: Double? = 0.7,
        maxOutputTokens: Int = 4096,
        maxContextTokens: Int = 128_000,
        thinkingBudget: Int? = nil,
        thinkingEffort: String? = nil,
        extendedCacheTTL: Bool = false,
        streaming: Bool = true,
        isValid: Bool = true,
        validationError: String? = nil,
        extraJSONOverrides: [String: AnyCodable]? = nil
    ) {
        self.id = id
        self.name = name
        self.providerID = providerID
        self.modelID = modelID
        self.temperature = temperature
        self.maxOutputTokens = maxOutputTokens
        self.maxContextTokens = maxContextTokens
        self.thinkingBudget = thinkingBudget
        self.thinkingEffort = thinkingEffort
        self.extendedCacheTTL = extendedCacheTTL
        self.streaming = streaming
        self.isValid = isValid
        self.validationError = validationError
        self.extraJSONOverrides = extraJSONOverrides
    }

    /// Backward-compatible decoder. Migration rules:
    /// - Legacy JSON with `"useDefaultTemperature": true` collapses to
    ///   `temperature = nil` regardless of the stored temperature value.
    /// - Legacy JSON with `"useDefaultTemperature": false` (or absent) keeps
    ///   the decoded temperature value.
    /// - `temperature` is now optional on the wire; legacy JSON has it present
    ///   (was non-optional), so decoding either form works.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        providerID = try container.decode(String.self, forKey: .providerID)
        modelID = try container.decode(String.self, forKey: .modelID)
        let rawTemperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        // Legacy migration: decode the now-removed useDefaultTemperature field
        // via a parallel keyset so it doesn't pollute the synthesized encoder.
        let legacyContainer = try decoder.container(keyedBy: LegacyCodingKeys.self)
        let legacyUseDefault = try legacyContainer.decodeIfPresent(Bool.self, forKey: .useDefaultTemperature) ?? false
        temperature = legacyUseDefault ? nil : rawTemperature
        maxOutputTokens = try container.decode(Int.self, forKey: .maxOutputTokens)
        maxContextTokens = try container.decode(Int.self, forKey: .maxContextTokens)
        thinkingBudget = try container.decodeIfPresent(Int.self, forKey: .thinkingBudget)
        thinkingEffort = try container.decodeIfPresent(String.self, forKey: .thinkingEffort)
        extendedCacheTTL = try container.decodeIfPresent(Bool.self, forKey: .extendedCacheTTL) ?? false
        streaming = try container.decode(Bool.self, forKey: .streaming)
        isValid = try container.decode(Bool.self, forKey: .isValid)
        validationError = try container.decodeIfPresent(String.self, forKey: .validationError)
        extraJSONOverrides = try container.decodeIfPresent([String: AnyCodable].self, forKey: .extraJSONOverrides)
    }

    /// Keys for synthesized Codable. `useDefaultTemperature` was removed in
    /// 0.0.21; its legacy decode lives in `LegacyCodingKeys` below.
    private enum CodingKeys: String, CodingKey {
        case id, name, providerID, modelID, temperature
        case maxOutputTokens, maxContextTokens, thinkingBudget, thinkingEffort, extendedCacheTTL
        case streaming, isValid, validationError, extraJSONOverrides
    }

    /// Side-channel keyset used by `init(from:)` to migrate the removed
    /// `useDefaultTemperature` field. Not used by the synthesized encoder.
    private enum LegacyCodingKeys: String, CodingKey {
        case useDefaultTemperature
    }
}

// MARK: - Backward-compat bridge

extension ModelConfiguration {
    /// **Deprecated.** Set `temperature = nil` directly to omit it from the request.
    ///
    /// Bridge for GUI consumers that pre-date the 0.0.21 change. Maps to
    /// `temperature == nil` on read; on write `true` clears temperature to nil,
    /// `false` is a no-op (set `temperature` to a value to assert a specific one).
    @available(*, deprecated, message: "Set `temperature = nil` to omit it from the request")
    public var useDefaultTemperature: Bool {
        get { temperature == nil }
        set {
            if newValue { temperature = nil }
            // Setting false intentionally does nothing — caller should assign
            // `temperature` directly. No previous value to restore.
        }
    }
}

// MARK: - Deterministic identity for projections

extension ModelConfiguration {
    /// A deterministic `id` for a config that is a PURE PROJECTION of a `(provider, model)` — e.g.
    /// the result of ``ModelConfigurationOverride/resolved(against:name:)``. Such a config is
    /// *derived*, not authored: recomputing it must yield an identical value, `id` included.
    ///
    /// The general `init(id: UUID = UUID())` default is deliberately random — a genuinely NEW config
    /// (e.g. one the user adds to the pool, looked up and persisted by `id`) needs a unique identity.
    /// A derived projection has the opposite requirement: a fresh random `id` each call makes an
    /// `Equatable` value compare unequal to its own recomputation, which thrashes any SwiftUI
    /// `.onChange`/`.task(id:)` observing it into a per-frame update loop. Deriving the id from the
    /// `(providerID, modelID)` it configures keeps the projection stable across recomputes and changes
    /// it exactly when the model does.
    public static func deterministicID(providerID: String, modelID: String) -> UUID {
        let digest = SHA256.hash(data: Data("\(providerID)\n\(modelID)".utf8))
        return digest.withUnsafeBytes { UUID(uuid: $0.load(as: uuid_t.self)) }
    }
}

// MARK: - Convenience Accessors

extension ModelConfiguration {
    /// Alias for `modelID`, matching the field name used by LLM provider APIs.
    public var model: String { modelID }
    /// Alias for `maxOutputTokens`, matching common provider API naming.
    public var maxTokens: Int { maxOutputTokens }
    /// Alias for `maxContextTokens`, the total context window budget in tokens.
    public var contextWindowSize: Int { maxContextTokens }
}
