import Foundation

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
    /// Sampling temperature.
    public var temperature: Double
    /// Maximum tokens to generate per response.
    public var maxOutputTokens: Int
    /// Total context window budget in tokens (for conversation pruning).
    public var maxContextTokens: Int
    /// Extended thinking token budget. Relevant for `.anthropic` and `.alibabaCloud` providers.
    public var thinkingBudget: Int?
    /// Use 1-hour prompt cache TTL instead of the default 5-minute ephemeral cache.
    /// Only relevant for `.anthropic` providers. Cached tokens cost 2x the base input price.
    public var extendedCacheTTL: Bool
    /// When true, temperature is omitted from the API request, letting the model use its default.
    /// Useful for models (e.g. Alibaba QVQ) that require their default temperature.
    public var useDefaultTemperature: Bool
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
        temperature: Double = 0.7,
        maxOutputTokens: Int = 4096,
        maxContextTokens: Int = 128_000,
        thinkingBudget: Int? = nil,
        extendedCacheTTL: Bool = false,
        useDefaultTemperature: Bool = false,
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
        self.extendedCacheTTL = extendedCacheTTL
        self.useDefaultTemperature = useDefaultTemperature
        self.streaming = streaming
        self.isValid = isValid
        self.validationError = validationError
        self.extraJSONOverrides = extraJSONOverrides
    }

    /// Backward-compatible decoder: old JSON without `extendedCacheTTL` or
    /// `extraJSONOverrides` keys still loads cleanly.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        providerID = try container.decode(String.self, forKey: .providerID)
        modelID = try container.decode(String.self, forKey: .modelID)
        temperature = try container.decode(Double.self, forKey: .temperature)
        maxOutputTokens = try container.decode(Int.self, forKey: .maxOutputTokens)
        maxContextTokens = try container.decode(Int.self, forKey: .maxContextTokens)
        thinkingBudget = try container.decodeIfPresent(Int.self, forKey: .thinkingBudget)
        extendedCacheTTL = try container.decodeIfPresent(Bool.self, forKey: .extendedCacheTTL) ?? false
        useDefaultTemperature = try container.decodeIfPresent(Bool.self, forKey: .useDefaultTemperature) ?? false
        streaming = try container.decode(Bool.self, forKey: .streaming)
        isValid = try container.decode(Bool.self, forKey: .isValid)
        validationError = try container.decodeIfPresent(String.self, forKey: .validationError)
        extraJSONOverrides = try container.decodeIfPresent([String: AnyCodable].self, forKey: .extraJSONOverrides)
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
