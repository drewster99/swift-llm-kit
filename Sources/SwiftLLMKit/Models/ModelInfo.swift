import Foundation

/// Metadata about a specific model available from a provider.
public struct ModelInfo: Identifiable, Sendable, Equatable {
    /// Composite identifier: `providerID/modelID`.
    public var id: String { "\(providerID)/\(modelID)" }

    /// The provider this model belongs to.
    public let providerID: String
    /// The raw model identifier as used in API calls.
    public let modelID: String
    /// Human-readable name (e.g. "Claude Opus 4.6").
    public var displayName: String
    /// When the model was created or last modified.
    public var createdAt: Date?
    /// Maximum input context window in tokens.
    public var maxInputTokens: Int?
    /// Maximum output tokens the model can generate.
    public var maxOutputTokens: Int?
    /// Feature capabilities of this model.
    public var capabilities: ModelCapabilities
    /// Compact size label, e.g. "8.6B" (Ollama).
    public var sizeLabel: String?
    /// Quantization level, e.g. "Q4_K_M" (Ollama).
    public var quantizationLabel: String?
    /// Rich pricing data for this model. Supports tiered pricing, cache
    /// pricing, and service tier variants.
    public var pricing: ModelPricing?
    /// Whether this model supports `/v1/chat/completions`. Defaults to `true` unless LiteLLM says otherwise.
    ///
    /// Endpoint reachability only — it does NOT mean the model can back an agent. Gemini's image
    /// models answer on the chat endpoint and are `true` here. Use ``isConversational`` to ask
    /// whether a model is a conversational text model.
    public var supportsChatCompletions: Bool
    /// The model's kind as reported by LiteLLM (`chat`, `embedding`, `image_generation`,
    /// `responses`, `ocr`, …), or `nil` when unknown — which is the norm for the ~63% of the
    /// catalog LiteLLM doesn't cover.
    ///
    /// Carried verbatim rather than reduced to a Bool so the UI can say *why* a model was
    /// disqualified ("this is an embedding model") instead of only that it was.
    public var mode: String?
    /// Per-(provider+model) runtime behavior knobs. Defaults to all-off — a
    /// model with no flags set behaves like a model from before this field
    /// existed. See `BehaviorFlags` for available knobs and the layering rules.
    public var behaviorFlags: BehaviorFlags

    // MARK: - Backward-compatible pricing accessors

    /// Input cost in USD per million tokens (base tier, uncached).
    /// Backed by ``pricing``. Setting this creates a base pricing tier if needed.
    public var inputCostPerMillionTokens: Double? {
        get { pricing?.base.input.map { $0 * 1_000_000 } }
        set {
            if pricing == nil && newValue != nil { pricing = ModelPricing() }
            pricing?.base.input = newValue.map { $0 / 1_000_000 }
        }
    }

    /// Output cost in USD per million tokens (base tier).
    /// Backed by ``pricing``. Setting this creates a base pricing tier if needed.
    public var outputCostPerMillionTokens: Double? {
        get { pricing?.base.output.map { $0 * 1_000_000 } }
        set {
            if pricing == nil && newValue != nil { pricing = ModelPricing() }
            pricing?.base.output = newValue.map { $0 / 1_000_000 }
        }
    }

    public init(
        providerID: String,
        modelID: String,
        displayName: String = "",
        createdAt: Date? = nil,
        maxInputTokens: Int? = nil,
        maxOutputTokens: Int? = nil,
        capabilities: ModelCapabilities = ModelCapabilities(),
        sizeLabel: String? = nil,
        quantizationLabel: String? = nil,
        pricing: ModelPricing? = nil,
        supportsChatCompletions: Bool = true,
        mode: String? = nil,
        behaviorFlags: BehaviorFlags = BehaviorFlags()
    ) {
        self.providerID = providerID
        self.modelID = modelID
        self.displayName = displayName.isEmpty ? modelID : displayName
        self.createdAt = createdAt
        self.maxInputTokens = maxInputTokens
        self.maxOutputTokens = maxOutputTokens
        self.capabilities = capabilities
        self.sizeLabel = sizeLabel
        self.quantizationLabel = quantizationLabel
        self.pricing = pricing
        self.supportsChatCompletions = supportsChatCompletions
        self.mode = mode
        self.behaviorFlags = behaviorFlags
    }

    /// Whether this model is a conversational text model — the kind an agent can drive.
    ///
    /// Distinct from ``supportsChatCompletions``, which only says the endpoint accepts it:
    /// Gemini's image models answer on `/v1/chat/completions` and are still not agent material.
    /// An unknown `mode` reports `true`, because most of the catalog has no LiteLLM entry and
    /// must stay usable.
    public var isConversational: Bool {
        guard let mode else { return true }
        return !Self.nonConversationalModes.contains(mode)
    }

    /// Kinds that can never back an agent. `chat` and `responses` are both text-generation kinds
    /// and are deliberately excluded — a `responses` model may still be chat-reachable, which
    /// ``supportsChatCompletions`` decides.
    static let nonConversationalModes: Set<String> = [
        "embedding", "image_generation", "video_generation", "audio_speech",
        "audio_transcription", "moderation", "ocr", "rerank"
    ]

    /// Whether the model was created/modified within the last 90 days.
    public var isNew: Bool {
        guard let createdAt else { return false }
        let cutoff = Calendar.current.date(byAdding: .day, value: -90, to: Date()) ?? Date.distantPast
        return createdAt > cutoff
    }
}

// MARK: - Codable (backward-compatible)

extension ModelInfo: Codable {
    private enum CodingKeys: String, CodingKey {
        case providerID, modelID, displayName, createdAt
        case maxInputTokens, maxOutputTokens, capabilities
        case sizeLabel, quantizationLabel, pricing, supportsChatCompletions
        case mode, behaviorFlags
        // Legacy keys for reading old persisted data
        case inputCostPerMillionTokens, outputCostPerMillionTokens
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        providerID = try container.decode(String.self, forKey: .providerID)
        modelID = try container.decode(String.self, forKey: .modelID)
        let name = try container.decodeIfPresent(String.self, forKey: .displayName) ?? ""
        displayName = name.isEmpty ? modelID : name
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt)
        maxInputTokens = try container.decodeIfPresent(Int.self, forKey: .maxInputTokens)
        maxOutputTokens = try container.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        capabilities = try container.decodeIfPresent(ModelCapabilities.self, forKey: .capabilities) ?? ModelCapabilities()
        sizeLabel = try container.decodeIfPresent(String.self, forKey: .sizeLabel)
        quantizationLabel = try container.decodeIfPresent(String.self, forKey: .quantizationLabel)
        supportsChatCompletions = try container.decodeIfPresent(Bool.self, forKey: .supportsChatCompletions) ?? true
        mode = try container.decodeIfPresent(String.self, forKey: .mode)
        behaviorFlags = try container.decodeIfPresent(BehaviorFlags.self, forKey: .behaviorFlags) ?? BehaviorFlags()

        // Read new pricing field, or fall back to legacy flat cost fields.
        if let p = try container.decodeIfPresent(ModelPricing.self, forKey: .pricing) {
            pricing = p
        } else {
            let legacyInput = try container.decodeIfPresent(Double.self, forKey: .inputCostPerMillionTokens)
            let legacyOutput = try container.decodeIfPresent(Double.self, forKey: .outputCostPerMillionTokens)
            if legacyInput != nil || legacyOutput != nil {
                pricing = ModelPricing(base: PricingTier(
                    input: legacyInput.map { $0 / 1_000_000 },
                    output: legacyOutput.map { $0 / 1_000_000 }
                ))
            } else {
                pricing = nil
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(providerID, forKey: .providerID)
        try container.encode(modelID, forKey: .modelID)
        try container.encode(displayName, forKey: .displayName)
        try container.encodeIfPresent(createdAt, forKey: .createdAt)
        try container.encodeIfPresent(maxInputTokens, forKey: .maxInputTokens)
        try container.encodeIfPresent(maxOutputTokens, forKey: .maxOutputTokens)
        try container.encode(capabilities, forKey: .capabilities)
        try container.encodeIfPresent(sizeLabel, forKey: .sizeLabel)
        try container.encodeIfPresent(quantizationLabel, forKey: .quantizationLabel)
        try container.encodeIfPresent(pricing, forKey: .pricing)
        try container.encode(supportsChatCompletions, forKey: .supportsChatCompletions)
        try container.encodeIfPresent(mode, forKey: .mode)
        // Skip encoding behavior flags when they're at defaults to keep on-disk
        // payloads compact and round-trippable through clients that don't know
        // the field. The all-default case loads as the same value either way.
        if !behaviorFlags.isAllDefault {
            try container.encode(behaviorFlags, forKey: .behaviorFlags)
        }
        // Legacy fields intentionally not written — new format only.
    }
}
