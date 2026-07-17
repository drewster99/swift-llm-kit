import Foundation

/// The default sampling parameters a provider publishes for a model — the values it applies when a
/// request omits them. Reference metadata, not constraints: unlike ``ModelInfo/maxTemperature``
/// (a ceiling the endpoint enforces), these are just the vendor's stated defaults, carried so a UI
/// can show or pre-fill them. Every field is optional because providers publish different subsets
/// (Gemini: temperature/topK/topP; Mistral: temperature only; OpenRouter: all six, often null).
public struct SamplingDefaults: Codable, Sendable, Equatable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var frequencyPenalty: Double?
    public var presencePenalty: Double?
    public var repetitionPenalty: Double?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        frequencyPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        repetitionPenalty: Double? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
    }

    /// Whether every field is unset — used to store `nil` rather than an all-empty struct.
    public var isEmpty: Bool {
        temperature == nil && topP == nil && topK == nil
            && frequencyPenalty == nil && presencePenalty == nil && repetitionPenalty == nil
    }
}
