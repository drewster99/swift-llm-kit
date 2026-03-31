import Foundation

/// Selects which LLM API protocol a provider endpoint speaks.
public enum ProviderAPIType: String, Codable, Sendable, CaseIterable, Equatable {
    case anthropic
    case openAICompatible
    case ollama
    case mistral
    case gemini
    case huggingFace
    case lmStudio
    case xAI
    case zAI

    /// The prefix LiteLLM uses for this provider's models (e.g. "mistral/" for `mistral/mistral-large-2512`).
    /// Returns `nil` for local-only providers that have no LiteLLM pricing data.
    public var liteLLMPrefix: String? {
        switch self {
        case .anthropic: return "anthropic"
        case .openAICompatible: return "openai"
        case .ollama: return "ollama"
        case .mistral: return "mistral"
        case .gemini: return "gemini"
        case .huggingFace: return nil
        case .lmStudio: return nil
        case .xAI: return "xai"
        case .zAI: return nil
        }
    }

    /// Human-readable name for display.
    public var displayName: String {
        switch self {
        case .anthropic: return "Anthropic"
        case .openAICompatible: return "OpenAI Compatible"
        case .ollama: return "Ollama"
        case .mistral: return "Mistral"
        case .gemini: return "Google Gemini"
        case .huggingFace: return "Hugging Face"
        case .lmStudio: return "LM Studio"
        case .xAI: return "xAI (Grok)"
        case .zAI: return "z.ai"
        }
    }
    /// The default API endpoint URL for this provider type.
    public var defaultEndpoint: URL { endpointPresets[0].url }

    /// All known endpoint presets for this provider type.
    /// The first entry is the default. Each entry has a human-readable label and URL.
    public var endpointPresets: [EndpointPreset] {
        switch self {
        case .anthropic:
            return [EndpointPreset("Anthropic", "https://api.anthropic.com")]
        case .openAICompatible:
            return [
                EndpointPreset("OpenAI", "https://api.openai.com/v1"),
                EndpointPreset("DeepSeek", "https://api.deepseek.com"),
            ]
        case .ollama:
            return [
                EndpointPreset("Ollama (local)", "http://localhost:11434/api"),
                EndpointPreset("Ollama (cloud)", "https://ollama.com/api"),
            ]
        case .mistral:
            return [EndpointPreset("Mistral", "https://api.mistral.ai/v1")]
        case .gemini:
            return [EndpointPreset("Google Gemini", "https://generativelanguage.googleapis.com/v1beta")]
        case .huggingFace:
            return [EndpointPreset("Hugging Face", "https://router.huggingface.co/v1")]
        case .lmStudio:
            return [EndpointPreset("LM Studio", "http://localhost:1234/v1")]
        case .xAI:
            return [EndpointPreset("xAI (Grok)", "https://api.x.ai/v1")]
        case .zAI:
            return [
                EndpointPreset("z.ai (General)", "https://api.z.ai/api/paas/v4"),
                EndpointPreset("z.ai (Coding)", "https://api.z.ai/api/coding/paas/v4"),
            ]
        }
    }

    /// All endpoint presets across all provider types, for building preset menus.
    public static var allEndpointPresets: [(apiType: ProviderAPIType, preset: EndpointPreset)] {
        allCases.flatMap { type in
            type.endpointPresets.map { (apiType: type, preset: $0) }
        }
    }
}

/// A known endpoint URL for an LLM provider.
public struct EndpointPreset: Sendable {
    /// Human-readable label (e.g. "Ollama (local)").
    public let label: String
    /// The endpoint URL.
    public let url: URL

    public init(_ label: String, _ urlString: String) {
        self.label = label
        guard let url = URL(string: urlString) else {
            preconditionFailure("Invalid endpoint URL literal: \(urlString)")
        }
        self.url = url
    }
}

/// Backward-compatible typealias — use `ProviderAPIType` in new code.
@available(*, deprecated, renamed: "ProviderAPIType")
public typealias ProviderType = ProviderAPIType
