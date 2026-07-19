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
    case metaModel
    case alibabaCloud
    case openRouter

    // MARK: - Forgiving Codable
    //
    // The synthesized `String` decode THROWS on any raw value the running binary doesn't recognize.
    // Because providers and configurations decode as ARRAYS, one unknown value fails the WHOLE array
    // and drops every entry. That is exactly the 2026-07-19 "all providers vanished" incident: a
    // persisted `providers.json` still held apiType `"metaLlama"` after that case was renamed to
    // `metaModel`; `loadProviders()` threw, so `providersLoadedOK` went false, built-in seeding was
    // skipped, and the entire provider list — built-in and custom alike — disappeared.
    //
    // Decode forgivingly instead: an unrecognized or legacy raw value falls back to
    // `.openAICompatible`, which is how every non-native provider is already treated internally, so
    // a single stale/newer value can never nuke the list again. Encoding stays the exact raw value.
    public init(from decoder: any Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = ProviderAPIType(rawValue: raw) ?? .openAICompatible
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
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
        case .metaModel: return "Meta Model API"
        case .alibabaCloud: return "Alibaba Cloud"
        case .openRouter: return "OpenRouter"
        }
    }
    /// The valid temperature range for this provider's API.
    /// Models may further constrain this (see `ModelInfo.temperatureRange`).
    public var temperatureRange: ClosedRange<Double> {
        switch self {
        case .anthropic:
            return 0...1
        case .gemini:
            return 0...2
        case .openAICompatible, .xAI, .zAI, .mistral, .huggingFace, .metaModel, .alibabaCloud, .openRouter:
            return 0...2
        case .ollama, .lmStudio:
            // Ollama/LM Studio accept wide ranges; models may clip internally
            return 0...5
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
        case .metaModel:
            return [EndpointPreset("Meta Model API", "https://api.meta.ai/v1")]
        case .alibabaCloud:
            return [
                EndpointPreset("Alibaba Cloud (US)", "https://dashscope-us.aliyuncs.com/compatible-mode/v1"),
                EndpointPreset("Alibaba Cloud (Singapore)", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
                EndpointPreset("Alibaba Cloud (Beijing)", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            ]
        case .openRouter:
            return [EndpointPreset("OpenRouter", "https://openrouter.ai/api/v1")]
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

// MARK: - URL Helpers

extension URL {
    /// Returns the URL with `/v1` appended if not already present.
    /// Anthropic endpoints accept both `https://api.anthropic.com` and `.../v1`.
    func ensureAnthropicV1() -> URL {
        path.hasSuffix("/v1") ? self : appendingPathComponent("v1")
    }

    /// Returns the URL with `/v1` stripped, if present.
    /// Used when the v1 prefix is added per-request (e.g. model listing).
    func strippingAnthropicV1() -> URL {
        path.hasSuffix("/v1") ? deletingLastPathComponent() : self
    }
}

/// Backward-compatible typealias — use `ProviderAPIType` in new code.
@available(*, deprecated, renamed: "ProviderAPIType")
public typealias ProviderType = ProviderAPIType
