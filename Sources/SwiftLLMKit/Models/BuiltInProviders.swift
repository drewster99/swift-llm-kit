import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "BuiltInProviders")

/// A well-known LLM provider preset bundled with SwiftLLMKit.
///
/// Built-in providers have stable IDs (e.g. `builtin.anthropic`) so that "Anthropic"
/// always means Anthropic regardless of user actions. Their `displayName`, `apiType`,
/// and `endpoint` are fixed; only the API key is user-editable.
public struct BuiltInProviderPreset: Codable, Sendable, Equatable {
    /// Stable identifier, e.g. `builtin.anthropic`.
    public let id: String
    /// Human-readable name shown in the providers list (e.g. "Anthropic", "Grok").
    /// May differ from the underlying `apiType.displayName` (e.g. `builtin.openai`
    /// uses the `openAICompatible` API type but displays as "OpenAI").
    public let displayName: String
    /// The API protocol this provider speaks.
    public let apiType: ProviderAPIType
    /// The official endpoint URL for this provider.
    public let endpoint: URL
    /// Whether this provider should be shown in the default ("popular") view.
    public let popular: Bool
    /// The `litellm_provider` value this provider's models are catalogued under, or `nil` when
    /// LiteLLM has no entries for it. See ``ModelProvider/liteLLMProviderName``.
    public let liteLLMProviderName: String?
}

/// Loader and registry for the bundled built-in provider list.
///
/// The list lives in `Resources/built_in_providers.json` and is loaded once on first
/// access. Use `BuiltInProviders.all` to enumerate every preset, `BuiltInProviders.popular`
/// for the default-shown subset, or `BuiltInProviders.preset(id:)` to look one up by ID.
public enum BuiltInProviders {
    /// Stable IDs for each built-in provider preset. These must match the `id` field
    /// for the corresponding entry in `Resources/built_in_providers.json`.
    public enum ID {
        public static let anthropic = "builtin.anthropic"
        public static let gemini = "builtin.gemini"
        public static let openai = "builtin.openai"
        public static let openRouter = "builtin.openrouter"
        public static let xAI = "builtin.xai"
        public static let alibabaCloud = "builtin.alibabacloud"
        public static let huggingFace = "builtin.huggingface"
        public static let lmStudio = "builtin.lmstudio"
        public static let metaLlama = "builtin.metallama"
        public static let mistral = "builtin.mistral"
        public static let ollama = "builtin.ollama"
        public static let ollamaCloud = "builtin.ollama-cloud"
        public static let zAI = "builtin.zai"
    }

    /// All bundled built-in provider presets, in the order declared in the JSON file.
    public static let all: [BuiltInProviderPreset] = loadFromBundle()

    /// Subset of `all` flagged as `popular: true` in the JSON.
    public static var popular: [BuiltInProviderPreset] {
        all.filter(\.popular)
    }

    /// Set of all built-in provider IDs, for fast `isBuiltIn(id:)` checks.
    public static let allIDs: Set<String> = Set(all.map(\.id))

    /// Returns true if the given provider ID belongs to a built-in preset.
    public static func isBuiltIn(id: String) -> Bool {
        allIDs.contains(id)
    }

    /// Returns the preset for a given ID, or `nil` if not a known built-in.
    public static func preset(id: String) -> BuiltInProviderPreset? {
        all.first { $0.id == id }
    }

    /// Returns a preset whose `apiType` and `endpoint` both match the given pair.
    /// Used by migration to find a built-in slot for an existing user provider.
    public static func preset(matchingAPIType apiType: ProviderAPIType, endpoint: URL) -> BuiltInProviderPreset? {
        all.first { $0.apiType == apiType && $0.endpoint == endpoint }
    }

    // MARK: - Loading

    private struct BundleFile: Decodable {
        let version: Int
        let providers: [BuiltInProviderPreset]
    }

    private static func loadFromBundle() -> [BuiltInProviderPreset] {
        guard let url = Bundle.module.url(forResource: "built_in_providers", withExtension: "json") else {
            logger.error("built_in_providers.json missing from SwiftLLMKit resource bundle")
            return []
        }
        do {
            let data = try Data(contentsOf: url)
            let file = try JSONDecoder().decode(BundleFile.self, from: data)
            return file.providers
        } catch {
            logger.error("Failed to decode built_in_providers.json: \(error.localizedDescription, privacy: .public)")
            return []
        }
    }
}

// MARK: - Convenience constructor

public extension ModelProvider {
    /// Creates a `ModelProvider` from a built-in preset.
    init(builtIn preset: BuiltInProviderPreset) {
        self.init(
            id: preset.id,
            name: preset.displayName,
            apiType: preset.apiType,
            endpoint: preset.endpoint,
            liteLLMProviderName: preset.liteLLMProviderName
        )
    }
}
