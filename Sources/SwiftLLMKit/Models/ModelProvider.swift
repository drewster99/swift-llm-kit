import Foundation

/// A configured LLM provider — connection details for a specific API endpoint.
/// API keys are stored separately in Keychain, not in this struct.
public struct ModelProvider: Codable, Identifiable, Sendable, Equatable {
    /// Stable identifier, e.g. "anthropic-1", "ollama-local".
    public let id: String
    /// User-facing name, e.g. "Anthropic", "My Ollama Server".
    public var name: String
    /// Which API protocol this endpoint speaks.
    public var apiType: ProviderAPIType
    /// Base URL for the provider's API.
    public var endpoint: URL
    /// Which LiteLLM provider this endpoint's models are catalogued under — matched against the
    /// `litellm_provider` FIELD of each entry in `model_prices_and_context_window.json`, NOT
    /// against the entry's key prefix.
    ///
    /// The field is authoritative and the key prefix is not: LiteLLM keys carry image sizes,
    /// quality tiers, and AWS inference-profile segments in the first position
    /// (`1024-x-1024/dall-e-2` is `openai`; `global.anthropic.claude-fable-5` is
    /// `bedrock_converse`), and LiteLLM's own resolver treats the key as a search term that the
    /// field then vetoes. Matching on the field also means provider-native models keyed bare
    /// (all 23 Anthropic entries, e.g. `claude-fable-5`) resolve with no special casing.
    ///
    /// `nil` means this provider's models are not catalogued by LiteLLM at all (e.g. LM Studio
    /// and Hugging Face have zero entries), so no enrichment is attempted.
    public var liteLLMProviderName: String?

    public init(id: String, name: String, apiType: ProviderAPIType, endpoint: URL, liteLLMProviderName: String? = nil) {
        self.id = id
        self.name = name
        self.apiType = apiType
        self.endpoint = endpoint
        self.liteLLMProviderName = liteLLMProviderName
    }
}
