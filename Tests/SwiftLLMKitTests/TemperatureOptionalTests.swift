import Foundation
import Testing
@testable import SwiftLLMKit

/// Coverage for the 0.0.21 temperature change:
/// `ModelConfiguration.temperature` is now `Double?` (was `Double`), and
/// `useDefaultTemperature` is gone as a stored property (a deprecated computed
/// bridge maps it to `temperature == nil`).
///
/// Locks the behavior on:
/// - Legacy JSON migration: `useDefaultTemperature: true` → `temperature = nil`.
/// - Each provider omits `temperature` from the outbound body when nil.
/// - Each provider sends `temperature` when set.
/// - The deprecated bridge maps reads/writes correctly.
@Suite("TokenConfiguration.temperature optionalization (0.0.21)")
struct TemperatureOptionalTests {

    private static let dummyKey: @Sendable () -> String = { "test" }

    private static func provider(_ apiType: ProviderAPIType, endpoint: String) throws -> ModelProvider {
        let url = try #require(URL(string: endpoint))
        return ModelProvider(id: "p", name: "p", apiType: apiType, endpoint: url)
    }

    private static func config(temperature: Double?) -> ModelConfiguration {
        ModelConfiguration(name: "t", providerID: "p", modelID: "m", temperature: temperature)
    }

    // MARK: - Codable backward-compat

    @Test func legacyJSON_useDefaultTemperatureTrue_decodesAsNilTemperature() throws {
        let legacy = """
        {
          "id": "11111111-1111-1111-1111-111111111111",
          "name": "legacy",
          "providerID": "p",
          "modelID": "m",
          "temperature": 0.7,
          "maxOutputTokens": 4096,
          "maxContextTokens": 128000,
          "useDefaultTemperature": true,
          "streaming": true,
          "isValid": true
        }
        """
        let decoded = try JSONDecoder().decode(ModelConfiguration.self, from: Data(legacy.utf8))
        #expect(decoded.temperature == nil)
    }

    @Test func legacyJSON_useDefaultTemperatureFalse_keepsDecodedTemperature() throws {
        let legacy = """
        {
          "id": "11111111-1111-1111-1111-111111111111",
          "name": "legacy",
          "providerID": "p",
          "modelID": "m",
          "temperature": 0.5,
          "maxOutputTokens": 4096,
          "maxContextTokens": 128000,
          "useDefaultTemperature": false,
          "streaming": true,
          "isValid": true
        }
        """
        let decoded = try JSONDecoder().decode(ModelConfiguration.self, from: Data(legacy.utf8))
        #expect(decoded.temperature == 0.5)
    }

    @Test func legacyJSON_useDefaultTemperatureAbsent_keepsDecodedTemperature() throws {
        let legacy = """
        {
          "id": "11111111-1111-1111-1111-111111111111",
          "name": "legacy",
          "providerID": "p",
          "modelID": "m",
          "temperature": 0.3,
          "maxOutputTokens": 4096,
          "maxContextTokens": 128000,
          "streaming": true,
          "isValid": true
        }
        """
        let decoded = try JSONDecoder().decode(ModelConfiguration.self, from: Data(legacy.utf8))
        #expect(decoded.temperature == 0.3)
    }

    @Test func encoded_doesNotEmitUseDefaultTemperature() throws {
        let config = ModelConfiguration(name: "t", providerID: "p", modelID: "m", temperature: nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(data: try encoder.encode(config), encoding: .utf8) ?? ""
        #expect(!json.contains("useDefaultTemperature"))
    }

    @Test func encoded_temperatureNil_omittedFromJSON() throws {
        let config = ModelConfiguration(name: "t", providerID: "p", modelID: "m", temperature: nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(data: try encoder.encode(config), encoding: .utf8) ?? ""
        #expect(!json.contains("\"temperature\""))
    }

    @Test func encoded_temperatureValue_presentInJSON() throws {
        let config = ModelConfiguration(name: "t", providerID: "p", modelID: "m", temperature: 0.42)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(data: try encoder.encode(config), encoding: .utf8) ?? ""
        #expect(json.contains("\"temperature\":0.42"))
    }

    // MARK: - Deprecated bridge (suppress deprecation warning by going through @available indirection)

    @Test func bridge_useDefaultTemperature_readsTemperatureIsNil() {
        var config = ModelConfiguration(name: "t", providerID: "p", modelID: "m", temperature: nil)
        // swiftformat:disable:next
        #expect(useDefaultBridge(config))
        config.temperature = 0.7
        #expect(!useDefaultBridge(config))
    }

    @Test func bridge_useDefaultTemperatureTrue_setsTemperatureNil() {
        var config = ModelConfiguration(name: "t", providerID: "p", modelID: "m", temperature: 0.7)
        setUseDefault(&config, true)
        #expect(config.temperature == nil)
    }

    // MARK: - Per-provider omission

    @Test func anthropicProvider_temperatureNil_bodyHasNoTemperature() throws {
        let p = try AnthropicProvider(
            configuration: Self.config(temperature: nil),
            provider: Self.provider(.anthropic, endpoint: "https://api.anthropic.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let body = try p.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["temperature"] == nil)
    }

    @Test func anthropicProvider_temperatureSet_bodyHasTemperature() throws {
        let p = try AnthropicProvider(
            configuration: Self.config(temperature: 0.5),
            provider: Self.provider(.anthropic, endpoint: "https://api.anthropic.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let body = try p.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["temperature"] as? Double == 0.5)
    }

    @Test func openAIProvider_temperatureNil_bodyHasNoTemperature() throws {
        let p = OpenAICompatibleProvider(
            configuration: Self.config(temperature: nil),
            provider: try Self.provider(.openAICompatible, endpoint: "https://api.openai.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let body = p.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["temperature"] == nil)
    }

    @Test func openAIProvider_temperatureSet_bodyHasTemperature() throws {
        let p = OpenAICompatibleProvider(
            configuration: Self.config(temperature: 0.3),
            provider: try Self.provider(.openAICompatible, endpoint: "https://api.openai.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let body = p.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["temperature"] as? Double == 0.3)
    }

    @Test func geminiProvider_temperatureNil_generationConfigHasNoTemperature() throws {
        let p = try GeminiProvider(
            configuration: Self.config(temperature: nil),
            provider: Self.provider(.gemini, endpoint: "https://generativelanguage.googleapis.com/v1beta"),
            readAPIKey: Self.dummyKey
        )
        let body = try p.buildRequestBody(messages: [.user("hi")], tools: [])
        let gc = try #require(body["generationConfig"] as? [String: Any])
        #expect(gc["temperature"] == nil)
        // maxOutputTokens still present
        #expect(gc["maxOutputTokens"] != nil)
    }

    @Test func geminiProvider_temperatureSet_generationConfigHasTemperature() throws {
        let p = try GeminiProvider(
            configuration: Self.config(temperature: 0.8),
            provider: Self.provider(.gemini, endpoint: "https://generativelanguage.googleapis.com/v1beta"),
            readAPIKey: Self.dummyKey
        )
        let body = try p.buildRequestBody(messages: [.user("hi")], tools: [])
        let gc = try #require(body["generationConfig"] as? [String: Any])
        #expect(gc["temperature"] as? Double == 0.8)
    }

    @Test func ollamaProvider_temperatureNil_optionsHasNoTemperature() {
        let p = OllamaProvider(
            configuration: Self.config(temperature: nil),
            provider: (try? Self.provider(.ollama, endpoint: "http://localhost:11434/api"))!,
            readAPIKey: Self.dummyKey
        )
        let body = p.buildRequestBody(messages: [.user("hi")], tools: [])
        let options = body["options"] as? [String: Any]
        #expect(options?["temperature"] == nil)
        #expect(options?["num_predict"] != nil)
    }

    @Test func ollamaProvider_temperatureSet_optionsHasTemperature() {
        let p = OllamaProvider(
            configuration: Self.config(temperature: 0.9),
            provider: (try? Self.provider(.ollama, endpoint: "http://localhost:11434/api"))!,
            readAPIKey: Self.dummyKey
        )
        let body = p.buildRequestBody(messages: [.user("hi")], tools: [])
        let options = body["options"] as? [String: Any]
        #expect(options?["temperature"] as? Double == 0.9)
    }

    // MARK: - Anthropic thinking override

    @Test func anthropicProvider_thinkingEnabled_alwaysSendsTemperature1() throws {
        // With thinking enabled, Anthropic requires temperature = 1.
        // Even if our config says nil, we send 1.0 because thinking demands it.
        let thinkingConfig = ModelConfiguration(
            name: "t",
            providerID: "p",
            modelID: "m",
            temperature: nil,
            thinkingBudget: 4096
        )
        let p = try AnthropicProvider(
            configuration: thinkingConfig,
            provider: Self.provider(.anthropic, endpoint: "https://api.anthropic.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let body = try p.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["temperature"] as? Double == 1.0)
    }
}

// MARK: - Bridge helpers (isolate the deprecation warning to these wrappers)

@available(*, deprecated)
private func useDefaultBridge(_ c: ModelConfiguration) -> Bool {
    c.useDefaultTemperature
}

@available(*, deprecated)
private func setUseDefault(_ c: inout ModelConfiguration, _ value: Bool) {
    c.useDefaultTemperature = value
}
