import Foundation
import Testing
@testable import SwiftLLMKit

/// Coverage for `ModelConfiguration.extraJSONOverrides` — caller-supplied
/// top-level JSON keys merged into the outbound provider request body. The
/// merge happens at the end of `buildRequestBody`, so user-supplied values win
/// over any defaults the provider sets.
@Suite("ModelConfiguration.extraJSONOverrides")
struct ExtraJSONOverridesTests {

    private static let dummyKey: @Sendable () -> String = { "test-key" }

    private static func configuration(
        modelID: String = "test-model",
        providerID: String = "test-provider",
        extras: [String: AnyCodable]? = nil
    ) -> ModelConfiguration {
        ModelConfiguration(
            name: "test",
            providerID: providerID,
            modelID: modelID,
            extraJSONOverrides: extras
        )
    }

    private static func provider(_ id: String, type: ProviderAPIType, endpoint: String) -> ModelProvider {
        ModelProvider(
            id: id,
            name: id,
            apiType: type,
            endpoint: URL(string: endpoint)!
        )
    }

    // MARK: - Per-provider merge tests

    @Test func anthropicProvider_mergesExtrasIntoBody() throws {
        let p = AnthropicProvider(
            configuration: Self.configuration(
                providerID: "builtin.anthropic",
                extras: [
                    "thinking": .dictionary([
                        "type": .string("enabled"),
                        "budget_tokens": .int(8000)
                    ]),
                    "metadata": .dictionary(["user_id": .string("agent-hydra")])
                ]
            ),
            provider: Self.provider("builtin.anthropic", type: .anthropic, endpoint: "https://api.anthropic.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let body = try p.buildRequestBody(messages: [LLMMessage(role: .user, text: "hi")], tools: [])
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["type"] as? String == "enabled")
        #expect(thinking["budget_tokens"] as? Int == 8000)
        let metadata = try #require(body["metadata"] as? [String: Any])
        #expect(metadata["user_id"] as? String == "agent-hydra")
    }

    @Test func openAICompatibleProvider_mergesExtrasIntoBody() {
        let p = OpenAICompatibleProvider(
            configuration: Self.configuration(
                providerID: "builtin.openai",
                extras: [
                    "reasoning_effort": .string("high"),
                    "service_tier": .string("priority"),
                    "top_logprobs": .int(5)
                ]
            ),
            provider: Self.provider("builtin.openai", type: .openAICompatible, endpoint: "https://api.openai.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let body = p.buildRequestBody(messages: [LLMMessage(role: .user, text: "hi")], tools: [])
        #expect(body["reasoning_effort"] as? String == "high")
        #expect(body["service_tier"] as? String == "priority")
        #expect(body["top_logprobs"] as? Int == 5)
    }

    @Test func geminiProvider_mergesExtrasIntoBody() throws {
        let p = GeminiProvider(
            configuration: Self.configuration(
                providerID: "builtin.gemini",
                extras: [
                    "safetySettings": .array([
                        .dictionary([
                            "category": .string("HARM_CATEGORY_HARASSMENT"),
                            "threshold": .string("BLOCK_NONE")
                        ])
                    ])
                ]
            ),
            provider: Self.provider("builtin.gemini", type: .gemini, endpoint: "https://generativelanguage.googleapis.com/v1beta"),
            readAPIKey: Self.dummyKey
        )
        let body = try p.buildRequestBody(messages: [LLMMessage(role: .user, text: "hi")], tools: [])
        let safety = try #require(body["safetySettings"] as? [[String: Any]])
        #expect(safety.count == 1)
        #expect(safety[0]["category"] as? String == "HARM_CATEGORY_HARASSMENT")
        #expect(safety[0]["threshold"] as? String == "BLOCK_NONE")
    }

    @Test func ollamaProvider_mergesExtrasIntoBody() {
        let p = OllamaProvider(
            configuration: Self.configuration(
                providerID: "builtin.ollama",
                extras: [
                    "keep_alive": .string("5m"),
                    "format": .string("json")
                ]
            ),
            provider: Self.provider("builtin.ollama", type: .ollama, endpoint: "http://localhost:11434/api"),
            readAPIKey: Self.dummyKey
        )
        let body = p.buildRequestBody(messages: [LLMMessage(role: .user, text: "hi")], tools: [])
        #expect(body["keep_alive"] as? String == "5m")
        #expect(body["format"] as? String == "json")
    }

    // MARK: - Override semantics

    @Test func extras_overrideProviderDefaults() throws {
        // `model` is set by every provider — extras should win.
        let p = AnthropicProvider(
            configuration: Self.configuration(
                modelID: "claude-default",
                providerID: "builtin.anthropic",
                extras: ["model": .string("claude-override")]
            ),
            provider: Self.provider("builtin.anthropic", type: .anthropic, endpoint: "https://api.anthropic.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let body = try p.buildRequestBody(messages: [LLMMessage(role: .user, text: "hi")], tools: [])
        #expect(body["model"] as? String == "claude-override")
    }

    @Test func extras_nilOrEmpty_leaveBodyUnchanged() throws {
        let p = AnthropicProvider(
            configuration: Self.configuration(providerID: "builtin.anthropic", extras: nil),
            provider: Self.provider("builtin.anthropic", type: .anthropic, endpoint: "https://api.anthropic.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let bodyNil = try p.buildRequestBody(messages: [LLMMessage(role: .user, text: "hi")], tools: [])

        let p2 = AnthropicProvider(
            configuration: Self.configuration(providerID: "builtin.anthropic", extras: [:]),
            provider: Self.provider("builtin.anthropic", type: .anthropic, endpoint: "https://api.anthropic.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let bodyEmpty = try p2.buildRequestBody(messages: [LLMMessage(role: .user, text: "hi")], tools: [])

        let aData = try JSONSerialization.data(withJSONObject: bodyNil, options: [.sortedKeys])
        let bData = try JSONSerialization.data(withJSONObject: bodyEmpty, options: [.sortedKeys])
        #expect(aData == bData)
    }

    // MARK: - Codable round-trip

    @Test func modelConfiguration_extras_codableRoundTrip() throws {
        let original = Self.configuration(
            providerID: "builtin.anthropic",
            extras: [
                "thinking": .dictionary([
                    "type": .string("enabled"),
                    "budget_tokens": .int(4000)
                ])
            ]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(original)
        let decoded = try JSONDecoder().decode(ModelConfiguration.self, from: data)
        #expect(decoded.extraJSONOverrides == original.extraJSONOverrides)
    }

    @Test func modelConfiguration_decodesLegacyJSONWithoutExtras() throws {
        let legacyJSON = """
        {
          "id": "11111111-1111-1111-1111-111111111111",
          "name": "legacy",
          "providerID": "builtin.openai",
          "modelID": "gpt-5",
          "temperature": 0.7,
          "maxOutputTokens": 4096,
          "maxContextTokens": 128000,
          "streaming": true,
          "isValid": true
        }
        """
        let decoded = try JSONDecoder().decode(ModelConfiguration.self, from: Data(legacyJSON.utf8))
        #expect(decoded.extraJSONOverrides == nil)
    }
}
