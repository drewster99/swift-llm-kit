import Foundation
import Testing
@testable import SwiftLLMKit

/// Coverage for the 0.0.24 `LLMMessage.Role.developer` case.
///
/// Per-provider behavior:
/// - **OpenAI-compatible** with `BehaviorFlags.supportsDeveloperRole = true`
///   emits `{"role": "developer", "content": ...}` on the wire (OpenAI's
///   o-series / GPT-5 dedicated role). Flag off → downgrade to `"system"`.
/// - **Anthropic / Gemini** have no native developer role. Both providers fold
///   `.developer` into their system field alongside `.system` messages.
@Suite("Developer role per-provider translation")
struct DeveloperRoleTests {

    private static let dummyKey: @Sendable () -> String = { "" }

    // MARK: - OpenAI-compatible: flag on → passthrough

    @Test("OpenAI-compatible with flag on emits role=developer on the wire")
    func openAIWithFlag_emitsDeveloperRole() throws {
        let provider = OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "gpt-5"),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .openAICompatible,
                endpoint: try #require(URL(string: "https://api.openai.com/v1"))
            ),
            readAPIKey: Self.dummyKey,
            behaviorFlags: BehaviorFlags(supportsDeveloperRole: true)
        )
        let messages: [LLMMessage] = [
            .system("you are X"),
            .developer("internal directives"),
            .user("hello")
        ]
        let body = provider.buildRequestBody(messages: messages, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])
        let roles = wire.compactMap { $0["role"] as? String }
        #expect(roles.contains("developer"),
                "with supportsDeveloperRole=true, developer role must appear on the wire; got \(roles)")
        let dev = try #require(wire.first { ($0["role"] as? String) == "developer" })
        #expect(dev["content"] as? String == "internal directives")
    }

    // MARK: - OpenAI-compatible: flag off → downgrade

    @Test("OpenAI-compatible with flag off downgrades developer to system role")
    func openAIWithoutFlag_downgradesToSystem() throws {
        let provider = OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "gpt-4o"),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .openAICompatible,
                endpoint: try #require(URL(string: "https://api.openai.com/v1"))
            ),
            readAPIKey: Self.dummyKey
            // Default flags: supportsDeveloperRole = false.
        )
        let messages: [LLMMessage] = [
            .system("you are X"),
            .developer("internal directives"),
            .user("hello")
        ]
        let body = provider.buildRequestBody(messages: messages, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])
        let roles = wire.compactMap { $0["role"] as? String }
        #expect(!roles.contains("developer"),
                "with supportsDeveloperRole=false, developer role must NOT appear; got \(roles)")
        // The developer content must still be on the wire — folded into system somewhere.
        let allContent = wire.compactMap { $0["content"] as? String }.joined(separator: "\n")
        #expect(allContent.contains("internal directives"),
                "developer message content must survive when downgraded to system")
    }

    // MARK: - Anthropic: developer folds into top-level system

    @Test("Anthropic folds developer messages into top-level system field")
    func anthropic_developerFoldsIntoSystem() throws {
        let provider = try AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "claude-opus-4-7"),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .anthropic,
                endpoint: try #require(URL(string: "https://api.anthropic.com/v1"))
            ),
            readAPIKey: Self.dummyKey
        )
        let messages: [LLMMessage] = [
            .system("system text"),
            .developer("developer text"),
            .user("hello")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        // System block exists with both pieces of content.
        let systemBlocks = try #require(body["system"] as? [[String: Any]])
        let combined = systemBlocks.compactMap { $0["text"] as? String }.joined(separator: "\n")
        #expect(combined.contains("system text"))
        #expect(combined.contains("developer text"))
        // The wire `messages` array must NOT include a developer entry.
        let wire = try #require(body["messages"] as? [[String: Any]])
        let roles = wire.compactMap { $0["role"] as? String }
        #expect(!roles.contains("developer"))
        #expect(!roles.contains("system"))
    }

    // MARK: - Gemini: developer folds into systemInstruction

    @Test("Gemini folds developer messages into systemInstruction")
    func gemini_developerFoldsIntoSystemInstruction() throws {
        let provider = try GeminiProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "gemini-2.5-pro"),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .gemini,
                endpoint: try #require(URL(string: "https://generativelanguage.googleapis.com/v1beta"))
            ),
            readAPIKey: Self.dummyKey
        )
        let messages: [LLMMessage] = [
            .system("system A"),
            .developer("developer B"),
            .user("hi")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let sysInstruction = try #require(body["systemInstruction"] as? [String: Any])
        let parts = try #require(sysInstruction["parts"] as? [[String: Any]])
        let combined = parts.compactMap { $0["text"] as? String }.joined(separator: "\n")
        #expect(combined.contains("system A"))
        #expect(combined.contains("developer B"))
        // The wire `contents` array must contain only the user turn.
        let contents = try #require(body["contents"] as? [[String: Any]])
        let roles = contents.compactMap { $0["role"] as? String }
        #expect(!roles.contains("developer"))
        #expect(roles == ["user"], "only the user turn should remain in contents, got \(roles)")
    }

    // MARK: - BehaviorFlags Codable round-trip

    @Test("BehaviorFlags Codable round-trips supportsDeveloperRole")
    func behaviorFlagsCodableRoundTrip() throws {
        let flags = BehaviorFlags(supportsDeveloperRole: true)
        let data = try JSONEncoder().encode(flags)
        let decoded = try JSONDecoder().decode(BehaviorFlags.self, from: data)
        #expect(decoded.supportsDeveloperRole == true)

        // Default (false) must NOT be serialized — keeps payloads compact.
        let defaultFlags = BehaviorFlags()
        let defaultData = try JSONEncoder().encode(defaultFlags)
        let json = String(data: defaultData, encoding: .utf8) ?? ""
        #expect(!json.contains("supportsDeveloperRole"),
                "default value must be omitted from JSON; got: \(json)")
    }

    // MARK: - LLMMessage Codable round-trip with .developer role

    @Test("LLMMessage Codable round-trips the .developer role")
    func developerRoleCodableRoundTrip() throws {
        let msg = LLMMessage.developer("internal")
        let data = try JSONEncoder().encode(msg)
        let decoded = try JSONDecoder().decode(LLMMessage.self, from: data)
        #expect(decoded.role == .developer)
        #expect(decoded.content.textValue == "internal")
    }
}
