import Foundation
import Testing
@testable import SwiftLLMKit

/// Coverage for `BehaviorFlags.supportsTrailingSystemMessage`.
///
/// A trailing `{role: system}` steering turn — the LAST message, immediately after a `.user` — is
/// left in place (emitted as role `system` at the tail) on models known to read one, instead of
/// being hoisted like every other system message. The flag is established by the probe's
/// `ModelProfile.trailingSystemMessage` finding; these tests verify the three hoisting providers
/// honor the resolved flag both ways. Gemini is out of scope: its `contents` shape has only
/// user/model roles plus a separate systemInstruction, so a trailing system turn is not expressible.
@Suite("Trailing system message per-provider")
struct TrailingSystemMessageTests {

    private static let dummyKey: @Sendable () -> String = { "" }

    /// A base system prompt, a user turn, then a trailing system nudge — the qualifying shape.
    private static let trailing: [LLMMessage] = [
        .system("BASE PROMPT"),
        .user("do the thing"),
        .system("TRAILING NUDGE")
    ]

    // MARK: - Positional helper

    @Test("trailingSystemTurnIndex fires only for a trailing system after a user, when allowed")
    func helperRule() {
        #expect(LLMMessage.trailingSystemTurnIndex(in: Self.trailing, allowed: true) == 2)
        #expect(LLMMessage.trailingSystemTurnIndex(in: Self.trailing, allowed: false) == nil)
        // Last message is not system.
        #expect(LLMMessage.trailingSystemTurnIndex(in: [.user("a"), .user("b")], allowed: true) == nil)
        // Trailing system is not preceded by a user.
        #expect(LLMMessage.trailingSystemTurnIndex(in: [.system("a"), .system("b")], allowed: true) == nil)
        // Fewer than two messages.
        #expect(LLMMessage.trailingSystemTurnIndex(in: [.system("only")], allowed: true) == nil)
    }

    // MARK: - Anthropic

    @Test("Anthropic keeps a trailing system turn in messages when the flag is on")
    func anthropic_flagOn_keepsTrailing() throws {
        let provider = try AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "claude-opus-4-8"),
            provider: ModelProvider(id: "p", name: "p", apiType: .anthropic,
                                    endpoint: try #require(URL(string: "https://api.anthropic.com/v1"))),
            readAPIKey: Self.dummyKey,
            behaviorFlags: BehaviorFlags(supportsTrailingSystemMessage: true)
        )
        let body = try provider.buildRequestBody(messages: Self.trailing, tools: [])
        // Top-level system holds the base prompt only — the trailing nudge is NOT hoisted.
        let systemBlocks = try #require(body["system"] as? [[String: Any]])
        let systemText = systemBlocks.compactMap { $0["text"] as? String }.joined(separator: "\n")
        #expect(systemText.contains("BASE PROMPT"))
        #expect(!systemText.contains("TRAILING NUDGE"))
        // The wire messages END with a role=system turn carrying the nudge.
        let wire = try #require(body["messages"] as? [[String: Any]])
        let last = try #require(wire.last)
        #expect(last["role"] as? String == "system")
        #expect(last["content"] as? String == "TRAILING NUDGE")
    }

    @Test("Anthropic hoists the trailing system turn when the flag is off")
    func anthropic_flagOff_hoists() throws {
        let provider = try AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "claude-opus-4-8"),
            provider: ModelProvider(id: "p", name: "p", apiType: .anthropic,
                                    endpoint: try #require(URL(string: "https://api.anthropic.com/v1"))),
            readAPIKey: Self.dummyKey
        )
        let body = try provider.buildRequestBody(messages: Self.trailing, tools: [])
        let systemBlocks = try #require(body["system"] as? [[String: Any]])
        let systemText = systemBlocks.compactMap { $0["text"] as? String }.joined(separator: "\n")
        #expect(systemText.contains("BASE PROMPT"))
        #expect(systemText.contains("TRAILING NUDGE"), "flag off must hoist the trailing nudge into system")
        let wire = try #require(body["messages"] as? [[String: Any]])
        #expect(!wire.compactMap { $0["role"] as? String }.contains("system"),
                "no system role should remain in the messages array")
    }

    // MARK: - OpenAI-compatible

    @Test("OpenAI-compatible keeps a trailing system turn last when the flag is on")
    func openAI_flagOn_keepsTrailing() throws {
        let provider = OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "some-model"),
            provider: ModelProvider(id: "p", name: "p", apiType: .openAICompatible,
                                    endpoint: try #require(URL(string: "https://api.example.com/v1"))),
            readAPIKey: Self.dummyKey,
            behaviorFlags: BehaviorFlags(supportsTrailingSystemMessage: true)
        )
        let body = provider.buildRequestBody(messages: Self.trailing, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])
        let last = try #require(wire.last)
        #expect(last["role"] as? String == "system")
        #expect(last["content"] as? String == "TRAILING NUDGE")
        // The base prompt is a distinct leading system message, not merged with the trailing nudge.
        let first = try #require(wire.first)
        #expect(first["role"] as? String == "system")
        #expect((first["content"] as? String)?.contains("BASE PROMPT") == true)
        #expect((first["content"] as? String)?.contains("TRAILING NUDGE") == false)
    }

    @Test("OpenAI-compatible consolidates the trailing system turn to the front when the flag is off")
    func openAI_flagOff_hoists() throws {
        let provider = OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "some-model"),
            provider: ModelProvider(id: "p", name: "p", apiType: .openAICompatible,
                                    endpoint: try #require(URL(string: "https://api.example.com/v1"))),
            readAPIKey: Self.dummyKey
        )
        let body = provider.buildRequestBody(messages: Self.trailing, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])
        let systemEntries = wire.filter { ($0["role"] as? String) == "system" }
        #expect(systemEntries.count == 1, "flag off must consolidate to one leading system message")
        let content = systemEntries.first?["content"] as? String ?? ""
        #expect(content.contains("BASE PROMPT"))
        #expect(content.contains("TRAILING NUDGE"))
        #expect(wire.last?["role"] as? String == "user", "with the flag off the last turn is the user turn")
    }

    // MARK: - Ollama

    @Test("Ollama keeps a trailing system turn last when the flag is on")
    func ollama_flagOn_keepsTrailing() throws {
        let provider = OllamaProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "llama3"),
            provider: ModelProvider(id: "p", name: "p", apiType: .ollama,
                                    endpoint: try #require(URL(string: "http://localhost:11434/api"))),
            readAPIKey: Self.dummyKey,
            behaviorFlags: BehaviorFlags(supportsTrailingSystemMessage: true)
        )
        let encoded = provider.buildEncodedMessagesForTesting(messages: Self.trailing, tools: [])
        #expect(encoded.last?["role"] as? String == "system")
        #expect(encoded.last?["content"] as? String == "TRAILING NUDGE")
        // Two system entries kept distinct: the leading base plus the trailing nudge.
        let systemEntries = encoded.filter { ($0["role"] as? String) == "system" }
        #expect(systemEntries.count == 2)
    }

    @Test("Ollama consolidates to one leading system message when the flag is off")
    func ollama_flagOff_hoists() throws {
        let provider = OllamaProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "llama3"),
            provider: ModelProvider(id: "p", name: "p", apiType: .ollama,
                                    endpoint: try #require(URL(string: "http://localhost:11434/api"))),
            readAPIKey: Self.dummyKey
        )
        let encoded = provider.buildEncodedMessagesForTesting(messages: Self.trailing, tools: [])
        let systemEntries = encoded.filter { ($0["role"] as? String) == "system" }
        #expect(systemEntries.count == 1)
        let content = systemEntries.first?["content"] as? String ?? ""
        #expect(content.contains("BASE PROMPT"))
        #expect(content.contains("TRAILING NUDGE"))
        #expect(encoded.last?["role"] as? String == "user")
    }

    // MARK: - Probe finding projects into the flag

    @Test("A probed trailing-system finding projects into the empirical behavior flag")
    func probeProjectsIntoFlag() {
        var profile = ModelProfile(providerID: "p", modelID: "m")
        profile.chat = .established(true)
        profile.trailingSystemMessage = .established(true, "echoed the code")
        let facts = profile.asEmpiricalFacts(includeAccountScoped: false)
        #expect(facts.behaviorFlags.supportsTrailingSystemMessage == true)

        var rejected = ModelProfile(providerID: "p", modelID: "m")
        rejected.chat = .established(true)
        rejected.trailingSystemMessage = .established(false, "role 'system' is not supported")
        let rejectedFacts = rejected.asEmpiricalFacts(includeAccountScoped: false)
        #expect(rejectedFacts.behaviorFlags.supportsTrailingSystemMessage == false)
    }

    // MARK: - Realistic probe shape (base system + trailing turn)

    @Test("makeTrailingSystemTurnTest builds a base system, a user turn, then a trailing system nonce")
    func probeTestShape() {
        let test = ModelProber.makeTrailingSystemTurnTest()
        #expect(test.messages.count == 3)
        #expect(test.messages[0].role == .system)   // base system prompt
        #expect(test.messages[1].role == .user)
        #expect(test.messages[2].role == .system)   // the trailing turn
        // The nonce lives ONLY in the trailing turn, so an echo proves that turn — not the base
        // system — was read.
        #expect(test.messages[2].content.textValue?.contains(test.nonce) == true)
        #expect(test.messages[0].content.textValue?.contains(test.nonce) == false)
    }

    @Test("The probe's messages, sent through a flag-on provider, land the nonce in a trailing system turn")
    func probeRoutesThroughConsumer() throws {
        let test = ModelProber.makeTrailingSystemTurnTest()
        let provider = OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "some-model"),
            provider: ModelProvider(id: "p", name: "p", apiType: .openAICompatible,
                                    endpoint: try #require(URL(string: "https://api.example.com/v1"))),
            readAPIKey: Self.dummyKey,
            behaviorFlags: BehaviorFlags(supportsTrailingSystemMessage: true)
        )
        let body = provider.buildRequestBody(messages: test.messages, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])
        let last = try #require(wire.last)
        #expect(last["role"] as? String == "system")
        #expect((last["content"] as? String)?.contains(test.nonce) == true)
        // The base system stays a distinct leading system message, not merged with the trailing nonce.
        #expect((wire.first?["content"] as? String)?.contains(test.nonce) == false)
    }
}
