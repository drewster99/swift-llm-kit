import Foundation
import Testing
@testable import SwiftLLMKit

/// Preserved-thinking prefix checks must not fail an agent loop.
///
/// On Claude Fable 5.1 / Opus 5.5 / Sonnet 5.5 / Mythos 5.1, an edited earlier message, a changed
/// `system`, or any change to `tools` invalidates later thinking blocks — a 400 by default for
/// accounts created on or after 2026-08-31. Agent loops do all three, so every thinking request to
/// Anthropic's API asks the API to drop an invalidated block instead (`drop_block`), with the beta
/// header that enables the field. Other Anthropic-format endpoints get the plain request.
@Suite("Thinking block binding")
struct ThinkingBlockBindingTests {

    private static let dummyKey: @Sendable () -> String = { "" }

    private func anthropic(endpoint: String, thinkingBudget: Int?, adaptive: Bool) throws -> AnthropicProvider {
        try AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m", thinkingBudget: thinkingBudget),
            provider: ModelProvider(id: "p", name: "p", apiType: .anthropic,
                                    endpoint: #require(URL(string: endpoint))),
            readAPIKey: Self.dummyKey,
            behaviorFlags: BehaviorFlags(requiresAdaptiveThinking: adaptive)
        )
    }

    @Test("adaptive thinking to Anthropic carries drop_block and the beta header")
    func adaptiveToAnthropic() throws {
        let provider = try anthropic(endpoint: "https://api.anthropic.com/v1", thinkingBudget: 4096, adaptive: true)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["type"] as? String == "adaptive")
        let binding = try #require(thinking["block_binding"] as? [String: Any])
        #expect(binding["prefix_mismatch_behavior"] as? String == "drop_block")
        #expect(AnthropicProvider.betaHeader(forBody: body, endpoint: try #require(URL(string: "https://api.anthropic.com/v1")))
                == "thinking-binding-controls-2026-08-01")
    }

    @Test("manual (budget) thinking to Anthropic carries it too")
    func manualToAnthropic() throws {
        let provider = try anthropic(endpoint: "https://api.anthropic.com/v1", thinkingBudget: 4096, adaptive: false)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["type"] as? String == "enabled")
        #expect(thinking["block_binding"] != nil)
    }

    @Test("a third-party Anthropic-compatible endpoint gets the plain thinking object and no beta")
    func thirdPartyEndpoint() throws {
        let endpoint = "https://anthropic-proxy.example.com/v1"
        let provider = try anthropic(endpoint: endpoint, thinkingBudget: 4096, adaptive: true)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["block_binding"] == nil)
        #expect(AnthropicProvider.betaHeader(forBody: body, endpoint: try #require(URL(string: endpoint))) == nil)
    }

    @Test("no thinking means no binding and no beta header")
    func thinkingOff() throws {
        let endpoint = "https://api.anthropic.com/v1"
        let provider = try anthropic(endpoint: endpoint, thinkingBudget: 0, adaptive: false)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["thinking"] == nil)
        #expect(AnthropicProvider.betaHeader(forBody: body, endpoint: try #require(URL(string: endpoint))) == nil)
    }
}
