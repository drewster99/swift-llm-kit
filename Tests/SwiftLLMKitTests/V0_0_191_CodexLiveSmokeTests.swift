import Foundation
import Testing
@testable import SwiftLLMKit

/// End-to-end checks against the real Codex endpoint, using the real credential.
///
/// Disabled unless `CODEX_LIVE_SMOKE=1`, because they need a signed-in `codex` CLI, reach the
/// network, and spend the user's ChatGPT allowance. Everything else in this suite runs off recorded
/// fixtures precisely so the normal suite needs none of that.
///
/// They exist because fixtures can only prove the parser handles what was recorded. The defect this
/// work actually shipped — argument deltas keyed by `item_id` on one side and `output_index` on the
/// other — passed every fixture test while being broken on the wire, because the fixture was
/// sparser than the real event. One live call would have caught it immediately.
///
///     CODEX_LIVE_SMOKE=1 swift test --filter CodexLiveSmoke
@Suite("Codex live smoke", .enabled(if: ProcessInfo.processInfo.environment["CODEX_LIVE_SMOKE"] == "1"))
struct CodexLiveSmokeTests {

    static let model = "gpt-5.5"

    static func makeProvider() -> CodexResponsesProvider {
        let provider = ModelProvider(
            id: "builtin.codex-chatgpt",
            name: "ChatGPT Subscription (Codex)",
            apiType: .codexChatGPT,
            endpoint: URL(fileURLWithPath: "/").appending(path: "x"))
        // The real endpoint, built the same way the app's built-in preset supplies it.
        let real = ModelProvider(
            id: provider.id, name: provider.name, apiType: .codexChatGPT,
            endpoint: URL(string: "https://chatgpt.com/backend-api/codex") ?? provider.endpoint)
        let config = ModelConfiguration(
            name: "smoke", providerID: real.id, modelID: model, maxOutputTokens: 512)
        // The flag the catalog states for every Codex model (`decodeCodexFacts`): the endpoint
        // rejects `temperature`, and this configuration carries the type's default 0.7. Built
        // without the catalog, the provider must be told what the catalog would have told it.
        var flags = BehaviorFlags()
        flags.mustNeverSendTemperatureParam = true
        return CodexResponsesProvider(configuration: config, provider: real, behaviorFlags: flags)
    }

    @Test("An image in a user turn is actually seen through the real provider")
    func liveImageTurn() async throws {
        // A solid red square; the answer must name both facts, the same grade the capability
        // probe applies. Before 2026-09-19 the serializer dropped the image and the model replied
        // "I don't see an image attached".
        let png = ProbeFixtures.makeShapePNG(shape: .square, red: 255, green: 0, blue: 0)
        let response = try await Self.makeProvider().send(
            messages: [.user("What shape is in this image, and what colour is it? Answer briefly.",
                             images: [LLMImageContent(data: png, mimeType: "image/png", detail: .low)])],
            tools: [])
        let text = (response.text ?? "").lowercased()
        #expect(text.contains("red"), "expected the colour, got: \(text)")
        #expect(text.contains("square"), "expected the shape, got: \(text)")
    }

    @Test("A plain turn returns text and usage through the real provider")
    func liveTextTurn() async throws {
        let response = try await Self.makeProvider().send(
            messages: [
                LLMMessage(role: .system, content: .text("Answer with one word.")),
                LLMMessage(role: .user, content: .text("Reply with exactly: OK"))
            ],
            tools: [])
        let text = try #require(response.text)
        #expect(text.localizedCaseInsensitiveContains("ok"), "got \(text)")
        let usage = try #require(response.usage, "usage must survive the real stream, not just fixtures")
        #expect(usage.inputTokens > 0)
        #expect(usage.outputTokens > 0)
        #expect(response.finishReason == "completed")
    }

    @Test("A tool call comes back with its arguments attached")
    func liveToolCall() async throws {
        let weather = LLMToolDefinition(
            name: "get_weather",
            description: "Get the current weather for a city",
            parameters: [
                "type": .string("object"),
                "properties": .dictionary(["city": .dictionary(["type": .string("string")])]),
                "required": .array([.string("city")])
            ])
        let response = try await Self.makeProvider().send(
            messages: [LLMMessage(role: .user, content: .text("What is the weather in Paris? Use the tool."))],
            tools: [weather])

        let call = try #require(response.toolCalls.first, "the model should have called the tool")
        #expect(call.name == "get_weather")
        #expect(!call.id.isEmpty, "call_id is what a later function_call_output must quote")
        // The regression that fixtures missed: arguments arrive as deltas keyed by item id, and a
        // key mismatch leaves this an empty object while everything else still looks healthy.
        #expect(call.arguments != "{}", "arguments did not join their call — check the item keying")
        let parsed = try #require(
            try JSONSerialization.jsonObject(with: Data(call.arguments.utf8)) as? [String: Any])
        #expect((parsed["city"] as? String)?.localizedCaseInsensitiveContains("paris") == true,
                "got \(call.arguments)")
    }

    @Test("A tool RESULT is accepted back, so a multi-turn agent loop can actually run")
    func liveToolResultRoundTrip() async throws {
        // The half a single call cannot prove: that our `function_call_output` encoding is one the
        // endpoint accepts. An agent loop is worthless if the second turn 400s.
        let weather = LLMToolDefinition(
            name: "get_weather", description: "Get the current weather for a city",
            parameters: [
                "type": .string("object"),
                "properties": .dictionary(["city": .dictionary(["type": .string("string")])]),
                "required": .array([.string("city")])
            ])
        let provider = Self.makeProvider()
        let first = try await provider.send(
            messages: [LLMMessage(role: .user, content: .text("Weather in Paris? Use the tool."))],
            tools: [weather])
        let call = try #require(first.toolCalls.first)

        let second = try await provider.send(
            messages: [
                LLMMessage(role: .user, content: .text("Weather in Paris? Use the tool.")),
                LLMMessage(role: .assistant, content: .toolCalls([call])),
                LLMMessage(role: .tool, content: .toolResult(
                    toolCallID: call.id, content: #"{"tempC": 17, "conditions": "cloudy"}"#))
            ],
            tools: [weather])
        let text = try #require(second.text, "the model should answer once it has the result")
        #expect(text.contains("17") || text.lowercased().contains("cloud"), "got \(text)")
    }
}
