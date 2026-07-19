import Foundation
import Testing
@testable import SwiftLLMKit

/// Reasoning models on some OpenAI-compatible hosts (Mistral's magistral family) return the
/// `content` field as an ARRAY of typed blocks — a `{"type":"text",...}` answer block beside a
/// `{"type":"thinking",...}` chain-of-thought block — rather than a plain string. Before the
/// 2026-07-18 fix, `parseResponse` read `content as? String`, which is nil for the array form,
/// so the answer was dropped entirely (every magistral reply came back empty; the capability
/// prober's tool round-trip check then recorded `toolResultRoundTrip=false` even though the model
/// had returned the exact identifier). These tests pin the array-form handling.
@Suite("OpenAICompatibleProvider array-form content parsing")
struct OpenAIArrayContentParseTests {

    private func provider() -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(
                id: "p", name: "p",
                apiType: .openAICompatible,
                endpoint: URL(string: "http://example.invalid/v1")!
            ),
            readAPIKey: { "" },
            behaviorFlags: BehaviorFlags()
        )
    }

    // MARK: - extractContent (pure)

    @Test("Plain string content → answer text, no inline reasoning")
    func stringContent() {
        let result = OpenAICompatibleProvider.extractContent("hello world")
        #expect(result.text == "hello world")
        #expect(result.reasoning == nil)
    }

    @Test("nil / non-string-non-array content → both nil")
    func nilContent() {
        #expect(OpenAICompatibleProvider.extractContent(nil).text == nil)
        #expect(OpenAICompatibleProvider.extractContent(nil).reasoning == nil)
        #expect(OpenAICompatibleProvider.extractContent(42).text == nil)
    }

    @Test("magistral block array → text block is the answer, thinking block is reasoning")
    func magistralArray() {
        let content: [[String: Any]] = [
            ["closed": true,
             "type": "thinking",
             "thinking": [["text": "The tool returned JNWMPFYV4. Reply with only the identifier.",
                           "type": "text"]]],
            ["text": "JNWMPFYV4", "type": "text"]
        ]
        let result = OpenAICompatibleProvider.extractContent(content)
        #expect(result.text == "JNWMPFYV4")
        #expect(result.reasoning == "The tool returned JNWMPFYV4. Reply with only the identifier.")
    }

    @Test("Multiple text blocks concatenate in order; multiple thinking pieces join with newlines")
    func multipleBlocks() {
        let content: [[String: Any]] = [
            ["type": "thinking", "thinking": [["text": "step one"], ["text": "step two"]]],
            ["type": "text", "text": "Hello, "],
            ["type": "text", "text": "world."]
        ]
        let result = OpenAICompatibleProvider.extractContent(content)
        #expect(result.text == "Hello, world.")
        #expect(result.reasoning == "step one\nstep two")
    }

    @Test("Array with only a thinking block → no answer text, reasoning present")
    func onlyThinking() {
        let content: [[String: Any]] = [
            ["type": "thinking", "thinking": [["text": "just thinking, no answer"]]]
        ]
        let result = OpenAICompatibleProvider.extractContent(content)
        #expect(result.text == nil)
        #expect(result.reasoning == "just thinking, no answer")
    }

    @Test("Unrecognized block carrying top-level text is kept as answer, not dropped")
    func unknownBlockKeepsText() {
        let content: [[String: Any]] = [["type": "future_kind", "text": "keep me"]]
        #expect(OpenAICompatibleProvider.extractContent(content).text == "keep me")
    }

    // MARK: - parseResponse (integration)

    @Test("parseResponse on a magistral round-trip reply extracts the identifier as text")
    func parseResponseMagistralRoundTrip() throws {
        let body = """
        {
          "choices": [{
            "finish_reason": "stop",
            "message": {
              "role": "assistant",
              "content": [
                {"closed": true, "type": "thinking",
                 "thinking": [{"text": "It returned the identifier JNWMPFYV4.", "type": "text"}]},
                {"text": "JNWMPFYV4", "type": "text"}
              ]
            }
          }]
        }
        """
        let data = Data(body.utf8)
        let response = try provider().parseResponse(data: data)
        #expect(response.text == "JNWMPFYV4")
        #expect(response.reasoning == "It returned the identifier JNWMPFYV4.")
        #expect(response.finishReason == "stop")
        // The round-trip probe's check is `text.contains(identifier)` — this is what regressed.
        #expect(response.text?.contains("JNWMPFYV4") == true)
    }

    @Test("parseResponse still handles plain string content (regression guard)")
    func parseResponseStringContent() throws {
        let body = """
        {"choices": [{"finish_reason": "stop",
          "message": {"role": "assistant", "content": "plain answer"}}]}
        """
        let response = try provider().parseResponse(data: Data(body.utf8))
        #expect(response.text == "plain answer")
        #expect(response.reasoning == nil)
    }

    @Test("parseResponse: array with only thinking salvages reasoning as text")
    func parseResponseThinkingOnlySalvage() throws {
        let body = """
        {"choices": [{"finish_reason": "stop",
          "message": {"role": "assistant",
            "content": [{"type": "thinking", "thinking": [{"text": "answer hidden in thought"}]}]}}]}
        """
        let response = try provider().parseResponse(data: Data(body.utf8))
        // No text block, no tool calls → the existing empty-content salvage promotes reasoning.
        #expect(response.text == "answer hidden in thought")
        #expect(response.reasoning == "answer hidden in thought")
    }
}
