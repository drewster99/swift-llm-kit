import Foundation
import Testing
@testable import SwiftLLMKit

/// Asserts that `OpenAICompatibleProvider.encodeMessage` round-trips
/// `reasoning_content` on the wire iff the `replayReasoningContent` behavior
/// flag is set. The flag exists because some "thinking" models (DeepSeek V4
/// Pro) REQUIRE the field on subsequent calls — HTTP 400
/// "The `reasoning_content` in the thinking mode must be passed back to the
/// API." — while others (deepseek-reasoner) reject it.
@Suite("OpenAICompatibleProvider reasoning_content replay")
struct OpenAIReasoningContentReplayTests {

    private func provider(flags: BehaviorFlags) -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(
                id: "p", name: "p",
                apiType: .openAICompatible,
                endpoint: URL(string: "http://example.invalid/v1")!
            ),
            readAPIKey: { "" },
            behaviorFlags: flags
        )
    }

    @Test("flag on + assistant message with reasoning → reasoning_content present")
    func emitsReasoningWhenFlagOn() {
        let p = provider(flags: BehaviorFlags(replayReasoningContent: true))
        let msg = LLMMessage(
            _role: .assistant,
            _content: .toolCalls([LLMToolCall(id: "x", name: "noop", arguments: "{}")]),
            _reasoning: "I should call the noop tool now."
        )
        let encoded = p.encodeMessage(msg)
        #expect(encoded["reasoning_content"] as? String == "I should call the noop tool now.")
        #expect(encoded["tool_calls"] != nil)
    }

    @Test("flag off + assistant message with reasoning → reasoning_content omitted")
    func skipsReasoningWhenFlagOff() {
        let p = provider(flags: BehaviorFlags())  // default: flag off
        let msg = LLMMessage(
            _role: .assistant,
            _content: .toolCalls([LLMToolCall(id: "x", name: "noop", arguments: "{}")]),
            _reasoning: "I should call the noop tool now."
        )
        let encoded = p.encodeMessage(msg)
        #expect(encoded["reasoning_content"] == nil,
                "reasoning_content must be omitted when the flag is off — deepseek-reasoner et al. reject it on replay")
    }

    @Test("flag on + mixed text+tools assistant message also gets reasoning_content")
    func emitsReasoningOnMixedAssistant() {
        let p = provider(flags: BehaviorFlags(replayReasoningContent: true))
        let msg = LLMMessage(
            _role: .assistant,
            _content: .mixed(text: "Calling noop.", toolCalls: [
                LLMToolCall(id: "x", name: "noop", arguments: "{}")
            ]),
            _reasoning: "Thinking step: invoke noop."
        )
        let encoded = p.encodeMessage(msg)
        #expect(encoded["reasoning_content"] as? String == "Thinking step: invoke noop.")
        #expect(encoded["content"] as? String == "Calling noop.")
    }

    @Test("flag on + user message with reasoning → reasoning_content NOT emitted")
    func skipsReasoningOnNonAssistantRole() {
        // Reasoning is only meaningful for assistant turns. Defensive guard:
        // if a user/system message somehow carries reasoning (e.g. it round-tripped
        // through persistence), don't send it back as if the model produced it.
        let p = provider(flags: BehaviorFlags(replayReasoningContent: true))
        let msg = LLMMessage(_role: .user, _content: .text("hello"), _reasoning: "stray")
        let encoded = p.encodeMessage(msg)
        #expect(encoded["reasoning_content"] == nil)
    }

    @Test("flag on + assistant text without reasoning → reasoning_content absent")
    func absentReasoningStaysAbsent() {
        let p = provider(flags: BehaviorFlags(replayReasoningContent: true))
        let msg = LLMMessage(_role: .assistant, _content: .text("ack"))
        let encoded = p.encodeMessage(msg)
        #expect(encoded["reasoning_content"] == nil)
    }

    @Test("LLMMessage round-trips reasoning through Codable")
    func codableRoundTrip() throws {
        let msg = LLMMessage(
            _role: .assistant,
            _content: .text("done"),
            _reasoning: "step 1; step 2"
        )
        let data = try JSONEncoder().encode(msg)
        let decoded = try JSONDecoder().decode(LLMMessage.self, from: data)
        #expect(decoded.reasoning == "step 1; step 2")
    }

    @Test("Decoding legacy LLMMessage JSON without reasoning gives nil reasoning")
    func legacyDecodeNoReasoning() throws {
        // Old persisted state: just role + content, no reasoning key. Must not
        // throw and must come back with reasoning = nil.
        let legacy = #"{"role":"assistant","content":{"type":"text","text":"hi"}}"#
        let data = legacy.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(LLMMessage.self, from: data)
        #expect(decoded.reasoning == nil)
        #expect(decoded.content.textValue == "hi")
    }

    @Test("BehaviorFlags Codable round-trips replayReasoningContent")
    func behaviorFlagsCodableRoundTrip() throws {
        let flags = BehaviorFlags(replayReasoningContent: true)
        let data = try JSONEncoder().encode(flags)
        let decoded = try JSONDecoder().decode(BehaviorFlags.self, from: data)
        #expect(decoded.replayReasoningContent == true)

        // Default flags should encode to {} (or near-empty) and decode back as default.
        let defaultFlags = BehaviorFlags()
        let defaultData = try JSONEncoder().encode(defaultFlags)
        let defaultDecoded = try JSONDecoder().decode(BehaviorFlags.self, from: defaultData)
        #expect(defaultDecoded.replayReasoningContent == false)
        #expect(defaultDecoded.isAllDefault)
    }
}
