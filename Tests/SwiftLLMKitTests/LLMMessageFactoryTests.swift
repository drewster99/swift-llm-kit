import Foundation
import Testing
@testable import SwiftLLMKit

/// Coverage for the 0.0.24 static factories on `LLMMessage`. The factories
/// are the misuse-resistant path: in particular `.assistant(from: response)`
/// is the only construction that automatically preserves
/// `LLMResponse.reasoning` AND `LLMResponse.continuation` without the caller
/// having to remember either. Tests lock the shapes and demonstrate that the
/// synthetic factories (deprecated) do NOT carry continuation.
@Suite("LLMMessage static factories")
struct LLMMessageFactoryTests {

    // MARK: - Plain role factories

    @Test func userFactory_setsRoleAndTextContent() {
        let msg = LLMMessage.user("hello")
        #expect(msg.role == .user)
        if case .text(let text) = msg.content {
            #expect(text == "hello")
        } else {
            Issue.record("expected .text content")
        }
        #expect(msg.images == nil)
        #expect(msg.reasoning == nil)
        #expect(msg.continuation == nil)
    }

    @Test func userFactoryWithImages_carriesImages() {
        let img = LLMImageContent(data: Data([0xFF, 0xD8]), mimeType: "image/jpeg")
        let msg = LLMMessage.user("describe", images: [img])
        #expect(msg.role == .user)
        #expect(msg.images?.count == 1)
        #expect(msg.images?.first?.mimeType == "image/jpeg")
    }

    @Test func systemFactory_setsSystemRoleAndText() {
        let msg = LLMMessage.system("you are helpful")
        #expect(msg.role == .system)
        #expect(msg.content.textValue == "you are helpful")
        #expect(msg.continuation == nil)
    }

    @Test func developerFactory_setsDeveloperRoleAndText() {
        let msg = LLMMessage.developer("internal instructions")
        #expect(msg.role == .developer)
        #expect(msg.content.textValue == "internal instructions")
    }

    @Test func toolResultFactory_setsToolRoleAndCallID() {
        let msg = LLMMessage.toolResult("ran cleanly", callID: "tc-1")
        #expect(msg.role == .tool)
        if case .toolResult(let id, let content) = msg.content {
            #expect(id == "tc-1")
            #expect(content == "ran cleanly")
        } else {
            Issue.record("expected .toolResult content")
        }
    }

    // MARK: - .assistant(from: response) — the load-bearing factory

    @Test func assistantFromResponse_textOnly_emitsTextContent() {
        let response = LLMResponse(text: "ok", toolCalls: [])
        let msg = LLMMessage.assistant(from: response)
        #expect(msg.role == .assistant)
        if case .text(let text) = msg.content {
            #expect(text == "ok")
        } else {
            Issue.record("expected .text content")
        }
        #expect(msg.reasoning == nil)
        #expect(msg.continuation == nil)
    }

    @Test func assistantFromResponse_nilTextNoTools_emitsEmptyTextContent() {
        // Defensive: a response with no text and no tool calls should still
        // yield a well-formed assistant message (not crash, not nil).
        let response = LLMResponse(text: nil, toolCalls: [])
        let msg = LLMMessage.assistant(from: response)
        #expect(msg.role == .assistant)
        if case .text(let text) = msg.content {
            #expect(text == "")
        } else {
            Issue.record("expected .text(\"\") content")
        }
    }

    @Test func assistantFromResponse_toolCallsOnly_emitsToolCallsContent() {
        let calls = [LLMToolCall(id: "x", name: "noop", arguments: "{}")]
        let response = LLMResponse(text: nil, toolCalls: calls)
        let msg = LLMMessage.assistant(from: response)
        if case .toolCalls(let outCalls) = msg.content {
            #expect(outCalls == calls)
        } else {
            Issue.record("expected .toolCalls content, got \(msg.content)")
        }
    }

    @Test func assistantFromResponse_emptyStringTextAndToolCalls_emitsToolCallsContent() {
        // Empty-string text + tool calls should collapse to .toolCalls, not .mixed
        // (which would carry an empty text part on the wire).
        let calls = [LLMToolCall(id: "x", name: "noop", arguments: "{}")]
        let response = LLMResponse(text: "", toolCalls: calls)
        let msg = LLMMessage.assistant(from: response)
        if case .toolCalls(let outCalls) = msg.content {
            #expect(outCalls == calls)
        } else {
            Issue.record("empty text + tool calls should collapse to .toolCalls, got \(msg.content)")
        }
    }

    @Test func assistantFromResponse_textPlusToolCalls_emitsMixedContent() {
        let calls = [LLMToolCall(id: "x", name: "noop", arguments: "{}")]
        let response = LLMResponse(text: "running tool", toolCalls: calls)
        let msg = LLMMessage.assistant(from: response)
        if case .mixed(let text, let outCalls) = msg.content {
            #expect(text == "running tool")
            #expect(outCalls == calls)
        } else {
            Issue.record("expected .mixed content, got \(msg.content)")
        }
    }

    @Test func assistantFromResponse_preservesReasoning() {
        let response = LLMResponse(
            text: "done",
            toolCalls: [],
            reasoning: "I considered the options"
        )
        let msg = LLMMessage.assistant(from: response)
        #expect(msg.reasoning == "I considered the options")
    }

    @Test func assistantFromResponse_preservesAnthropicContinuation() {
        // The whole point of the factory: continuation flows through without
        // the caller having to remember it.
        let blocks = [AnthropicThinkingBlock(thinking: "step 1", signature: "abc==")]
        let continuation = ProviderContinuation(anthropicThinkingBlocks: blocks)
        let response = LLMResponse(
            text: "answer",
            toolCalls: [],
            continuation: continuation
        )
        let msg = LLMMessage.assistant(from: response)
        #expect(msg.continuation?.anthropicThinkingBlocks == blocks)
    }

    @Test func assistantFromResponse_preservesGeminiContinuation() {
        let sigs = ["0": "sig-zero", "1": "sig-one"]
        let continuation = makeLegacyGeminiContinuation(sigs: sigs)
        let response = LLMResponse(text: "answer", toolCalls: [], continuation: continuation)
        let msg = LLMMessage.assistant(from: response)
        #expect(readLegacyGeminiSignatures(msg.continuation) == sigs)
    }

    // MARK: - Synthetic assistant factories — DO NOT carry continuation

    @Test func syntheticAssistantText_dropsContinuationByDesign() {
        // The deprecated synthetic factory cannot carry continuation —
        // documenting that here so the contract is locked.
        let msg = useDeprecatedSyntheticAssistantText("hi", reasoning: "thinking out loud")
        #expect(msg.role == .assistant)
        #expect(msg.reasoning == "thinking out loud")
        #expect(msg.continuation == nil)
    }

    @Test func syntheticAssistantToolCalls_dropsContinuationByDesign() {
        let calls = [LLMToolCall(id: "x", name: "noop", arguments: "{}")]
        let msg = useDeprecatedSyntheticAssistantToolCalls(calls, text: "running")
        if case .mixed(let text, let outCalls) = msg.content {
            #expect(text == "running")
            #expect(outCalls == calls)
        } else {
            Issue.record("expected .mixed when text is non-empty")
        }
        #expect(msg.continuation == nil)
    }

    @Test func syntheticAssistantToolCalls_emptyText_emitsPureToolCallsContent() {
        let calls = [LLMToolCall(id: "x", name: "noop", arguments: "{}")]
        let msg = useDeprecatedSyntheticAssistantToolCalls(calls, text: nil)
        if case .toolCalls(let outCalls) = msg.content {
            #expect(outCalls == calls)
        } else {
            Issue.record("nil text should yield .toolCalls, not .mixed")
        }
    }

    // MARK: - Codable round-trip with continuation

    @Test func codableRoundTrip_preservesAnthropicContinuation() throws {
        let blocks = [
            AnthropicThinkingBlock(thinking: "step 1", signature: "abc=="),
            AnthropicThinkingBlock(thinking: "step 2", signature: "def==")
        ]
        let response = LLMResponse(
            text: "answer",
            toolCalls: [],
            continuation: ProviderContinuation(anthropicThinkingBlocks: blocks)
        )
        let msg = LLMMessage.assistant(from: response)

        let data = try JSONEncoder().encode(msg)
        let decoded = try JSONDecoder().decode(LLMMessage.self, from: data)
        #expect(decoded.continuation?.anthropicThinkingBlocks == blocks)
    }

    @Test func codableRoundTrip_preservesGeminiContinuation() throws {
        let sigs = ["0": "sig-A", "1": "sig-B", "2": "sig-C"]
        let response = LLMResponse(
            text: "answer",
            toolCalls: [],
            continuation: makeLegacyGeminiContinuation(sigs: sigs)
        )
        let msg = LLMMessage.assistant(from: response)

        let data = try JSONEncoder().encode(msg)
        let decoded = try JSONDecoder().decode(LLMMessage.self, from: data)
        #expect(readLegacyGeminiSignatures(decoded.continuation) == sigs)
    }

    @Test func codableRoundTrip_emptyContinuation_omittedFromJSON() throws {
        // isEmpty continuation should not pollute the encoded payload.
        let msg = LLMMessage.user("hello")
        let data = try JSONEncoder().encode(msg)
        let json = String(data: data, encoding: .utf8) ?? ""
        #expect(!json.contains("continuation"))
    }

    @Test func legacyJSON_withoutContinuation_decodesCleanly() throws {
        // Pre-0.0.24 persisted state: just role + content. Must not throw and
        // continuation must come back as nil.
        let legacy = #"{"role":"assistant","content":{"type":"text","text":"hi"}}"#
        let decoded = try JSONDecoder().decode(LLMMessage.self, from: Data(legacy.utf8))
        #expect(decoded.continuation == nil)
        #expect(decoded.content.textValue == "hi")
    }
}

// MARK: - Deprecation-bridge wrappers (isolate deprecation warnings to one place)

@available(*, deprecated)
private func useDeprecatedSyntheticAssistantText(_ text: String, reasoning: String?) -> LLMMessage {
    LLMMessage.assistant(text: text, reasoning: reasoning)
}

@available(*, deprecated)
private func useDeprecatedSyntheticAssistantToolCalls(
    _ toolCalls: [LLMToolCall],
    text: String?
) -> LLMMessage {
    LLMMessage.assistant(toolCalls: toolCalls, text: text)
}

// MARK: - 0.0.26 deprecation bridges for legacy Gemini sigs field

/// Constructs a continuation populating only the legacy
/// `geminiThoughtSignatures` field. Wraps the deprecated init so the
/// deprecation warning is contained to this one helper.
@available(*, deprecated)
private func makeLegacyGeminiContinuation(sigs: [String: String]) -> ProviderContinuation {
    ProviderContinuation(geminiThoughtSignatures: sigs)
}

/// Reads the legacy `geminiThoughtSignatures` field, isolated in this
/// `@available(*, deprecated)` helper so test bodies stay warning-free.
@available(*, deprecated)
private func readLegacyGeminiSignatures(_ continuation: ProviderContinuation?) -> [String: String]? {
    continuation?.geminiThoughtSignatures
}
