import Foundation
import Testing
@testable import SwiftLLMKit

/// Encoding-side coverage for the Gemini 3 strict thought-signature rule
/// (see `GeminiThoughtSignatureLiveTests` for the live-API repro).
///
/// Gemini 3 rejects an assistant step with 400 INVALID_ARGUMENT unless its
/// first `functionCall` part carries a `thought_signature`. Whenever that
/// part is bare — turns recorded from a NON-Gemini model (no
/// `geminiResponseParts`, rendered from content shape), Gemini 2.5-era
/// captures that carried no signature, or legacy index-keyed signatures
/// that landed elsewhere — the encoder must attach Google's documented
/// validation-bypass dummy. Real captured signatures must never be
/// overwritten.
@Suite("Gemini functionCall dummy signature encoding")
struct GeminiFunctionCallDummySignatureTests {

    private static let dummyKey: @Sendable () -> String = { "" }

    private func gemini() throws -> GeminiProvider {
        try GeminiProvider(
            configuration: ModelConfiguration(
                name: "t", providerID: "p", modelID: "gemini-3.5-flash"
            ),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .gemini,
                endpoint: try #require(URL(string: "https://generativelanguage.googleapis.com/v1beta"))
            ),
            readAPIKey: Self.dummyKey
        )
    }

    /// Extracts the encoded parts of the model-role content entry at `index`
    /// among the request body's model entries.
    private func modelParts(
        body: [String: Any],
        entry index: Int = 0
    ) throws -> [[String: Any]] {
        let contents = try #require(body["contents"] as? [[String: Any]])
        let modelEntries = contents.filter { ($0["role"] as? String) == "model" }
        let entry = try #require(modelEntries.indices.contains(index) ? modelEntries[index] : nil)
        return try #require(entry["parts"] as? [[String: Any]])
    }

    @Test("unsigned .toolCalls turn gets the dummy signature on its first functionCall part")
    func toolCallsTurn_getsDummySignature() throws {
        let messages: [LLMMessage] = [
            .user("go"),
            LLMMessage(_role: .assistant, _content: .toolCalls([
                LLMToolCall(id: "id-1", name: "tool_a", arguments: "{}"),
                LLMToolCall(id: "id-2", name: "tool_b", arguments: "{}")
            ])),
            .toolResult("ok", callID: "id-1"),
            .toolResult("ok", callID: "id-2")
        ]

        let body = try gemini().buildRequestBody(messages: messages, tools: [])
        let parts = try modelParts(body: body)

        #expect(parts[0]["thoughtSignature"] as? String == GeminiProvider.functionCallDummySignature)
        // Only the FIRST functionCall part of the step needs a signature.
        #expect(parts[1]["thoughtSignature"] == nil)
    }

    @Test("unsigned .mixed turn signs the first functionCall part, not the text part")
    func mixedTurn_signsFunctionCallNotText() throws {
        let messages: [LLMMessage] = [
            .user("go"),
            LLMMessage(_role: .assistant, _content: .mixed(
                text: "calling now",
                toolCalls: [LLMToolCall(id: "id-1", name: "tool_a", arguments: "{}")]
            )),
            .toolResult("ok", callID: "id-1")
        ]

        let body = try gemini().buildRequestBody(messages: messages, tools: [])
        let parts = try modelParts(body: body)

        #expect(parts[0]["text"] as? String == "calling now")
        #expect(parts[0]["thoughtSignature"] == nil)
        #expect(parts[1]["thoughtSignature"] as? String == GeminiProvider.functionCallDummySignature)
    }

    @Test("text-only assistant turn gets no signature")
    func textOnlyTurn_unchanged() throws {
        let messages: [LLMMessage] = [
            .user("hi"),
            LLMMessage(_role: .assistant, _content: .text("hello")),
            .user("bye")
        ]

        let body = try gemini().buildRequestBody(messages: messages, tools: [])
        let parts = try modelParts(body: body)

        #expect(parts.count == 1)
        #expect(parts[0]["thoughtSignature"] == nil)
    }

    @Test("real captured Gemini signature is replayed, not overwritten by the dummy")
    func capturedSignature_notOverwritten() throws {
        var assistant = LLMMessage(_role: .assistant, _content: .toolCalls([
            LLMToolCall(id: "id-1", name: "tool_a", arguments: "{}")
        ]))
        assistant.continuation = ProviderContinuation(geminiResponseParts: [
            GeminiResponsePart(
                functionCall: GeminiFunctionCall(name: "tool_a", argsJSON: "{}"),
                thoughtSignature: "real-signature"
            )
        ])
        let messages: [LLMMessage] = [
            .user("go"),
            assistant,
            .toolResult("ok", callID: "id-1")
        ]

        let body = try gemini().buildRequestBody(messages: messages, tools: [])
        let parts = try modelParts(body: body)

        #expect(parts[0]["thoughtSignature"] as? String == "real-signature")
    }

    @Test("saved parallel-call parts: unsigned first call gets the dummy, later real signature untouched")
    func capturedParallelParts_firstUnsigned_dummyOnlyOnFirst() throws {
        var assistant = LLMMessage(_role: .assistant, _content: .toolCalls([
            LLMToolCall(id: "id-1", name: "tool_a", arguments: "{}"),
            LLMToolCall(id: "id-2", name: "tool_b", arguments: "{}")
        ]))
        assistant.continuation = ProviderContinuation(geminiResponseParts: [
            GeminiResponsePart(functionCall: GeminiFunctionCall(name: "tool_a", argsJSON: "{}")),
            GeminiResponsePart(
                functionCall: GeminiFunctionCall(name: "tool_b", argsJSON: "{}"),
                thoughtSignature: "real-signature-on-b"
            )
        ])
        let messages: [LLMMessage] = [
            .user("go"),
            assistant,
            .toolResult("ok", callID: "id-1"),
            .toolResult("ok", callID: "id-2")
        ]

        let body = try gemini().buildRequestBody(messages: messages, tools: [])
        let parts = try modelParts(body: body)

        #expect(parts[0]["thoughtSignature"] as? String == GeminiProvider.functionCallDummySignature)
        #expect(parts[1]["thoughtSignature"] as? String == "real-signature-on-b")
    }

    @Test("legacy .mixed turn: real signature on the text part stays; first functionCall gets the dummy")
    func legacyMixedTurn_textSignatureKept_functionCallGetsDummy() throws {
        var assistant = LLMMessage(_role: .assistant, _content: .mixed(
            text: "thinking aloud",
            toolCalls: [LLMToolCall(id: "id-1", name: "tool_a", arguments: "{}")]
        ))
        assistant.continuation = ProviderContinuation(geminiThoughtSignatures: ["0": "sig-on-text"])
        let messages: [LLMMessage] = [
            .user("go"),
            assistant,
            .toolResult("ok", callID: "id-1")
        ]

        let body = try gemini().buildRequestBody(messages: messages, tools: [])
        let parts = try modelParts(body: body)

        #expect(parts[0]["text"] as? String == "thinking aloud")
        #expect(parts[0]["thoughtSignature"] as? String == "sig-on-text")
        #expect(parts[1]["thoughtSignature"] as? String == GeminiProvider.functionCallDummySignature)
    }

    @Test("consecutive assistant turns merged into one model entry keep each step's dummy signature")
    func consecutiveAssistantTurns_mergedEntryKeepsBothSignatures() throws {
        // Signing runs per message BEFORE mergeConsecutiveSameRole collapses
        // same-role entries, so both former first-functionCall parts keep
        // their dummy after the merge — no call ends up unsigned.
        let messages: [LLMMessage] = [
            .user("go"),
            LLMMessage(_role: .assistant, _content: .toolCalls([
                LLMToolCall(id: "id-1", name: "tool_a", arguments: "{}")
            ])),
            LLMMessage(_role: .assistant, _content: .toolCalls([
                LLMToolCall(id: "id-2", name: "tool_b", arguments: "{}")
            ])),
            .toolResult("ok", callID: "id-1"),
            .toolResult("ok", callID: "id-2")
        ]

        let body = try gemini().buildRequestBody(messages: messages, tools: [])
        let parts = try modelParts(body: body)

        #expect(parts.count == 2)
        #expect(parts[0]["thoughtSignature"] as? String == GeminiProvider.functionCallDummySignature)
        #expect(parts[1]["thoughtSignature"] as? String == GeminiProvider.functionCallDummySignature)
    }

    @Test("captured Gemini parts WITHOUT a signature (2.5-era capture) get the dummy on replay")
    func capturedPartsWithoutSignature_getDummy() throws {
        var assistant = LLMMessage(_role: .assistant, _content: .toolCalls([
            LLMToolCall(id: "id-1", name: "tool_a", arguments: "{}")
        ]))
        assistant.continuation = ProviderContinuation(geminiResponseParts: [
            GeminiResponsePart(functionCall: GeminiFunctionCall(name: "tool_a", argsJSON: "{}"))
        ])
        let messages: [LLMMessage] = [
            .user("go"),
            assistant,
            .toolResult("ok", callID: "id-1")
        ]

        let body = try gemini().buildRequestBody(messages: messages, tools: [])
        let parts = try modelParts(body: body)

        #expect(parts[0]["thoughtSignature"] as? String == GeminiProvider.functionCallDummySignature)
    }
}
