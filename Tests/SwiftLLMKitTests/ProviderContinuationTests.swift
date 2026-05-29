import Foundation
import Testing
@testable import SwiftLLMKit

/// 0.0.24 thinking-continuity tests for the two providers that need it:
/// - **Anthropic**: thinking blocks with `signature` parsed into
///   `ProviderContinuation.anthropicThinkingBlocks`, replayed verbatim at the
///   start of the next assistant turn.
/// - **Gemini 2.5**: `thoughtSignature` per response part captured into
///   `ProviderContinuation.geminiThoughtSignatures` (keyed by part index),
///   re-attached to the matching outgoing parts.
///
/// Without these tests the 0.0.22 regression (which broke Gemini thinking
/// continuity in multi-turn tool-use) could silently come back.
@Suite("Provider continuation parse + replay")
struct ProviderContinuationTests {

    private static let dummyKey: @Sendable () -> String = { "" }

    // MARK: - Anthropic

    private func anthropic(_ overrides: [String: AnyCodable]? = nil) throws -> AnthropicProvider {
        try AnthropicProvider(
            configuration: ModelConfiguration(
                name: "t", providerID: "p", modelID: "claude-opus-4-7",
                extraJSONOverrides: overrides
            ),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .anthropic,
                endpoint: try #require(URL(string: "https://api.anthropic.com/v1"))
            ),
            readAPIKey: Self.dummyKey
        )
    }

    @Test("Anthropic parseResponse extracts thinking blocks into continuation")
    func anthropic_parseResponse_capturesThinkingBlocks() throws {
        let provider = try anthropic()
        let body: [String: Any] = [
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-7",
            "stop_reason": "end_turn",
            "usage": ["input_tokens": 10, "output_tokens": 5],
            "content": [
                ["type": "thinking", "thinking": "I am thinking", "signature": "sig-1"],
                ["type": "thinking", "thinking": "more thoughts", "signature": "sig-2"],
                ["type": "text", "text": "Here is my answer."]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let response = try provider.parseResponse(data: data)

        #expect(response.text == "Here is my answer.")
        let blocks = try #require(response.continuation?.anthropicThinkingBlocks)
        #expect(blocks.count == 2)
        #expect(blocks[0] == AnthropicThinkingBlock(thinking: "I am thinking", signature: "sig-1"))
        #expect(blocks[1] == AnthropicThinkingBlock(thinking: "more thoughts", signature: "sig-2"))
        // The text of the last thinking block is also surfaced as reasoning.
        #expect(response.reasoning == "more thoughts")
    }

    @Test("Anthropic parseResponse with no thinking blocks → continuation is nil")
    func anthropic_parseResponse_noThinking_continuationNil() throws {
        let provider = try anthropic()
        let body: [String: Any] = [
            "content": [
                ["type": "text", "text": "plain answer"]
            ],
            "usage": ["input_tokens": 1, "output_tokens": 1]
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let response = try provider.parseResponse(data: data)
        #expect(response.text == "plain answer")
        #expect(response.continuation == nil)
    }

    @Test("Anthropic parseResponse skips thinking blocks with empty signature")
    func anthropic_parseResponse_emptySignature_dropped() throws {
        // No signature → nothing to replay; we drop the block from continuation
        // (and the thinking text alone is useless for continuity).
        let provider = try anthropic()
        let body: [String: Any] = [
            "content": [
                ["type": "thinking", "thinking": "no sig here", "signature": ""],
                ["type": "text", "text": "plain"]
            ],
            "usage": ["input_tokens": 1, "output_tokens": 1]
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let response = try provider.parseResponse(data: data)
        #expect(response.continuation == nil)
    }

    @Test("Anthropic buildRequestBody replays thinking blocks verbatim at start of assistant turn")
    func anthropic_buildRequestBody_replaysThinkingBlocks() throws {
        let provider = try anthropic()
        let blocks = [
            AnthropicThinkingBlock(thinking: "first thought", signature: "sig-A"),
            AnthropicThinkingBlock(thinking: "second thought", signature: "sig-B")
        ]
        let response = LLMResponse(
            text: "answer text",
            toolCalls: [],
            continuation: ProviderContinuation(anthropicThinkingBlocks: blocks)
        )
        let assistant = LLMMessage.assistant(from: response)

        let messages: [LLMMessage] = [
            .user("first question"),
            assistant,
            .user("follow up")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])

        // wire[1] is the assistant turn; its content must be a content-block
        // array whose first two entries are the thinking blocks, followed by
        // the text block.
        let assistantWire = try #require(wire.first { ($0["role"] as? String) == "assistant" })
        let content = try #require(assistantWire["content"] as? [[String: Any]])
        #expect(content.count == 3, "expected 2 thinking + 1 text, got \(content.count) blocks")
        #expect(content[0]["type"] as? String == "thinking")
        #expect(content[0]["thinking"] as? String == "first thought")
        #expect(content[0]["signature"] as? String == "sig-A")
        #expect(content[1]["type"] as? String == "thinking")
        #expect(content[1]["signature"] as? String == "sig-B")
        #expect(content[2]["type"] as? String == "text")
        #expect(content[2]["text"] as? String == "answer text")
    }

    @Test("Anthropic buildRequestBody replays thinking blocks before tool_use calls")
    func anthropic_buildRequestBody_replaysThinkingBeforeToolCalls() throws {
        let provider = try anthropic()
        let blocks = [AnthropicThinkingBlock(thinking: "I should call x", signature: "sig-X")]
        let toolCalls = [LLMToolCall(id: "call-1", name: "x", arguments: "{}")]
        let response = LLMResponse(
            text: nil,
            toolCalls: toolCalls,
            continuation: ProviderContinuation(anthropicThinkingBlocks: blocks)
        )
        let assistant = LLMMessage.assistant(from: response)

        let messages: [LLMMessage] = [
            .user("do x"),
            assistant,
            .toolResult("did x", callID: "call-1")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])

        let assistantWire = try #require(wire.first { ($0["role"] as? String) == "assistant" })
        let content = try #require(assistantWire["content"] as? [[String: Any]])
        #expect(content.count == 2, "expected 1 thinking + 1 tool_use, got \(content.count)")
        #expect(content[0]["type"] as? String == "thinking")
        #expect(content[0]["signature"] as? String == "sig-X")
        #expect(content[1]["type"] as? String == "tool_use")
        #expect(content[1]["id"] as? String == "call-1")
    }

    @Test("Anthropic buildRequestBody — no continuation → no thinking blocks emitted")
    func anthropic_buildRequestBody_noContinuation_noThinkingBlocks() throws {
        let provider = try anthropic()
        // Synthetic assistant turn with no continuation.
        let assistant = LLMMessage(_role: .assistant, _content: .text("answer"))
        let messages: [LLMMessage] = [.user("ask"), assistant, .user("again")]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])
        let assistantWire = try #require(wire.first { ($0["role"] as? String) == "assistant" })
        // No thinking blocks → content is a plain string, not a block array.
        #expect(assistantWire["content"] is String)
    }

    // MARK: - Gemini

    private func gemini(_ overrides: [String: AnyCodable]? = nil) throws -> GeminiProvider {
        try GeminiProvider(
            configuration: ModelConfiguration(
                name: "t", providerID: "p", modelID: "gemini-2.5-pro",
                extraJSONOverrides: overrides
            ),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .gemini,
                endpoint: try #require(URL(string: "https://generativelanguage.googleapis.com/v1beta"))
            ),
            readAPIKey: Self.dummyKey
        )
    }

    // Note: legacy `geminiThoughtSignatures` is still populated by the parser
    // for backward-compat with 0.0.24/0.0.25 consumers; the new field is
    // `geminiResponseParts` covered in V0_0_26_Tests. This test exercises
    // the legacy field via an @available(*, deprecated) bridge to keep the
    // test body warning-free.
    @Test("Gemini parseResponse populates legacy geminiThoughtSignatures (0.0.24/0.0.25 compat)")
    func gemini_parseResponse_capturesThoughtSignaturesPerPartIndex() throws {
        let provider = try gemini()
        let body: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [
                            ["text": "first text", "thoughtSignature": "sig-0"],
                            ["text": " second text", "thoughtSignature": "sig-1"],
                            // Part 2 has no signature.
                            ["text": " third"],
                            // Part 3 has a function call + signature.
                            [
                                "functionCall": ["name": "do_thing", "args": [:] as [String: Any]],
                                "thoughtSignature": "sig-3"
                            ]
                        ] as [[String: Any]]
                    ] as [String: Any],
                    "finishReason": "STOP"
                ] as [String: Any]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let response = try provider.parseResponse(data: data)

        let sigs = try #require(legacySignaturesBridge(response.continuation))
        #expect(sigs["0"] == "sig-0")
        #expect(sigs["1"] == "sig-1")
        #expect(sigs["2"] == nil, "part 2 had no signature, must not appear in map")
        #expect(sigs["3"] == "sig-3")

        // Tool call from part 3 is parsed.
        #expect(response.toolCalls.count == 1)
        #expect(response.toolCalls.first?.name == "do_thing")
    }

    @Test("Gemini parseResponse with no thoughtSignatures → continuation is nil")
    func gemini_parseResponse_noSignatures_continuationNil() throws {
        let provider = try gemini()
        let body: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [["text": "plain answer"]] as [[String: Any]]
                    ] as [String: Any]
                ] as [String: Any]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let response = try provider.parseResponse(data: data)
        #expect(response.continuation == nil)
    }

    @Test("Gemini parseResponse drops empty-string thoughtSignature values")
    func gemini_parseResponse_emptySignatureDropped() throws {
        let provider = try gemini()
        let body: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [
                            ["text": "x", "thoughtSignature": ""]
                        ] as [[String: Any]]
                    ] as [String: Any]
                ] as [String: Any]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let response = try provider.parseResponse(data: data)
        #expect(response.continuation == nil)
    }

    @Test("Gemini buildRequestBody re-attaches thoughtSignature to outgoing assistant parts")
    func gemini_buildRequestBody_replaysSignaturesByPartIndex() throws {
        let provider = try gemini()
        // Assistant response with tool calls (becomes part-array on the wire).
        let toolCalls = [
            LLMToolCall(id: "id-A", name: "tool_a", arguments: "{\"x\":1}"),
            LLMToolCall(id: "id-B", name: "tool_b", arguments: "{\"y\":2}")
        ]
        let sigs = ["0": "sig-zero", "1": "sig-one"]
        let response = LLMResponse(
            text: nil,
            toolCalls: toolCalls,
            continuation: ProviderContinuation(geminiThoughtSignatures: sigs)
        )
        let assistant = LLMMessage.assistant(from: response)

        let messages: [LLMMessage] = [
            .user("call both tools"),
            assistant,
            .toolResult("A done", callID: "id-A"),
            .toolResult("B done", callID: "id-B")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let modelTurn = try #require(contents.first { ($0["role"] as? String) == "model" })
        let parts = try #require(modelTurn["parts"] as? [[String: Any]])
        #expect(parts.count == 2)
        #expect(parts[0]["thoughtSignature"] as? String == "sig-zero")
        #expect(parts[1]["thoughtSignature"] as? String == "sig-one")
        // Make sure the part bodies (functionCall) are preserved alongside the signature.
        let fc0 = try #require(parts[0]["functionCall"] as? [String: Any])
        #expect(fc0["name"] as? String == "tool_a")
        let fc1 = try #require(parts[1]["functionCall"] as? [String: Any])
        #expect(fc1["name"] as? String == "tool_b")
    }

    @Test("Gemini buildRequestBody attaches signature on a text-only assistant part")
    func gemini_buildRequestBody_replaysSignatureOnTextPart() throws {
        let provider = try gemini()
        let sigs = ["0": "sig-text"]
        let response = LLMResponse(
            text: "an answer",
            toolCalls: [],
            continuation: ProviderContinuation(geminiThoughtSignatures: sigs)
        )
        let assistant = LLMMessage.assistant(from: response)
        let messages: [LLMMessage] = [.user("ask"), assistant, .user("again")]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let modelTurn = try #require(contents.first { ($0["role"] as? String) == "model" })
        let parts = try #require(modelTurn["parts"] as? [[String: Any]])
        #expect(parts.count == 1)
        #expect(parts[0]["thoughtSignature"] as? String == "sig-text")
        #expect(parts[0]["text"] as? String == "an answer")
    }

    @Test("Gemini buildRequestBody — no continuation → no thoughtSignature on parts")
    func gemini_buildRequestBody_noContinuation_noSignaturesEmitted() throws {
        let provider = try gemini()
        let assistant = LLMMessage(_role: .assistant, _content: .text("answer"))
        let messages: [LLMMessage] = [.user("ask"), assistant, .user("more")]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let modelTurn = try #require(contents.first { ($0["role"] as? String) == "model" })
        let parts = try #require(modelTurn["parts"] as? [[String: Any]])
        #expect(parts[0]["thoughtSignature"] == nil)
    }

    @Test("Gemini buildRequestBody — partial signatures map only attaches at known indexes")
    func gemini_buildRequestBody_partialSignaturesOnlyAttachToKnownIndexes() throws {
        // Half the parts in the response had signatures (e.g. only some
        // functionCalls were "thoughts" parts). The encoder must attach
        // signatures only at the indexes that had them; other parts go bare.
        let provider = try gemini()
        let toolCalls = [
            LLMToolCall(id: "id-A", name: "tool_a", arguments: "{}"),
            LLMToolCall(id: "id-B", name: "tool_b", arguments: "{}"),
            LLMToolCall(id: "id-C", name: "tool_c", arguments: "{}")
        ]
        let sigs = ["1": "sig-only-on-B"]
        let response = LLMResponse(
            text: nil,
            toolCalls: toolCalls,
            continuation: ProviderContinuation(geminiThoughtSignatures: sigs)
        )
        let assistant = LLMMessage.assistant(from: response)
        let messages: [LLMMessage] = [
            .user("call all"),
            assistant,
            .toolResult("A", callID: "id-A"),
            .toolResult("B", callID: "id-B"),
            .toolResult("C", callID: "id-C")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let modelTurn = try #require(contents.first { ($0["role"] as? String) == "model" })
        let parts = try #require(modelTurn["parts"] as? [[String: Any]])
        #expect(parts.count == 3)
        #expect(parts[0]["thoughtSignature"] == nil)
        #expect(parts[1]["thoughtSignature"] as? String == "sig-only-on-B")
        #expect(parts[2]["thoughtSignature"] == nil)
    }
}

// MARK: - 0.0.26 deprecation bridge

/// Reads the legacy 0.0.24/0.0.25 `geminiThoughtSignatures` field. Isolated
/// in an `@available(*, deprecated)` helper so the test bodies that
/// intentionally exercise the backward-compat path stay warning-free.
@available(*, deprecated)
private func legacySignaturesBridge(_ continuation: ProviderContinuation?) -> [String: String]? {
    continuation?.geminiThoughtSignatures
}
