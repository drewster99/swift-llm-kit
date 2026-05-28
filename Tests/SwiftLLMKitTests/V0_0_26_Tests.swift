import Foundation
import Testing
@testable import SwiftLLMKit

/// Tests for the 0.0.26 release:
/// 1. Gemini per-part thoughtSignature redesign — replace position-keyed
///    `geminiThoughtSignatures` with the structurally-faithful
///    `geminiResponseParts` payload that preserves the original part array
///    verbatim, eliminating the shape-fragility that 0.0.24's design had
///    when `.assistant(from:)` collapsed a multi-part response.
/// 2. Anthropic empty-content fix — substitute a single space for empty
///    assistant text content (HTTP 400 workaround) and skip empty text
///    blocks in the content-array path.
@Suite("0.0.26: Gemini parts + Anthropic empty-content")
struct V0_0_26_Tests {

    private static let dummyKey: @Sendable () -> String = { "" }

    private func gemini() throws -> GeminiProvider {
        try GeminiProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "gemini-2.5-pro"),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .gemini,
                endpoint: #require(URL(string: "https://generativelanguage.googleapis.com/v1beta"))
            ),
            readAPIKey: Self.dummyKey
        )
    }

    private func anthropic() throws -> AnthropicProvider {
        try AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "claude-sonnet-4-6"),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .anthropic,
                endpoint: #require(URL(string: "https://api.anthropic.com/v1"))
            ),
            readAPIKey: Self.dummyKey
        )
    }

    // MARK: - Gemini parts capture

    @Test("Gemini parseResponse captures full parts array into geminiResponseParts")
    func gemini_parsePopulatesNewPartsField() throws {
        let provider = try gemini()
        let body: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [
                            ["text": "answer text", "thoughtSignature": "sig-text"],
                            [
                                "functionCall": ["name": "tool_x", "args": ["a": 1] as [String: Any]] as [String: Any],
                                "thoughtSignature": "sig-fc"
                            ]
                        ] as [[String: Any]]
                    ] as [String: Any]
                ] as [String: Any]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let response = try provider.parseResponse(data: data)

        let parts = try #require(response.continuation?.geminiResponseParts)
        #expect(parts.count == 2)
        #expect(parts[0].text == "answer text")
        #expect(parts[0].thoughtSignature == "sig-text")
        #expect(parts[0].functionCall == nil)
        #expect(parts[1].text == nil)
        #expect(parts[1].thoughtSignature == "sig-fc")
        #expect(parts[1].functionCall?.name == "tool_x")
    }

    @Test("Gemini parseResponse still populates legacy geminiThoughtSignatures (backward-compat)")
    func gemini_parsePopulatesLegacyFieldToo() throws {
        let provider = try gemini()
        let body: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [
                            ["text": "x", "thoughtSignature": "sig-0"],
                            [
                                "functionCall": ["name": "f", "args": [:] as [String: Any]] as [String: Any],
                                "thoughtSignature": "sig-1"
                            ]
                        ] as [[String: Any]]
                    ] as [String: Any]
                ] as [String: Any]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: body)
        let response = try provider.parseResponse(data: data)
        let legacySigs = try #require(legacySignatures(response.continuation))
        #expect(legacySigs["0"] == "sig-0")
        #expect(legacySigs["1"] == "sig-1")
    }

    // MARK: - Gemini parts replay (the shape-fragility fix)

    @Test("Gemini encoder emits saved parts verbatim, not derived from content shape")
    func gemini_encoderUsesSavedPartsWhenAvailable() throws {
        // The scenario that 0.0.24's position-keying broke:
        // - Response shape: [thoughtTextPart sig=sig-thought, fcA sig=sig-A, fcB sig=sig-B]
        // - Factory `.assistant(from:)` would see text="..." + 2 tool calls and
        //   emit .mixed content. Replay would emit [textPart, fcA, fcB] and
        //   attach sigs by enumerated index — but the text in the replay is
        //   the CONCATENATED text, not the original thought-text part, so the
        //   sig at original-index 0 (on the thought text) would correctly
        //   attach to the new text part (lucky), but if text was empty (a pure
        //   thought-sig-only first part), the factory would have emitted
        //   .toolCalls([fcA, fcB]) — only 2 parts — and sig "0" (thought)
        //   would attach to fcA, mis-mapping every subsequent sig.
        // With the parts-verbatim replay, we get faithful round-trip regardless
        // of content-shape collapse.

        let provider = try gemini()
        // Build the worst-case shape: empty leading text + 2 tool calls, each
        // with its own signature, plus a leading thought-text-only part.
        let savedParts = [
            GeminiResponsePart(text: "", thoughtSignature: "sig-thought"),
            GeminiResponsePart(
                functionCall: GeminiFunctionCall(name: "tool_a", argsJSON: "{}"),
                thoughtSignature: "sig-A"
            ),
            GeminiResponsePart(
                functionCall: GeminiFunctionCall(name: "tool_b", argsJSON: "{}"),
                thoughtSignature: "sig-B"
            )
        ]
        let continuation = ProviderContinuation(geminiResponseParts: savedParts)

        // The user-facing LLMMessage has only the 2 tool calls (the factory
        // would collapse text="" + toolCalls into .toolCalls). Sigs on the
        // wire MUST still come from the saved parts, not the content.
        let response = LLMResponse(
            text: nil,
            toolCalls: [
                LLMToolCall(id: "id-A", name: "tool_a", arguments: "{}"),
                LLMToolCall(id: "id-B", name: "tool_b", arguments: "{}")
            ],
            continuation: continuation
        )
        let assistant = LLMMessage.assistant(from: response)
        let messages: [LLMMessage] = [
            .user("call both"),
            assistant,
            .toolResult("A done", callID: "id-A"),
            .toolResult("B done", callID: "id-B")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let modelTurn = try #require(contents.first { ($0["role"] as? String) == "model" })
        let parts = try #require(modelTurn["parts"] as? [[String: Any]])

        // Three parts on the wire — matches the SAVED parts, not the collapsed
        // 2-part content shape.
        #expect(parts.count == 3, "parts count must match saved parts, not collapsed content shape")
        #expect(parts[0]["text"] as? String == "")
        #expect(parts[0]["thoughtSignature"] as? String == "sig-thought")
        let fcA = try #require(parts[1]["functionCall"] as? [String: Any])
        #expect(fcA["name"] as? String == "tool_a")
        #expect(parts[1]["thoughtSignature"] as? String == "sig-A")
        let fcB = try #require(parts[2]["functionCall"] as? [String: Any])
        #expect(fcB["name"] as? String == "tool_b")
        #expect(parts[2]["thoughtSignature"] as? String == "sig-B")
    }

    @Test("Gemini round-trip — parse → replay produces matching wire-shape parts")
    func gemini_roundTripPartsAreFaithful() throws {
        // Parse a response that had multiple parts with sigs, then encode the
        // resulting assistant message back and confirm the encoded parts
        // match the source bytes.
        let provider = try gemini()
        let responseBody: [String: Any] = [
            "candidates": [
                [
                    "content": [
                        "parts": [
                            ["text": "narrate", "thoughtSignature": "sig-A"],
                            [
                                "functionCall": ["name": "tool_one", "args": ["x": 1] as [String: Any]] as [String: Any],
                                "thoughtSignature": "sig-B"
                            ]
                        ] as [[String: Any]]
                    ] as [String: Any]
                ] as [String: Any]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: responseBody)
        let response = try provider.parseResponse(data: data)
        let assistant = LLMMessage.assistant(from: response)

        let messages: [LLMMessage] = [
            .user("ask"),
            assistant,
            .toolResult("result", callID: response.toolCalls.first?.id ?? "missing")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let modelTurn = try #require(contents.first { ($0["role"] as? String) == "model" })
        let parts = try #require(modelTurn["parts"] as? [[String: Any]])

        #expect(parts.count == 2)
        #expect(parts[0]["text"] as? String == "narrate")
        #expect(parts[0]["thoughtSignature"] as? String == "sig-A")
        let fc = try #require(parts[1]["functionCall"] as? [String: Any])
        #expect(fc["name"] as? String == "tool_one")
        #expect(parts[1]["thoughtSignature"] as? String == "sig-B")
    }

    @Test("Gemini encoder falls back to legacy sigs when only legacy field present")
    func gemini_legacyFallbackPath() throws {
        // Locks the backward-compat path: a saved conversation from 0.0.24/0.0.25
        // (no `geminiResponseParts`, only legacy `geminiThoughtSignatures`)
        // still gets sigs attached via position-keying when replayed.
        let provider = try gemini()
        let continuation = ProviderContinuation(
            geminiResponseParts: nil,
            geminiThoughtSignatures: legacySigsDict(["0": "legacy-sig"])
        )
        let response = LLMResponse(
            text: "answer",
            toolCalls: [],
            continuation: continuation
        )
        let assistant = LLMMessage.assistant(from: response)
        let messages: [LLMMessage] = [.user("ask"), assistant, .user("again")]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let modelTurn = try #require(contents.first { ($0["role"] as? String) == "model" })
        let parts = try #require(modelTurn["parts"] as? [[String: Any]])
        #expect(parts[0]["text"] as? String == "answer")
        #expect(parts[0]["thoughtSignature"] as? String == "legacy-sig")
    }

    // MARK: - Codable

    @Test("ProviderContinuation Codable round-trips geminiResponseParts")
    func continuation_codableRoundTripPreservesParts() throws {
        let parts = [
            GeminiResponsePart(text: "x", thoughtSignature: "sig-1"),
            GeminiResponsePart(
                functionCall: GeminiFunctionCall(name: "f", argsJSON: "{\"a\":1}"),
                thoughtSignature: "sig-2"
            )
        ]
        let continuation = ProviderContinuation(geminiResponseParts: parts)
        let data = try JSONEncoder().encode(continuation)
        let decoded = try JSONDecoder().decode(ProviderContinuation.self, from: data)
        #expect(decoded.geminiResponseParts == parts)
    }

    @Test("Legacy JSON with only geminiThoughtSignatures still decodes")
    func continuation_legacyDecodeStillWorks() throws {
        let legacyJSON = #"{"geminiThoughtSignatures":{"0":"sig-A","1":"sig-B"}}"#
        let decoded = try JSONDecoder().decode(
            ProviderContinuation.self,
            from: Data(legacyJSON.utf8)
        )
        let legacy = try #require(legacySignatures(decoded))
        #expect(legacy["0"] == "sig-A")
        #expect(legacy["1"] == "sig-B")
        #expect(decoded.geminiResponseParts == nil)
    }

    // MARK: - Anthropic empty-content fix

    @Test("Anthropic encoder substitutes single space for empty .text content")
    func anthropic_emptyTextBecomesSingleSpace() throws {
        let provider = try anthropic()
        // Build an assistant turn with empty text — what
        // .assistant(from: LLMResponse(text: nil, toolCalls: [])) produces.
        let messages: [LLMMessage] = [
            .user("ask"),
            .assistant(from: LLMResponse(text: nil, toolCalls: [])),
            .user("follow up")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])
        let assistantWire = try #require(wire.first { ($0["role"] as? String) == "assistant" })
        // Plain-text path with no thinking blocks / images → content is a
        // single space, not an empty string (Anthropic rejects "").
        #expect(assistantWire["content"] as? String == " ",
                "empty assistant text must become \" \" to avoid Anthropic HTTP 400")
    }

    @Test("Anthropic encoder skips empty text block in content-array path")
    func anthropic_emptyTextBlockSkippedInArrayPath() throws {
        let provider = try anthropic()
        // Assistant with empty text but DOES have thinking blocks — the
        // content-array path. The empty {"type":"text","text":""} entry must
        // not be emitted (Anthropic rejects it).
        let blocks = [AnthropicThinkingBlock(thinking: "thinking", signature: "sig")]
        let response = LLMResponse(
            text: nil,
            toolCalls: [],
            continuation: ProviderContinuation(anthropicThinkingBlocks: blocks)
        )
        let messages: [LLMMessage] = [
            .user("ask"),
            .assistant(from: response),
            .user("follow up")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])
        let assistantWire = try #require(wire.first { ($0["role"] as? String) == "assistant" })
        let content = try #require(assistantWire["content"] as? [[String: Any]])
        // Only the thinking block — no text block at all.
        #expect(content.count == 1, "expected only the thinking block; empty text block must be skipped")
        #expect(content[0]["type"] as? String == "thinking")
    }

    @Test("Anthropic encoder skips empty text block in .mixed content")
    func anthropic_emptyTextSkippedInMixedContent() throws {
        let provider = try anthropic()
        // .mixed with empty text + tool calls. The empty text block must not
        // be emitted; only the tool_use blocks.
        let assistant = LLMMessage(
            _role: .assistant,
            _content: .mixed(
                text: "",
                toolCalls: [LLMToolCall(id: "id-x", name: "x", arguments: "{}")]
            )
        )
        let messages: [LLMMessage] = [
            .user("ask"),
            assistant,
            .toolResult("ok", callID: "id-x")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])
        let assistantWire = try #require(wire.first { ($0["role"] as? String) == "assistant" })
        let content = try #require(assistantWire["content"] as? [[String: Any]])
        #expect(content.count == 1, "expected only the tool_use block; empty text block must be skipped")
        #expect(content[0]["type"] as? String == "tool_use")
    }

    @Test("Anthropic encoder still emits non-empty text content normally")
    func anthropic_nonEmptyTextUnchanged() throws {
        let provider = try anthropic()
        let messages: [LLMMessage] = [
            .user("ask"),
            LLMMessage(_role: .assistant, _content: .text("real text")),
            .user("again")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let wire = try #require(body["messages"] as? [[String: Any]])
        let assistantWire = try #require(wire.first { ($0["role"] as? String) == "assistant" })
        #expect(assistantWire["content"] as? String == "real text",
                "non-empty text must pass through unchanged")
    }
}

// MARK: - Deprecation bridges (isolate warnings from the test bodies)

@available(*, deprecated)
private func legacySignatures(_ continuation: ProviderContinuation?) -> [String: String]? {
    continuation?.geminiThoughtSignatures
}

@available(*, deprecated)
private func legacySigsDict(_ pairs: [String: String]) -> [String: String]? {
    pairs
}
