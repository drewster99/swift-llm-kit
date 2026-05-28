import Foundation
import Testing
@testable import SwiftLLMKit

/// 0.0.24 redo of the 0.0.22 functionResponse.name fix (which had to be
/// reverted because it broke thinking-continuity in 0.0.23).
///
/// Gemini matches a `functionResponse` to its originating `functionCall` by
/// **name**, NOT by ID — IDs aren't part of Gemini's wire schema at all. So
/// when the agent loop hands back a tool result keyed by our internal
/// `toolCallID`, the encoder must look up the actual function name from prior
/// assistant turns and put THAT in `functionResponse.name`.
///
/// Serial single-call conversations happen to work without this (Gemini falls
/// back to positional pairing), but parallel calls fail without correct names.
@Suite("Gemini functionResponse.name resolution")
struct GeminiFunctionResponseNameTests {

    private static let dummyKey: @Sendable () -> String = { "" }

    private func gemini() throws -> GeminiProvider {
        try GeminiProvider(
            configuration: ModelConfiguration(
                name: "t", providerID: "p", modelID: "gemini-2.5-pro"
            ),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .gemini,
                endpoint: try #require(URL(string: "https://generativelanguage.googleapis.com/v1beta"))
            ),
            readAPIKey: Self.dummyKey
        )
    }

    // MARK: - buildToolNameLookup walker

    @Test("buildToolNameLookup collects names from .toolCalls assistant turns")
    func buildToolNameLookup_toolCallsContent() {
        let messages: [LLMMessage] = [
            .user("call foo and bar"),
            LLMMessage(_role: .assistant, _content: .toolCalls([
                LLMToolCall(id: "id-1", name: "foo", arguments: "{}"),
                LLMToolCall(id: "id-2", name: "bar", arguments: "{}")
            ]))
        ]
        let map = GeminiProvider.buildToolNameLookup(messages)
        #expect(map["id-1"] == "foo")
        #expect(map["id-2"] == "bar")
        #expect(map.count == 2)
    }

    @Test("buildToolNameLookup collects names from .mixed assistant turns")
    func buildToolNameLookup_mixedContent() {
        let messages: [LLMMessage] = [
            .user("with prelude"),
            LLMMessage(_role: .assistant, _content: .mixed(
                text: "calling foo now",
                toolCalls: [LLMToolCall(id: "id-X", name: "foo", arguments: "{}")]
            ))
        ]
        let map = GeminiProvider.buildToolNameLookup(messages)
        #expect(map["id-X"] == "foo")
    }

    @Test("buildToolNameLookup ignores tool-result and text-only messages")
    func buildToolNameLookup_ignoresOtherContent() {
        let messages: [LLMMessage] = [
            .user("hi"),
            LLMMessage(_role: .assistant, _content: .text("hello back")),
            .toolResult("doesn't appear", callID: "id-from-tool")
        ]
        let map = GeminiProvider.buildToolNameLookup(messages)
        #expect(map.isEmpty)
    }

    @Test("buildToolNameLookup accumulates across turns")
    func buildToolNameLookup_accumulates() {
        let messages: [LLMMessage] = [
            .user("turn 1"),
            LLMMessage(_role: .assistant, _content: .toolCalls([
                LLMToolCall(id: "id-1", name: "tool_a", arguments: "{}")
            ])),
            .toolResult("ok", callID: "id-1"),
            .user("turn 2"),
            LLMMessage(_role: .assistant, _content: .toolCalls([
                LLMToolCall(id: "id-2", name: "tool_b", arguments: "{}"),
                LLMToolCall(id: "id-3", name: "tool_c", arguments: "{}")
            ]))
        ]
        let map = GeminiProvider.buildToolNameLookup(messages)
        #expect(map["id-1"] == "tool_a")
        #expect(map["id-2"] == "tool_b")
        #expect(map["id-3"] == "tool_c")
    }

    // MARK: - functionResponse.name on the wire

    @Test("functionResponse.name uses the originating functionCall name, not the toolCallID")
    func functionResponseName_usesOriginatingName() throws {
        let provider = try gemini()
        let messages: [LLMMessage] = [
            .user("do x"),
            LLMMessage(_role: .assistant, _content: .toolCalls([
                LLMToolCall(id: "tc-opaque-uuid-1", name: "do_x", arguments: "{}")
            ])),
            .toolResult("did x", callID: "tc-opaque-uuid-1")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        // Find the content whose parts contain a functionResponse.
        let withFR = try #require(contents.first { content in
            guard let parts = content["parts"] as? [[String: Any]] else { return false }
            return parts.contains { $0["functionResponse"] != nil }
        })
        let parts = try #require(withFR["parts"] as? [[String: Any]])
        let fr = try #require(parts.compactMap { $0["functionResponse"] as? [String: Any] }.first)
        #expect(fr["name"] as? String == "do_x",
                "functionResponse.name must be the original tool name, not the call ID")
    }

    // MARK: - Parallel tool calls — the case where positional pairing fails

    @Test("parallel tool calls — each functionResponse.name matches its functionCall.name")
    func parallelToolCalls_namesPair() throws {
        let provider = try gemini()
        let messages: [LLMMessage] = [
            .user("call both"),
            LLMMessage(_role: .assistant, _content: .toolCalls([
                LLMToolCall(id: "uuid-A", name: "tool_a", arguments: "{}"),
                LLMToolCall(id: "uuid-B", name: "tool_b", arguments: "{}")
            ])),
            // Tool results come back in REVERSE order (parallel execution
            // doesn't guarantee ordering). Name-based lookup must still pair
            // them correctly.
            .toolResult("B done", callID: "uuid-B"),
            .toolResult("A done", callID: "uuid-A")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])

        // Walk the wire and collect the (name, content) pairs of each
        // functionResponse, in wire order.
        var pairs: [(name: String, content: String)] = []
        for content in contents {
            guard let parts = content["parts"] as? [[String: Any]] else { continue }
            for part in parts {
                guard let fr = part["functionResponse"] as? [String: Any] else { continue }
                let name = (fr["name"] as? String) ?? ""
                let respDict = (fr["response"] as? [String: Any]) ?? [:]
                let inner = (respDict["content"] as? String) ?? ""
                pairs.append((name, inner))
            }
        }
        #expect(pairs.count == 2)
        // The wire pairing must reflect ID-based name lookup, not positional.
        // tool result for uuid-B came first → its function name should be tool_b.
        #expect(pairs[0].name == "tool_b")
        #expect(pairs[0].content == "B done")
        #expect(pairs[1].name == "tool_a")
        #expect(pairs[1].content == "A done")
    }

    // MARK: - Orphan tool result (no matching prior call)

    @Test("orphan tool result falls back to using toolCallID as functionResponse.name")
    func orphanToolResult_fallsBackToCallID() throws {
        // No prior assistant turn with this ID. The encoder must not crash;
        // it falls back to using the ID as the name (and logs a warning).
        // Gemini will likely reject this on its end, but that's the model's
        // problem — the encoder must produce well-formed JSON.
        let provider = try gemini()
        let messages: [LLMMessage] = [
            .user("hi"),
            .toolResult("orphan", callID: "ghost-id")
        ]
        let body = try provider.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let withFR = try #require(contents.first { content in
            guard let parts = content["parts"] as? [[String: Any]] else { return false }
            return parts.contains { $0["functionResponse"] != nil }
        })
        let parts = try #require(withFR["parts"] as? [[String: Any]])
        let fr = try #require(parts.compactMap { $0["functionResponse"] as? [String: Any] }.first)
        #expect(fr["name"] as? String == "ghost-id")
    }
}
