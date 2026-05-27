import Foundation
import Testing
@testable import SwiftLLMKit

/// Regression coverage for the 0.0.22 Gemini `functionResponse.name` fix.
///
/// Gemini's wire schema has no tool-call-IDs at all — it pairs tool calls to
/// their results via the `name` field. Previously
/// `GeminiProvider.encodeContent` put the toolCallID in `functionResponse.name`,
/// which: (a) works for serial calls because Gemini falls back to positional
/// pairing; (b) silently fails for parallel calls because name is the only
/// signal.
///
/// The fix walks the conversation once to build a [toolCallID → functionName]
/// lookup from prior assistant `.toolCalls` / `.mixed` turns, then resolves
/// each `.toolResult`'s name at request-build time.
@Suite("GeminiProvider functionResponse.name correctness (0.0.22)")
struct GeminiFunctionResponseNameTests {

    private static let dummyKey: @Sendable () -> String = { "test" }

    private static func provider() throws -> GeminiProvider {
        let endpoint = try #require(URL(string: "https://generativelanguage.googleapis.com/v1beta"))
        return GeminiProvider(
            configuration: ModelConfiguration(
                name: "t", providerID: "p", modelID: "gemini-2.5-pro"
            ),
            provider: ModelProvider(id: "p", name: "p", apiType: .gemini, endpoint: endpoint),
            readAPIKey: dummyKey
        )
    }

    // MARK: - Lookup helper

    @Test func buildToolNameLookup_walksToolCallsAndMixed() {
        let messages: [LLMMessage] = [
            LLMMessage(role: .user, text: "hi"),
            LLMMessage(role: .assistant, content: .toolCalls([
                LLMToolCall(id: "call_1", name: "add", arguments: "{}"),
                LLMToolCall(id: "call_2", name: "lookup", arguments: "{}")
            ])),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "call_1", content: "ok")),
            LLMMessage(role: .assistant, content: .mixed(text: "thinking", toolCalls: [
                LLMToolCall(id: "call_3", name: "weather", arguments: "{}")
            ])),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "call_3", content: "ok"))
        ]
        let map = GeminiProvider.buildToolNameLookup(messages)
        #expect(map == [
            "call_1": "add",
            "call_2": "lookup",
            "call_3": "weather"
        ])
    }

    @Test func buildToolNameLookup_emptyConversation_emptyMap() {
        #expect(GeminiProvider.buildToolNameLookup([]).isEmpty)
    }

    @Test func buildToolNameLookup_noToolTurns_emptyMap() {
        let messages: [LLMMessage] = [
            LLMMessage(role: .user, text: "hi"),
            LLMMessage(role: .assistant, text: "hello")
        ]
        #expect(GeminiProvider.buildToolNameLookup(messages).isEmpty)
    }

    @Test func buildToolNameLookup_duplicateIDs_lastWins() {
        // Shouldn't happen in well-formed conversations, but verify the rule.
        let messages: [LLMMessage] = [
            LLMMessage(role: .assistant, content: .toolCalls([
                LLMToolCall(id: "dup", name: "first", arguments: "{}")
            ])),
            LLMMessage(role: .assistant, content: .toolCalls([
                LLMToolCall(id: "dup", name: "second", arguments: "{}")
            ]))
        ]
        #expect(GeminiProvider.buildToolNameLookup(messages)["dup"] == "second")
    }

    // MARK: - Wire-format request body assertions

    @Test func toolResult_functionResponseName_resolvesToFunctionNameNotCallID() throws {
        let p = try Self.provider()
        let messages: [LLMMessage] = [
            LLMMessage(role: .user, text: "what's the weather?"),
            LLMMessage(role: .assistant, content: .toolCalls([
                LLMToolCall(id: "toolu_01ABC", name: "get_weather", arguments: #"{"city":"SF"}"#)
            ])),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "toolu_01ABC", content: "62F"))
        ]
        let body = try p.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        // Find the functionResponse part.
        let toolResultPart = contents.compactMap { content -> [String: Any]? in
            let parts = content["parts"] as? [[String: Any]] ?? []
            return parts.first { $0["functionResponse"] != nil }?["functionResponse"] as? [String: Any]
        }.first
        let fr = try #require(toolResultPart)
        #expect(fr["name"] as? String == "get_weather",
                "functionResponse.name must be the function name, not the toolCallID")
    }

    @Test func toolResult_fromMixedAssistant_alsoResolves() throws {
        let p = try Self.provider()
        let messages: [LLMMessage] = [
            LLMMessage(role: .user, text: "do it"),
            LLMMessage(role: .assistant, content: .mixed(
                text: "I'll calculate",
                toolCalls: [LLMToolCall(id: "call_xyz", name: "add", arguments: #"{"a":1,"b":2}"#)]
            )),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "call_xyz", content: "3"))
        ]
        let body = try p.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let toolResultPart = contents.compactMap { content -> [String: Any]? in
            let parts = content["parts"] as? [[String: Any]] ?? []
            return parts.first { $0["functionResponse"] != nil }?["functionResponse"] as? [String: Any]
        }.first
        let fr = try #require(toolResultPart)
        #expect(fr["name"] as? String == "add")
    }

    @Test func parallelToolCalls_eachResponseHasCorrectName() throws {
        // Critical case: two tool calls in one assistant turn, two results in
        // the next user turn. Gemini pairs by name when calls are parallel.
        // If both responses had the same wrong name (or just toolCallIDs),
        // pairing would silently break.
        let p = try Self.provider()
        let messages: [LLMMessage] = [
            LLMMessage(role: .user, text: "weather and time?"),
            LLMMessage(role: .assistant, content: .toolCalls([
                LLMToolCall(id: "call_a", name: "get_weather", arguments: "{}"),
                LLMToolCall(id: "call_b", name: "get_time", arguments: "{}")
            ])),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "call_a", content: "62F")),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "call_b", content: "3pm"))
        ]
        let body = try p.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        // Collect every functionResponse name in encounter order.
        var names: [String] = []
        for content in contents {
            let parts = content["parts"] as? [[String: Any]] ?? []
            for part in parts {
                if let fr = part["functionResponse"] as? [String: Any],
                   let name = fr["name"] as? String {
                    names.append(name)
                }
            }
        }
        #expect(names == ["get_weather", "get_time"],
                "parallel tool responses must each carry the right function name")
    }

    @Test func orphanToolResult_fallsBackToToolCallID_doesNotCrash() throws {
        // Malformed conversation: tool result with no preceding tool call.
        // Other consumers may have code that "works" with the bug; preserve
        // back-compat by falling back to the toolCallID (with a logged warning).
        let p = try Self.provider()
        let messages: [LLMMessage] = [
            LLMMessage(role: .user, text: "q"),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "orphan_id", content: "ignored"))
        ]
        let body = try p.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let toolResultPart = contents.compactMap { content -> [String: Any]? in
            let parts = content["parts"] as? [[String: Any]] ?? []
            return parts.first { $0["functionResponse"] != nil }?["functionResponse"] as? [String: Any]
        }.first
        let fr = try #require(toolResultPart)
        #expect(fr["name"] as? String == "orphan_id",
                "fallback to toolCallID when no matching prior call")
    }

    @Test func crossProviderIDFormats_allResolvedToCorrectFunctionName() throws {
        // Hydra rotation case: assistant turn could have toolu_ (Anthropic),
        // call_ (OpenAI), or UUID (Gemini) IDs from previous rotations. The
        // lookup is by literal ID string, not format, so all should resolve.
        let p = try Self.provider()
        let messages: [LLMMessage] = [
            LLMMessage(role: .user, text: "..."),
            LLMMessage(role: .assistant, content: .toolCalls([
                LLMToolCall(id: "toolu_01XYZ", name: "anthropic_call", arguments: "{}"),
                LLMToolCall(id: "call_openai", name: "openai_call", arguments: "{}"),
                LLMToolCall(id: "10F80F64-C871-4B20-B654-AF430059CF23", name: "gemini_call", arguments: "{}")
            ])),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "toolu_01XYZ", content: "a")),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "call_openai", content: "b")),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "10F80F64-C871-4B20-B654-AF430059CF23", content: "c"))
        ]
        let body = try p.buildRequestBody(messages: messages, tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        var names: [String] = []
        for content in contents {
            let parts = content["parts"] as? [[String: Any]] ?? []
            for part in parts where part["functionResponse"] != nil {
                if let fr = part["functionResponse"] as? [String: Any],
                   let name = fr["name"] as? String {
                    names.append(name)
                }
            }
        }
        #expect(names == ["anthropic_call", "openai_call", "gemini_call"])
    }
}
