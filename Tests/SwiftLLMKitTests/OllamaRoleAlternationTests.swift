import Foundation
import Testing
@testable import SwiftLLMKit

/// Asserts that the messages OllamaProvider would put on the wire for a given
/// agent conversation match what gemma3:27b's strict user/assistant chat
/// template will accept. The 400 we keep seeing in the live app —
/// "Conversation roles must alternate user/assistant/user/assistant/..." —
/// fires whenever any non-system role besides user/assistant survives, or two
/// same-role messages appear back-to-back, in the encoded request.
@Suite("OllamaProvider role alternation")
struct OllamaRoleAlternationTests {

    @Test("after a tool result the encoded request must not contain role=tool")
    func toolRoleMustNotSurviveWhenToolsAreDefined() {
        let encoded = encode(messages: liveAppSequence(), tools: standardTools())
        let roles = encoded.compactMap { $0["role"] as? String }
        let hasTool = roles.contains("tool")
        if hasTool {
            print("DEBUG roles after merge: \(roles)")
        }
        #expect(!hasTool,
                "Encoded request kept role=tool — gemma3:27b's chat template knows only user/assistant and rejects this with HTTP 400.")
    }

    @Test("non-system roles strictly alternate user/assistant after merge")
    func nonSystemRolesStrictlyAlternate() {
        let encoded = encode(messages: liveAppSequence(), tools: standardTools())
        let nonSystem = encoded.compactMap { $0["role"] as? String }.filter { $0 != "system" }
        for (i, role) in nonSystem.enumerated() where i > 0 {
            #expect(role != nonSystem[i - 1],
                    "Two consecutive `\(role)` messages at position \(i). Full sequence: \(nonSystem.joined(separator: ", "))")
            // Also ensure only user/assistant appear post-system.
            #expect(role == "user" || role == "assistant",
                    "Non-system role `\(role)` at position \(i) — gemma3 only accepts user/assistant.")
        }
    }

    @Test("text+tool_call assistant collapses with prior assistant text")
    func consecutiveAssistantMessagesMerge() {
        let messages: [LLMMessage] = [
            .system("sys"),
            .user("hi"),
            syntheticAssistantText("thinking aloud"),
            syntheticAssistantToolCalls([
                LLMToolCall(id: "abc", name: "noop", arguments: "{}")
            ])
        ]
        let encoded = encode(messages: messages, tools: standardTools())
        let assistantCount = encoded.filter { ($0["role"] as? String) == "assistant" }.count
        #expect(assistantCount == 1, "Expected exactly one assistant message after merging consecutive same-role; got \(assistantCount).")
    }

    @Test("user message arriving after a tool result must collapse alternation")
    func userAfterToolResultCollapses() {
        // Mirrors the live trace: assistant emitted a bad tool call, the tool
        // returned an error, then the user typed another message before the
        // next LLM turn. The encoded request must end up with strict
        // user/assistant alternation, no surviving `tool` role.
        let messages: [LLMMessage] = [
            .system("sys"),
            .user("yes"),
            syntheticAssistantMixed(
                text: "Calling run_task.",
                toolCalls: [LLMToolCall(id: "tc1", name: "run_task", arguments: "{}")]
            ),
            .toolResult("Tool error: missing task_id", callID: "tc1"),
            .user("actually use task abc-123")
        ]
        let encoded = encode(messages: messages, tools: standardTools())
        let roles = encoded.compactMap { $0["role"] as? String }
        if roles.contains("tool") {
            print("DEBUG roles: \(roles)")
        }
        #expect(!roles.contains("tool"))
        let nonSystem = roles.filter { $0 != "system" }
        for (i, role) in nonSystem.enumerated() where i > 0 {
            #expect(role != nonSystem[i - 1], "Two `\(role)` in a row at \(i): \(nonSystem)")
        }
    }

    // MARK: - Fixtures

    private func liveAppSequence() -> [LLMMessage] {
        [
            .system("You are Smith."),
            syntheticAssistantText("Drew, I see we have a pending task."),
            .user("[USER (Drew)]: yes"),
            syntheticAssistantMixed(
                text: "Okay, calling run_task.",
                toolCalls: [LLMToolCall(id: "tc1", name: "run_task", arguments: "{\"instructions\":\"go\"}")]
            ),
            .toolResult("Tool error: Missing required argument: task_id", callID: "tc1")
        ]
    }

    private func standardTools() -> [LLMToolDefinition] {
        [
            LLMToolDefinition(
                name: "run_task",
                description: "Run a task",
                parameters: [
                    "type": .string("object"),
                    "properties": .dictionary([
                        "task_id": .dictionary(["type": .string("string")]),
                        "instructions": .dictionary(["type": .string("string")])
                    ]),
                    "required": .array([.string("task_id"), .string("instructions")])
                ]
            )
        ]
    }

    private func encode(messages: [LLMMessage], tools: [LLMToolDefinition]) -> [[String: Any]] {
        let provider = OllamaProvider(
            configuration: ModelConfiguration(name: "gemma", providerID: "p", modelID: "gemma3:27b"),
            provider: ModelProvider(id: "p", name: "p", apiType: .ollama, endpoint: URL(string: "http://example.invalid/api")!),
            readAPIKey: { "" }
        )
        return provider.buildEncodedMessagesForTesting(messages: messages, tools: tools)
    }
}

// MARK: - Synthetic message helpers (use the internal init to avoid the deprecation
// warnings on the public synthetic .assistant(...) factories — these tests are
// specifically validating role-handling behavior on hand-built messages).

private func syntheticAssistantText(_ text: String) -> LLMMessage {
    LLMMessage(_role: .assistant, _content: .text(text))
}

private func syntheticAssistantToolCalls(_ calls: [LLMToolCall]) -> LLMMessage {
    LLMMessage(_role: .assistant, _content: .toolCalls(calls))
}

private func syntheticAssistantMixed(text: String, toolCalls: [LLMToolCall]) -> LLMMessage {
    LLMMessage(_role: .assistant, _content: .mixed(text: text, toolCalls: toolCalls))
}

// MARK: - Documents are refused, not dropped

/// Ollama's chat API carries `images` and nothing else. `encodeMessage` has no `documents`
/// branch and `normalizeMessages`' merge path rebuilds messages with images only, so before the
/// guard a caller passing `documents:` got a 200 and an answer about a file the model was never
/// shown. The other three providers all encode documents; this one silently discarded them.
@Suite("Ollama refuses document attachments")
struct OllamaDocumentRefusalTests {

    private func makeProvider() -> OllamaProvider {
        OllamaProvider(
            configuration: ModelConfiguration(name: "gemma", providerID: "p", modelID: "gemma3:27b"),
            provider: ModelProvider(id: "p", name: "p", apiType: .ollama,
                                    endpoint: URL(string: "http://example.invalid/api")!),
            readAPIKey: { "" }
        )
    }

    @Test("A message carrying a document throws instead of silently dropping it")
    func documentThrows() async throws {
        let pdf = LLMDocumentContent(data: Data("%PDF-1.4".utf8), mimeType: "application/pdf")
        let message = LLMMessage.user("Summarize this", images: [], documents: [pdf])
        await #expect(throws: LLMProviderError.self) {
            _ = try await makeProvider().send(messages: [message], tools: [])
        }
    }

    /// The endpoint is unreachable by construction, so a message WITHOUT documents must fail for
    /// a transport reason — proving the guard above fired on the documents, not on the URL.
    @Test("The refusal names documents, and a document-free message gets past it")
    func refusalIsAboutDocumentsNotTransport() async throws {
        let pdf = LLMDocumentContent(data: Data("%PDF-1.4".utf8), mimeType: "application/pdf")
        do {
            _ = try await makeProvider().send(messages: [.user("hi", images: [], documents: [pdf])], tools: [])
            Issue.record("expected a refusal")
        } catch let error as LLMProviderError {
            guard case .invalidRequest(let detail) = error else {
                Issue.record("expected .invalidRequest, got \(error)"); return
            }
            #expect(detail.lowercased().contains("document"))
        }

        // Same call, no documents: must NOT be .invalidRequest (it dies at the network instead).
        do {
            _ = try await makeProvider().send(messages: [.user("hi")], tools: [])
            Issue.record("unreachable endpoint should have failed")
        } catch let error as LLMProviderError {
            if case .invalidRequest = error {
                Issue.record("document guard fired on a message with no documents")
            }
        } catch {
            // A transport error is the expected outcome here.
        }
    }
}
