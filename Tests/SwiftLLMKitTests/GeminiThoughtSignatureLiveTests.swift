import Foundation
import Testing
@testable import SwiftLLMKit

/// Live-API regression test for Gemini 3.x strict thought-signature
/// validation (hydra-style cross-model rotation).
///
/// Gemini 3 models require the first `functionCall` part in each step of the
/// current turn to carry a `thought_signature`; without one the API rejects
/// the request with 400 INVALID_ARGUMENT ("Function call is missing a
/// thought_signature in functionCall parts"). An assistant tool-call turn
/// recorded from a NON-Gemini model has no signature to replay, so the
/// encoder must attach Google's documented validation-bypass dummy signature
/// (https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures).
///
/// Runs only when `GEMINI_API_KEY` is set in the environment — it performs a
/// real network call. Without the dummy-signature fix in
/// `GeminiProvider.encodeContent`, this test fails with the 400 above.
@Suite(
    "Gemini thought-signature live",
    .enabled(if: ProcessInfo.processInfo.environment["GEMINI_API_KEY"]?.isEmpty == false)
)
struct GeminiThoughtSignatureLiveTests {

    private static let apiKey: @Sendable () -> String = {
        ProcessInfo.processInfo.environment["GEMINI_API_KEY"] ?? ""
    }

    private func gemini(modelID: String) throws -> GeminiProvider {
        try GeminiProvider(
            configuration: ModelConfiguration(
                name: "live-test",
                providerID: "gemini",
                modelID: modelID,
                // Thinking models draw thought tokens from the same
                // maxOutputTokens budget as visible output; a small budget
                // risks MAX_TOKENS with an empty candidate, which throws
                // malformedResponse for reasons unrelated to the
                // thought-signature validation this suite exercises.
                maxOutputTokens: 8192
            ),
            provider: ModelProvider(
                id: "gemini",
                name: "Google Gemini",
                apiType: .gemini,
                endpoint: try #require(URL(string: "https://generativelanguage.googleapis.com/v1beta"))
            ),
            readAPIKey: Self.apiKey
        )
    }

    private static let taskAcknowledgedTool = LLMToolDefinition(
        name: "task_acknowledged",
        description: "Acknowledge that the user's task was received.",
        parameters: [
            "type": .string("object"),
            "properties": .dictionary([:])
        ]
    )

    private static let logNoteTool = LLMToolDefinition(
        name: "log_note",
        description: "Record a short note in the task log.",
        parameters: [
            "type": .string("object"),
            "properties": .dictionary([
                "note": .dictionary(["type": .string("string")])
            ])
        ]
    )

    /// Both generations must accept the dummy: 3.5 is the strict-validation
    /// model the fix targets; 2.5 (still routed to by hydra configs) treats
    /// signatures as optional and must not reject the added field.
    /// Override with GEMINI_LIVE_TEST_MODELS (comma-separated model IDs)
    /// as these defaults age out.
    private static let liveModels: [String] = {
        if let override = ProcessInfo.processInfo.environment["GEMINI_LIVE_TEST_MODELS"] {
            let models = override
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            if !models.isEmpty {
                return models
            }
        }
        return ["gemini-3.5-flash", "gemini-2.5-flash"]
    }()

    @Test("foreign (unsigned) assistant tool-call turn is accepted", arguments: liveModels)
    func foreignToolCallTurn_succeeds(modelID: String) async throws {
        // Assistant turn constructed the way MessageEngine reconstructs
        // history recorded from a non-Gemini model: tool calls present,
        // `continuation` nil — no thoughtSignature available to replay.
        let foreignAssistantTurn = LLMMessage(
            _role: .assistant,
            _content: .toolCalls([
                LLMToolCall(id: "call_1", name: "task_acknowledged", arguments: "{}")
            ])
        )

        let messages: [LLMMessage] = [
            .user("Please acknowledge my task using the task_acknowledged tool, then reply with the single word DONE."),
            foreignAssistantTurn,
            .toolResult("acknowledged", callID: "call_1")
        ]

        let response = try await gemini(modelID: modelID).send(
            messages: messages,
            tools: [Self.taskAcknowledgedTool]
        )

        // Any non-throwing response proves strict validation passed; the
        // text assertion just guards against an empty-candidate edge.
        #expect(response.text?.isEmpty == false || !response.toolCalls.isEmpty)
    }

    @Test("foreign (unsigned) PARALLEL tool-call turn is accepted", arguments: liveModels)
    func foreignParallelToolCallTurn_succeeds(modelID: String) async throws {
        // Parallel calls exercise the stricter wire rules: only the FIRST
        // functionCall part carries the (dummy) signature, and the function
        // responses must follow as a group — FC1+sig, FC2, FR1, FR2.
        let foreignAssistantTurn = LLMMessage(
            _role: .assistant,
            _content: .toolCalls([
                LLMToolCall(id: "call_1", name: "task_acknowledged", arguments: "{}"),
                LLMToolCall(id: "call_2", name: "log_note", arguments: #"{"note":"starting"}"#)
            ])
        )

        let messages: [LLMMessage] = [
            .user("Acknowledge my task with task_acknowledged AND log the note 'starting' with log_note, then reply with the single word DONE."),
            foreignAssistantTurn,
            .toolResult("acknowledged", callID: "call_1"),
            .toolResult("logged", callID: "call_2")
        ]

        let response = try await gemini(modelID: modelID).send(
            messages: messages,
            tools: [Self.taskAcknowledgedTool, Self.logNoteTool]
        )

        #expect(response.text?.isEmpty == false || !response.toolCalls.isEmpty)
    }

    @Test("real 2.5-captured continuation replays into strict 3.5 mid-tool-loop")
    func capturedContinuation_rotatesIntoStrictModel() async throws {
        // Harvest a REAL tool-call continuation from 2.5 (toolChoice
        // .required pins the call; 2.5 with thinking usually attaches a
        // genuine thoughtSignature). This validates the actual hydra
        // rotation path — real 2.5 signatures, or a real capture whose bare
        // first functionCall gets the dummy, inside the current turn of a
        // strict 3.5 request. A fabricated signature can't test this.
        let userTurn = LLMMessage.user("Log the note 'starting' using the log_note tool, then reply with the single word DONE.")
        let harvest = try await gemini(modelID: "gemini-2.5-flash").send(
            messages: [userTurn],
            tools: [Self.logNoteTool],
            overrides: LLMCallOverrides(toolChoice: .required)
        )
        guard let call = harvest.toolCalls.first else {
            Issue.record("Harvest returned no tool call despite toolChoice .required — cannot exercise the rotation replay")
            return
        }
        if harvest.continuation?.geminiResponseParts == nil {
            // Still a valid rotation shape (content-shape replay + dummy),
            // but note that the real-signature leg went unexercised.
            print("note: 2.5 harvest carried no thoughtSignature; replay exercises only the dummy path")
        }

        let messages: [LLMMessage] = [
            userTurn,
            .assistant(from: harvest),
            .toolResult("logged", callID: call.id)
        ]

        let response = try await gemini(modelID: "gemini-3.5-flash").send(
            messages: messages,
            tools: [Self.logNoteTool]
        )

        #expect(response.text?.isEmpty == false || !response.toolCalls.isEmpty)
    }
}
