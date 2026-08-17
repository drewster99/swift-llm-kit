import Foundation
import Testing
@testable import SwiftLLMKit

/// 0.0.182 — `LLMResponse.finishReason` populated by every provider adapter,
/// plus the `hitOutputTokenLimit` accessor that normalizes across them.
///
/// Only the OpenAI-compatible adapter surfaced `finishReason` before this;
/// Anthropic, Gemini, and Ollama parsed their response bodies and dropped the
/// field. A caller asking "was this generation cut off?" therefore got `nil`
/// from three of the four adapters and could only infer truncation from token
/// counts.
///
/// The field stays verbatim per provider (documented on `LLMResponse`), so the
/// vocabulary differs — OpenAI/Ollama `"length"`, Anthropic `"max_tokens"`,
/// Gemini `"MAX_TOKENS"`. `hitOutputTokenLimit` owns that table so callers
/// don't each re-derive it.
@Suite("0.0.182: finishReason across providers")
struct V0_0_182_FinishReasonTests {

    private static let dummyKey: @Sendable () -> String = { "test" }

    private static func anthropic() throws -> AnthropicProvider {
        let endpoint = try #require(URL(string: "https://api.anthropic.com/v1"))
        return AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(id: "p", name: "p", apiType: .anthropic, endpoint: endpoint),
            readAPIKey: dummyKey
        )
    }

    private static func gemini() throws -> GeminiProvider {
        let endpoint = try #require(URL(string: "https://generativelanguage.googleapis.com/v1beta"))
        return GeminiProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(id: "p", name: "p", apiType: .gemini, endpoint: endpoint),
            readAPIKey: dummyKey
        )
    }

    private static func ollama() throws -> OllamaProvider {
        let endpoint = try #require(URL(string: "http://example.invalid/api"))
        return OllamaProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(id: "p", name: "p", apiType: .ollama, endpoint: endpoint),
            readAPIKey: dummyKey
        )
    }

    private static func openAI() throws -> OpenAICompatibleProvider {
        let endpoint = try #require(URL(string: "https://api.openai.com/v1"))
        return OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(id: "p", name: "p", apiType: .openAICompatible, endpoint: endpoint),
            readAPIKey: dummyKey
        )
    }

    // MARK: - Anthropic

    private static func anthropicBody(stopReason: String) -> String {
        """
        {
          "id": "msg_1",
          "type": "message",
          "role": "assistant",
          "content": [{"type":"text","text":"hi"}],
          "stop_reason": "\(stopReason)",
          "usage": {"input_tokens": 10, "output_tokens": 4000}
        }
        """
    }

    @Test func anthropic_surfacesStopReason() throws {
        let provider = try Self.anthropic()
        let data = try #require(Self.anthropicBody(stopReason: "end_turn").data(using: .utf8))
        let response = try provider.parseResponse(data: data)
        #expect(response.finishReason == "end_turn")
        #expect(response.hitOutputTokenLimit == false)
    }

    @Test func anthropic_maxTokensMeansTruncated() throws {
        let provider = try Self.anthropic()
        let data = try #require(Self.anthropicBody(stopReason: "max_tokens").data(using: .utf8))
        let response = try provider.parseResponse(data: data)
        #expect(response.finishReason == "max_tokens")
        #expect(response.hitOutputTokenLimit)
    }

    @Test func anthropic_missingStopReasonStaysNil() throws {
        // Absent field must not be invented — a caller can't distinguish
        // "finished cleanly" from "provider said nothing" if we guess.
        let provider = try Self.anthropic()
        let json = """
        {
          "id": "msg_1", "type": "message", "role": "assistant",
          "content": [{"type":"text","text":"hi"}]
        }
        """
        let data = try #require(json.data(using: .utf8))
        let response = try provider.parseResponse(data: data)
        #expect(response.finishReason == nil)
        #expect(response.hitOutputTokenLimit == false)
    }

    // MARK: - Gemini

    private static func geminiBody(finishReason: String) -> String {
        """
        {
          "candidates": [{
            "content": {"parts": [{"text": "hi"}], "role": "model"},
            "finishReason": "\(finishReason)"
          }],
          "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20}
        }
        """
    }

    @Test func gemini_surfacesFinishReason() throws {
        let provider = try Self.gemini()
        let data = try #require(Self.geminiBody(finishReason: "STOP").data(using: .utf8))
        let response = try provider.parseResponse(data: data)
        #expect(response.finishReason == "STOP")
        #expect(response.hitOutputTokenLimit == false)
    }

    @Test func gemini_maxTokensMeansTruncated() throws {
        // Gemini shouts its reason; the accessor must fold case.
        let provider = try Self.gemini()
        let data = try #require(Self.geminiBody(finishReason: "MAX_TOKENS").data(using: .utf8))
        let response = try provider.parseResponse(data: data)
        #expect(response.finishReason == "MAX_TOKENS")
        #expect(response.hitOutputTokenLimit)
    }

    @Test func gemini_errorFinishReasonWithoutContentStillCarriesReason() throws {
        // The early-return error path (no content block) must not drop the
        // reason — it's the only clue to why the response is unusable.
        let provider = try Self.gemini()
        let json = """
        {
          "candidates": [{
            "finishReason": "MALFORMED_FUNCTION_CALL",
            "finishMessage": "bad call"
          }]
        }
        """
        let data = try #require(json.data(using: .utf8))
        let response = try provider.parseResponse(data: data)
        #expect(response.finishReason == "MALFORMED_FUNCTION_CALL")
        #expect(response.hitOutputTokenLimit == false)
    }

    // MARK: - Ollama

    private static func ollamaBody(doneReason: String) -> String {
        """
        {
          "model": "m",
          "message": {"role": "assistant", "content": "hi"},
          "done": true,
          "done_reason": "\(doneReason)",
          "prompt_eval_count": 10,
          "eval_count": 20
        }
        """
    }

    @Test func ollama_surfacesDoneReason() throws {
        let provider = try Self.ollama()
        let data = try #require(Self.ollamaBody(doneReason: "stop").data(using: .utf8))
        let response = try provider.parseResponse(data: data)
        #expect(response.finishReason == "stop")
        #expect(response.hitOutputTokenLimit == false)
    }

    @Test func ollama_lengthMeansTruncated() throws {
        let provider = try Self.ollama()
        let data = try #require(Self.ollamaBody(doneReason: "length").data(using: .utf8))
        let response = try provider.parseResponse(data: data)
        #expect(response.finishReason == "length")
        #expect(response.hitOutputTokenLimit)
    }

    // MARK: - OpenAI (pre-existing behavior, pinned against regression)

    @Test func openAI_lengthMeansTruncated() throws {
        let provider = try Self.openAI()
        let json = """
        {
          "id": "chatcmpl-1",
          "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "length"
          }],
          "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        """
        let data = try #require(json.data(using: .utf8))
        let response = try provider.parseResponse(data: data)
        #expect(response.finishReason == "length")
        #expect(response.hitOutputTokenLimit)
    }

    // MARK: - The normalizing accessor

    @Test func hitOutputTokenLimit_coversEveryProviderVocabulary() {
        #expect(LLMResponse(finishReason: "length").hitOutputTokenLimit)      // OpenAI, Ollama
        #expect(LLMResponse(finishReason: "max_tokens").hitOutputTokenLimit)  // Anthropic
        #expect(LLMResponse(finishReason: "MAX_TOKENS").hitOutputTokenLimit)  // Gemini
    }

    @Test func hitOutputTokenLimit_isFalseForCleanAndUnrelatedStops() {
        #expect(LLMResponse(finishReason: "stop").hitOutputTokenLimit == false)
        #expect(LLMResponse(finishReason: "end_turn").hitOutputTokenLimit == false)
        #expect(LLMResponse(finishReason: "STOP").hitOutputTokenLimit == false)
        #expect(LLMResponse(finishReason: "tool_use").hitOutputTokenLimit == false)
        #expect(LLMResponse(finishReason: "stop_sequence").hitOutputTokenLimit == false)
        #expect(LLMResponse(finishReason: "refusal").hitOutputTokenLimit == false)
    }

    @Test func hitOutputTokenLimit_isFalseWhenProviderReportedNothing() {
        // Silence isn't evidence of truncation; guessing would make the flag
        // untrustworthy for the callers that gate on it.
        #expect(LLMResponse(finishReason: nil).hitOutputTokenLimit == false)
        #expect(LLMResponse().hitOutputTokenLimit == false)
    }
}
