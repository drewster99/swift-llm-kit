import Foundation
import Testing
@testable import SwiftLLMKit

/// Coverage for the cross-provider `TokenUsage` parsing in each provider's
/// `parseResponse`. Locks in two recent behavioral changes:
///
/// - **0.0.17:** `reasoningTokens` surfaced for OpenAI (`completion_tokens_details.reasoning_tokens`)
///   and Gemini (`usageMetadata.thoughtsTokenCount`).
/// - **0.0.19:** `inputTokens` normalized so Anthropic reports the FULL prompt
///   (uncached + cache_read + cache_write), matching OpenAI / Gemini semantics.
///   Without this, `cacheReadTokens / inputTokens` as a hit-rate computation
///   would mean different things per provider.
///
/// Also asserts Codable backward-compat: legacy persisted `TokenUsage` JSON
/// without `reasoningTokens` still decodes cleanly (defaults to 0).
@Suite("TokenUsage parsing across providers")
struct TokenUsageParserTests {

    private static let dummyKey: @Sendable () -> String = { "test" }

    private static func anthropic() throws -> AnthropicProvider {
        let endpoint = try #require(URL(string: "https://api.anthropic.com/v1"))
        return AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(id: "p", name: "p", apiType: .anthropic, endpoint: endpoint),
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
    private static func gemini() throws -> GeminiProvider {
        let endpoint = try #require(URL(string: "https://generativelanguage.googleapis.com/v1beta"))
        return GeminiProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(id: "p", name: "p", apiType: .gemini, endpoint: endpoint),
            readAPIKey: dummyKey
        )
    }

    // MARK: - Anthropic — inputTokens normalization

    @Test func anthropic_inputTokens_includesCacheReadAndWrite() throws {
        // Wire reports uncached=100, cache_read=2000, cache_creation=300.
        // Pre-0.0.19 we'd surface inputTokens=100 (wire passthrough).
        // 0.0.19+: inputTokens=100+2000+300=2400 (full prompt input).
        let json = """
        {
          "id": "msg_1",
          "type": "message",
          "role": "assistant",
          "content": [{"type":"text","text":"hi"}],
          "stop_reason": "end_turn",
          "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 2000,
            "cache_creation_input_tokens": 300
          }
        }
        """
        let response = try Self.anthropic().parseResponse(data: Data(json.utf8))
        let usage = try #require(response.usage)
        #expect(usage.inputTokens == 2400, "inputTokens should be the FULL prompt (uncached + read + write)")
        #expect(usage.cacheReadTokens == 2000)
        #expect(usage.cacheWriteTokens == 300)
        #expect(usage.outputTokens == 50)
        // Hit rate math works uniformly: cacheRead / inputTokens.
        let hitRate = Double(usage.cacheReadTokens) / Double(usage.inputTokens)
        #expect(abs(hitRate - 2000.0 / 2400.0) < 0.0001)
    }

    @Test func anthropic_inputTokens_noCache_isJustUncached() throws {
        let json = """
        {
          "id": "msg_1",
          "type": "message",
          "role": "assistant",
          "content": [{"type":"text","text":"hi"}],
          "stop_reason": "end_turn",
          "usage": {"input_tokens": 500, "output_tokens": 20}
        }
        """
        let response = try Self.anthropic().parseResponse(data: Data(json.utf8))
        let usage = try #require(response.usage)
        #expect(usage.inputTokens == 500)
        #expect(usage.cacheReadTokens == 0)
        #expect(usage.cacheWriteTokens == 0)
    }

    @Test func anthropic_reasoningTokens_alwaysZero() throws {
        // Anthropic folds thinking into output_tokens, never reports it separately.
        let json = """
        {
          "id": "msg_1", "type": "message", "role": "assistant",
          "content": [{"type":"text","text":"hi"}],
          "stop_reason": "end_turn",
          "usage": {"input_tokens": 100, "output_tokens": 500}
        }
        """
        let response = try Self.anthropic().parseResponse(data: Data(json.utf8))
        #expect(response.usage?.reasoningTokens == 0)
    }

    // MARK: - OpenAI — reasoning tokens

    @Test func openAI_reasoningTokens_parsedFromCompletionTokensDetails() throws {
        let json = """
        {
          "id": "chatcmpl-1",
          "object": "chat.completion",
          "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
          }],
          "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 60,
            "completion_tokens_details": {"reasoning_tokens": 40},
            "prompt_tokens_details": {"cached_tokens": 150}
          }
        }
        """
        let response = try Self.openAI().parseResponse(data: Data(json.utf8))
        let usage = try #require(response.usage)
        #expect(usage.inputTokens == 200)             // full prompt (wire passthrough)
        #expect(usage.outputTokens == 60)
        #expect(usage.reasoningTokens == 40)
        #expect(usage.cacheReadTokens == 150)
    }

    @Test func openAI_noReasoningOrCacheDetails_zeros() throws {
        let json = """
        {
          "id": "chatcmpl-1", "object": "chat.completion",
          "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
          }],
          "usage": {"prompt_tokens": 50, "completion_tokens": 10}
        }
        """
        let response = try Self.openAI().parseResponse(data: Data(json.utf8))
        #expect(response.usage?.reasoningTokens == 0)
        #expect(response.usage?.cacheReadTokens == 0)
    }

    // MARK: - Gemini — reasoning tokens

    @Test func gemini_reasoningTokens_parsedFromThoughtsTokenCount() throws {
        let json = """
        {
          "candidates": [{
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "finishReason": "STOP"
          }],
          "usageMetadata": {
            "promptTokenCount": 1500,
            "candidatesTokenCount": 30,
            "cachedContentTokenCount": 1200,
            "thoughtsTokenCount": 800,
            "totalTokenCount": 2330
          }
        }
        """
        let response = try Self.gemini().parseResponse(data: Data(json.utf8))
        let usage = try #require(response.usage)
        #expect(usage.inputTokens == 1500)
        #expect(usage.outputTokens == 30)
        #expect(usage.reasoningTokens == 800)
        #expect(usage.cacheReadTokens == 1200)
    }

    @Test func gemini_noThoughts_reasoningIsZero() throws {
        let json = """
        {
          "candidates": [{
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "finishReason": "STOP"
          }],
          "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10, "totalTokenCount": 60}
        }
        """
        let response = try Self.gemini().parseResponse(data: Data(json.utf8))
        #expect(response.usage?.reasoningTokens == 0)
        #expect(response.usage?.cacheReadTokens == 0)
    }

    // MARK: - Codable backward-compat

    @Test func tokenUsage_legacyJSONWithoutReasoningTokens_decodesAsZero() throws {
        // A previously-persisted TokenUsage missing the new reasoningTokens field
        // (added in 0.0.17) must still decode — we use decodeIfPresent.
        let legacy = """
        {
          "inputTokens": 100,
          "outputTokens": 50,
          "cacheReadTokens": 0,
          "cacheWriteTokens": 0
        }
        """
        let decoded = try JSONDecoder().decode(TokenUsage.self, from: Data(legacy.utf8))
        #expect(decoded.reasoningTokens == 0)
        #expect(decoded.inputTokens == 100)
    }
}
