import Foundation
import Testing
@testable import SwiftLLMKit

/// 0.0.206 — a content-policy refusal spelled as a finish/stop reason on an HTTP 200 is thrown as
/// the same typed `responseFailed` the Responses endpoint reports (0.0.205's `cyber_policy`), so a
/// retry policy classifies every provider's refusal from one accessor instead of retrying a
/// withheld answer until its budget runs out.
@Suite("0.0.206: content-policy refusals across providers")
struct V0_0_206_ContentPolicyRefusalTests {

    private static let dummyKey: @Sendable () -> String = { "test" }

    private static func provider<P>(_ apiType: ProviderAPIType, _ endpoint: String, _ make: (ModelConfiguration, ModelProvider, @escaping @Sendable () -> String) -> P) throws -> P {
        let url = try #require(URL(string: endpoint))
        return make(
            ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            ModelProvider(id: "p", name: "p", apiType: apiType, endpoint: url),
            dummyKey)
    }

    private static func expectRefusal(_ code: LLMProviderError.ContentPolicyRefusalCode, _ body: () throws -> LLMResponse) {
        do {
            _ = try body()
            Issue.record("expected a throw for \(code.rawValue)")
        } catch let error as LLMProviderError {
            #expect(error.contentPolicyRefusal == code)
        } catch {
            Issue.record("unexpected error type \(error)")
        }
    }

    @Test func openAI_contentFilterIsARefusal() throws {
        let provider = try Self.provider(.openAICompatible, "https://api.openai.com/v1") {
            OpenAICompatibleProvider(configuration: $0, provider: $1, readAPIKey: $2)
        }
        let json = """
        {"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":null},"finish_reason":"content_filter"}]}
        """
        Self.expectRefusal(.contentFilter) { try provider.parseResponse(data: try #require(json.data(using: .utf8))) }
    }

    @Test func openAI_ordinaryStopsStillParse() throws {
        let provider = try Self.provider(.openAICompatible, "https://api.openai.com/v1") {
            OpenAICompatibleProvider(configuration: $0, provider: $1, readAPIKey: $2)
        }
        let json = """
        {"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}
        """
        let response = try provider.parseResponse(data: try #require(json.data(using: .utf8)))
        #expect(response.text == "hi")
    }

    @Test func anthropic_refusalStopReasonIsARefusal() throws {
        let provider = try Self.provider(.anthropic, "https://api.anthropic.com/v1") {
            AnthropicProvider(configuration: $0, provider: $1, readAPIKey: $2)
        }
        let json = """
        {"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"I"}],"stop_reason":"refusal","usage":{"input_tokens":10,"output_tokens":1}}
        """
        Self.expectRefusal(.anthropicRefusal) { try provider.parseResponse(data: try #require(json.data(using: .utf8))) }
    }

    @Test func gemini_safetyFinishReasonIsARefusal() throws {
        let provider = try Self.provider(.gemini, "https://generativelanguage.googleapis.com/v1beta") {
            GeminiProvider(configuration: $0, provider: $1, readAPIKey: $2)
        }
        let json = """
        {"candidates":[{"finishReason":"SAFETY","safetyRatings":[]}]}
        """
        Self.expectRefusal(.geminiSafety) { try provider.parseResponse(data: try #require(json.data(using: .utf8))) }
    }

    @Test func gemini_blockedPromptIsARefusalNotAMalformedBody() throws {
        let provider = try Self.provider(.gemini, "https://generativelanguage.googleapis.com/v1beta") {
            GeminiProvider(configuration: $0, provider: $1, readAPIKey: $2)
        }
        let json = """
        {"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"},"usageMetadata":{"promptTokenCount":10}}
        """
        Self.expectRefusal(.geminiProhibitedContent) { try provider.parseResponse(data: try #require(json.data(using: .utf8))) }
    }

    @Test func gemini_otherErrorFinishReasonsStillReturnText() throws {
        let provider = try Self.provider(.gemini, "https://generativelanguage.googleapis.com/v1beta") {
            GeminiProvider(configuration: $0, provider: $1, readAPIKey: $2)
        }
        let json = """
        {"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"bad call"}]}
        """
        let response = try provider.parseResponse(data: try #require(json.data(using: .utf8)))
        #expect(response.finishReason == "MALFORMED_FUNCTION_CALL")
    }

    @Test func errorObjectCodesAreNotFinishReasons() {
        #expect(LLMProviderError.contentPolicyRefusal(finishReason: "cyber_policy") == nil)
        #expect(LLMProviderError.contentPolicyRefusal(finishReason: "stop") == nil)
        #expect(LLMProviderError.contentPolicyRefusal(finishReason: nil) == nil)
        #expect(LLMProviderError.contentPolicyRefusal(finishReason: "content_filter") == .contentFilter)
    }
}
