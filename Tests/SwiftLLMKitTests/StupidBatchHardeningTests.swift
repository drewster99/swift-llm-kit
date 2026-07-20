import Testing
import Foundation
@testable import SwiftLLMKit

/// Pins the 2026-07-19 /stupid-batch hardening fixes: non-finite Retry-After, digit-overflow
/// fall-through in the limit extractor, and non-container tool_call arguments that would otherwise
/// raise an uncatchable JSONSerialization NSException.
@Suite("Stupid-batch hardening")
struct StupidBatchHardeningTests {

    // #4 — parseRetryAfter rejects non-finite values.
    @Test("parseRetryAfter rejects inf/1e999/nan, keeps finite deltas")
    func retryAfterNonFinite() {
        #expect(LLMProviderError.parseRetryAfter("inf") == nil)
        #expect(LLMProviderError.parseRetryAfter("infinity") == nil)
        #expect(LLMProviderError.parseRetryAfter("1e999") == nil)   // overflows to +inf
        #expect(LLMProviderError.parseRetryAfter("nan") == nil)
        #expect(LLMProviderError.parseRetryAfter("120") == 120)
        #expect(LLMProviderError.parseRetryAfter("1.5") == 1.5)
    }

    // #5 — an overflowing digit run in the first pattern falls through to a later valid pattern.
    @Test("A limit extractor overflow falls through instead of returning nil")
    func firstCaptureOverflowFallsThrough() {
        // reportedLimitHint tries the 范围[..] pattern first, then range[..]. An unparseable
        // (overflowing) first range must not short-circuit the whole list.
        let body = "范围[1, 999999999999999999999999999], range[0, 65536]"
        #expect(LLMProviderError.reportedLimitHint(inBody: body) == 65536)
    }

    // #9 — a tool_call whose `arguments` is a JSON non-container (null/number/bool) must not crash;
    // it defaults to "{}". (isValidJSONObject gates the uncatchable JSONSerialization exception.)
    @Test("Non-container tool_call arguments default to {} without crashing")
    func nonContainerToolArgumentsDefault() throws {
        let provider = OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(id: "p", name: "p", apiType: .openAICompatible,
                                    endpoint: URL(string: "http://example.invalid/v1")!),
            readAPIKey: { "" },
            behaviorFlags: BehaviorFlags()
        )
        for rawArgs in ["42", "null", "true"] {
            let body = """
            {"choices":[{"message":{"role":"assistant","tool_calls":[
              {"id":"x","function":{"name":"foo","arguments":\(rawArgs)}}]}}]}
            """
            let response = try provider.parseResponse(data: Data(body.utf8))
            #expect(response.toolCalls.count == 1)
            #expect(response.toolCalls.first?.arguments == "{}", "args \(rawArgs) should default to {}")
        }
    }
}
