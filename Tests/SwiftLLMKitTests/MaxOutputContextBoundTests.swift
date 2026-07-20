import Testing
import Foundation
@testable import SwiftLLMKit

/// The HF-router max-output family (2026-07-19 audit): providers that bound `max_tokens` only by
/// context length or a fixed router parameter ceiling were making the probe record a fabricated
/// output cap — the deepinfra param ceiling (~10M), context-minus-input, or a flatly impossible
/// value above context. These pin the three fixes: broadened context-bound parsing, param-ceiling
/// recognition, and the "never record output ≥ context" sanity guard.
@Suite("Max-output: context-bound and param-ceiling handling")
struct MaxOutputContextBoundTests {

    // MARK: - Pure parse helpers

    @Test("reportedContextLengthBound matches both 'is N' and 'of N' phrasings")
    func contextBoundParsing() {
        #expect(LLMProviderError.reportedContextLengthBound(inBody: "maximum context length is 40960 tokens") == 40960)
        #expect(LLMProviderError.reportedContextLengthBound(inBody:
            "Requested token count exceeds the model's maximum context length of 131072 tokens") == 131072)
        #expect(LLMProviderError.reportedContextLengthBound(inBody: "temperature is out of range") == nil)
    }

    @Test("reportedParameterCeiling catches the router param limit, distinct from a real output cap")
    func paramCeilingParsing() {
        #expect(LLMProviderError.reportedParameterCeiling(inBody: "Input should be less than or equal to 10000000") == 10_000_000)
        #expect(LLMProviderError.reportedParameterCeiling(inBody: "Failed to process request") == nil)
        // A real model cap ("must be between 0 and N") is NOT a param ceiling and IS a max-output limit.
        #expect(LLMProviderError.reportedParameterCeiling(inBody: "max_tokens must be between 0 and 65536") == nil)
        #expect(LLMProviderError.reportedMaxOutputTokenLimit(inBody: "Input should be less than or equal to 10000000") == nil)
    }

    // MARK: - probeMaxOutputTokens end-to-end

    /// deepinfra: the 100M probe is rejected with the router's absolute param ceiling. Must NOT search
    /// (would converge to ~10M) — record context-bound when we know the window, else inconclusive.
    private final class ParamCeilingProvider: LLMProvider, @unchecked Sendable {
        let lock = NSLock(); private var _calls = 0
        var callCount: Int { lock.withLock { _calls } }
        func send(messages: [LLMMessage], tools: [LLMToolDefinition], overrides: LLMCallOverrides) async throws -> LLMResponse {
            lock.withLock { _calls += 1 }
            throw LLMProviderError.httpError(statusCode: 422, body: "Input should be less than or equal to 10000000", url: nil, retryAfter: nil)
        }
    }

    @Test("A param-ceiling rejection records context-bound and never searches (context known)")
    func paramCeilingWithContext() async {
        let provider = ParamCeilingProvider()
        let result = await ModelProber.probeMaxOutputTokens(
            llm: provider, modelID: "gpt-oss-120b:deepinfra", maxContextTokens: 131072)
        #expect(result.cap.status == .inconclusive, "must not fabricate an output cap from a param ceiling")
        #expect(result.contextBound?.value == 131072)
        #expect(provider.callCount == 1, "the single 100M probe — no binary search")
    }

    @Test("A param-ceiling rejection with unknown context is inconclusive, still no search")
    func paramCeilingWithoutContext() async {
        let provider = ParamCeilingProvider()
        let result = await ModelProber.probeMaxOutputTokens(
            llm: provider, modelID: "x:deepinfra", maxContextTokens: nil)
        #expect(result.cap.status == .inconclusive)
        #expect(result.contextBound == nil)
        #expect(provider.callCount == 1)
    }

    /// together/HF-router: the 100M probe returns an explicit context message. One call, context-bound.
    private struct ContextMessageProvider: LLMProvider {
        func send(messages: [LLMMessage], tools: [LLMToolDefinition], overrides: LLMCallOverrides) async throws -> LLMResponse {
            throw LLMProviderError.httpError(statusCode: 400,
                body: "Requested token count exceeds the model's maximum context length of 131072 tokens", url: nil, retryAfter: nil)
        }
    }

    @Test("An explicit 'maximum context length of N' settles context-bound in one call")
    func explicitContextMessage() async {
        let result = await ModelProber.probeMaxOutputTokens(
            llm: ContextMessageProvider(), modelID: "gpt-oss-20b:together", maxContextTokens: 131072)
        #expect(result.contextBound?.value == 131072)
        #expect(result.cap.status == .inconclusive)
    }

    /// nscale-style: rejects above the context-minus-input ceiling with only a GENERIC error (no
    /// parseable message). The search converges to context-minus-input; the sanity guard must catch
    /// that it's within an input-sized margin of context and record context-bound, not the number.
    private struct GenericMasqueradeProvider: LLMProvider {
        let ceiling: Int   // = context - input tokens
        func send(messages: [LLMMessage], tools: [LLMToolDefinition], overrides: LLMCallOverrides) async throws -> LLMResponse {
            if (overrides.maxOutputTokens ?? 512) > ceiling {
                throw LLMProviderError.httpError(statusCode: 400, body: "Input validation error", url: nil, retryAfter: nil)
            }
            return LLMResponse(text: "ok", toolCalls: [], reasoning: nil, usage: nil, continuation: nil)
        }
    }

    @Test("A generic-error search that reaches context-minus-input is recorded context-bound, not a cap")
    func genericMasqueradeGuarded() async {
        // context 40960, input ~205 → the endpoint accepts up to 40755, i.e. context-minus-input.
        let result = await ModelProber.probeMaxOutputTokens(
            llm: GenericMasqueradeProvider(ceiling: 40755), modelID: "Qwen3-32B:deepinfra", maxContextTokens: 40960)
        #expect(result.cap.status == .inconclusive, "context-minus-input must not be recorded as an output cap")
        #expect(result.contextBound?.value == 40960)
    }

    @Test("Without a known context window the same search still returns a value (best effort, no guard)")
    func noContextNoGuard() async {
        let result = await ModelProber.probeMaxOutputTokens(
            llm: GenericMasqueradeProvider(ceiling: 40755), modelID: "x", maxContextTokens: nil)
        // No context to compare against, so the guard can't fire — the search value stands.
        #expect(result.cap.status == .established)
        #expect((result.cap.value ?? 0) <= 40755)
    }

    /// codex review: a param ceiling BELOW the context window is a genuine per-model output cap
    /// (a validator enforcing the real max_tokens), not the router's absolute limit — record it.
    private struct ParamCeilingBelowContextProvider: LLMProvider {
        func send(messages: [LLMMessage], tools: [LLMToolDefinition], overrides: LLMCallOverrides) async throws -> LLMResponse {
            throw LLMProviderError.httpError(statusCode: 422, body: "Input should be less than or equal to 8192", url: nil, retryAfter: nil)
        }
    }

    @Test("A param ceiling below the context window is recorded as the real output cap")
    func paramCeilingBelowContextIsRealCap() async {
        let result = await ModelProber.probeMaxOutputTokens(
            llm: ParamCeilingBelowContextProvider(), modelID: "m", maxContextTokens: 131072)
        #expect(result.cap.status == .established)
        #expect(result.cap.value == 8192)
        #expect(result.contextBound == nil)
    }

    /// codex review: the fixed 2048 margin must not discard a genuine cap on a SMALL context window.
    /// A 4096-context model with a real 2048 output cap (gap 2048) must survive as an output cap.
    @Test("A genuine cap on a small context window is not over-guarded (proportional margin)")
    func smallContextGenuineCapSurvives() async {
        let result = await ModelProber.probeMaxOutputTokens(
            llm: GenericMasqueradeProvider(ceiling: 2048), modelID: "m", maxContextTokens: 4096)
        #expect(result.cap.status == .established)
        #expect((result.cap.value ?? 0) <= 2048)
        #expect((result.cap.value ?? 0) >= 2048 - max(2, 2048 / 100))
        #expect(result.contextBound == nil, "a half-of-context cap on a 4096 window must not read as context-bound")
    }

    @Test("A genuine output cap well below context is still recorded as an output cap, not guarded away")
    func genuineCapSurvives() async {
        // ceiling 4096 with a 131072 context — gap is far larger than the input margin, so it's a real
        // per-model cap (cohere/groq style), not a masquerade.
        let result = await ModelProber.probeMaxOutputTokens(
            llm: GenericMasqueradeProvider(ceiling: 4096), modelID: "command-r7b", maxContextTokens: 131072)
        #expect(result.cap.status == .established)
        #expect((result.cap.value ?? 0) <= 4096)
        #expect((result.cap.value ?? 0) >= 4096 - max(2, 4096 / 100))
        #expect(result.contextBound == nil, "a genuine cap must not be reclassified as context-bound")
    }
}
