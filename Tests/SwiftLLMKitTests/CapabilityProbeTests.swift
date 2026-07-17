import Testing
import Foundation
@testable import SwiftLLMKit

/// The probe exists to produce evidence rather than repeat claims, so its grading has to be
/// stricter about what counts as knowing something than the rest of the codebase is.
@Suite("Capability probe grading")
struct CapabilityProbeGradingTests {

    // MARK: - Verdict -> Bool?

    /// The whole point of the tri-state: a probe that couldn't reach a model has learned nothing,
    /// and must not be recorded as having learned "no". Writing false here would be worse than
    /// writing nothing, because it looks like a measurement.
    @Test("An inconclusive probe yields nil, never false")
    func inconclusiveIsNilNotFalse() {
        #expect(makeResult(.inconclusive).toolUse == nil)
    }

    @Test("Only the two positive verdicts prove tool calling")
    func positiveVerdicts() {
        #expect(makeResult(.roundTripCompleted).toolUse == true)
        // Emitting the call is proof of tool calling even if the round-trip didn't finish.
        #expect(makeResult(.toolCallOnly).toolUse == true)
    }

    @Test("Refusing to call, or being refused, both mean no")
    func negativeVerdicts() {
        #expect(makeResult(.noToolCall).toolUse == false)
        #expect(makeResult(.rejected).toolUse == false)
    }

    // MARK: - Classifying a failure

    /// A refusal that names tools is the endpoint answering the question we asked.
    @Test("A 4xx naming tools is a capability answer")
    func refusalNamingToolsIsCapabilityAnswer() {
        #expect(CapabilityProbe.classifyFailure(
            httpError(400, "tools are not supported for this model")) == .refusedTools)
        #expect(CapabilityProbe.classifyFailure(
            httpError(400, "Unknown parameter: 'function_call'")) == .refusedTools)
    }

    /// The regression that matters. On the probe's first live run it sent `temperature: 0` to
    /// claude-fable-5, which replied 400 "temperature is deprecated for this model" — and an
    /// any-4xx-is-a-rejection rule recorded "claude-fable-5 cannot call tools". A flat lie about a
    /// flagship model, manufactured by our own request. A refusal we provoked is not evidence.
    @Test("A 4xx about our own parameters is NOT a capability answer")
    func refusalAboutOurRequestIsNotEvidence() {
        #expect(CapabilityProbe.classifyFailure(
            httpError(400, #"{"error":{"message":"`temperature` is deprecated for this model."}}"#))
            == .refusedOurRequest)
        #expect(CapabilityProbe.classifyFailure(httpError(401, "invalid x-api-key")) == .refusedOurRequest)
        #expect(CapabilityProbe.classifyFailure(httpError(404, "model not found")) == .refusedOurRequest)
    }

    /// Rate limiting says the endpoint is busy, not that the model is incapable. Reading it as a
    /// refusal would record "no tool calling" for a model we merely asked too fast.
    @Test("429 says busy, not incapable")
    func rateLimitIsNoAnswer() {
        #expect(CapabilityProbe.classifyFailure(httpError(429, "rate limit exceeded")) == .noAnswer)
    }

    /// A 5xx is the endpoint breaking; a timeout is us not reaching it. Neither says anything.
    @Test("5xx and transport failures establish nothing")
    func serverAndTransportAreNoAnswer() {
        #expect(CapabilityProbe.classifyFailure(httpError(500, "internal error")) == .noAnswer)
        #expect(CapabilityProbe.classifyFailure(httpError(503, "overloaded")) == .noAnswer)
        #expect(CapabilityProbe.classifyFailure(URLError(.timedOut)) == .noAnswer)
        #expect(CapabilityProbe.classifyFailure(URLError(.notConnectedToInternet)) == .noAnswer)
        #expect(CapabilityProbe.classifyFailure(LLMProviderError.invalidResponse) == .noAnswer)
    }

    /// The status code alone loses the finding: "tools are not supported" and "temperature is
    /// deprecated" are entirely different answers behind the same 400 — which is exactly how the
    /// claude-fable-5 false negative was spotted.
    @Test("Rejection detail keeps the endpoint's own words")
    func rejectionDetailKeepsBody() {
        let detail = CapabilityProbe.rejectionDetail(
            httpError(400, "tools are not supported for this model"))
        #expect(detail.contains("400"))
        #expect(detail.contains("tools are not supported"))
    }

    // MARK: - The probe's own inputs

    /// A fixed identifier could be echoed straight from the prompt, or parroted from training
    /// data, and would be indistinguishable from having actually called the tool.
    @Test("Identifiers are random, 9 chars, and visually unambiguous")
    func identifierShape() {
        let ids = (0..<50).map { _ in CapabilityProbe.makeIdentifier() }
        for id in ids {
            #expect(id.count == 9)
            // O/0 and I/1 are excluded so a verdict is never lost to reading a log wrong.
            #expect(!id.contains("O") && !id.contains("0") && !id.contains("I") && !id.contains("1"))
            #expect(id.allSatisfy { $0.isUppercase || $0.isNumber })
        }
        #expect(Set(ids).count > 45, "identifiers must not repeat across runs")
    }

    /// Parameters the model has to construct add a way to fail that has nothing to do with
    /// whether it can call a tool — which is the only thing being measured.
    @Test("The probe tool takes no parameters")
    func probeToolIsParameterless() {
        let tool = CapabilityProbe.makeProbeTool()
        #expect(tool.name == CapabilityProbe.probeToolName)
        #expect(tool.parameters["type"] == .string("object"))
        #expect(tool.parameters["properties"] == .dictionary([:]))
        #expect(tool.parameters["required"] == .array([]))
    }

    // MARK: - Helpers

    private func makeResult(_ verdict: CapabilityProbe.ToolCallVerdict) -> CapabilityProbe.ToolCallResult {
        CapabilityProbe.ToolCallResult(
            providerID: "p", modelID: "m", verdict: verdict, toolChoiceForced: true,
            expectedIdentifier: "ABC234XYZ", returnedText: nil, calledTools: [],
            errorDescription: nil, duration: 0
        )
    }

    private func httpError(_ code: Int, _ body: String = "body") -> LLMProviderError {
        .httpError(statusCode: code, body: body, url: nil, retryAfter: nil)
    }
}
