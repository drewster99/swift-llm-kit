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

    // MARK: - Rejection vs failure to ask

    /// A 4xx is the server having read the request and declined; that is a capability fact.
    @Test("A 4xx is a rejection")
    func fourxxIsRejection() {
        for code in [400, 401, 403, 404, 422] {
            #expect(CapabilityProbe.looksLikeRejection(httpError(code)), "\(code) should be a rejection")
        }
    }

    /// Rate limiting says the endpoint is busy, not that the model can't do this. Reading it as a
    /// refusal would record "no tool calling" for a model we merely asked too fast.
    @Test("429 is NOT a rejection — it says busy, not incapable")
    func rateLimitIsNotRejection() {
        #expect(!CapabilityProbe.looksLikeRejection(httpError(429)))
    }

    /// A 5xx is the endpoint breaking, which says nothing about the model.
    @Test("5xx and transport errors are not rejections")
    func serverAndTransportAreNotRejections() {
        #expect(!CapabilityProbe.looksLikeRejection(httpError(500)))
        #expect(!CapabilityProbe.looksLikeRejection(httpError(503)))
        #expect(!CapabilityProbe.looksLikeRejection(URLError(.timedOut)))
        #expect(!CapabilityProbe.looksLikeRejection(URLError(.notConnectedToInternet)))
        #expect(!CapabilityProbe.looksLikeRejection(LLMProviderError.invalidResponse))
    }

    /// The status code alone loses the finding: "tools are not supported for this model" and
    /// "unknown parameter 'tool_choice'" are different answers behind the same 400.
    @Test("Rejection detail keeps the endpoint's own words")
    func rejectionDetailKeepsBody() {
        let detail = CapabilityProbe.rejectionDetail(
            LLMProviderError.httpError(statusCode: 400, body: "tools are not supported for this model", url: nil, retryAfter: nil)
        )
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

    private func httpError(_ code: Int) -> LLMProviderError {
        .httpError(statusCode: code, body: "body", url: nil, retryAfter: nil)
    }
}
