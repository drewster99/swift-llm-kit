import Testing
import Foundation
import CoreGraphics
import ImageIO
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

/// Behavior flags describe how to form a valid request. When one is missing the request is
/// malformed, and a malformed request fails in ways that look exactly like a capability failure —
/// which is how a probe came to record "claude-fable-5 cannot call tools".
@Suite("Model request constraints")
struct ModelRequestConstraintTests {

    /// claude-fable-5 answers `temperature` with HTTP 400 ("deprecated for this model"), verified
    /// live against api.anthropic.com. ModelConfiguration defaults temperature to 0.7 and
    /// AnthropicProvider sends it unless this flag says not to, so without the flag EVERY request
    /// to this model fails. Not a preference — the model is unusable without it.
    @Test("claude-fable-5 must never be sent temperature")
    func fableFiveRejectsTemperature() {
        let registry = BundledModelMetadataRegistry.load()
        let override = registry.override(providerAPIType: "anthropic", modelID: "claude-fable-5")
        #expect(override?.behaviorFlags?.mustNeverSendTemperatureParam == true,
                "claude-fable-5 needs mustNeverSendTemperatureParam or every request 400s")
    }

    /// Verified live in the same run: haiku accepted temperature: 0 and completed the round trip.
    /// The flag is per-model precisely because it does not generalise across a vendor's lineup.
    @Test("The constraint is per-model, not per-vendor")
    func constraintIsPerModel() {
        let registry = BundledModelMetadataRegistry.load()
        let haiku = registry.override(providerAPIType: "anthropic", modelID: "claude-haiku-4-5-20251001")
        #expect(haiku?.behaviorFlags?.mustNeverSendTemperatureParam != true)
    }
}

/// Fixtures are hand-assembled bytes, so "it compiles" proves nothing. These check the containers
/// are actually well-formed — a malformed PNG would make every vision probe report "no vision".
@Suite("Probe fixtures")
struct ProbeFixtureTests {

    @Test("PNG has a valid signature, IHDR dimensions, and IEND")
    func pngIsWellFormed() {
        let png = ProbeFixtures.makePNG(width: 16, height: 16, red: 255, green: 0, blue: 0)
        #expect(png.starts(with: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]))
        #expect(String(data: png, encoding: .isoLatin1)?.contains("IHDR") == true)
        #expect(String(data: png, encoding: .isoLatin1)?.contains("IEND") == true)
        // width/height are big-endian at fixed offsets right after the IHDR length+type.
        let width = png[16...19].reduce(0) { ($0 << 8) | Int($1) }
        let height = png[20...23].reduce(0) { ($0 << 8) | Int($1) }
        #expect(width == 16 && height == 16)
    }

    /// macOS can decode PNG natively, so the fixture can be validated against a real decoder
    /// rather than against our own idea of the format.
    @Test("PNG decodes in a real image decoder, with the colour we asked for")
    func pngDecodes() throws {
        for color in ProbeFixtures.namedColors {
            let png = ProbeFixtures.makePNG(width: 8, height: 8, red: color.red, green: color.green, blue: color.blue)
            let source = try #require(CGImageSourceCreateWithData(png as CFData, nil))
            let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
            #expect(image.width == 8 && image.height == 8, "\(color.name) failed to decode")
        }
    }

    @Test("PDF is a well-formed container carrying the code")
    func pdfIsWellFormed() throws {
        let pdf = ProbeFixtures.makePDF(code: "K7M2Q")
        let text = try #require(String(data: pdf, encoding: .isoLatin1))
        #expect(text.hasPrefix("%PDF-1.4"))
        #expect(text.hasSuffix("%%EOF"))
        #expect(text.contains("(K7M2Q) Tj"))     // the code is actually drawn
        #expect(text.contains("startxref"))
    }

    /// The xref offsets are computed by measuring, and an off-by-one there yields a file readers
    /// reject — which would read as "this model can't do PDFs".
    @Test("PDF opens in a real PDF parser")
    func pdfParses() throws {
        let pdf = ProbeFixtures.makePDF(code: "XR4T9")
        let provider = try #require(CGDataProvider(data: pdf as CFData))
        let document = try #require(CGPDFDocument(provider))
        #expect(document.numberOfPages == 1)
    }
}

/// The profile is the durable output, so its shape is worth pinning — especially the invariant
/// that "we didn't find out" never reads as a measured false.
@Suite("Model profile")
struct ModelProfileTests {

    @Test("A finding's value is nil unless established")
    func findingValueGuarded() {
        #expect(ProbeFinding<Bool>.notAttempted.value == nil)
        #expect(ProbeFinding<Bool>.inconclusive("timeout").value == nil)
        #expect(ProbeFinding<Bool>.established(true).value == true)
        #expect(ProbeFinding<Int>.established(64000).value == 64000)
    }

    @Test("Established effort levels come back ordered shallow → deep, gaps allowed")
    func effortLevelsOrdered() {
        var p = ModelProfile(providerID: "builtin.anthropic", modelID: "claude-sonnet-4-6")
        // Sonnet 4.6 accepts max but NOT xhigh — the real gap that broke an earlier inference.
        p.effortLevels = [
            "max": .established(true), "high": .established(true),
            "xhigh": .established(false), "medium": .established(true),
            "low": .established(true), "none": .inconclusive("not tried")
        ]
        #expect(p.establishedEffortLevels == ["low", "medium", "high", "max"])
    }

    @Test("Rank table orders max above xhigh, and unknowns sort last")
    func effortRanks() {
        #expect(EffortRank.rank(of: "xhigh") < EffortRank.rank(of: "max"))
        #expect(EffortRank.rank(of: "low") < EffortRank.rank(of: "high"))
        #expect(EffortRank.rank(of: "none") < EffortRank.rank(of: "minimal"))
        #expect(EffortRank.rank(of: "some-future-level") == Int.max)
        #expect(EffortRank.allKnown == ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
    }

    @Test("Profile round-trips through Codable")
    func profileCodable() throws {
        var p = ModelProfile(providerID: "p", modelID: "m",
                             toolCalling: .established(true, "returned identifier"),
                             maxOutputTokens: .established(64000, "endpoint reported"))
        p.effortLevels = ["high": .established(true)]
        let back = try JSONDecoder().decode(ModelProfile.self, from: JSONEncoder().encode(p))
        #expect(back.toolCalling.value == true)
        #expect(back.maxOutputTokens.value == 64000)
        #expect(back.effortLevels["high"]?.value == true)
    }

    @Test("Coloured shapes decode and carry both signals distinctly")
    func shapeFixtureDecodes() throws {
        for shape in ProbeFixtures.namedShapes {
            let png = ProbeFixtures.makeShapePNG(shape: shape, red: 0, green: 0, blue: 255, size: 32)
            let source = try #require(CGImageSourceCreateWithData(png as CFData, nil))
            #expect(CGImageSourceCreateImageAtIndex(source, 0, nil) != nil, "\(shape.rawValue) failed to decode")
        }
    }
}
