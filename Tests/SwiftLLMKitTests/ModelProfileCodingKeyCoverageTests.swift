import Foundation
import Testing
@testable import SwiftLLMKit

/// `ModelProfile` carries an explicit `CodingKeys` for one reason — mapping
/// `trailingSystemMessage` back to the `trailingSystemTurn` key already on disk — and that one
/// mapping costs the type its synthesized key list. A stored property with a default value (every
/// optional has one) that is missing a case is then silently never persisted, and new probe
/// dimensions land as exactly that shape: an optional `ProbeFinding`, nil until probed.
///
/// These tests turn that silence into a failure.
@Suite("ModelProfile coding-key coverage")
struct ModelProfileCodingKeyCoverageTests {

    /// A profile with EVERY field set to something distinguishable from its default, so a property
    /// missing from `CodingKeys` shows up as an absent key rather than a coincidentally-equal one.
    private func fullyPopulatedProfile() -> ModelProfile {
        ModelProfile(
            providerID: "anthropic",
            modelID: "claude-opus-5",
            probedAt: Date(timeIntervalSince1970: 1_700_000_000),
            displayName: "Claude Opus 5",
            createdAt: Date(timeIntervalSince1970: 1_600_000_000),
            pricing: ModelPricing(base: PricingTier(input: 3, output: 15)),
            deprecatedOn: Date(timeIntervalSince1970: 1_800_000_000),
            maxTemperature: 2,
            samplingDefaults: SamplingDefaults(temperature: 1),
            isFree: false,
            benchmarks: ModelBenchmarks(artificialAnalysis: .init(intelligenceIndex: 70)),
            isAvailable: .established(true, "called it"),
            isAccessDenied: .established(false, "no refusal"),
            maxContextTokens: .decoded(200_000),
            chat: .established(true, "answered"),
            toolCalling: .established(true, "emitted a call"),
            toolResultRoundTrip: .established(true, "used the result"),
            vision: .established(true, "read the code"),
            pdfInput: .established(false, "rejected the part"),
            acceptsTemperature: .established(true, "accepted"),
            maxOutputTokens: .established(64_000, "endpoint said so"),
            maxOutputBoundedByContext: .established(8_192, "context-bound"),
            trailingSystemMessage: .established(true, "echoed the nonce"),
            generalEffortLevels: ["low": .established(true, "accepted")],
            reasoningEffortLevels: ["high": .established(false, "rejected")],
            capabilityFindings: [ModelCapability.structuredOutputJSONObject.rawValue: .established(true, "returned JSON")],
            maxThinkingBudgetTokens: .established(32_000, "largest accepted budget"),
            callCount: 12,
            duration: 34.5
        )
    }

    /// The load-bearing one, and it uses REFLECTION rather than a round trip on purpose.
    ///
    /// A round-trip equality check cannot see this bug: a property missing from `CodingKeys` is
    /// never encoded, decodes back to its declared default, and compares equal to a fixture that
    /// also left it at its default — which a fixture written before that property existed always
    /// does. (Verified: adding an uncovered property leaves a round-trip test green.) Comparing the
    /// encoded key set against `Mirror`'s stored-property labels needs no fixture update at all, so
    /// it still fires for a property added years after this test was written.
    @Test("Every stored property is covered by CodingKeys — checked by reflection, not by fixture")
    func everyStoredPropertyIsEncoded() throws {
        let profile = fullyPopulatedProfile()
        let object = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(profile)) as? [String: Any]
        )
        // The single deliberate divergence between property name and persisted key.
        let persistedKey = ["trailingSystemMessage": "trailingSystemTurn"]
        let expected = Set(Mirror(reflecting: profile).children.compactMap(\.label).map { persistedKey[$0] ?? $0 })
        let missing = expected.subtracting(object.keys).sorted()
        #expect(missing.isEmpty, """
            \(missing.joined(separator: ", ")) is a stored property with no CodingKeys case, so it is \
            silently never persisted. Add it to ModelProfile.CodingKeys.
            """)
    }

    @Test("A fully-populated profile round-trips unchanged")
    func roundTripPreservesEveryField() throws {
        let original = fullyPopulatedProfile()
        let decoded = try JSONDecoder().decode(ModelProfile.self, from: JSONEncoder().encode(original))
        #expect(decoded == original)
    }

    @Test("The trailing-system finding is persisted under its original key, not the renamed property")
    func trailingSystemKeepsItsPersistedKey() throws {
        let encoded = try JSONEncoder().encode(fullyPopulatedProfile())
        let object = try #require(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["trailingSystemTurn"] != nil,
                "the property was renamed; the key was not — probe records on disk already use this one")
        #expect(object["trailingSystemMessage"] == nil)
    }

    /// The migration this rename had to preserve: 11 real Anthropic-gated findings, each costing a
    /// live API call, were already written under the old key before the property was renamed.
    @Test("A probe record written before the rename keeps its established finding")
    func legacyRecordKeepsItsFinding() throws {
        let legacy = """
        {
          "providerID": "anthropic",
          "modelID": "claude-opus-5",
          "probedAt": 1700000000,
          "isAvailable": {"status": "notAttempted"},
          "isAccessDenied": {"status": "notAttempted"},
          "maxContextTokens": {"status": "notAttempted"},
          "chat": {"status": "notAttempted"},
          "toolCalling": {"status": "notAttempted"},
          "toolResultRoundTrip": {"status": "notAttempted"},
          "vision": {"status": "notAttempted"},
          "pdfInput": {"status": "notAttempted"},
          "acceptsTemperature": {"status": "notAttempted"},
          "maxOutputTokens": {"status": "notAttempted"},
          "trailingSystemTurn": {"status": "established", "value": true, "evidence": "echoed the code carried by the trailing system turn", "source": "probed", "duration": 1.73},
          "generalEffortLevels": {},
          "reasoningEffortLevels": {},
          "capabilityFindings": {},
          "callCount": 1,
          "duration": 1.73
        }
        """
        let decoded = try JSONDecoder().decode(ModelProfile.self, from: Data(legacy.utf8))
        #expect(decoded.trailingSystemMessage?.value == true, "the finding must survive the API rename")
        #expect(decoded.trailingSystemMessage?.evidence?.isEmpty == false)
    }

    @Test("A record predating the field entirely still decodes, with the finding absent")
    func recordPredatingTheFieldDecodes() throws {
        var object = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(fullyPopulatedProfile())) as? [String: Any]
        )
        object.removeValue(forKey: "trailingSystemTurn")
        let decoded = try JSONDecoder().decode(
            ModelProfile.self,
            from: try JSONSerialization.data(withJSONObject: object)
        )
        #expect(decoded.trailingSystemMessage == nil, "absent means unprobed, not false")
    }
}
