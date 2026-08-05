import Testing
import Foundation
@testable import SwiftLLMKit

/// The 2026-07-30 `ModelCapabilities` work: 17 loose `Bool` stored properties became a tri-state
/// `[ModelCapability: Bool]` (present = known true/false, absent = unknown). These pin what must hold:
/// the public boolean API stays a 2-state view (unknown reads false) so no existing reader breaks;
/// the tri-state truth is reachable via the subscript; the Codable form serializes only KNOWN
/// capabilities and still reads legacy full-Bool objects; and the filter predicate acts on KNOWN
/// facts only — an unmeasured capability never disqualifies a model.
@Suite("ModelCapabilities tri-state + availability filtering")
struct ModelCapabilityTriStateTests {

    // MARK: - Known / unknown distinction

    @Test("Known-true init: named flags are known-true, the rest are UNKNOWN (not false)")
    func trueInitLeavesRestUnknown() {
        let caps = ModelCapabilities(toolUse: true, vision: true, toolResultRoundTrip: true)
        #expect(caps.toolUse)
        #expect(caps.vision)
        #expect(caps.toolResultRoundTrip)
        #expect(caps.state(of: .toolUse) == true)
        #expect(caps.isKnown(.toolUse))
        // The crux: an unmentioned capability is UNKNOWN, not known-false.
        #expect(caps.state(of: .reasoning) == nil)
        #expect(!caps.isKnown(.reasoning))
        #expect(!caps.reasoning)                 // 2-state view still reads false
        #expect(caps.asSet == [.toolUse, .vision, .toolResultRoundTrip])
        #expect(caps.contains(.toolUse))
        #expect(!caps.contains(.reasoning))
    }

    @Test("Set initializer marks its members known-true, equivalent to the boolean one")
    func setInitMatchesTrueOnlyBoolInit() {
        let fromSet = ModelCapabilities([.reasoning, .pdfInput])
        let fromBools = ModelCapabilities(reasoning: true, pdfInput: true)
        #expect(fromSet == fromBools)
        #expect(fromSet.state(of: .reasoning) == true)
        #expect(fromSet.state(of: .vision) == nil)
    }

    @Test("Subscript reaches all three states; nil clears to unknown")
    func subscriptTriState() {
        var caps = ModelCapabilities()
        #expect(caps[.toolUse] == nil)
        caps[.toolUse] = true
        #expect(caps[.toolUse] == true)
        #expect(caps.toolUse)
        caps[.toolUse] = false
        #expect(caps[.toolUse] == false)
        #expect(caps.isKnown(.toolUse))
        #expect(!caps.toolUse)                   // 2-state view: known-false reads false
        caps[.toolUse] = nil
        #expect(caps[.toolUse] == nil)
        #expect(!caps.isKnown(.toolUse))
        #expect(!caps.toolUse)                   // unknown also reads false
    }

    @Test("Boolean setter records an EXPLICIT true/false, never unknown")
    func boolSetterRecordsExplicit() {
        var caps = ModelCapabilities()
        caps.toolUse = false                     // deliberate statement of false
        #expect(caps.state(of: .toolUse) == false)
        #expect(caps.isKnown(.toolUse))
        #expect(caps.asSet.isEmpty)              // known-false is not "on"
        caps.vision = true
        #expect(caps.state(of: .vision) == true)
        #expect(caps.asSet == [.vision])
    }

    @Test("enabledLabels lists only known-true flags, in canonical case order")
    func enabledLabelsOrder() {
        let caps = ModelCapabilities(toolUse: true, vision: true)
        #expect(caps.enabledLabels == [ModelCapability.toolUse.label, ModelCapability.vision.label])
    }

    @Test("Every capability round-trips through its known-true accessor")
    func everyCaseRoundTrips() {
        for capability in ModelCapability.allCases {
            let caps = ModelCapabilities([capability])
            #expect(caps.contains(capability))
            #expect(caps.state(of: capability) == true)
            #expect(caps.asSet == [capability])
            // `.chat` is the baseline and is deliberately omitted from the label summary.
            #expect(caps.enabledLabels == (capability == .chat ? [] : [capability.label]))
        }
    }

    @Test("enabledLabels omits the .chat baseline but keeps other known-true capabilities")
    func enabledLabelsOmitsChat() {
        let caps = ModelCapabilities([.chat, .toolUse, .vision])
        #expect(caps.enabledLabels == [ModelCapability.toolUse.label, ModelCapability.vision.label])
        #expect(caps.contains(.chat))   // still a real, queryable capability
    }

    @Test("demotingKnownFalse turns known-false to unknown, keeping trues and the exceptions")
    func demotingKnownFalseFailsOpen() {
        var caps = ModelCapabilities(toolUse: true)   // known-true
        caps[.vision] = false                          // known-false (to be demoted)
        caps[.chat] = false                            // known-false but KEPT (reliable)
        // reasoning left unknown
        let demoted = caps.demotingKnownFalse(keeping: [.chat])
        #expect(demoted.state(of: .toolUse) == true)   // known-true survives
        #expect(demoted.state(of: .vision) == nil)     // known-false demoted to unknown
        #expect(demoted.state(of: .chat) == false)     // kept: reliable
        #expect(demoted.state(of: .reasoning) == nil)  // unchanged
    }

    // MARK: - Codable

    @Test("Encodes ONLY known capabilities; unknowns are omitted")
    func encodesOnlyKnownCapabilities() throws {
        var caps = ModelCapabilities(toolUse: true, vision: true)
        caps[.reasoning] = false                 // an explicit known-false is written too
        let data = try JSONEncoder().encode(caps)
        let obj = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Bool])
        #expect(obj["toolUse"] == true)
        #expect(obj["vision"] == true)
        #expect(obj["reasoning"] == false)
        #expect(obj["pdfInput"] == nil)          // unknown ⇒ absent from the wire
        #expect(obj.count == 3)
    }

    @Test("A fresh all-unknown value encodes to an empty object")
    func allUnknownEncodesEmpty() throws {
        let data = try JSONEncoder().encode(ModelCapabilities())
        let obj = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Bool])
        #expect(obj.isEmpty)
    }

    @Test("Decodes a legacy full boolean object as all-known")
    func decodesLegacyFullObject() throws {
        let json = """
        {"toolUse":true,"vision":false,"reasoning":true,"codeExecution":false,"promptCaching":false,
         "computerUse":false,"audioInput":false,"audioOutput":false,"videoInput":false,
         "responseSchema":false,"parallelToolCalls":true,"pdfInput":true,"webSearch":false,
         "systemMessages":true,"assistantPrefill":false,"toolChoice":false,"toolResultRoundTrip":true}
        """
        let caps = try JSONDecoder().decode(ModelCapabilities.self, from: Data(json.utf8))
        #expect(caps.asSet == [.toolUse, .reasoning, .parallelToolCalls, .pdfInput, .systemMessages, .toolResultRoundTrip])
        #expect(caps.state(of: .vision) == false)   // legacy false decodes as KNOWN-false
        #expect(caps.isKnown(.vision))
    }

    @Test("Decodes a partial object: absent keys are UNKNOWN")
    func decodesPartialObjectLeavesRestUnknown() throws {
        let json = #"{"toolUse":true,"reasoning":true}"#
        let caps = try JSONDecoder().decode(ModelCapabilities.self, from: Data(json.utf8))
        #expect(caps.state(of: .toolUse) == true)
        #expect(caps.state(of: .reasoning) == true)
        #expect(caps.state(of: .vision) == nil)     // absent ⇒ unknown, not false
        #expect(caps.asSet == [.toolUse, .reasoning])
    }

    @Test("Round-trips true / false / unknown distinctly")
    func codableRoundTripPreservesTriState() throws {
        var original = ModelCapabilities(toolUse: true)
        original[.vision] = false                    // known-false
        // reasoning left unknown
        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(ModelCapabilities.self, from: data)
        #expect(restored == original)
        #expect(restored.state(of: .toolUse) == true)
        #expect(restored.state(of: .vision) == false)
        #expect(restored.state(of: .reasoning) == nil)
    }

    // MARK: - Availability states

    private func model(isAvailable: Bool? = nil, isAccessDenied: Bool? = nil, deprecatedOn: Date? = nil) -> ModelInfo {
        ModelInfo(
            providerID: "p",
            modelID: "m",
            deprecatedOn: deprecatedOn,
            isAvailable: isAvailable,
            isAccessDenied: isAccessDenied
        )
    }

    @Test("A normal model has no special availability states")
    func normalModelHasNoStates() {
        #expect(model().availabilityStates.isEmpty)
    }

    @Test("isAvailable == false yields .isUnavailable; nil does not")
    func unavailableState() {
        #expect(model(isAvailable: false).availabilityStates == [.isUnavailable])
        #expect(model(isAvailable: nil).availabilityStates.isEmpty)
        #expect(model(isAvailable: true).availabilityStates.isEmpty)
    }

    @Test("isAccessDenied == true yields .isAccessDenied; nil/false do not")
    func accessDeniedState() {
        #expect(model(isAccessDenied: true).availabilityStates == [.isAccessDenied])
        #expect(model(isAccessDenied: false).availabilityStates.isEmpty)
        #expect(model(isAccessDenied: nil).availabilityStates.isEmpty)
    }

    @Test("Future deprecation date is .isFutureDeprecated; past is .isDeprecated")
    func deprecationStates() {
        let future = Date(timeIntervalSinceNow: 86_400)
        let past = Date(timeIntervalSinceNow: -86_400)
        #expect(model(deprecatedOn: future).availabilityStates == [.isFutureDeprecated])
        #expect(model(deprecatedOn: past).availabilityStates == [.isDeprecated])
    }

    @Test(".all never appears in a model's set (future/past deprecation are mutually exclusive)")
    func allNeverPresentOnAModel() {
        let past = Date(timeIntervalSinceNow: -86_400)
        let states = model(isAvailable: false, isAccessDenied: true, deprecatedOn: past).availabilityStates
        #expect(states == [.isUnavailable, .isAccessDenied, .isDeprecated])
        #expect(!states.contains(.all))
    }

    // MARK: - satisfies() filter predicate (tri-state)

    @Test("Required capability rejects ONLY a known-false; unknown and true pass")
    func requiredRejectsOnlyKnownFalse() {
        var known = model(); known.capabilities = ModelCapabilities(toolUse: true)
        #expect(known.satisfies(requiredCapabilities: [.toolUse], mustNotBePresent: [], includedAvailabilityStates: []))

        let unknown = model()   // toolUse unmeasured
        #expect(unknown.satisfies(requiredCapabilities: [.toolUse], mustNotBePresent: [], includedAvailabilityStates: []))

        var denied = model(); denied.capabilities[.toolUse] = false   // KNOWN cannot
        #expect(!denied.satisfies(requiredCapabilities: [.toolUse], mustNotBePresent: [], includedAvailabilityStates: []))
    }

    @Test("Forbidden capability rejects ONLY a known-true; unknown and false pass")
    func forbiddenRejectsOnlyKnownTrue() {
        var hasVision = model(); hasVision.capabilities = ModelCapabilities(vision: true)
        #expect(!hasVision.satisfies(requiredCapabilities: [], mustNotBePresent: [.vision], includedAvailabilityStates: []))

        let unknown = model()   // vision unmeasured
        #expect(unknown.satisfies(requiredCapabilities: [], mustNotBePresent: [.vision], includedAvailabilityStates: []))

        var knownNo = model(); knownNo.capabilities[.vision] = false
        #expect(knownNo.satisfies(requiredCapabilities: [], mustNotBePresent: [.vision], includedAvailabilityStates: []))
    }

    @Test("A normal model passes the availability gate regardless of the include-set")
    func normalModelAlwaysPassesAvailabilityGate() {
        #expect(model().satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: []))
    }

    @Test("A model in a special state is excluded unless that state is included (or .all)")
    func specialStateGate() {
        let unavailable = model(isAvailable: false)
        #expect(!unavailable.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: []))
        #expect(unavailable.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.isUnavailable]))
        #expect(unavailable.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.all]))
        #expect(!unavailable.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.isDeprecated]))
    }

    @Test("A model with multiple special states needs all of them included")
    func multipleStatesNeedAllIncluded() {
        let m = model(isAvailable: false, isAccessDenied: true)
        #expect(!m.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.isUnavailable]))
        #expect(m.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.isUnavailable, .isAccessDenied]))
        #expect(m.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.all]))
    }

    @Test("ModelRequirements bundle is Equatable across its three fields")
    func requirementsBundleEquatable() {
        let a = ModelRequirements(requiredCapabilities: [.toolUse], mustNotBePresent: [.vision], includedAvailabilityStates: [.all])
        let b = ModelRequirements(requiredCapabilities: [.toolUse], mustNotBePresent: [.vision], includedAvailabilityStates: [.all])
        #expect(a == b)
    }
}

/// Keeps `isEmpiricallyProbed` and the "NOT PROBED" doc comments telling the same story.
///
/// Two independent statements of the same fact, deliberately: the property is what code reads, the
/// comment is what a person reads at the definition when deciding whether a capability can be
/// trusted. Either drifting alone is the failure mode — a probe gets written and the comment still
/// warns it does not exist, or a capability is added and neither is filled in.
@Suite("Which capabilities a probe can establish")
struct EmpiricallyProbedCapabilityTests {

    /// The unprobed set, restated independently of the property so a change to either has to be
    /// deliberate. Same reasoning as the wire-string table in `ChannelMessageKindTests`.
    private static let expectedUnprobed: Set<ModelCapability> = [
        .thinkingSupportsTokenBudget, .batch, .codeExecution, .promptCaching, .computerUse,
        .audioInput, .audioOutput, .videoInput, .webSearch, .toolChoiceSupported
    ]

    @Test("The property matches the independently-stated set")
    func propertyMatchesTheTable() {
        let unprobed = Set(ModelCapability.allCases.filter { !$0.isEmpiricallyProbed })
        #expect(unprobed == Self.expectedUnprobed)
    }

    /// Every unprobed capability says so where it is DEFINED — the question is asked while reading
    /// the enum, not while reading the prober.
    @Test("Every unprobed capability is annotated at its definition, and no probed one is")
    func annotationsMatchTheProperty() throws {
        let source = try String(contentsOf: Self.capabilitiesSourceURL, encoding: .utf8)
        for capability in ModelCapability.allCases {
            // The doc comment block immediately preceding `case x = "x"`.
            guard let caseRange = source.range(of: "case \(capability.rawValue) = \"") else {
                Issue.record("no case declaration found for \(capability.rawValue)"); continue
            }
            let preceding = source[source.startIndex..<caseRange.lowerBound]
            // `dropFirst` on the reversed lines discards the partial line the range cuts through —
            // the indentation before `case`, which is not a doc line and would end the walk
            // immediately, making every doc block read as empty.
            let docBlock = preceding.split(separator: "\n", omittingEmptySubsequences: false)
                .reversed().dropFirst()
                .prefix { $0.trimmingCharacters(in: .whitespaces).hasPrefix("///") }
                .joined(separator: "\n")
            let annotated = docBlock.contains("NOT PROBED")
            let detail = "\(capability.rawValue): annotated=\(annotated) but "
                       + "isEmpiricallyProbed=\(capability.isEmpiricallyProbed)"
            #expect(annotated == !capability.isEmpiricallyProbed, "\(detail)")
        }
    }

    private static var capabilitiesSourceURL: URL {
        URL(fileURLWithPath: #filePath)                       // …/Tests/SwiftLLMKitTests/<this>.swift
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/SwiftLLMKit/Models/ModelCapabilities.swift")
    }
}
