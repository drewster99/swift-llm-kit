import Testing
import Foundation
@testable import SwiftLLMKit

/// The 2026-07-30 `ModelCapabilities` refactor: 17 loose `Bool` stored properties became one
/// `Set<ModelCapability>` with the booleans preserved as computed get/set. These pin the two things
/// that must not regress — the public boolean API stays behaviourally identical, and the on-disk
/// Codable JSON stays byte-compatible — plus the new availability-state derivation and the
/// capability/availability filter predicate that `availableModels(…)` delegates to.
@Suite("ModelCapabilities set-backing + availability filtering")
struct ModelCapabilitySetBackingTests {

    // MARK: - Boolean API ⇄ Set

    @Test("Boolean initializer populates the set; unmentioned flags are off")
    func boolInitMapsToSet() {
        let caps = ModelCapabilities(toolUse: true, vision: true, toolResultRoundTrip: true)
        #expect(caps.toolUse)
        #expect(caps.vision)
        #expect(caps.toolResultRoundTrip)
        #expect(!caps.reasoning)
        #expect(!caps.pdfInput)
        #expect(caps.asSet == [.toolUse, .vision, .toolResultRoundTrip])
        #expect(caps.contains(.toolUse))
        #expect(!caps.contains(.reasoning))
    }

    @Test("Set initializer is equivalent to the boolean one")
    func setInitMatchesBoolInit() {
        let fromSet = ModelCapabilities([.reasoning, .pdfInput])
        let fromBools = ModelCapabilities(reasoning: true, pdfInput: true)
        #expect(fromSet == fromBools)
        #expect(fromSet.reasoning)
        #expect(fromSet.pdfInput)
    }

    @Test("Setters insert and remove from the backing set")
    func settersMutateSet() {
        var caps = ModelCapabilities()
        #expect(caps.asSet.isEmpty)
        caps.toolUse = true
        #expect(caps.contains(.toolUse))
        caps.vision = true
        caps.toolUse = false
        #expect(!caps.contains(.toolUse))
        #expect(caps.contains(.vision))
        #expect(caps.asSet == [.vision])
    }

    @Test("enabledLabels reflects only enabled flags, in canonical case order")
    func enabledLabelsOrder() {
        let caps = ModelCapabilities(toolUse: true, vision: true)
        // enabledLabels iterates ModelCapability.allCases, so order is canonical regardless of how
        // the flags were set: toolUse precedes vision.
        #expect(caps.enabledLabels == [ModelCapability.toolUse.label, ModelCapability.vision.label])
    }

    @Test("Every capability case has a matching boolean accessor round-trip")
    func everyCaseRoundTripsThroughItsBool() {
        for capability in ModelCapability.allCases {
            let caps = ModelCapabilities([capability])
            #expect(caps.contains(capability))
            #expect(caps.asSet == [capability])
            #expect(caps.enabledLabels == [capability.label])
        }
    }

    // MARK: - Codable backward compatibility

    @Test("Encoded JSON is the flat per-capability boolean object, unchanged")
    func encodesToFlatBooleanObject() throws {
        let caps = ModelCapabilities(toolUse: true, vision: true)
        let data = try JSONEncoder().encode(caps)
        let obj = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Bool])
        #expect(obj["toolUse"] == true)
        #expect(obj["vision"] == true)
        #expect(obj["reasoning"] == false)
        #expect(obj["toolResultRoundTrip"] == false)
        // All 17 keys are present (the wire format never omitted any).
        #expect(obj.count == ModelCapability.allCases.count)
    }

    @Test("Decodes a legacy full boolean object")
    func decodesLegacyFullObject() throws {
        let json = """
        {"toolUse":true,"vision":false,"reasoning":true,"codeExecution":false,"promptCaching":false,
         "computerUse":false,"audioInput":false,"audioOutput":false,"videoInput":false,
         "responseSchema":false,"parallelToolCalls":true,"pdfInput":true,"webSearch":false,
         "systemMessages":true,"assistantPrefill":false,"toolChoice":false,"toolResultRoundTrip":true}
        """
        let caps = try JSONDecoder().decode(ModelCapabilities.self, from: Data(json.utf8))
        #expect(caps.asSet == [.toolUse, .reasoning, .parallelToolCalls, .pdfInput, .systemMessages, .toolResultRoundTrip])
    }

    @Test("Decodes a partial object with missing keys defaulting to false")
    func decodesPartialObject() throws {
        let json = #"{"toolUse":true,"reasoning":true}"#
        let caps = try JSONDecoder().decode(ModelCapabilities.self, from: Data(json.utf8))
        #expect(caps.toolUse)
        #expect(caps.reasoning)
        #expect(!caps.vision)
        #expect(caps.asSet == [.toolUse, .reasoning])
    }

    @Test("Round-trips through Codable unchanged")
    func codableRoundTrip() throws {
        let original = ModelCapabilities(
            toolUse: true, vision: true, reasoning: true, pdfInput: true, toolChoice: true, toolResultRoundTrip: true
        )
        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(ModelCapabilities.self, from: data)
        #expect(restored == original)
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
        // Most-decorated possible model: unavailable + access denied + a deprecation date.
        let past = Date(timeIntervalSinceNow: -86_400)
        let states = model(isAvailable: false, isAccessDenied: true, deprecatedOn: past).availabilityStates
        #expect(states == [.isUnavailable, .isAccessDenied, .isDeprecated])
        #expect(!states.contains(.all))
    }

    // MARK: - satisfies() filter predicate (what availableModels delegates to)

    @Test("Required capabilities must all be present")
    func requiredCapabilitiesGate() {
        var m = model()
        m.capabilities = ModelCapabilities(toolUse: true)
        #expect(m.satisfies(requiredCapabilities: [.toolUse], mustNotBePresent: [], includedAvailabilityStates: []))
        #expect(!m.satisfies(requiredCapabilities: [.toolUse, .vision], mustNotBePresent: [], includedAvailabilityStates: []))
    }

    @Test("mustNotBePresent capabilities exclude a model")
    func mustNotBePresentGate() {
        var m = model()
        m.capabilities = ModelCapabilities(toolUse: true, vision: true)
        #expect(!m.satisfies(requiredCapabilities: [.toolUse], mustNotBePresent: [.vision], includedAvailabilityStates: []))
        #expect(m.satisfies(requiredCapabilities: [.toolUse], mustNotBePresent: [.reasoning], includedAvailabilityStates: []))
    }

    @Test("A normal model passes regardless of the include-set")
    func normalModelAlwaysPassesAvailabilityGate() {
        let m = model()
        #expect(m.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: []))
    }

    @Test("A model in a special state is excluded unless that state is included (or .all)")
    func specialStateGate() {
        let unavailable = model(isAvailable: false)
        #expect(!unavailable.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: []))
        #expect(unavailable.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.isUnavailable]))
        #expect(unavailable.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.all]))
        // Including a DIFFERENT state does not admit it.
        #expect(!unavailable.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.isDeprecated]))
    }

    @Test("A model with multiple special states needs all of them included")
    func multipleStatesNeedAllIncluded() {
        let m = model(isAvailable: false, isAccessDenied: true)
        #expect(!m.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.isUnavailable]))
        #expect(m.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.isUnavailable, .isAccessDenied]))
        #expect(m.satisfies(requiredCapabilities: [], mustNotBePresent: [], includedAvailabilityStates: [.all]))
    }

    @Test("ModelRequirements forwards its three fields")
    func requirementsBundleEquatable() {
        let a = ModelRequirements(requiredCapabilities: [.toolUse], mustNotBePresent: [.vision], includedAvailabilityStates: [.all])
        let b = ModelRequirements(requiredCapabilities: [.toolUse], mustNotBePresent: [.vision], includedAvailabilityStates: [.all])
        #expect(a == b)
    }
}
