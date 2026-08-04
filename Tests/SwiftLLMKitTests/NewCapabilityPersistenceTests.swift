import Testing
import Foundation
@testable import SwiftLLMKit

/// The nine capabilities added in this work must survive persistence in BOTH containers, or a user
/// override for them is inert and a probed finding never reaches the catalog — the same silent
/// shape as the ModelMetadataOverride bug.
@Suite("New capabilities round-trip in both containers")
struct NewCapabilityPersistenceTests {

    private static let added: [ModelCapability] = [
        .reasoningEnableable, .reasoningDisableable, .thinkingKeepAll, .thinkingBudgetTokens,
        .structuredOutputJSONObject, .toolChoiceRequired, .toolChoiceNone,
        .toolChoiceSpecificFunction, .strictToolDefinitions
    ]

    @Test("ModelCapabilities keeps each new capability's known state, both directions")
    func capabilitiesRoundTrip() throws {
        var caps = ModelCapabilities()
        for (index, capability) in Self.added.enumerated() { caps[capability] = index.isMultiple(of: 2) }
        let restored = try JSONDecoder().decode(
            ModelCapabilities.self, from: JSONEncoder().encode(caps))
        for (index, capability) in Self.added.enumerated() {
            #expect(restored[capability] == index.isMultiple(of: 2),
                    "\(capability.rawValue) lost its state — a stated NO must survive as NO")
        }
    }

    @Test("ModelCapabilitiesOverride persists each new capability")
    func overrideRoundTrips() throws {
        var patch = ModelCapabilitiesOverride()
        for capability in Self.added { patch[capability] = true }
        let restored = try JSONDecoder().decode(
            ModelCapabilitiesOverride.self, from: JSONEncoder().encode(patch))
        for capability in Self.added {
            #expect(restored[capability] == true,
                    "\(capability.rawValue) is not persisted, so an override for it is inert")
        }
    }

    @Test("An override for a new capability applies to a fresh capability set")
    func overrideApplies() throws {
        var patch = ModelCapabilitiesOverride()
        patch[.strictToolDefinitions] = true
        patch[.toolChoiceRequired] = false
        var caps = ModelCapabilities()
        patch.apply(to: &caps, forceReplace: true)
        #expect(caps[.strictToolDefinitions] == true)
        #expect(caps[.toolChoiceRequired] == false, "a stated NO must apply, not just a YES")
    }
}
