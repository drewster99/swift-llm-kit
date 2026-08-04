import Testing
import Foundation
@testable import SwiftLLMKit

/// The nine capabilities added in this work must survive persistence in BOTH containers, or a user
/// override for them is inert and a probed finding never reaches the catalog — the same silent
/// shape as the ModelMetadataOverride bug.
@Suite("New capabilities round-trip in both containers")
struct NewCapabilityPersistenceTests {

    private static let added: [ModelCapability] = [
        .reasoningCanBeEnabled, .reasoningCanBeDisabled, .thinkingSupportsKeepAll, .thinkingSupportsTokenBudget,
        .structuredOutputSupportsJSONObject, .toolChoiceSupportsValueRequired, .toolChoiceSupportsValueNone,
        .toolChoiceSupportsNamedFunction, .toolDefinitionsSupportStrict
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
        patch[.toolDefinitionsSupportStrict] = true
        patch[.toolChoiceSupportsValueRequired] = false
        var caps = ModelCapabilities()
        patch.apply(to: &caps, forceReplace: true)
        #expect(caps[.toolDefinitionsSupportStrict] == true)
        #expect(caps[.toolChoiceSupportsValueRequired] == false, "a stated NO must apply, not just a YES")
    }
}

/// `ModelCapability` is `String`-backed and its rawValues ARE the persisted keys — in
/// `model_catalog.json`, in every probe record's `capabilityFindings`, and in the tri-state
/// `ModelCapabilities` Codable. They were IMPLICIT, so renaming a case silently rewrote the key and
/// orphaned every record using it; `vision` alone appears in 1,414 of them.
///
/// Every rawValue is pinned explicitly now, and this table is asserted independently of the enum so
/// the two must be changed together. Renaming a CASE is free; changing a WIRE STRING fails here.
@Suite("ModelCapability wire strings are frozen")
struct ModelCapabilityWireStringTests {

    /// Written out by hand on purpose — deriving it from the enum would assert nothing.
    private static let expected: [ModelCapability: String] = [
        .chat: "chat", .batch: "batch", .toolUse: "toolUse", .vision: "vision",
        .reasoning: "reasoning", .codeExecution: "codeExecution", .promptCaching: "promptCaching",
        .computerUse: "computerUse", .audioInput: "audioInput", .audioOutput: "audioOutput",
        .videoInput: "videoInput", .parallelToolCalls: "parallelToolCalls", .pdfInput: "pdfInput",
        .webSearch: "webSearch", .systemMessages: "systemMessages",
        .assistantPrefill: "assistantPrefill", .toolResultRoundTrip: "toolResultRoundTrip",
        // Migrated to match the case names by scripts/migrate_capability_wire_names.py, a
        // one-time pass over the probe records and the model catalog.
        .toolChoiceSupported: "toolChoiceSupported",
        .toolChoiceSupportsValueRequired: "toolChoiceSupportsValueRequired",
        .toolChoiceSupportsValueNone: "toolChoiceSupportsValueNone",
        .toolChoiceSupportsNamedFunction: "toolChoiceSupportsNamedFunction",
        .structuredOutputSupportsJSONObject: "structuredOutputSupportsJSONObject",
        .structuredOutputSupportsJSONSchema: "structuredOutputSupportsJSONSchema",
        .thinkingSupportsKeepAll: "thinkingSupportsKeepAll",
        .thinkingSupportsTokenBudget: "thinkingSupportsTokenBudget",
        .toolDefinitionsSupportStrict: "toolDefinitionsSupportStrict",
        .reasoningCanBeEnabled: "reasoningCanBeEnabled",
        .reasoningCanBeDisabled: "reasoningCanBeDisabled"
    ]

    @Test("Every case's wire string matches the pinned table")
    func wireStringsMatch() {
        for (capability, wire) in Self.expected {
            #expect(capability.rawValue == wire,
                    "\(capability) persists as '\(capability.rawValue)' but the table says '\(wire)' — changing a wire string orphans every record already using the old one")
        }
    }

    @Test("The table covers every case, so a new one cannot ship unpinned")
    func tableIsComplete() {
        let pinned = Set(Self.expected.keys)
        let all = Set(ModelCapability.allCases)
        #expect(all.subtracting(pinned).isEmpty,
                "unpinned: \(all.subtracting(pinned).map(\.rawValue).sorted())")
        #expect(pinned.subtracting(all).isEmpty)
    }

    /// Unrenamed capabilities must still decode exactly as before — the migration touched only
    /// the eleven that moved.
    @Test("Untouched capabilities keep their original wire strings")
    func untouchedCapabilitiesUnchanged() throws {
        let json = #"{"vision": true, "toolUse": false, "pdfInput": true}"#
        let caps = try JSONDecoder().decode(ModelCapabilities.self, from: Data(json.utf8))
        #expect(caps[.vision] == true)
        #expect(caps[.toolUse] == false)
        #expect(caps[.pdfInput] == true)
    }

    /// A record still carrying a legacy spelling decodes with that capability ABSENT, not wrong —
    /// `init(rawValue:)` returns nil for an unknown key and the decoder skips it. That is the
    /// failure mode the one-time migration exists to prevent, and why it verifies zero survivors.
    @Test("A legacy spelling is dropped, never silently mapped to something else")
    func legacySpellingIsDroppedNotMisread() throws {
        let stale = #"{"vision": true, "toolChoiceRequired": false}"#
        let caps = try JSONDecoder().decode(ModelCapabilities.self, from: Data(stale.utf8))
        #expect(caps[.vision] == true)
        #expect(caps[.toolChoiceSupportsValueRequired] == nil, "unknown key skipped, not misread")
    }
}
