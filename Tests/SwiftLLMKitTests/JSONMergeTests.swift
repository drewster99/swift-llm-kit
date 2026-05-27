import Foundation
import Testing
@testable import SwiftLLMKit

/// Unit coverage for `mergeJSONOverrides` — the deep-merge helper applied to
/// `ModelConfiguration.extraJSONOverrides` by every provider's
/// `buildRequestBody`. The merge rule: dict-valued keys recurse; everything
/// else (scalars, arrays, strings, type mismatches) replaces outright.
@Suite("mergeJSONOverrides deep-merge semantics")
struct JSONMergeTests {

    @Test func scalarOverride_replacesBodyValue() {
        var body: [String: Any] = ["temperature": 0.7, "model": "claude"]
        let overrides: [String: AnyCodable] = ["temperature": .double(0.2)]
        mergeJSONOverrides(&body, with: overrides)
        #expect(body["temperature"] as? Double == 0.2)
        #expect(body["model"] as? String == "claude")  // unchanged
    }

    @Test func arrayOverride_replacesBodyValue() {
        var body: [String: Any] = ["stop": ["END"]]
        let overrides: [String: AnyCodable] = ["stop": .array([.string("DONE"), .string("FIN")])]
        mergeJSONOverrides(&body, with: overrides)
        let result = body["stop"] as? [Any]
        #expect(result?.count == 2)
    }

    @Test func dictOverride_mergesSubkeys_preservingBodySubkeys() {
        // The whole point — Gemini's generationConfig case.
        var body: [String: Any] = [
            "generationConfig": [
                "temperature": 0.7,
                "maxOutputTokens": 4096
            ] as [String: Any]
        ]
        let overrides: [String: AnyCodable] = [
            "generationConfig": .dictionary([
                "thinkingConfig": .dictionary([
                    "thinkingBudget": .int(8000)
                ])
            ])
        ]
        mergeJSONOverrides(&body, with: overrides)
        let gc = body["generationConfig"] as? [String: Any]
        #expect(gc?["temperature"] as? Double == 0.7)
        #expect(gc?["maxOutputTokens"] as? Int == 4096)
        let tc = gc?["thinkingConfig"] as? [String: Any]
        #expect(tc?["thinkingBudget"] as? Int == 8000)
    }

    @Test func dictOverride_subkeyConflict_overrideWins() {
        var body: [String: Any] = [
            "generationConfig": ["temperature": 0.7, "topP": 0.9] as [String: Any]
        ]
        let overrides: [String: AnyCodable] = [
            "generationConfig": .dictionary(["temperature": .double(0.1)])
        ]
        mergeJSONOverrides(&body, with: overrides)
        let gc = body["generationConfig"] as? [String: Any]
        #expect(gc?["temperature"] as? Double == 0.1)
        #expect(gc?["topP"] as? Double == 0.9)
    }

    @Test func typeMismatch_dictOverrideOnScalarBody_replaces() {
        var body: [String: Any] = ["thing": "hello"]
        let overrides: [String: AnyCodable] = [
            "thing": .dictionary(["k": .string("v")])
        ]
        mergeJSONOverrides(&body, with: overrides)
        let result = body["thing"] as? [String: Any]
        #expect(result?["k"] as? String == "v")
    }

    @Test func typeMismatch_scalarOverrideOnDictBody_replaces() {
        var body: [String: Any] = ["thing": ["nested": "val"] as [String: Any]]
        let overrides: [String: AnyCodable] = ["thing": .string("flat")]
        mergeJSONOverrides(&body, with: overrides)
        #expect(body["thing"] as? String == "flat")
    }

    @Test func recursesArbitraryDepth() {
        var body: [String: Any] = [
            "a": ["b": ["c": ["d": "original"] as [String: Any]] as [String: Any]] as [String: Any]
        ]
        let overrides: [String: AnyCodable] = [
            "a": .dictionary([
                "b": .dictionary([
                    "c": .dictionary([
                        "e": .string("added")
                    ])
                ])
            ])
        ]
        mergeJSONOverrides(&body, with: overrides)
        let leaf = body["a"] as? [String: Any]
        let b = leaf?["b"] as? [String: Any]
        let c = b?["c"] as? [String: Any]
        #expect(c?["d"] as? String == "original")  // preserved
        #expect(c?["e"] as? String == "added")     // added
    }

    @Test func emptyOverrides_leavesBodyUnchanged() throws {
        var body: [String: Any] = ["a": 1, "b": ["c": 2] as [String: Any]]
        let dataBefore = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        mergeJSONOverrides(&body, with: [:])
        let dataAfter = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        #expect(dataBefore == dataAfter)
    }
}
