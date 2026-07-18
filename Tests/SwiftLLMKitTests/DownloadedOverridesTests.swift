import Testing
import Foundation
@testable import SwiftLLMKit

/// The downloaded-overrides slot: an external file in the bundled schema that supersedes the
/// bundled registry inside the ONE downloaded-overrides layer.
@Suite("Downloaded overrides registry")
struct DownloadedOverridesRegistryTests {

    @Test("Overlaying replaces same-key entries and keeps the rest")
    func overlaySemantics() {
        let bundled = BundledModelMetadataRegistry(
            entries: ["anthropic/claude-x": ModelMetadataOverride(maxOutputTokens: 1000)],
            providerEntries: ["builtin.a/m1": ModelMetadataOverride(maxOutputTokens: 2000)],
            providerDefaults: ["builtin.a": ModelMetadataOverride(supportsChatCompletions: true)]
        )
        let downloaded = BundledModelMetadataRegistry(
            entries: ["anthropic/claude-x": ModelMetadataOverride(maxOutputTokens: 1500)],  // supersedes
            providerEntries: ["builtin.a/m2": ModelMetadataOverride(maxOutputTokens: 3000)], // adds
            providerDefaults: [:]
        )
        let effective = bundled.overlaying(downloaded)
        #expect(effective.override(providerAPIType: "anthropic", modelID: "claude-x")?.maxOutputTokens == 1500)
        #expect(effective.override(providerID: "builtin.a", modelID: "m1")?.maxOutputTokens == 2000)  // kept
        #expect(effective.override(providerID: "builtin.a", modelID: "m2")?.maxOutputTokens == 3000)  // added
        #expect(effective.defaults(providerID: "builtin.a")?.supportsChatCompletions == true)          // kept
    }

    @Test("load(from:) reads the bundled schema from an external file; absent file is nil")
    func externalFileLoad() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("downloaded-overrides-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("downloaded_overrides.json")

        #expect(BundledModelMetadataRegistry.load(from: url) == nil)   // absent = no registry

        let json = #"""
        {"version": 2,
         "entries": {"zai/glm-x": {"maxOutputTokens": 4096}},
         "providerEntries": {"builtin.anthropic/claude-old-snapshot": {"hidden": true}}}
        """#
        try Data(json.utf8).write(to: url)
        let loaded = try #require(BundledModelMetadataRegistry.load(from: url))
        #expect(loaded.override(providerAPIType: "zai", modelID: "glm-x")?.maxOutputTokens == 4096)
        #expect(loaded.override(providerID: "builtin.anthropic", modelID: "claude-old-snapshot")?.hidden == true)
    }

    @Test("providerScopedModelIDs enumerates only the providerID axis")
    func enumerationAxis() {
        let registry = BundledModelMetadataRegistry(
            entries: ["anthropic/apitype-model": ModelMetadataOverride(maxOutputTokens: 1)],
            providerEntries: [
                "builtin.a/added-model": ModelMetadataOverride(maxOutputTokens: 2),
                "builtin.b/other": ModelMetadataOverride(maxOutputTokens: 3),
            ],
            providerDefaults: ["builtin.a": ModelMetadataOverride(supportsChatCompletions: true)]
        )
        #expect(registry.providerScopedModelIDs(providerID: "builtin.a") == ["added-model"])
        // apiType entries and provider defaults never materialize models.
    }
}

/// `hidden` flows override → facts → merge → materialized ModelInfo.
@Suite("Hidden flag propagation")
struct HiddenFlagTests {
    @Test("A hidden override reaches the materialized model; absence stays nil")
    func hiddenFlows() {
        let override = ModelMetadataOverride(hidden: true)
        var downloaded = ModelFacts()
        downloaded.overlay(override.asFacts)
        let merged = ModelFactsMerger.merge(authoritative: ModelFacts(), downloadedOverrides: downloaded)
        #expect(merged.merged.hidden == true)
        #expect(merged.provenance["hidden"] == .downloadedOverrides)
        let info = merged.merged.materialize(providerID: "p", modelID: "m")
        #expect(info.hidden == true)

        let plain = ModelFactsMerger.merge(authoritative: ModelFacts()).merged.materialize(providerID: "p", modelID: "m")
        #expect(plain.hidden == nil)
    }

    @Test("hidden round-trips through ModelMetadataOverride Codable")
    func hiddenCodable() throws {
        let original = ModelMetadataOverride(maxOutputTokens: 9, hidden: true)
        let decoded = try JSONDecoder().decode(ModelMetadataOverride.self, from: JSONEncoder().encode(original))
        #expect(decoded == original)
        #expect(decoded.hidden == true)
    }
}
