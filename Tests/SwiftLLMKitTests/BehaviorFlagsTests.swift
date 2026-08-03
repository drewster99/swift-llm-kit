import Testing
import Foundation
@testable import SwiftLLMKit

// MARK: - BehaviorFlags Codable

@Suite("BehaviorFlags Codable")
struct BehaviorFlagsCodableTests {

    @Test("default-valued flags encode to an empty JSON object")
    func defaultsEncodeEmpty() throws {
        let flags = BehaviorFlags()
        #expect(flags.isAllDefault)
        let data = try JSONEncoder().encode(flags)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?.isEmpty == true,
                "All-default flags should serialize to {} so older clients ignore unknown keys cleanly")
    }

    @Test("non-default flags encode only the set fields")
    func nonDefaultsEncodeSparsely() throws {
        let flags = BehaviorFlags(glmTemplateSalvage: true)
        let data = try JSONEncoder().encode(flags)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["glmTemplateSalvage"] as? Bool == true)
        #expect(json["useMaxCompletionTokens"] == nil,
                "False fields shouldn't appear in JSON")
        #expect(json["disableParallelToolCalls"] == nil)
        #expect(json["extras"] == nil)
    }

    @Test("round-trip preserves every field")
    func roundTripPreservesAllFields() throws {
        let original = BehaviorFlags(
            glmTemplateSalvage: true,
            useMaxCompletionTokens: true,
            disableParallelToolCalls: true,
            extras: ["responseFormat": "json_object", "bypassReasoning": "true"]
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(BehaviorFlags.self, from: data)
        #expect(decoded == original)
    }

    @Test("decoding a JSON missing every key yields all defaults")
    func decodeMissingKeysGivesDefaults() throws {
        let json = "{}".data(using: .utf8)!
        let decoded = try JSONDecoder().decode(BehaviorFlags.self, from: json)
        #expect(decoded.isAllDefault)
    }

    @Test("decoding a JSON with unknown keys ignores them and decodes known fields")
    func decodeUnknownKeysIgnored() throws {
        // Forward-compat: a future client adds new flag keys; old clients should
        // decode the known ones cleanly without throwing.
        let json = """
            {
              "glmTemplateSalvage": true,
              "futureFlag": "foo",
              "anotherFutureFlag": 42
            }
            """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(BehaviorFlags.self, from: json)
        #expect(decoded.glmTemplateSalvage)
        #expect(!decoded.useMaxCompletionTokens)
    }

    @Test("isAllDefault flips false when any single flag turns on")
    func isAllDefaultDetection() {
        #expect(BehaviorFlags().isAllDefault)
        #expect(!BehaviorFlags(glmTemplateSalvage: true).isAllDefault)
        #expect(!BehaviorFlags(useMaxCompletionTokens: true).isAllDefault)
        #expect(!BehaviorFlags(disableParallelToolCalls: true).isAllDefault)
        #expect(!BehaviorFlags(extras: ["k": "v"]).isAllDefault)
    }
}

// MARK: - BehaviorFlagsOverride apply

@Suite("BehaviorFlagsOverride apply")
struct BehaviorFlagsOverrideTests {

    @Test("gap-fill upgrades false→true for each non-nil patch field")
    func gapFillUpgrades() {
        var flags = BehaviorFlags()
        let patch = BehaviorFlagsOverride(
            glmTemplateSalvage: true,
            useMaxCompletionTokens: true,
            disableParallelToolCalls: true
        )
        patch.apply(to: &flags, forceReplace: false)
        #expect(flags.glmTemplateSalvage)
        #expect(flags.useMaxCompletionTokens)
        #expect(flags.disableParallelToolCalls)
    }

    @Test("gap-fill does NOT downgrade true→false")
    func gapFillCannotDowngrade() {
        var flags = BehaviorFlags(glmTemplateSalvage: true, useMaxCompletionTokens: true)
        let patch = BehaviorFlagsOverride(
            glmTemplateSalvage: false,
            useMaxCompletionTokens: false
        )
        patch.apply(to: &flags, forceReplace: false)
        #expect(flags.glmTemplateSalvage,
                "Gap-fill must not turn off a flag a higher-priority layer set")
        #expect(flags.useMaxCompletionTokens)
    }

    @Test("force-replace replaces with any non-nil value, including false")
    func forceReplaceCanDowngrade() {
        var flags = BehaviorFlags(glmTemplateSalvage: true, useMaxCompletionTokens: true)
        let patch = BehaviorFlagsOverride(
            glmTemplateSalvage: false,
            useMaxCompletionTokens: nil
        )
        patch.apply(to: &flags, forceReplace: true)
        #expect(!flags.glmTemplateSalvage,
                "User override (force-replace) must be able to disable a bundled flag")
        #expect(flags.useMaxCompletionTokens,
                "nil patch fields don't touch the existing value")
    }

    @Test("nil patch fields leave the existing flag unchanged")
    func nilPatchSkipsField() {
        var flags = BehaviorFlags(glmTemplateSalvage: true, disableParallelToolCalls: true)
        let patch = BehaviorFlagsOverride(useMaxCompletionTokens: true)
        patch.apply(to: &flags, forceReplace: true)
        #expect(flags.glmTemplateSalvage)
        #expect(flags.useMaxCompletionTokens)
        #expect(flags.disableParallelToolCalls)
    }

    @Test("extras gap-fill: only fills keys not already present")
    func extrasGapFillSkipsExistingKeys() {
        var flags = BehaviorFlags(extras: ["responseFormat": "text"])
        let patch = BehaviorFlagsOverride(extras: [
            "responseFormat": "json_object",  // existing key — should NOT replace
            "bypassReasoning": "true"          // new key — fills
        ])
        patch.apply(to: &flags, forceReplace: false)
        #expect(flags.extras["responseFormat"] == "text")
        #expect(flags.extras["bypassReasoning"] == "true")
    }

    @Test("extras force-replace: wholesale replacement, can clear existing")
    func extrasForceReplace() {
        var flags = BehaviorFlags(extras: ["a": "1", "b": "2"])
        let patch = BehaviorFlagsOverride(extras: ["c": "3"])
        patch.apply(to: &flags, forceReplace: true)
        #expect(flags.extras == ["c": "3"],
                "Force-replace overwrites the entire extras dictionary")
    }

    @Test("isEmpty correctly detects an all-nil patch")
    func isEmptyDetection() {
        #expect(BehaviorFlagsOverride().isEmpty)
        #expect(!BehaviorFlagsOverride(glmTemplateSalvage: true).isEmpty)
        #expect(!BehaviorFlagsOverride(extras: ["k": "v"]).isEmpty)
        #expect(BehaviorFlagsOverride(extras: [:]).isEmpty,
                "Empty extras dict counts as no patch")
    }
}

// MARK: - ModelMetadataOverride flag propagation

@Suite("ModelMetadataOverride behavior-flag propagation")
struct ModelMetadataOverrideFlagsTests {

    @Test("apply propagates BehaviorFlagsOverride into ModelInfo.behaviorFlags")
    func applyPropagatesFlags() {
        var model = ModelInfo(providerID: "test", modelID: "m1")
        let override = ModelMetadataOverride(
            behaviorFlags: BehaviorFlagsOverride(glmTemplateSalvage: true)
        )
        override.apply(to: &model, forceReplace: false)
        #expect(model.behaviorFlags.glmTemplateSalvage)
    }

    @Test("apply with no behaviorFlags patch leaves model.behaviorFlags untouched")
    func applyWithoutFlagsLeavesUntouched() {
        var model = ModelInfo(
            providerID: "test", modelID: "m1",
            behaviorFlags: BehaviorFlags(glmTemplateSalvage: true)
        )
        let override = ModelMetadataOverride(displayName: "Pretty Name")
        override.apply(to: &model, forceReplace: true)
        #expect(model.displayName == "Pretty Name")
        #expect(model.behaviorFlags.glmTemplateSalvage,
                "An override that doesn't touch flags must not reset them")
    }

    @Test("force-replace through ModelMetadataOverride downgrades flags as expected")
    func forceReplaceFromTopLevel() {
        var model = ModelInfo(
            providerID: "test", modelID: "m1",
            behaviorFlags: BehaviorFlags(glmTemplateSalvage: true, useMaxCompletionTokens: true)
        )
        let override = ModelMetadataOverride(
            behaviorFlags: BehaviorFlagsOverride(glmTemplateSalvage: false)
        )
        override.apply(to: &model, forceReplace: true)
        #expect(!model.behaviorFlags.glmTemplateSalvage)
        #expect(model.behaviorFlags.useMaxCompletionTokens,
                "User override of one flag must not zero the others")
    }
}

// MARK: - ModelInfo Codable round-trip with flags

@Suite("ModelInfo Codable with BehaviorFlags")
struct ModelInfoBehaviorFlagsCodableTests {

    @Test("round-trip preserves non-default behaviorFlags")
    func roundTripPreservesNonDefault() throws {
        let original = ModelInfo(
            providerID: "p", modelID: "m",
            behaviorFlags: BehaviorFlags(glmTemplateSalvage: true)
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ModelInfo.self, from: data)
        #expect(decoded.behaviorFlags.glmTemplateSalvage)
    }

    @Test("default behaviorFlags are NOT serialized — keeps payloads compact")
    func defaultsOmittedOnEncode() throws {
        let info = ModelInfo(providerID: "p", modelID: "m")
        let data = try JSONEncoder().encode(info)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["behaviorFlags"] == nil,
                "ModelInfo must skip encoding behaviorFlags when isAllDefault to keep on-disk JSON small")
    }

    @Test("round-trip preserves every metadata-audit field")
    func roundTripPreservesAllNewFields() throws {
        let original = ModelInfo(
            providerID: "p", modelID: "m", displayName: "M",
            maxInputTokens: 200_000, maxOutputTokens: 64_000,
            pricing: ModelPricing(base: PricingTier(input: 0.000005, output: 0.000025)),
            generalEffort: .levels(["low", "medium", "high"]),
            deprecatedOn: Date(timeIntervalSince1970: 1_788_177_600),
            deprecationReplacement: "m2",
            maxTemperature: 2,
            modelDescription: "a model",
            samplingDefaults: SamplingDefaults(temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1),
            isFree: false,
            benchmarks: ModelBenchmarks(
                artificialAnalysis: .init(intelligenceIndex: 57.1, codingIndex: 76.2, agenticIndex: 50.1),
                designArena: [.init(arena: "agents", category: "gamedev", elo: 1200, rank: 7, winRate: 47.8)]),
            huggingFaceID: "org/M",
            hidden: true,
            isAvailable: false,
            isAccessDenied: true
        )
        let decoded = try JSONDecoder().decode(ModelInfo.self, from: JSONEncoder().encode(original))
        #expect(decoded == original, "a full ModelInfo must survive an encode/decode round-trip unchanged")
    }

    @Test("decoding ModelInfo without behaviorFlags falls back to defaults")
    func legacyJSONDecodesWithDefaults() throws {
        let legacyJSON = """
            {
              "providerID": "p",
              "modelID": "m",
              "displayName": "m",
              "capabilities": {},
              "supportsChatCompletions": true
            }
            """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ModelInfo.self, from: legacyJSON)
        #expect(decoded.behaviorFlags.isAllDefault)
    }
}

// MARK: - BundledModelMetadataRegistry schema v2

@Suite("BundledModelMetadataRegistry schema v2")
struct BundledModelMetadataRegistrySchemaTests {

    /// Manually decodes a JSON string through the same path the registry uses.
    /// Mirrors the runtime loader so the schema-fallback assertions stay honest.
    private func parse(_ json: String) throws -> BundledRegistryFixture {
        struct Wrapper: Codable {
            let version: Int
            let entries: [String: ModelMetadataOverride]
            let providerEntries: [String: ModelMetadataOverride]?
            let providerDefaults: [String: ModelMetadataOverride]?
        }
        let data = json.data(using: .utf8)!
        let w = try JSONDecoder().decode(Wrapper.self, from: data)
        return BundledRegistryFixture(
            entries: w.entries,
            providerEntries: w.providerEntries ?? [:],
            providerDefaults: w.providerDefaults ?? [:]
        )
    }

    private struct BundledRegistryFixture {
        let entries: [String: ModelMetadataOverride]
        let providerEntries: [String: ModelMetadataOverride]
        let providerDefaults: [String: ModelMetadataOverride]
    }

    @Test("v1 JSON (entries only) decodes cleanly with empty providerEntries / providerDefaults")
    func v1Compatible() throws {
        let v1 = """
            {
              "version": 1,
              "entries": {
                "anthropic/claude-test": {
                  "maxOutputTokens": 4096
                }
              }
            }
            """
        let fixture = try parse(v1)
        #expect(fixture.entries["anthropic/claude-test"]?.maxOutputTokens == 4096)
        #expect(fixture.providerEntries.isEmpty)
        #expect(fixture.providerDefaults.isEmpty)
    }

    @Test("v2 JSON parses all three lookup axes")
    func v2AllAxes() throws {
        let v2 = """
            {
              "version": 2,
              "entries": {
                "zAI/glm-5": { "behaviorFlags": { "glmTemplateSalvage": true } }
              },
              "providerEntries": {
                "builtin.ollama-cloud/glm-4-plus": { "behaviorFlags": { "glmTemplateSalvage": true } }
              },
              "providerDefaults": {
                "builtin.openai": { "behaviorFlags": { "useMaxCompletionTokens": true } }
              }
            }
            """
        let fixture = try parse(v2)
        #expect(fixture.entries["zAI/glm-5"]?.behaviorFlags?.glmTemplateSalvage == true)
        #expect(fixture.providerEntries["builtin.ollama-cloud/glm-4-plus"]?.behaviorFlags?.glmTemplateSalvage == true)
        #expect(fixture.providerDefaults["builtin.openai"]?.behaviorFlags?.useMaxCompletionTokens == true)
    }

    @Test("the bundled JSON shipped in Resources loads without errors and has the expected providers")
    func shippedJSONLoads() {
        let registry = BundledModelMetadataRegistry.load()
        // We don't assert exact count here — entries can grow as new GLM models are added —
        // just that the file loaded and the GLM/OpenAI defaults landed somewhere.
        #expect(registry.providerDefaults["builtin.openai"]?.behaviorFlags?.useMaxCompletionTokens == true,
                "OpenAI provider-wide default for useMaxCompletionTokens should be present")
        let hasZAIGLM = registry.entries.keys.contains(where: { $0.hasPrefix("zAI/glm-") })
        #expect(hasZAIGLM, "Bundled entries should include z.ai's GLM models with glmTemplateSalvage")
    }
}

// MARK: - Layered resolution end-to-end

@Suite("Layered behavior-flags resolution")
struct LayeredResolutionTests {

    /// Simulates the layering applied in `LLMKitManager.fetchProviderModels`:
    /// providerDefaults → providerEntries → entries → user.
    private func resolve(
        providerDefaults: BehaviorFlagsOverride?,
        providerEntry: BehaviorFlagsOverride?,
        apiTypeEntry: BehaviorFlagsOverride?,
        userOverride: BehaviorFlagsOverride?
    ) -> BehaviorFlags {
        var flags = BehaviorFlags()
        providerDefaults?.apply(to: &flags, forceReplace: false)
        providerEntry?.apply(to: &flags, forceReplace: false)
        apiTypeEntry?.apply(to: &flags, forceReplace: false)
        userOverride?.apply(to: &flags, forceReplace: true)
        return flags
    }

    @Test("with no overrides anywhere, flags resolve to all-defaults")
    func noOverridesGivesDefaults() {
        let flags = resolve(providerDefaults: nil, providerEntry: nil, apiTypeEntry: nil, userOverride: nil)
        #expect(flags.isAllDefault)
    }

    @Test("provider-wide default flag flows through to per-model resolution")
    func providerDefaultFlowsThrough() {
        let flags = resolve(
            providerDefaults: BehaviorFlagsOverride(useMaxCompletionTokens: true),
            providerEntry: nil, apiTypeEntry: nil, userOverride: nil
        )
        #expect(flags.useMaxCompletionTokens)
    }

    @Test("user override force-disables a bundled flag")
    func userOverrideDisablesBundledFlag() {
        let flags = resolve(
            providerDefaults: BehaviorFlagsOverride(useMaxCompletionTokens: true),
            providerEntry: nil, apiTypeEntry: nil,
            userOverride: BehaviorFlagsOverride(useMaxCompletionTokens: false)
        )
        #expect(!flags.useMaxCompletionTokens,
                "User override (force-replace) must beat bundled provider-default")
    }

    @Test("multiple bundled layers union their flags")
    func bundledLayersUnion() {
        let flags = resolve(
            providerDefaults: BehaviorFlagsOverride(useMaxCompletionTokens: true),
            providerEntry: BehaviorFlagsOverride(glmTemplateSalvage: true),
            apiTypeEntry: BehaviorFlagsOverride(disableParallelToolCalls: true),
            userOverride: nil
        )
        #expect(flags.useMaxCompletionTokens)
        #expect(flags.glmTemplateSalvage)
        #expect(flags.disableParallelToolCalls)
    }

    @Test("user can selectively disable one bundled flag without affecting others")
    func userOverrideIsScoped() {
        let flags = resolve(
            providerDefaults: BehaviorFlagsOverride(useMaxCompletionTokens: true),
            providerEntry: BehaviorFlagsOverride(glmTemplateSalvage: true),
            apiTypeEntry: nil,
            userOverride: BehaviorFlagsOverride(glmTemplateSalvage: false)
        )
        #expect(flags.useMaxCompletionTokens,
                "Disabling glmTemplateSalvage must not zero the unrelated useMaxCompletionTokens flag")
        #expect(!flags.glmTemplateSalvage)
    }
}
