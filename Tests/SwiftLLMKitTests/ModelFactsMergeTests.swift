import Testing
import Foundation
@testable import SwiftLLMKit

/// The field-descriptor table is the single source of truth for what merges, so a field that
/// exists on the record but isn't registered is a silent wrong-merge. This test makes it a red
/// test instead: every stored property of `ModelFacts` (and every per-flag property of its two
/// containers) must appear in the table.
@Suite("ModelFacts field table completeness")
struct ModelFactsFieldTableTests {

    @Test("Every ModelFacts stored property is registered in the field table")
    func everyFieldRegistered() {
        let registered = Set(ModelFactsFieldTable.fields.map(\.name))

        // Top-level stored properties. The two containers are covered per-flag below.
        for child in Mirror(reflecting: ModelFacts()).children {
            guard let label = child.label else { continue }
            if label == "capabilities" || label == "behaviorFlags" { continue }
            #expect(registered.contains(label), "ModelFacts.\(label) is not in ModelFactsFieldTable")
        }
        for child in Mirror(reflecting: ModelCapabilitiesOverride()).children {
            guard let label = child.label else { continue }
            #expect(registered.contains("capabilities.\(label)"),
                    "capabilities.\(label) is not in ModelFactsFieldTable")
        }
        for child in Mirror(reflecting: BehaviorFlagsOverride()).children {
            guard let label = child.label else { continue }
            #expect(registered.contains("behaviorFlags.\(label)"),
                    "behaviorFlags.\(label) is not in ModelFactsFieldTable")
        }
    }

    @Test("A fresh ModelFacts is silent; a single stated field is not")
    func silenceDetection() {
        #expect(ModelFacts().isSilent)
        var facts = ModelFacts()
        facts.capabilities.toolUse = false     // a stated FALSE is a statement, not silence
        #expect(!facts.isSilent)
    }
}

/// The five-layer merge: authoritative base, empirical gap-fill, downloaded overrides FORCE,
/// LiteLLM gap-fill, user overrides FORCE — per field, with provenance and visible disagreements.
@Suite("Five-layer facts merge")
struct ModelFactsMergerTests {

    @Test("Gap-fill layers only fill unknowns; authoritative wins fields it states")
    func authoritativeWins() {
        var authoritative = ModelFacts()
        authoritative.maxInputTokens = 200_000
        authoritative.capabilities.vision = false      // vendor-stated NO

        var enrichment = ModelFacts()
        enrichment.maxInputTokens = 128_000            // LiteLLM's different claim
        enrichment.capabilities.vision = true          // LiteLLM says yes
        enrichment.capabilities.toolUse = true         // vendor was silent on this

        let result = ModelFactsMerger.merge(authoritative: authoritative, enrichment: enrichment)
        #expect(result.merged.maxInputTokens == 200_000)
        #expect(result.merged.capabilities.vision == false)   // stated false SURVIVES enrichment true
        #expect(result.merged.capabilities.toolUse == true)   // gap genuinely filled
        #expect(result.provenance["maxInputTokens"] == .authoritative)
        #expect(result.provenance["capabilities.toolUse"] == .enrichment)
        // The overridden claims stay visible as disagreements.
        #expect(result.disagreements.contains { $0.field == "capabilities.vision" && $0.dissentingLayer == .enrichment })
    }

    @Test("Downloaded overrides FORCE over authoritative and probe, and shield from LiteLLM")
    func downloadedOverridesForce() {
        var authoritative = ModelFacts()
        authoritative.maxOutputTokens = 8192          // vendor claim (wrong, say)
        var empirical = ModelFacts()
        empirical.capabilities.pdfInput = false       // probe result (wrong, say)
        var downloaded = ModelFacts()
        downloaded.maxOutputTokens = 4096             // curated fix beats the vendor
        downloaded.capabilities.pdfInput = true       // curated fix beats the probe
        var enrichment = ModelFacts()
        enrichment.maxOutputTokens = 16_384           // LiteLLM must NOT touch a forced field

        let result = ModelFactsMerger.merge(
            authoritative: authoritative, empirical: empirical,
            downloadedOverrides: downloaded, enrichment: enrichment)
        #expect(result.merged.maxOutputTokens == 4096)
        #expect(result.merged.capabilities.pdfInput == true)
        #expect(result.provenance["maxOutputTokens"] == .downloadedOverrides)
        #expect(result.provenance["capabilities.pdfInput"] == .downloadedOverrides)
    }

    @Test("User overrides beat everything, including downloaded overrides")
    func userAlwaysWins() {
        var authoritative = ModelFacts()
        authoritative.capabilities.toolUse = true
        var downloaded = ModelFacts()
        downloaded.capabilities.toolUse = true
        var user = ModelFacts()
        user.capabilities.toolUse = false             // the user says no; the user wins

        let result = ModelFactsMerger.merge(
            authoritative: authoritative, downloadedOverrides: downloaded, userOverrides: user)
        #expect(result.merged.capabilities.toolUse == false)
        #expect(result.provenance["capabilities.toolUse"] == .userOverrides)
    }

    @Test("Empirical fills what the vendor left unknown but never displaces a vendor statement")
    func empiricalGapFills() {
        var authoritative = ModelFacts()
        authoritative.capabilities.vision = true          // vendor states vision
        var empirical = ModelFacts()
        empirical.capabilities.vision = false             // probe disagrees — visible, not winning
        empirical.capabilities.toolUse = true             // vendor silent — probe fills

        let result = ModelFactsMerger.merge(authoritative: authoritative, empirical: empirical)
        #expect(result.merged.capabilities.vision == true)
        #expect(result.merged.capabilities.toolUse == true)
        #expect(result.provenance["capabilities.toolUse"] == .empirical)
        #expect(result.disagreements.contains { $0.field == "capabilities.vision" && $0.dissentingLayer == .empirical })
    }

    @Test("Composites merge atomically: a stated pricing blocks another layer's pricing wholesale")
    func atomicComposites() {
        var authoritative = ModelFacts()
        authoritative.pricing = ModelPricing(base: PricingTier(input: 0.000002, output: 0.000006))
        var enrichment = ModelFacts()
        // LiteLLM has cache rates the vendor didn't state — but stitching them in would
        // fabricate a pricing structure that exists nowhere. Whole-value-or-nothing.
        enrichment.pricing = ModelPricing(base: PricingTier(input: 0.000009, output: 0.000027, cacheRead: 0.000001))

        let result = ModelFactsMerger.merge(authoritative: authoritative, enrichment: enrichment)
        #expect(result.merged.pricing?.base.input == 0.000002)
        #expect(result.merged.pricing?.base.cacheRead == nil)    // NOT stitched in
    }

    @Test("Materialization applies the historical defaults only for still-unknown fields")
    func materializationDefaults() {
        let merged = ModelFactsMerger.merge(authoritative: ModelFacts()).merged
        let info = merged.materialize(providerID: "p", modelID: "m")
        #expect(info.supportsChatCompletions == true)   // nil → true (historical default)
        #expect(info.capabilities.toolUse == false)     // nil → false
        #expect(info.validEffortLevels == [])           // nil → []
        #expect(info.displayName == "m")                // nil → modelID
    }

    @Test("Provenance is empty for fields no layer stated")
    func provenanceOnlyForStatedFields() {
        let result = ModelFactsMerger.merge(authoritative: ModelFacts())
        #expect(result.provenance.isEmpty)
        #expect(result.disagreements.isEmpty)
        #expect(result.layers.isEmpty)
    }
}

/// The stated-facts audit, pinned per provider: a decoder may only state what its vendor actually
/// said. These assert the tri-state distinction directly on the facts (the ModelInfo seams
/// flatten nil → false and can't see it).
@Suite("Decoder stated-facts audit")
struct DecoderStatedFactsTests {
    private let service = ModelFetchService()

    @Test("Anthropic: capabilities block states both directions — but NEVER tool use")
    func anthropicToolUseStaysUnknown() throws {
        let body = #"""
        {"data":[{"id":"claude-test","display_name":"Claude Test","created_at":"2026-01-01T00:00:00Z",
          "max_input_tokens":1000000,"max_tokens":128000,"type":"model",
          "capabilities":{"image_input":{"supported":true},"pdf_input":{"supported":false},
            "thinking":{"supported":true,"types":{"adaptive":{"supported":true},"enabled":{"supported":false}}},
            "code_execution":{"supported":true},"structured_outputs":{"supported":true},
            "effort":{"supported":true,"high":{"supported":true},"max":{"supported":true}}}}]}
        """#
        let facts = try #require(try service.decodeModelFactsForTesting(from: Data(body.utf8), apiType: .anthropic).first).facts
        #expect(facts.capabilities.vision == true)          // stated
        #expect(facts.capabilities.pdfInput == false)       // stated NO — believed
        #expect(facts.capabilities.toolUse == nil)          // the block has no tool key: UNKNOWN, not false
        #expect(facts.validEffortLevels == ["high", "max"]) // rank order
        #expect(facts.behaviorFlags.requiresAdaptiveThinking == true)   // derived from enabled=false
        #expect(facts.behaviorFlags.mustNeverSendTemperatureParam == true)
    }

    @Test("Anthropic: effort.supported=false is a stated empty ladder, absent effort is unknown")
    func anthropicEffortStatement() throws {
        let noEffort = #"{"data":[{"id":"haiku-test","capabilities":{"effort":{"supported":false}}}]}"#
        let factsNo = try #require(try service.decodeModelFactsForTesting(from: Data(noEffort.utf8), apiType: .anthropic).first).facts
        #expect(factsNo.validEffortLevels == [])            // a STATEMENT: no efforts

        let silent = #"{"data":[{"id":"old-test"}]}"#
        let factsSilent = try #require(try service.decodeModelFactsForTesting(from: Data(silent.utf8), apiType: .anthropic).first).facts
        #expect(factsSilent.validEffortLevels == nil)       // said nothing
    }

    @Test("Mistral: a present leaf is stated both directions; an absent leaf is unknown")
    func mistralTriState() throws {
        let body = #"""
        {"data":[{"id":"m1","capabilities":{"function_calling":false,"completion_chat":true}},
                 {"id":"m2"}]}
        """#
        let decoded = try service.decodeModelFactsForTesting(from: Data(body.utf8), apiType: .mistral)
        let m1 = try #require(decoded.first { $0.modelID == "m1" }).facts
        #expect(m1.capabilities.toolUse == false)           // stated NO — believed (kills the old `?? false`)
        #expect(m1.supportsChatCompletions == true)
        #expect(m1.capabilities.vision == nil)              // leaf absent → unknown
        let m2 = try #require(decoded.first { $0.modelID == "m2" }).facts
        #expect(m2.capabilities.toolUse == nil)             // whole block absent → all unknown
        #expect(m2.supportsChatCompletions == nil)
    }

    @Test("HuggingFace: stated supports_tools=false is believed, absent is unknown")
    func huggingFaceTriState() throws {
        let body = #"""
        {"data":[{"id":"org/m","providers":[
          {"provider":"a","status":"live","context_length":1000,"supports_tools":false},
          {"provider":"b","status":"live","context_length":2000}]}]}
        """#
        let decoded = try service.decodeModelFactsForTesting(from: Data(body.utf8), apiType: .huggingFace)
        let a = try #require(decoded.first { $0.modelID == "org/m:a" }).facts
        #expect(a.capabilities.toolUse == false)            // stated false no longer discarded
        let b = try #require(decoded.first { $0.modelID == "org/m:b" }).facts
        #expect(b.capabilities.toolUse == nil)
    }

    @Test("OpenRouter: empty arrays say nothing; non-empty arrays state both directions")
    func openRouterArraySemantics() throws {
        let body = #"""
        {"data":[{"id":"a/no-arrays","context_length":1000},
                 {"id":"b/full","context_length":1000,
                  "architecture":{"input_modalities":["text"]},
                  "supported_parameters":["temperature"]}]}
        """#
        let decoded = try service.decodeModelFactsForTesting(from: Data(body.utf8), apiType: .openRouter)
        let bare = try #require(decoded.first { $0.modelID == "a/no-arrays" }).facts
        #expect(bare.capabilities.vision == nil)
        #expect(bare.capabilities.toolUse == nil)
        #expect(bare.validEffortLevels == nil)              // absent reasoning block ≠ "no efforts"
        let full = try #require(decoded.first { $0.modelID == "b/full" }).facts
        #expect(full.capabilities.vision == false)          // enumerated list without "image" = stated no
        #expect(full.capabilities.toolUse == false)         // enumerated params without "tools" = stated no
    }

    @Test("Plain OpenAI: states almost nothing — the honest record")
    func openAIStatesAlmostNothing() throws {
        let body = #"{"data":[{"id":"gpt-test","created":1700000000,"owned_by":"openai"}]}"#
        let facts = try #require(try service.decodeModelFactsForTesting(from: Data(body.utf8), apiType: .openAICompatible).first).facts
        #expect(facts.capabilities.vision == nil)
        #expect(facts.capabilities.toolUse == nil)
        #expect(facts.supportsChatCompletions == nil)
        #expect(facts.maxInputTokens == nil)
    }

    @Test("Ollama: tool use is positive-only — absence is a hint, not a statement")
    func ollamaPositiveOnly() throws {
        let body = #"""
        {"models":[{"name":"with-tools","size":1,"modified_at":"2026-01-01T00:00:00Z","capabilities":["tools"]},
                   {"name":"without","size":1,"modified_at":"2026-01-01T00:00:00Z","capabilities":["vision"]}]}
        """#
        let decoded = try service.decodeModelFactsForTesting(from: Data(body.utf8), apiType: .ollama)
        #expect(try #require(decoded.first { $0.modelID == "with-tools" }).facts.capabilities.toolUse == true)
        #expect(try #require(decoded.first { $0.modelID == "without" }).facts.capabilities.toolUse == nil)
    }

    @Test("Gemini: absent methods list is unknown chat, present list states both directions")
    func geminiMethodsTriState() throws {
        let body = #"""
        {"models":[{"name":"models/g-chat","supportedGenerationMethods":["generateContent"]},
                   {"name":"models/g-embed","supportedGenerationMethods":["embedContent"]},
                   {"name":"models/g-silent"}]}
        """#
        let decoded = try service.decodeModelFactsForTesting(from: Data(body.utf8), apiType: .gemini)
        #expect(try #require(decoded.first { $0.modelID == "g-chat" }).facts.supportsChatCompletions == true)
        #expect(try #require(decoded.first { $0.modelID == "g-embed" }).facts.supportsChatCompletions == false)
        #expect(try #require(decoded.first { $0.modelID == "g-silent" }).facts.supportsChatCompletions == nil)
    }
}

/// Intentional divergences from the old flattened pipeline — places where the old behavior was
/// wrong and the layered merge deliberately differs. Named as such so nobody "fixes" them back.
@Suite("Intentional divergences from the flattened pipeline")
struct IntentionalDivergenceTests {

    @Test("A vendor-stated capability NO now survives a LiteLLM YES (old pipeline upgraded it)")
    func vendorNoSurvivesLiteLLMYes() {
        // Old: Mistral states function_calling=false → base false; LiteLLM gap-fill
        // "upgrades false→true" because false-meant-unknown. New: the stated false is knowledge.
        var authoritative = ModelFacts()
        authoritative.capabilities.toolUse = false
        var enrichment = ModelFacts()
        enrichment.capabilities.toolUse = true
        let merged = ModelFactsMerger.merge(authoritative: authoritative, enrichment: enrichment).merged
        #expect(merged.capabilities.toolUse == false)
    }

    @Test("A vendor-stated chat YES now survives LiteLLM's chat NO (old pipeline downgraded it)")
    func vendorChatYesSurvivesLiteLLMNo() {
        var authoritative = ModelFacts()
        authoritative.supportsChatCompletions = true    // Gemini stating generateContent
        var enrichment = ModelFacts()
        enrichment.supportsChatCompletions = false      // LiteLLM's mode-derived claim
        let merged = ModelFactsMerger.merge(authoritative: authoritative, enrichment: enrichment).merged
        #expect(merged.supportsChatCompletions == true)
    }

    @Test("Downloaded overrides are now FORCE: they can fix a wrong vendor claim")
    func downloadedOverridesCanFix() {
        // Old: bundled entries were gap-fill and could not repair a wrong vendor value.
        // New: they exist precisely to fix wrong claims, with evidence.
        var authoritative = ModelFacts()
        authoritative.maxOutputTokens = 100_000         // vendor overstates
        var downloaded = ModelFacts()
        downloaded.maxOutputTokens = 64_000             // curated correction
        let merged = ModelFactsMerger.merge(authoritative: authoritative, downloadedOverrides: downloaded).merged
        #expect(merged.maxOutputTokens == 64_000)
    }
}

/// LiteLLM's claims arrive positives-only — its schema omits what it doesn't know, so absence
/// can never become a stated false in the enrichment layer.
@Suite("LiteLLM enrichment conversion")
struct LiteLLMFactsConversionTests {

    @Test("Capability positives convert; absences stay unknown; chat-negative is kept")
    func conversionSemantics() {
        // mode "embedding" makes the computed supportsChatCompletions false — the one genuine
        // negative LiteLLM states.
        let entry = LiteLLMEntry(
            maxInputTokens: 128_000, maxOutputTokens: 16_384, pricing: nil,
            supportsToolUse: true, supportsVision: false, supportsReasoning: false,
            supportsPromptCaching: false, supportsComputerUse: false, supportsAudioInput: false,
            supportsAudioOutput: false, supportsVideoInput: false, supportsResponseSchema: false,
            supportsParallelToolCalls: false, supportsPdfInput: false, supportsWebSearch: false,
            supportsSystemMessages: false, supportsAssistantPrefill: false, supportsToolChoice: false,
            mode: "embedding", supportedEndpoints: nil
        )
        let facts = entry.asFacts
        #expect(facts.capabilities.toolUse == true)     // positive converts
        #expect(facts.capabilities.vision == nil)       // false-in-LiteLLM = didn't say → nil
        #expect(facts.supportsChatCompletions == false) // the one genuine negative it states
        #expect(facts.mode == "embedding")
        #expect(facts.maxInputTokens == 128_000)
    }
}

/// The downloaded-overrides layer is built by folding three key axes with `overlay` — which must
/// be FORCE (later wins), or provider-wide defaults would block the per-model entries meant to
/// refine them.
@Suite("Override-axis overlay ordering")
struct OverlayOrderingTests {
    @Test("overlay is force: the most specific axis, applied last, wins")
    func mostSpecificWins() {
        var folded = ModelFacts()
        var providerWide = ModelFacts()
        providerWide.behaviorFlags.useMaxCompletionTokens = true    // all-provider default
        providerWide.maxOutputTokens = 4096
        var apiTypeScoped = ModelFacts()
        apiTypeScoped.maxOutputTokens = 8192                        // apiType refinement
        var providerScoped = ModelFacts()
        providerScoped.maxOutputTokens = 16_384                     // per-model, most specific

        folded.overlay(providerWide)
        folded.overlay(apiTypeScoped)
        folded.overlay(providerScoped)

        #expect(folded.maxOutputTokens == 16_384)                   // most specific won
        #expect(folded.behaviorFlags.useMaxCompletionTokens == true) // untouched fields survive
    }
}
