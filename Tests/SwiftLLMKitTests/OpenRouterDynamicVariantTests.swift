import Foundation
import Testing
@testable import SwiftLLMKit

/// OpenRouter's dynamic routing suffixes appear in `/models` nowhere, so the decoder states them
/// itself. These pin the two halves that matter: that the rows exist (a model nobody can see is a
/// model nobody can pick or probe) and that they inherit enough to resolve pricing.
@Suite("OpenRouter dynamic routing variants")
struct OpenRouterDynamicVariantTests {

    private func decode(_ body: String) throws -> [ModelInfo] {
        try ModelFetchService().decodeOpenRouterModelsForTesting(
            from: Data(body.utf8), providerID: "builtin.openrouter")
    }

    @Test("A base model yields itself plus :floor and :nitro")
    func baseModelGainsBothVariants() throws {
        let body = #"{"data":[{"id":"qwen/qwen3.5-397b-a17b","name":"Qwen3.5 397B","context_length":262144,"pricing":{"prompt":"0.00000039","completion":"0.00000234"}}]}"#
        let ids = try decode(body).map(\.modelID)
        #expect(ids == [
            "qwen/qwen3.5-397b-a17b",
            "qwen/qwen3.5-397b-a17b:floor",
            "qwen/qwen3.5-397b-a17b:nitro",
        ])
    }

    /// `foo/model:free:floor` is not a slug OpenRouter accepts. A colon is the marker of an
    /// already-suffixed id, so static variants (`:free`, `:batch`, `:thinking`, `:extended`) are
    /// passed through untouched.
    @Test("Static variants do not stack another suffix")
    func staticVariantsAreNotStacked() throws {
        let body = #"{"data":[{"id":"a/b:free","name":"B free"},{"id":"a/b:batch","name":"B batch"},{"id":"a/b:thinking","name":"B thinking"}]}"#
        let ids = try decode(body).map(\.modelID)
        #expect(ids == ["a/b:batch", "a/b:free", "a/b:thinking"])
        #expect(!ids.contains { $0.hasSuffix(":floor") || $0.hasSuffix(":nitro") })
    }

    /// A suffixed id absent from the catalog makes an exact-match pricing lookup miss, and a
    /// missed price reads as a free request rather than an error. Inheriting the base row is what
    /// keeps that from happening.
    @Test("Variants inherit the base model's pricing and capabilities")
    func variantsInheritFacts() throws {
        let body = #"{"data":[{"id":"x/y","name":"Y","context_length":128000,"supported_parameters":["tools","tool_choice"],"top_provider":{"max_completion_tokens":32000},"pricing":{"prompt":"0.000005","completion":"0.000025"}}]}"#
        let models = try decode(body)
        let base = try #require(models.first { $0.modelID == "x/y" })
        for variant in ["x/y:floor", "x/y:nitro"] {
            let m = try #require(models.first { $0.modelID == variant })
            #expect(m.pricing?.base.input == base.pricing?.base.input)
            #expect(m.pricing?.base.output == base.pricing?.base.output)
            #expect(m.maxInputTokens == base.maxInputTokens)
            #expect(m.maxOutputTokens == base.maxOutputTokens)
            #expect(m.capabilities.toolUse == true)
        }
    }

    @Test("Display names distinguish the variants from the base")
    func displayNamesAreDistinct() throws {
        let body = #"{"data":[{"id":"x/y","name":"Why"}]}"#
        let models = try decode(body)
        #expect(models.first { $0.modelID == "x/y" }?.displayName == "Why")
        #expect(models.first { $0.modelID == "x/y:floor" }?.displayName == "Why (floor — cheapest route)")
        #expect(models.first { $0.modelID == "x/y:nitro" }?.displayName == "Why (nitro — fastest route)")
    }

    /// The base description is kept, not replaced — the variant only prepends what it changes.
    @Test("Variant descriptions lead with the routing note and keep the original")
    func descriptionsExplainTheRouting() throws {
        let body = #"{"data":[{"id":"x/y","name":"Why","description":"A model."}]}"#
        let models = try decode(body)
        let floor = try #require(models.first { $0.modelID == "x/y:floor" }?.modelDescription)
        #expect(floor.hasPrefix("Routing variant:"))
        #expect(floor.contains("sorts providers by price"))
        #expect(floor.hasSuffix("A model."))
        let nitro = try #require(models.first { $0.modelID == "x/y:nitro" }?.modelDescription)
        #expect(nitro.contains("sorts providers by throughput"))
    }

    /// A model with no description of its own must still get the routing note, not an empty
    /// string or a stray separator.
    @Test("A description-less base still explains its variants")
    func descriptionlessBaseStillExplained() throws {
        let models = try decode(#"{"data":[{"id":"x/y","name":"Why"}]}"#)
        let floor = try #require(models.first { $0.modelID == "x/y:floor" }?.modelDescription)
        #expect(floor.hasPrefix("Routing variant:"))
        #expect(!floor.hasSuffix("\n\n"))
    }

    /// `:exacto` is documented but never exercised against the live API here, and `:online` is
    /// deprecated. Neither is offered until someone confirms it works.
    @Test("Only floor and nitro are synthesized")
    func onlyTwoVariantsExist() throws {
        #expect(OpenRouterDynamicVariant.allCases.map(\.rawValue) == ["floor", "nitro"])
        let ids = try decode(#"{"data":[{"id":"x/y","name":"Why"}]}"#).map(\.modelID)
        #expect(!ids.contains { $0.contains(":exacto") || $0.contains(":online") })
    }

    /// The suffix and `provider.sort` are two spellings of one instruction; a consumer sending
    /// the body field instead of the suffix must get the same routing.
    @Test("Each variant maps to its provider.sort equivalent")
    func providerSortMapping() {
        #expect(OpenRouterDynamicVariant.floor.suffix == ":floor")
        #expect(OpenRouterDynamicVariant.nitro.suffix == ":nitro")
        #expect(OpenRouterDynamicVariant.floor.providerSortValue == "price")
        #expect(OpenRouterDynamicVariant.nitro.providerSortValue == "throughput")
    }

    /// The static set moves — `:extended` is documented with zero entries today — so a suffix
    /// being dynamic now is no promise it stays that way. If OpenRouter ever lists one, its own
    /// entry must stand alone rather than being doubled by the row we would have invented.
    @Test("A listed variant is never shadowed by a synthesized duplicate")
    func statedVariantWins() throws {
        let body = #"{"data":[{"id":"x/y","name":"Why"},{"id":"x/y:floor","name":"Why floor, as listed","description":"Stated by OpenRouter."}]}"#
        let models = try decode(body)
        #expect(models.map(\.modelID) == ["x/y", "x/y:floor", "x/y:nitro"])
        let floor = try #require(models.first { $0.modelID == "x/y:floor" })
        #expect(floor.displayName == "Why floor, as listed")
        #expect(floor.modelDescription == "Stated by OpenRouter.")
    }

    @Test("Suffix eligibility is decided by the presence of a colon")
    func suffixEligibility() {
        #expect(OpenRouterDynamicVariant.acceptsDynamicSuffix(modelID: "a/b"))
        #expect(!OpenRouterDynamicVariant.acceptsDynamicSuffix(modelID: "a/b:free"))
        #expect(!OpenRouterDynamicVariant.acceptsDynamicSuffix(modelID: "a/b:floor"))
    }
}
