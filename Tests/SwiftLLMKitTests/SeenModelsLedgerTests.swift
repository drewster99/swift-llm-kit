import Testing
import Foundation
@testable import SwiftLLMKit

/// The discovery ledger's load-bearing rule: seeding is PER PROVIDER — a provider's first-ever
/// listing seeds silently and reports NOTHING as new. That covers all three storm scenarios:
/// an existing install's first run of this feature, a fresh install's second provider, and
/// pasting a new provider key whose catalog lists hundreds of models.
@Suite("Seen-models ledger")
struct SeenModelsLedgerTests {

    @Test("A provider's first listing seeds silently and reports nothing new")
    func firstListingSeedsSilently() {
        var ledger = SeenModelsLedger()
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        let fresh = ledger.observe(providerID: "p", keys: ["p/a", "p/b"], at: now)
        #expect(fresh.isEmpty, "a first listing must never report discoveries")
        #expect(ledger.seededAt == now)
        #expect(ledger.seenKeys == ["p/a", "p/b"])
    }

    @Test("A SECOND provider's first listing is also silent (the fresh-install storm case)")
    func secondProviderFirstListingIsSilent() {
        var ledger = SeenModelsLedger()
        ledger.observe(providerID: "p", keys: ["p/a"], at: Date(timeIntervalSince1970: 0))
        // Provider q's entire 343-model catalog arrives for the first time: NOT discoveries.
        let fresh = ledger.observe(providerID: "q", keys: ["q/1", "q/2", "q/3"], at: Date(timeIntervalSince1970: 50))
        #expect(fresh.isEmpty, "a newly added provider's first catalog must seed, not storm")
        #expect(ledger.seenKeys.contains("q/2"))
    }

    @Test("After a provider is seeded, only genuinely unseen keys are reported")
    func discoveryAfterSeeding() {
        var ledger = SeenModelsLedger()
        ledger.observe(providerID: "p", keys: ["p/a"], at: Date(timeIntervalSince1970: 0))
        let fresh = ledger.observe(providerID: "p", keys: ["p/a", "p/new"], at: Date(timeIntervalSince1970: 100))
        #expect(fresh == ["p/new"])
        #expect(ledger.seenKeys == ["p/a", "p/new"])
    }

    @Test("Keys are never removed: a delisted-then-relisted model is not a discovery")
    func delistingIsNotForgetting() {
        var ledger = SeenModelsLedger()
        ledger.observe(providerID: "p", keys: ["p/a", "p/gone"], at: Date(timeIntervalSince1970: 0))
        // "p/gone" absent from this observation — stays seen.
        _ = ledger.observe(providerID: "p", keys: ["p/a"], at: Date(timeIntervalSince1970: 100))
        #expect(ledger.seenKeys.contains("p/gone"))
        // Relisting it later reports nothing.
        let fresh = ledger.observe(providerID: "p", keys: ["p/a", "p/gone"], at: Date(timeIntervalSince1970: 200))
        #expect(fresh.isEmpty)
    }

    @Test("A seeded provider with an empty first listing discovers on the next listing")
    func emptyFirstListingStillSeeds() {
        var ledger = SeenModelsLedger()
        // Provider was reachable but listed nothing (e.g. local Ollama with no models pulled).
        ledger.observe(providerID: "p", keys: [], at: Date(timeIntervalSince1970: 0))
        let fresh = ledger.observe(providerID: "p", keys: ["p/a"], at: Date(timeIntervalSince1970: 100))
        #expect(fresh == ["p/a"], "the provider was seeded (empty); a later listing is a real discovery")
    }

    @Test("Round-trips through Codable")
    func codableRoundTrip() throws {
        let original = SeenModelsLedger(
            seededAt: Date(timeIntervalSince1970: 1_800_000_000),
            seededProviders: ["p", "q"],
            seenKeys: ["p/a", "q/b"]
        )
        let decoded = try JSONDecoder().decode(SeenModelsLedger.self, from: JSONEncoder().encode(original))
        #expect(decoded == original)
    }

    @Test("An older-schema ledger missing keys decodes to defaults instead of throwing")
    func missingKeysDecodeToDefaults() throws {
        let legacy = #"{"seededAt": 800000000}"#
        let decoded = try JSONDecoder().decode(SeenModelsLedger.self, from: Data(legacy.utf8))
        #expect(decoded.seededProviders.isEmpty)
        #expect(decoded.seenKeys.isEmpty)
    }
}
