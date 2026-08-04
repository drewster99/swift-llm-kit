import Testing
import Foundation
@testable import SwiftLLMKit

/// The probe → empirical-facts projection: only `established` + `source == .probed` findings
/// project, and the deliberately-excluded fields stay on the record for the inspector.
@Suite("Probe evidence projection")
struct ProbeProjectionTests {

    @Test("Only probed established findings project; decoded and inconclusive never do")
    func projectionSourceRules() {
        var profile = ModelProfile(providerID: "p", modelID: "m")
        profile.toolCalling = .established(true, "round trip completed")     // probed
        profile.vision = .decoded(true, "capabilities.image_input")          // decoded echo
        profile.pdfInput = .inconclusive("rate limited")                     // learned nothing
        profile.maxOutputTokens = .established(97922, "binary search")

        let facts = profile.asEmpiricalFacts(includeAccountScoped: false)
        #expect(facts.capabilities.toolUse == true)
        #expect(facts.capabilities.vision == nil, "decoded echoes must not wear empirical authority")
        #expect(facts.capabilities.pdfInput == nil)
        #expect(facts.maxOutputTokens == 97922)
    }

    @Test("acceptsTemperature projects only its NEGATIVE, as the no-temperature flag")
    func temperatureProjection() {
        var rejects = ModelProfile(providerID: "p", modelID: "m")
        rejects.acceptsTemperature = .established(false, "400 naming temperature")
        #expect(rejects.asEmpiricalFacts(includeAccountScoped: false).behaviorFlags.mustNeverSendTemperatureParam == true)

        // A probed TRUE is conditional on the flags active at probe time — never projected.
        var accepts = ModelProfile(providerID: "p", modelID: "m")
        accepts.acceptsTemperature = .established(true, "accepted temperature")
        #expect(accepts.asEmpiricalFacts(includeAccountScoped: false).behaviorFlags.mustNeverSendTemperatureParam == nil)
    }

    @Test("Promoted fields: model-scoped always project; account-scoped only when gated in")
    func promotionRules() {
        var profile = ModelProfile(providerID: "p", modelID: "m")
        profile.isAvailable = .established(false, "404 no longer available")
        profile.toolResultRoundTrip = .established(false, "no identifier")
        profile.isAccessDenied = .established(true, "Model.AccessDenied")
        profile.reasoningEffortLevels = ["high": .established(true, "accepted")]   // PARTIAL ladder

        // Model-scoped: always. Account-scoped: excluded without the gate.
        let modelScoped = profile.asEmpiricalFacts(includeAccountScoped: false)
        #expect(modelScoped.isAvailable == false)
        #expect(modelScoped.capabilities.toolResultRoundTrip == false)
        #expect(modelScoped.isAccessDenied == nil)
        #expect(modelScoped.reasoningEffort == nil)

        // Account-scoped gate on: isAccessDenied projects; the PARTIAL effort ladder still
        // must not (it would understate the ladder).
        let accountScoped = profile.asEmpiricalFacts(includeAccountScoped: true)
        #expect(accountScoped.isAccessDenied == true)
        #expect(accountScoped.reasoningEffort == nil, "a partial ladder must never project")
    }

    /// Capability findings and the budget ceiling are MODEL-scoped: whether a model accepts
    /// `json_object` does not depend on whose key asked. They were briefly nested inside the
    /// account-scoped gate, which silently discarded every one of them from downloaded records —
    /// those are projected with the gate false on purpose.
    @Test("Probed capabilities project regardless of the account-scoped gate")
    func capabilityFindingsAreModelScoped() {
        var profile = ModelProfile(providerID: "p", modelID: "m")
        profile[.structuredOutputSupportsJSONObject] = .established(true, "returned the requested JSON")
        profile[.toolDefinitionsSupportStrict] = .established(false, "rejected strict")
        profile.maxThinkingBudgetTokens = .established(32_000, "largest accepted")

        for includeAccountScoped in [true, false] {
            let facts = profile.asEmpiricalFacts(includeAccountScoped: includeAccountScoped)
            #expect(facts.capabilities.structuredOutputSupportsJSONObject == true,
                    "dropped with includeAccountScoped=\(includeAccountScoped)")
            #expect(facts.capabilities.toolDefinitionsSupportStrict == false,
                    "a stated NO must project too, not just a YES")
            #expect(facts.maxThinkingBudgetTokens == 32_000)
        }
    }

    /// Only PROBED findings project. A decoded seed is already in the authoritative layer, and
    /// re-projecting it as empirical would let it outrank the vendor that stated it.
    @Test("A decoded capability finding does not project as empirical")
    func decodedFindingsDoNotProject() {
        var profile = ModelProfile(providerID: "p", modelID: "m")
        profile[.structuredOutputSupportsJSONObject] = .decoded(true, "stated in /models")
        let facts = profile.asEmpiricalFacts(includeAccountScoped: true)
        #expect(facts.capabilities.structuredOutputSupportsJSONObject == nil)
    }

    @Test("Effort levels project only from a COMPLETE-ladder probe")
    func completeLadderProjection() {
        var profile = ModelProfile(providerID: "p", modelID: "m")
        for level in EffortRank.allKnown {
            profile.reasoningEffortLevels[level] = .established(["low", "medium", "high"].contains(level), "probed")
        }
        let facts = profile.asEmpiricalFacts(includeAccountScoped: true)
        #expect(facts.reasoningEffort == .levels(["low", "medium", "high"]))
        #expect(facts.generalEffort == nil, "the untouched construct stays silent")
    }

    @Test("The completeness gate: decoded-only or inconclusive-only profiles are not persistable")
    func completenessGate() {
        var decodedOnly = ModelProfile(providerID: "p", modelID: "m")
        decodedOnly.vision = .decoded(true, "payload")
        #expect(!decodedOnly.hasEstablishedProbedFindings)

        var gutted = ModelProfile(providerID: "p", modelID: "m")
        gutted.chat = .inconclusive("429 everywhere")
        #expect(!gutted.hasEstablishedProbedFindings)

        var real = ModelProfile(providerID: "p", modelID: "m")
        real.chat = .established(true, "echoed the identifier")
        #expect(real.hasEstablishedProbedFindings)
    }
}

/// The store: per-record files, atomic replace, complete-run-only, cross-run replace-wholesale.
@Suite("Probe record store")
struct ProbeRecordStoreTests {
    private func makeTempStore() throws -> ProbeRecordStore {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("probe-store-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return ProbeRecordStore(baseDirectory: dir)
    }

    private var probedProfile: ModelProfile {
        var profile = ModelProfile(providerID: "builtin.zai", modelID: "glm-5.2")
        profile.chat = .established(true, "echoed")
        profile.toolCalling = .established(true, "round trip")
        return profile
    }

    @Test("Upsert writes a record; a later complete run replaces it wholesale")
    func upsertAndReplace() throws {
        let store = try makeTempStore()
        let key = ProbeRecordKey(apiType: "zai", host: "api.z.ai", modelID: "glm-5.2")

        #expect(try store.upsert(profile: probedProfile, key: key, providerID: "builtin.zai", proberVersion: 1))
        let first = try #require(store.record(forKey: key))
        #expect(first.profile.toolCalling.value == true)
        #expect(first.proberVersion == 1)
        #expect(first.schemaVersion == ProbeRecord.currentSchemaVersion)

        // A newer complete run with a DIFFERENT shape replaces wholesale — no stitching.
        var rerun = ModelProfile(providerID: "builtin.zai", modelID: "glm-5.2")
        rerun.chat = .established(true, "echoed")
        rerun.vision = .established(false, "rejected the image")
        #expect(try store.upsert(profile: rerun, key: key, providerID: "builtin.zai", proberVersion: 2))
        let second = try #require(store.record(forKey: key))
        #expect(second.proberVersion == 2)
        #expect(second.profile.vision.value == false)
        #expect(second.profile.toolCalling.status == .notAttempted, "replace-wholesale: no cross-run stitching")
    }

    @Test("A run with no established probed findings is rejected and does not clobber")
    func abortedRunRejected() throws {
        let store = try makeTempStore()
        let key = ProbeRecordKey(apiType: "zai", host: "api.z.ai", modelID: "glm-5.2")
        #expect(try store.upsert(profile: probedProfile, key: key, providerID: "builtin.zai", proberVersion: 1))

        var gutted = ModelProfile(providerID: "builtin.zai", modelID: "glm-5.2")
        gutted.chat = .inconclusive("rate limited")
        #expect(try !store.upsert(profile: gutted, key: key, providerID: "builtin.zai", proberVersion: 1))
        #expect(store.record(forKey: key)?.profile.toolCalling.value == true, "real record survived")
    }

    @Test("Keys with slashes and colons map to distinct, stable filenames")
    func fileNameSafety() {
        let a = ProbeRecordKey(apiType: "huggingFace", host: "router.huggingface.co", modelID: "zai-org/GLM-5.2:novita")
        let b = ProbeRecordKey(apiType: "huggingFace", host: "router.huggingface.co", modelID: "zai-org/GLM-5.2:together")
        #expect(a.fileName != b.fileName)
        #expect(!a.fileName.contains("/"))
        #expect(a.fileName == a.fileName, "deterministic")
    }

    @Test("Host normalization keeps ports and lowercases; providerID is not part of identity")
    func keyNormalization() throws {
        let localA = ProbeRecordKey(apiType: .openAICompatible, endpoint: try #require(URL(string: "http://Localhost:8080/v1")), modelID: "m")
        let localB = ProbeRecordKey(apiType: .openAICompatible, endpoint: try #require(URL(string: "http://localhost:8081/v1")), modelID: "m")
        #expect(localA.host == "localhost:8080")
        #expect(localA != localB, "different ports are different servers")
    }

    @Test("loadAll skips unreadable files instead of failing the layer")
    func corruptRecordSkipped() throws {
        let store = try makeTempStore()
        let key = ProbeRecordKey(apiType: "zai", host: "api.z.ai", modelID: "glm-5.2")
        _ = try store.upsert(profile: probedProfile, key: key, providerID: "builtin.zai", proberVersion: 1)
        try Data("not json".utf8).write(to: store.directory.appendingPathComponent("garbage.json"))
        let records = store.loadAll()
        #expect(records.count == 1)
    }
}

/// Local + downloaded evidence combine per-field with the newer record winning — and an
/// established finding never losing to the newer run's unknown.
@Suite("Probe evidence combining")
struct ProbeEvidenceCombinerTests {
    private func record(_ profile: ModelProfile, at time: TimeInterval) -> ProbeRecord {
        ProbeRecord(
            proberVersion: 1, recordedAt: Date(timeIntervalSince1970: time),
            key: ProbeRecordKey(apiType: "zai", host: "api.z.ai", modelID: "glm-5.2"),
            providerID: "builtin.zai", profile: profile
        )
    }

    @Test("Newer record wins per field; older fills what the newer left unknown")
    func newestWinsPerField() {
        var older = ModelProfile(providerID: "p", modelID: "m")
        older.toolCalling = .established(true, "old run")
        older.vision = .established(true, "old run saw it")
        var newer = ModelProfile(providerID: "p", modelID: "m")
        newer.toolCalling = .established(false, "new run — vendor changed something")
        // newer never attempted vision (policy halt) — older's established survives.

        let combined = ProbeEvidenceCombiner.combinedFacts(
            local: record(newer, at: 1000), downloaded: record(older, at: 0), forProviderID: "builtin.zai")
        #expect(combined.capabilities.toolUse == false)   // newer established wins
        #expect(combined.capabilities.vision == true)      // established never loses to unknown
    }

    @Test("Direction is symmetric: a newer DOWNLOADED record beats an older local one")
    func downloadedCanBeNewer() {
        var local = ModelProfile(providerID: "p", modelID: "m")
        local.pdfInput = .established(false, "probed before the encoder fix")
        var downloaded = ModelProfile(providerID: "p", modelID: "m")
        downloaded.pdfInput = .established(true, "probed with the fixed encoder")

        let combined = ProbeEvidenceCombiner.combinedFacts(
            local: record(local, at: 0), downloaded: record(downloaded, at: 1000), forProviderID: "builtin.zai")
        #expect(combined.capabilities.pdfInput == true)
    }

    @Test("A single source passes through; no sources is silence")
    func degenerateCases() {
        var profile = ModelProfile(providerID: "p", modelID: "m")
        profile.chat = .established(true, "echoed")
        #expect(ProbeEvidenceCombiner.combinedFacts(local: record(profile, at: 0), downloaded: nil, forProviderID: "builtin.zai").capabilities.chat == true)
        #expect(ProbeEvidenceCombiner.combinedFacts(local: nil, downloaded: nil, forProviderID: "builtin.zai").isSilent)
    }
}

/// The two golden cases the design demands, end to end through the five-layer merge.
@Suite("Golden probe-layer merges")
struct GoldenProbeMergeTests {

    @Test("glm-5.2: vendor silent on tools, probe established true — the probe is the only truth")
    func glmProbeOnlyTruth() {
        // z.ai's /models is bare (id/created/owned_by): the authoritative layer is silent on tools.
        let authoritative = ModelFacts()
        var profile = ModelProfile(providerID: "builtin.zai", modelID: "glm-5.2")
        profile.chat = .established(true, "echoed the identifier")
        profile.toolCalling = .established(true, "returned the identifier")
        profile.maxOutputTokens = .established(97922, "binary search")

        let result = ModelFactsMerger.merge(authoritative: authoritative, empirical: profile.asEmpiricalFacts(includeAccountScoped: true))
        #expect(result.merged.capabilities.toolUse == true)
        #expect(result.merged.maxOutputTokens == 97922)
        #expect(result.provenance["capabilities.toolUse"] == .empirical)

        let info = result.merged.materialize(providerID: "builtin.zai", modelID: "glm-5.2")
        #expect(info.capabilities.toolUse == true)
    }

    @Test("Dead-but-listed Gemini model: the probe fabricates nothing into the merge")
    func deadButListedFabricatesNothing() {
        // Gemini still lists the model with generateContent — authoritative states chat=true.
        var authoritative = ModelFacts()
        authoritative.capabilities.chat = true
        authoritative.maxInputTokens = 1_048_576

        // The probe halted at the reachability gate: isAvailable=false established, everything
        // else untouched. The ONLY empirical claim that may flow is the honest isAvailable=false
        // (now promoted into the merge); no capability may be fabricated.
        var profile = ModelProfile(providerID: "builtin.gemini", modelID: "gemini-2.0-flash-lite")
        profile.chat = .decoded(true, "supportedGenerationMethods")
        profile.isAvailable = .established(false, "HTTP 404 no longer available")

        let empirical = profile.asEmpiricalFacts(includeAccountScoped: true)
        #expect(empirical.isAvailable == false)
        #expect(empirical.capabilities.vision == nil)

        let result = ModelFactsMerger.merge(authoritative: authoritative, empirical: empirical)
        #expect(result.merged.capabilities.chat == true)   // the listing's claim, unchanged
        #expect(result.merged.capabilities.vision == nil)         // nothing fabricated
        #expect(result.merged.isAvailable == false)               // the dead model IS marked dead
        #expect(result.provenance["isAvailable"] == .empirical)
        let info = result.merged.materialize(providerID: "builtin.gemini", modelID: "gemini-2.0-flash-lite")
        #expect(info.isAvailable == false, "the merged catalog now carries the death certificate")
        #expect(profile.hasEstablishedProbedFindings)
    }
}

/// Export stripping: account-scoped findings never leave the machine.
@Suite("Probe record export stripping")
struct ProbeRecordExportTests {
    @Test("isAccessDenied and effort levels are stripped; model-scoped findings survive")
    func stripsAccountScopedFields() {
        var profile = ModelProfile(providerID: "builtin.alibaba", modelID: "qwen-max")
        profile.toolCalling = .established(true, "round trip")
        profile.isAccessDenied = .established(true, "Model.AccessDenied — THIS account's key")
        profile.reasoningEffortLevels = ["high": .established(true, "tier-gated acceptance")]
        let record = ProbeRecord(
            proberVersion: 1, recordedAt: Date(timeIntervalSince1970: 0),
            key: ProbeRecordKey(apiType: "alibabaCloud", host: "dashscope.aliyuncs.com", modelID: "qwen-max"),
            providerID: "builtin.alibaba", profile: profile
        )
        let exported = record.strippedForExport
        #expect(exported.profile.isAccessDenied.status == .notAttempted)
        #expect(exported.profile.reasoningEffortLevels.isEmpty)
        #expect(exported.profile.generalEffortLevels.isEmpty)
        #expect(exported.profile.toolCalling.value == true, "model-scoped evidence survives export")
        // The original is untouched (value semantics).
        #expect(record.profile.isAccessDenied.value == true)
    }
}

/// Prober version outranks recency: a fixed prober's findings must not lose to a later run of a
/// broken one.
@Suite("Prober version precedence")
struct ProberVersionPrecedenceTests {
    private func record(_ profile: ModelProfile, at time: TimeInterval, proberVersion: Int) -> ProbeRecord {
        ProbeRecord(
            proberVersion: proberVersion, recordedAt: Date(timeIntervalSince1970: time),
            key: ProbeRecordKey(apiType: "zai", host: "api.z.ai", modelID: "glm-5.2"),
            providerID: "builtin.zai", profile: profile
        )
    }

    @Test("A newer-TIMESTAMP record from an OLDER prober loses to a fixed prober's record")
    func versionBeatsTimestamp() {
        var fixedProber = ModelProfile(providerID: "p", modelID: "m")
        fixedProber.pdfInput = .established(true, "probed with the fixed encoder")     // v2, older
        var brokenProber = ModelProfile(providerID: "p", modelID: "m")
        brokenProber.pdfInput = .established(false, "our own encoder bug")             // v1, newer

        let combined = ProbeEvidenceCombiner.combinedFacts(
            local: record(fixedProber, at: 500, proberVersion: 2),
            downloaded: record(brokenProber, at: 1000, proberVersion: 1), forProviderID: "builtin.zai")
        #expect(combined.capabilities.pdfInput == true, "proberVersion outranks recency")
    }

    @Test("Same prober version: newest wins; exact ties go to local")
    func sameVersionNewestWins() {
        var older = ModelProfile(providerID: "p", modelID: "m")
        older.vision = .established(true, "old")
        var newer = ModelProfile(providerID: "p", modelID: "m")
        newer.vision = .established(false, "new")
        let combined = ProbeEvidenceCombiner.combinedFacts(
            local: record(older, at: 0, proberVersion: 1),
            downloaded: record(newer, at: 1000, proberVersion: 1), forProviderID: "builtin.zai")
        #expect(combined.capabilities.vision == false)
    }
}

/// The facts-based seed preserves tri-state: a field seeds as decoded exactly when the vendor
/// stated it — never fabricated from materialization's nil→false flattening.
@Suite("Facts-based probe seeding")
struct FactsSeedTests {
    @Test("Unstated fields stay notAttempted (the probe will run); stated fields seed both directions")
    func triStateSeeding() {
        var facts = ModelFacts()
        facts.capabilities.vision = true       // stated yes
        facts.capabilities.pdfInput = nil      // NOT stated (e.g. a payload missing the leaf)
        facts.capabilities.toolUse = false     // stated NO (e.g. Mistral function_calling=false)
        facts.capabilities.chat = true
        facts.maxInputTokens = 200_000

        let seed = ModelProber.seedProfile(
            fromDecodedFacts: DecodedModelFacts(modelID: "m", facts: facts), providerID: "p")
        #expect(seed.vision.value == true && seed.vision.source == .decoded)
        #expect(seed.pdfInput.status == .notAttempted, "unstated must stay probe-able — no fabricated decoded(false)")
        #expect(seed.toolCalling.value == false, "a stated no is believed")
        #expect(seed.chat.value == true)
        #expect(seed.maxContextTokens.value == 200_000)
        #expect(seed.isAvailable.value == true && seed.isAvailable.source == .decoded)
    }

    @Test("Future-schema probe records are rejected, not trusted")
    func futureSchemaRejected() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("schema-tests-\(UUID().uuidString)", isDirectory: true)
        let store = ProbeRecordStore(baseDirectory: dir)
        var profile = ModelProfile(providerID: "p", modelID: "m")
        profile.chat = .established(true, "echoed")
        let key = ProbeRecordKey(apiType: "zai", host: "h", modelID: "m")
        _ = try store.upsert(profile: profile, key: key, providerID: "p", proberVersion: 1)

        // Rewrite the record claiming a future schema; it must be skipped by both readers.
        let url = store.directory.appendingPathComponent(key.fileName)
        var text = try String(contentsOf: url, encoding: .utf8)
        text = text.replacingOccurrences(
            of: "\"schemaVersion\" : \(ProbeRecord.currentSchemaVersion)", with: "\"schemaVersion\" : 99")
        try text.write(to: url, atomically: true, encoding: .utf8)
        #expect(store.record(forKey: key) == nil)
        #expect(store.loadAll().isEmpty)
    }
}

/// The account-scoping hardening and the capability round-trip, pinned.
@Suite("Account-scoping and capability serialization")
struct AccountScopingHardeningTests {
    @Test("A downloaded record NEVER contributes account-scoped facts, even with a matching builtin providerID")
    func downloadedNeverAccountScoped() {
        // A shipped record that (wrongly) escaped export-stripping, carrying the dev account's
        // access denial, annotated with a builtin providerID identical on every machine.
        var profile = ModelProfile(providerID: "builtin.alibaba", modelID: "qwen-max")
        profile.chat = .established(true, "echoed")
        profile.isAccessDenied = .established(true, "the DEV account's restriction")
        let shipped = ProbeRecord(
            proberVersion: 1, recordedAt: Date(timeIntervalSince1970: 1000),
            key: ProbeRecordKey(apiType: "alibabaCloud", host: "dashscope.aliyuncs.com", modelID: "qwen-max"),
            providerID: "builtin.alibaba", profile: profile
        )
        let combined = ProbeEvidenceCombiner.combinedFacts(
            local: nil, downloaded: shipped, forProviderID: "builtin.alibaba")
        #expect(combined.capabilities.chat == true)     // model-scoped flows
        #expect(combined.isAccessDenied == nil, "another account's restriction must never project")
    }

    @Test("toolResultRoundTrip survives a ModelCapabilities encode/decode round-trip")
    func roundTripCapabilityCodable() throws {
        let original = ModelCapabilities(toolUse: true, toolResultRoundTrip: true)
        let decoded = try JSONDecoder().decode(ModelCapabilities.self, from: JSONEncoder().encode(original))
        #expect(decoded == original)
        #expect(decoded.toolResultRoundTrip == true)
    }
}

/// The self-healing prune: a bug-fix re-probe that halts at authoritatively-non-chat with no
/// probed findings must be able to remove a stale record it can no longer overwrite — but only
/// when that record holds no real capability measurement.
@Suite("Self-healing prune")
struct SelfHealingPruneTests {
    private func makeTempStore() throws -> ProbeRecordStore {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("prune-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return ProbeRecordStore(baseDirectory: dir)
    }

    /// A record shaped exactly like the Gemini gone-bug: the only probed finding is a (now wrong)
    /// isAvailable; there is no capability measurement.
    private var bugRecordProfile: ModelProfile {
        var p = ModelProfile(providerID: "builtin.gemini", modelID: "gemini-embedding-001")
        p.isAvailable = .established(false, "HTTP 404: ... not supported for generateContent")
        return p
    }

    @Test("delete removes an existing record and reports it")
    func deleteWorks() throws {
        let store = try makeTempStore()
        let key = ProbeRecordKey(apiType: "gemini", host: "generativelanguage.googleapis.com", modelID: "gemini-embedding-001")
        _ = try store.upsert(profile: bugRecordProfile, key: key, providerID: "builtin.gemini", proberVersion: 3)
        #expect(store.record(forKey: key) != nil)
        #expect(try store.delete(forKey: key) == true)
        #expect(store.record(forKey: key) == nil)
        #expect(try store.delete(forKey: key) == false)   // idempotent: nothing left to delete
    }

    @Test("hasProbedCapabilityFindings excludes availability/access flags")
    func capabilityFindingsExcludeAvailability() {
        // The bug record: isAvailable probed, no capability → false.
        #expect(!bugRecordProfile.hasProbedCapabilityFindings)
        #expect(bugRecordProfile.hasEstablishedProbedFindings)   // isAvailable still counts here

        // A record with a real probed capability → true, must never be pruned.
        var real = ModelProfile(providerID: "p", modelID: "m")
        real.vision = .established(true, "named 'blue triangle'")
        #expect(real.hasProbedCapabilityFindings)
    }

    @Test("isAuthoritativelyNonChat only when the payload decoded chat=false")
    func authoritativeNonChat() {
        var decodedNonChat = ModelProfile(providerID: "p", modelID: "m")
        decodedNonChat.chat = .decoded(false, "provider /models payload")       // decoded false
        #expect(decodedNonChat.isAuthoritativelyNonChat)

        var probedNonChat = ModelProfile(providerID: "p", modelID: "m")
        probedNonChat.chat = .established(false, "not a chat model")            // probed false (source .probed)
        #expect(!probedNonChat.isAuthoritativelyNonChat)                        // probed, not decoded

        #expect(!ModelProfile(providerID: "p", modelID: "m").isAuthoritativelyNonChat)  // notAttempted
    }
}

/// The app-bundled probe seed: shipped evidence a fresh install starts from.
@Suite("Bundled probe seed")
struct BundledProbeSeedTests {
    @Test("The bundled seed loads and decodes into valid records")
    func bundledSeedLoads() {
        let seed = ProbeRecordStore.bundledSeedRecords()
        #expect(!seed.isEmpty, "bundled_probe_records.json should ship with records")
        // Every record decoded cleanly and is at or below the current schema.
        #expect(seed.allSatisfy { $0.schemaVersion <= ProbeRecord.currentSchemaVersion })
        // The four probed providers are represented.
        let providers = Set(seed.map(\.providerID))
        #expect(providers.contains("builtin.openai"))
        #expect(providers.contains("builtin.anthropic"))
        // A sample record carries a real profile (chat finding present) and no error-trace refs.
        if let anthropic = seed.first(where: { $0.providerID == "builtin.anthropic" }) {
            #expect(anthropic.profile.chat.status != .notAttempted)
        }
        #expect(seed.allSatisfy { !($0.profile.isAvailable.evidence ?? "").contains("ref:") },
                "shipped evidence should have provider error-trace refs stripped")
    }

    @Test("The seed is account-scoped-stripped: no established isAccessDenied")
    func seedIsAccountStripped() {
        let seed = ProbeRecordStore.bundledSeedRecords()
        #expect(seed.allSatisfy { $0.profile.isAccessDenied.status != .established },
                "account-scoped access-denial must never ship in the bundled seed")
    }

    @Test("strippedForExport removes provider error-trace refs but keeps the message")
    func stripsEvidenceTraceRefs() {
        var profile = ModelProfile(providerID: "p", modelID: "m")
        profile.isAvailable = .established(false, "HTTP 410: {\"error\":\"glm-4.6 was retired (ref: 8c022417-b1b1-4d91-a7f2-61fd7c3dad86)\"}")
        profile.vision = .inconclusive("HTTP 500: {\"error\":\"Internal Server Error (ref: f0b70fbe-81cb-4a35-94e5-57e3605bfadf)\"}")
        let stripped = ProbeRecord(proberVersion: ModelProber.proberVersion, recordedAt: Date(),
                                   key: ProbeRecordKey(apiType: "ollama", host: "h", modelID: "m"),
                                   providerID: "p", profile: profile).strippedForExport
        #expect(!(stripped.profile.isAvailable.evidence ?? "").contains("ref:"))
        #expect(!(stripped.profile.vision.evidence ?? "").contains("ref:"))
        #expect((stripped.profile.isAvailable.evidence ?? "").contains("retired"), "the human-readable message must survive")
    }
}
