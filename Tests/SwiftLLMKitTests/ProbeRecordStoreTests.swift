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

        let facts = profile.asEmpiricalFacts
        #expect(facts.capabilities.toolUse == true)
        #expect(facts.capabilities.vision == nil, "decoded echoes must not wear empirical authority")
        #expect(facts.capabilities.pdfInput == nil)
        #expect(facts.maxOutputTokens == 97922)
    }

    @Test("acceptsTemperature projects only its NEGATIVE, as the no-temperature flag")
    func temperatureProjection() {
        var rejects = ModelProfile(providerID: "p", modelID: "m")
        rejects.acceptsTemperature = .established(false, "400 naming temperature")
        #expect(rejects.asEmpiricalFacts.behaviorFlags.mustNeverSendTemperatureParam == true)

        // A probed TRUE is conditional on the flags active at probe time — never projected.
        var accepts = ModelProfile(providerID: "p", modelID: "m")
        accepts.acceptsTemperature = .established(true, "accepted temperature")
        #expect(accepts.asEmpiricalFacts.behaviorFlags.mustNeverSendTemperatureParam == nil)
    }

    @Test("Record-only fields (isAvailable, round-trip, efforts) never enter the merged facts")
    func recordOnlyFieldsExcluded() {
        var profile = ModelProfile(providerID: "p", modelID: "m")
        profile.isAvailable = .established(false, "404 no longer available")
        profile.toolResultRoundTrip = .established(false, "no identifier")
        profile.effortLevels = ["high": .established(true, "accepted")]
        let facts = profile.asEmpiricalFacts
        #expect(facts.isSilent, "none of these fields have a ModelFacts home yet — record-only")
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
            local: record(newer, at: 1000), downloaded: record(older, at: 0))
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
            local: record(local, at: 0), downloaded: record(downloaded, at: 1000))
        #expect(combined.capabilities.pdfInput == true)
    }

    @Test("A single source passes through; no sources is silence")
    func degenerateCases() {
        var profile = ModelProfile(providerID: "p", modelID: "m")
        profile.chat = .established(true, "echoed")
        #expect(ProbeEvidenceCombiner.combinedFacts(local: record(profile, at: 0), downloaded: nil).supportsChatCompletions == true)
        #expect(ProbeEvidenceCombiner.combinedFacts(local: nil, downloaded: nil).isSilent)
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

        let result = ModelFactsMerger.merge(authoritative: authoritative, empirical: profile.asEmpiricalFacts)
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
        authoritative.supportsChatCompletions = true
        authoritative.maxInputTokens = 1_048_576

        // The probe halted at the reachability gate: isAvailable=false established, everything
        // else untouched. The projection has no mergeable fields — so the merge must carry ZERO
        // empirical claims, and isAvailable stays on the record for the inspector.
        var profile = ModelProfile(providerID: "builtin.gemini", modelID: "gemini-2.0-flash-lite")
        profile.chat = .decoded(true, "supportedGenerationMethods")
        profile.isAvailable = .established(false, "HTTP 404 no longer available")

        let empirical = profile.asEmpiricalFacts
        #expect(empirical.isSilent)

        let result = ModelFactsMerger.merge(authoritative: authoritative, empirical: empirical)
        #expect(result.merged.supportsChatCompletions == true)   // the listing's claim, unchanged
        #expect(result.merged.capabilities.vision == nil)         // nothing fabricated
        #expect(result.provenance.values.allSatisfy { $0 == .authoritative })
        // The empirical truth is preserved on the record itself (hasEstablishedProbedFindings
        // means this run IS persistable — the reachability verdict is real evidence).
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
        profile.effortLevels = ["high": .established(true, "tier-gated acceptance")]
        let record = ProbeRecord(
            proberVersion: 1, recordedAt: Date(timeIntervalSince1970: 0),
            key: ProbeRecordKey(apiType: "alibabaCloud", host: "dashscope.aliyuncs.com", modelID: "qwen-max"),
            providerID: "builtin.alibaba", profile: profile
        )
        let exported = record.strippedForExport
        #expect(exported.profile.isAccessDenied.status == .notAttempted)
        #expect(exported.profile.effortLevels.isEmpty)
        #expect(exported.profile.toolCalling.value == true, "model-scoped evidence survives export")
        // The original is untouched (value semantics).
        #expect(record.profile.isAccessDenied.value == true)
    }
}
