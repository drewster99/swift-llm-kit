import Foundation
import CryptoKit
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "ProbeRecordStore")

/// Model-scoped identity for probe evidence: `(apiType, normalized endpoint host, modelID)`.
///
/// NOT keyed by providerID: user-created providers carry local UUIDs, so providerID-keyed
/// evidence dies when a provider row is deleted and recreated, and downloaded (shipped) records
/// could never address them at all. The same modelID on a different HOST is genuinely a
/// different thing (glm-5.2 on z.ai vs Ollama-cloud vs a vLLM box differ in caps and limits),
/// so the host stays in the key; HuggingFace's per-provider `model:provider` composites are
/// disambiguated by their modelID alone but still carry the router host.
public struct ProbeRecordKey: Codable, Sendable, Equatable, Hashable {
    public let apiType: String
    /// Lowercased host (+ `:port` when non-default). Empty when the endpoint has no host.
    public let host: String
    public let modelID: String

    public init(apiType: ProviderAPIType, endpoint: URL, modelID: String) {
        self.apiType = apiType.rawValue
        self.host = Self.normalizedHost(of: endpoint)
        self.modelID = modelID
    }

    public init(apiType: String, host: String, modelID: String) {
        self.apiType = apiType
        self.host = host
        self.modelID = modelID
    }

    /// Lowercased host with the port appended when present — "127.0.0.1:8080" and
    /// "127.0.0.1:8081" are different servers.
    public static func normalizedHost(of endpoint: URL) -> String {
        let host = endpoint.host?.lowercased() ?? ""
        if let port = endpoint.port { return "\(host):\(port)" }
        return host
    }

    /// Stable string form, also used to derive the record's filename.
    public var storageKey: String { "\(apiType)|\(host)|\(modelID)" }

    /// Filesystem-safe, collision-resistant filename: a short human slug for browsability plus a
    /// hash of the full key for uniqueness (modelIDs contain `/` and `:` freely).
    public var fileName: String {
        let slug = modelID.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "." ? $0 : "_" }
            .prefix(40)
        let digest = SHA256.hash(data: Data(storageKey.utf8))
        let hash = digest.map { String(format: "%02x", $0) }.joined().prefix(16)
        return "\(String(slug))-\(hash).json"
    }
}

/// One COMPLETE probe run for one model, as persisted. Complete-run-replace is the store's
/// contract: a record is a single run with one honest timestamp — no cross-run stitching, no
/// mixed vintages. (A run that established nothing probed is not worth a record and is rejected
/// at the store boundary.)
public struct ProbeRecord: Codable, Sendable, Equatable {
    /// Bump when the record's serialized shape changes incompatibly.
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    /// The prober that produced this run — findings from an older prober are suspect once the
    /// request-forming code changes (a pre-native-PDF prober recorded pdfInput=false that OUR
    /// encoder caused). Comparisons/staleness policies key off this.
    public var proberVersion: Int
    /// When the store accepted the run. The store stamps this itself — `ModelProfile.probedAt`
    /// is set at seed construction, before any call completed, and is not a completion time.
    public var recordedAt: Date
    public var key: ProbeRecordKey
    /// The provider that ran the probe — an ANNOTATION for account-scoped findings
    /// (isAccessDenied is about this provider's key), never part of the identity.
    public var providerID: String
    /// The full profile, evidence strings and all — the audit trail is the point.
    public var profile: ModelProfile

    public init(proberVersion: Int, recordedAt: Date, key: ProbeRecordKey, providerID: String, profile: ModelProfile) {
        self.schemaVersion = Self.currentSchemaVersion
        self.proberVersion = proberVersion
        self.recordedAt = recordedAt
        self.key = key
        self.providerID = providerID
        self.profile = profile
    }
}

/// The local probe store: one JSON file per record under `<base>/probes/`, written atomically.
///
/// Per-record files are the cross-process discipline: the GUI app and the headless eval runner
/// are separate processes that both write here, and with one file per (apiType, host, modelID)
/// their races collapse to same-record last-writer-wins — a CLI sweep can never clobber the
/// records a GUI probe wrote for OTHER models (the failure mode of a single whole-store file).
public struct ProbeRecordStore: Sendable {
    public let directory: URL

    public init(baseDirectory: URL) {
        self.directory = baseDirectory.appendingPathComponent("probes", isDirectory: true)
    }

    /// Persists a completed run, replacing any existing record for the same key. Rejects (returns
    /// false) a profile with no established probed findings — an aborted/rate-gutted run
    /// established nothing and must not overwrite a real record.
    @discardableResult
    public func upsert(profile: ModelProfile, key: ProbeRecordKey, providerID: String, proberVersion: Int) throws -> Bool {
        guard profile.hasEstablishedProbedFindings else {
            logger.info("Skipping probe record for \(key.storageKey, privacy: .public): no established probed findings")
            return false
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let record = ProbeRecord(
            proberVersion: proberVersion, recordedAt: Date(),
            key: key, providerID: providerID, profile: profile
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(record)
        try data.write(to: directory.appendingPathComponent(key.fileName), options: .atomic)
        return true
    }

    /// Loads every readable record. Unreadable files are skipped with a log line, never fatal —
    /// one corrupt record must not take down the whole empirical layer.
    public func loadAll() -> [ProbeRecord] {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ) else { return [] }
        var records: [ProbeRecord] = []
        for file in files where file.pathExtension == "json" {
            do {
                let data = try Data(contentsOf: file)
                let record = try JSONDecoder().decode(ProbeRecord.self, from: data)
                guard record.schemaVersion <= ProbeRecord.currentSchemaVersion else {
                    logger.error("Skipping probe record \(file.lastPathComponent, privacy: .public): schema v\(record.schemaVersion) is newer than supported v\(ProbeRecord.currentSchemaVersion)")
                    continue
                }
                records.append(record)
            } catch {
                logger.error("Skipping unreadable probe record \(file.lastPathComponent, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        }
        return records
    }

    public func record(forKey key: ProbeRecordKey) -> ProbeRecord? {
        let url = directory.appendingPathComponent(key.fileName)
        guard let data = try? Data(contentsOf: url),
              let record = try? JSONDecoder().decode(ProbeRecord.self, from: data),
              // A future schema that still happens to decode must not be trusted as current
              // empirical truth — its field semantics may have changed.
              record.schemaVersion <= ProbeRecord.currentSchemaVersion else { return nil }
        return record
    }
}

// MARK: - Probe evidence → empirical facts

extension ModelProfile {
    /// Whether any finding was actually established BY PROBING — the completeness gate for
    /// persisting a run. Decoded findings don't count: they're echoes of a /models payload.
    public var hasEstablishedProbedFindings: Bool {
        func counts(_ f: ProbeFinding<some Any>) -> Bool { f.status == .established && f.source == .probed }
        return counts(chat) || counts(toolCalling) || counts(toolResultRoundTrip)
            || counts(vision) || counts(pdfInput) || counts(acceptsTemperature)
            || counts(maxOutputTokens) || counts(maxContextTokens)
            || counts(isAvailable) || counts(isAccessDenied)
            || effortLevels.values.contains { counts($0) }
    }

    /// Projects this profile into an empirical-layer ``ModelFacts`` record.
    ///
    /// Only findings that are `established` AND `source == .probed` project — decoded findings
    /// are echoes of some day's /models payload, and letting them into the empirical layer would
    /// launder catalog claims into empirical authority (they are already represented, fresher,
    /// by the authoritative layer).
    ///
    /// `includeAccountScoped` gates the findings that are facts about the PROBING KEY rather than
    /// the model — `isAccessDenied` and effort-level acceptance (account-tier-gated on some
    /// providers). The combiner passes true only when the record was probed by the provider being
    /// composed; a shared host probed under a different key says nothing about this one.
    ///
    /// Projection rules for the trickier fields:
    /// - `effortLevels`: projected ONLY when the run attempted the complete known ladder — a
    ///   partial probe would understate `validEffortLevels`, and `[]` reads as "no effort knob"
    ///   downstream, making validation reject levels the model accepts.
    /// - `acceptsTemperature == true`: conditional on the behavior flags active at probe time
    ///   (with mustNeverSendTemperatureParam set, the provider omits the parameter and the probe
    ///   trivially "passes"), so only the NEGATIVE is projected, as the flag derivation.
    /// - `isAvailable`: only the probe supplies it (the listing is not proof), and only explicit
    ///   gone-signals ever established false — a transient failure never reaches `established`.
    public func asEmpiricalFacts(includeAccountScoped: Bool) -> ModelFacts {
        var facts = ModelFacts()
        func probed<T>(_ finding: ProbeFinding<T>) -> T? {
            guard finding.status == .established, finding.source == .probed else { return nil }
            return finding.value
        }
        facts.supportsChatCompletions = probed(chat)
        facts.capabilities.toolUse = probed(toolCalling)
        facts.capabilities.toolResultRoundTrip = probed(toolResultRoundTrip)
        facts.capabilities.vision = probed(vision)
        facts.capabilities.pdfInput = probed(pdfInput)
        facts.maxOutputTokens = probed(maxOutputTokens)
        facts.maxInputTokens = probed(maxContextTokens)
        facts.isAvailable = probed(isAvailable)
        if probed(acceptsTemperature) == false {
            facts.behaviorFlags.mustNeverSendTemperatureParam = true
        }
        if includeAccountScoped {
            facts.isAccessDenied = probed(isAccessDenied)
            // Complete-ladder gate: every known level must have an established probed answer
            // (accepted or rejected) before the set of accepted levels can claim to BE the ladder.
            let established = effortLevels.filter { $0.value.status == .established && $0.value.source == .probed }
            if Set(established.keys).isSuperset(of: EffortRank.table.keys) {
                facts.validEffortLevels = established
                    .filter { $0.value.value == true }
                    .map(\.key)
                    .sorted { EffortRank.rank(of: $0) < EffortRank.rank(of: $1) }
            }
        }
        return facts
    }
}

extension ProbeRecord {
    /// This record with account-scoped findings removed — the shape safe to ship or share.
    ///
    /// `isAccessDenied` is a fact about the PROBING ACCOUNT's key (dashboard enablement), and
    /// effort acceptance can be account-tier-gated: shipping either would export one account's
    /// restrictions as everyone's truth — worse, under newest-wins a shipped `isAccessDenied:
    /// false` probed on a dev key would overwrite a user's true `established(true)`. Everything
    /// model-scoped (tool calling, vision, PDF, limits, chat) survives intact.
    public var strippedForExport: ProbeRecord {
        var stripped = self
        stripped.profile.isAccessDenied = .notAttempted
        stripped.profile.effortLevels = [:]
        return stripped
    }
}

// MARK: - Combining local + downloaded probe evidence

public enum ProbeEvidenceCombiner {
    /// Combines the local and downloaded records for one key into the empirical layer's facts,
    /// composed for the given provider.
    ///
    /// Records are whole complete runs, so "newest established wins per field" reduces to:
    /// project both, take the newer record's facts, and gap-fill from the older — an established
    /// finding never loses to the newer run's unknown, and where both established, newer wins.
    ///
    /// Account-scoped findings (isAccessDenied, effort acceptance) project only from a record
    /// probed by `providerID`'s own key; downloaded records never include them (export-stripped,
    /// and the gate here is the belt to that strip's suspenders).
    public static func combinedFacts(local: ProbeRecord?, downloaded: ProbeRecord?, forProviderID providerID: String) -> ModelFacts {
        // Account-scoped findings come ONLY from the LOCAL record, and only when it was probed by
        // the composing provider's own key. A downloaded record never qualifies regardless of its
        // providerID annotation — builtin provider IDs are identical on every machine, so a
        // shipped record that escaped export-stripping would otherwise sail through the gate.
        func facts(of record: ProbeRecord, isLocal: Bool) -> ModelFacts {
            record.profile.asEmpiricalFacts(includeAccountScoped: isLocal && record.providerID == providerID)
        }
        switch (local, downloaded) {
        case (nil, nil):
            return ModelFacts()
        case (let localOnly?, nil):
            return facts(of: localOnly, isLocal: true)
        case (nil, let downloadedOnly?):
            return facts(of: downloadedOnly, isLocal: false)
        case (let local?, let downloaded?):
            // A higher proberVersion beats a newer timestamp: a record from a FIXED prober
            // outranks a later run of a prober whose request-forming code was wrong (the
            // pre-native-PDF prober recorded pdfInput=false that our own encoder caused — a
            // later shipped record from that prober must not overwrite a corrected local one).
            // Same version → newest wins; ties → local wins.
            let localWins = (local.proberVersion, local.recordedAt.timeIntervalSince1970)
                >= (downloaded.proberVersion, downloaded.recordedAt.timeIntervalSince1970)
            var combined = localWins
                ? facts(of: local, isLocal: true)
                : facts(of: downloaded, isLocal: false)
            let olderFacts = localWins
                ? facts(of: downloaded, isLocal: false)
                : facts(of: local, isLocal: true)
            for field in ModelFactsFieldTable.fields
            where !field.isSet(combined) && field.isSet(olderFacts) {
                field.copy(olderFacts, &combined)
            }
            return combined
        }
    }
}
