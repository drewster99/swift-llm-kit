import Testing
import Foundation
@testable import SwiftLLMKit

/// `fetchedAt` / `lastProbedAt` (2026-07-31): provenance timestamps surfaced on `ModelInfo` so the
/// Models UI reads "list fetched" / "capabilities probed" off the model record, not a sidecar.
/// Both are optional and hand-coded in `ModelInfo`'s Codable, so these pin round-trip, back-compat
/// decode of catalogs written before the fields existed, and the encodeIfPresent (nil = absent).
@Suite("ModelInfo provenance timestamps")
struct ModelInfoProvenanceTests {

    private func roundTrip(_ info: ModelInfo) throws -> ModelInfo {
        let data = try JSONEncoder().encode(info)
        return try JSONDecoder().decode(ModelInfo.self, from: data)
    }

    @Test("Both timestamps survive a JSON round-trip")
    func roundTripsBoth() throws {
        let fetched = Date(timeIntervalSince1970: 1_780_000_000)
        let probed = Date(timeIntervalSince1970: 1_779_000_000)
        let info = ModelInfo(providerID: "p", modelID: "m", fetchedAt: fetched, lastProbedAt: probed)
        let decoded = try roundTrip(info)
        #expect(decoded.fetchedAt == fetched)
        #expect(decoded.lastProbedAt == probed)
    }

    @Test("Absent when nil, and a catalog written before the fields decodes to nil")
    func absentAndBackCompat() throws {
        let info = ModelInfo(providerID: "p", modelID: "m")
        let data = try JSONEncoder().encode(info)
        let json = String(decoding: data, as: UTF8.self)
        #expect(!json.contains("fetchedAt"))
        #expect(!json.contains("lastProbedAt"))

        // A pre-field catalog entry: valid ModelInfo JSON with neither key present.
        let legacy = #"{"providerID":"p","modelID":"m","displayName":"m","capabilities":{}}"#
        let decoded = try JSONDecoder().decode(ModelInfo.self, from: Data(legacy.utf8))
        #expect(decoded.fetchedAt == nil)
        #expect(decoded.lastProbedAt == nil)
    }

    @Test("One set, one nil round-trips independently")
    func mixedNil() throws {
        let fetched = Date(timeIntervalSince1970: 1_780_500_000)
        let info = ModelInfo(providerID: "p", modelID: "m", fetchedAt: fetched)
        let decoded = try roundTrip(info)
        #expect(decoded.fetchedAt == fetched)
        #expect(decoded.lastProbedAt == nil)
    }
}
