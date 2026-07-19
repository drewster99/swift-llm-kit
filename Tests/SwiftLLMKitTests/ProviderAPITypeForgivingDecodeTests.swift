import Testing
import Foundation
@testable import SwiftLLMKit

/// `ProviderAPIType` must decode forgivingly: an unknown or legacy raw value degrades to
/// `.openAICompatible` rather than throwing. Regression coverage for the 2026-07-19 incident where a
/// persisted `providers.json` still held apiType `"metaLlama"` after that case was renamed to
/// `metaModel` — the throw failed the whole array decode, `providersLoadedOK` went false, built-in
/// seeding was skipped, and every provider (built-in and custom) vanished.
@Suite("ProviderAPIType forgiving decode")
struct ProviderAPITypeForgivingDecodeTests {

    private func decode(_ raw: String) throws -> ProviderAPIType {
        try JSONDecoder().decode(ProviderAPIType.self, from: Data("\"\(raw)\"".utf8))
    }

    @Test("Known raw values decode to their case")
    func knownValues() throws {
        #expect(try decode("openAICompatible") == .openAICompatible)
        #expect(try decode("metaModel") == .metaModel)
        #expect(try decode("anthropic") == .anthropic)
        #expect(try decode("ollama") == .ollama)
    }

    @Test("Legacy 'metaLlama' degrades to openAICompatible, does not throw")
    func legacyMetaLlama() throws {
        #expect(try decode("metaLlama") == .openAICompatible)
    }

    @Test("An unknown future raw value degrades to openAICompatible")
    func unknownFutureValue() throws {
        #expect(try decode("someProviderFromTheFuture") == .openAICompatible)
    }

    @Test("Encoding round-trips the exact raw value")
    func encodeRoundTrips() throws {
        for type in ProviderAPIType.allCases {
            let data = try JSONEncoder().encode(type)
            #expect(String(data: data, encoding: .utf8) == "\"\(type.rawValue)\"")
            #expect(try JSONDecoder().decode(ProviderAPIType.self, from: data) == type)
        }
    }

    /// The incident itself: an ARRAY holding one legacy value must decode in FULL, not drop every
    /// element. Mirrors the real `providers.json` shape (`builtin.metallama` + real providers).
    @Test("A provider array with a legacy apiType decodes fully — one bad entry never drops the rest")
    func arrayWithLegacyEntrySurvives() throws {
        struct Row: Codable, Equatable {
            let id: String
            let apiType: ProviderAPIType
        }
        let json = """
        [
          {"id": "builtin.ollama-cloud", "apiType": "ollama"},
          {"id": "builtin.metallama", "apiType": "metaLlama"},
          {"id": "provider-1566D9B4", "apiType": "openAICompatible"},
          {"id": "builtin.meta-model-api", "apiType": "metaModel"}
        ]
        """
        let rows = try JSONDecoder().decode([Row].self, from: Data(json.utf8))
        #expect(rows.count == 4, "all four entries must survive; the legacy metaLlama must not fail the array")
        #expect(rows[0].apiType == .ollama)
        #expect(rows[1].apiType == .openAICompatible)   // legacy metaLlama → openAICompatible
        #expect(rows[2].apiType == .openAICompatible)
        #expect(rows[3].apiType == .metaModel)
    }
}
