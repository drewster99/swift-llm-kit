import Testing
import Foundation
@testable import SwiftLLMKit

// MARK: - LLMRequestLogger filename stamps

/// Serialized because every test here draws from the same process-wide counter.
@Suite("LLMRequestLogger unique stamps", .serialized)
struct LLMRequestLoggerStampTests {

    /// Pulls the trailing ordinal out of `yyyy-MM-dd_HH-mm-ss.SSS-NNNNN`.
    private func ordinal(of stamp: String) throws -> Int {
        let suffix = try #require(stamp.split(separator: "-").last.map(String.init))
        return try #require(Int(suffix))
    }

    @Test("a stamp is a filename-safe timestamp plus a zero-padded ordinal")
    func stampShape() throws {
        let stamp = LLMRequestLogger.uniqueStamp()
        let shape = try Regex(#"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}-\d{5,}$"#)
        #expect(stamp.wholeMatch(of: shape) != nil, "unexpected stamp shape: \(stamp)")
    }

    @Test("consecutive stamps carry strictly increasing ordinals")
    func ordinalsIncreaseMonotonically() throws {
        let stamps = (0..<50).map { _ in LLMRequestLogger.uniqueStamp() }
        let ordinals = try stamps.map { try ordinal(of: $0) }
        #expect(zip(ordinals, ordinals.dropFirst()).allSatisfy { $0 < $1 },
                "the ordinal must record the order calls were made, not just break ties")
        #expect(Set(stamps).count == stamps.count)
    }

    /// The regression this exists for: a parallel tool-call batch fires several provider requests
    /// inside one millisecond. When the filename was the millisecond timestamp alone they all
    /// resolved to the same path and `write(to:atomically:)` let the last writer win, silently
    /// destroying the very requests worth diffing against each other.
    @Test("concurrent stamps never collide, even inside a single millisecond")
    func concurrentStampsAreUnique() async throws {
        let callCount = 500
        let stamps = await withTaskGroup(of: String.self) { group in
            for _ in 0..<callCount {
                group.addTask { LLMRequestLogger.uniqueStamp() }
            }
            var collected: [String] = []
            for await stamp in group { collected.append(stamp) }
            return collected
        }

        #expect(stamps.count == callCount)
        #expect(Set(stamps).count == callCount,
                "stamps collided — concurrent request logs would overwrite each other on disk")

        // Guards the guard: if nothing shared a millisecond, the collision path went untested.
        let millisecondPrefixes = Set(stamps.map { $0.split(separator: "-").dropLast().joined(separator: "-") })
        #expect(millisecondPrefixes.count < callCount,
                "no two stamps landed in the same millisecond, so this test proved nothing")
    }
}

// MARK: - Request/response pairing

@Suite("LLMRequestLogger request/response pairing")
struct LLMRequestLoggerPairingTests {

    private func token(stamp: String, model: String?) -> LLMRequestLogger.RequestLogToken {
        LLMRequestLogger.RequestLogToken(stamp: stamp, modelSegment: model, sentAt: Date())
    }

    @Test("a response reuses the request's stem, so the pair sorts adjacent")
    func responseSharesRequestStem() {
        let sent = token(stamp: "2026-07-27_15-45-47.831-00042", model: "qwen3.5:397b")
        let requestStem = LLMRequestLogger.fileStem(stamp: sent.stamp, label: "OpenAI", modelSegment: sent.modelSegment)
        let responseStem = LLMRequestLogger.fileStem(stamp: sent.stamp, label: "OpenAI", modelSegment: sent.modelSegment)

        #expect(requestStem == responseStem)
        #expect(requestStem == "2026-07-27_15-45-47.831-00042_OpenAI_qwen3.5:397b")
        // Sorting the two real filenames must put the request first, then its own response —
        // not some other call's response that happened to arrive in between.
        #expect(["\(responseStem)_response.json", "\(requestStem)_request.json"].sorted()
                == ["\(requestStem)_request.json", "\(responseStem)_response.json"])
    }

    @Test("concurrent calls' pairs never interleave")
    func concurrentPairsStaySeparate() {
        let first = token(stamp: "2026-07-27_15-45-47.831-00042", model: "qwen3.5:397b")
        let second = token(stamp: "2026-07-27_15-45-47.831-00043", model: "qwen3.5:397b")
        let names = [first, second].flatMap { sent -> [String] in
            let stem = LLMRequestLogger.fileStem(stamp: sent.stamp, label: "OpenAI", modelSegment: sent.modelSegment)
            return ["\(stem)_request.json", "\(stem)_response.json"]
        }
        #expect(Set(names).count == 4)
        #expect(names.sorted() == names, "a sorted listing must read request, response, request, response")
    }

    @Test("a bodyless request pairs without a model segment")
    func bodylessPairOmitsModel() {
        let stem = LLMRequestLogger.fileStem(stamp: "2026-07-27_15-45-47.831-00042", label: "LiteLLM", modelSegment: nil)
        #expect(stem == "2026-07-27_15-45-47.831-00042_LiteLLM")
    }

    /// An empty model must collapse the same way `nil` does, or a request logged with `model: ""`
    /// writes `<stamp>_<label>__request.json` and its response `<stamp>_<label>_response.json` —
    /// a silently broken pair.
    @Test("an empty model segment collapses rather than leaving a dangling separator")
    func emptyModelSegmentCollapses() {
        let withEmpty = LLMRequestLogger.fileStem(stamp: "S", label: "OpenAI", modelSegment: "")
        let withNil = LLMRequestLogger.fileStem(stamp: "S", label: "OpenAI", modelSegment: nil)
        #expect(withEmpty == withNil)
    }
}
