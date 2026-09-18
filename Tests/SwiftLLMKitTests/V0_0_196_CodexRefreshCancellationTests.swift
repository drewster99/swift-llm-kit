import Foundation
import Testing
@testable import SwiftLLMKit

/// The single-flight's integrity under cancellation, and the cooldown's refusal to be armed by one.
///
/// The slot and the settled state are owned by the REFRESH, not by the caller that started it. That
/// is not a style preference: a caller-owned slot only holds because awaiting an unstructured task
/// is not a cancellation point and `Task {}` does not inherit cancellation — two language details
/// nothing in this package pins. These tests pin the OUTCOMES instead, so the guarantee survives a
/// refactor that changes how the refresh is launched.
@Suite("Codex refresh cancellation")
struct CodexRefreshCancellationTests {

    /// Counts attempts and holds each one open long enough for a second caller to arrive.
    actor SlowTransport {
        private(set) var calls = 0
        let body: [String: Any]
        init(body: [String: Any]) { self.body = body }
        func handle(_ request: URLRequest) async throws -> (Data, URLResponse) {
            calls += 1
            try await Task.sleep(for: .milliseconds(400))
            let data = try JSONSerialization.data(withJSONObject: body)
            let response = HTTPURLResponse(
                url: request.url ?? URL(fileURLWithPath: "/"),
                statusCode: 200, httpVersion: nil, headerFields: nil)
            return (data, response ?? URLResponse())
        }
    }

    /// Fails every attempt the way an abandoned one does.
    actor CancellingTransport {
        private(set) var calls = 0
        let error: any Error
        init(error: any Error) { self.error = error }
        func handle(_ request: URLRequest) async throws -> (Data, URLResponse) {
            calls += 1
            throw error
        }
    }

    static func expiredStore(_ now: Date) throws -> CodexAuthStore {
        let store = CodexAuthTests.tempStore()
        try store.save(CodexAuthTokens(
            accessToken: try CodexAuthTests.makeJWT(exp: now),
            refreshToken: "r", accountId: "acct"))
        return store
    }

    @Test("A cancelled initiator does not release the slot while its refresh is still running")
    func cancelledInitiatorHoldsTheSlot() async throws {
        let now = Date()
        let store = try Self.expiredStore(now)
        let spy = SlowTransport(
            body: ["access_token": try CodexAuthTests.makeJWT(exp: now.addingTimeInterval(3600))])
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        let initiator = Task { _ = try await coordinator.validTokens(now: now) }
        try await Task.sleep(for: .milliseconds(120))
        initiator.cancel()
        try await Task.sleep(for: .milliseconds(80))

        // Arrives while the refresh is still in flight. It must JOIN, not start its own.
        let second = Task { _ = try await coordinator.validTokens(now: now) }
        _ = await initiator.result
        _ = await second.result

        #expect(await spy.calls == 1, """
            the slot was freed with the refresh still running, so a second caller started another \
            one — two concurrent writers of auth.json, which is what signs the user out
            """)
    }

    @Test("A cancelled refresh never arms the cooldown, in either spelling")
    func cancellationDoesNotArmTheCooldown() async throws {
        // Both are "abandoned", not "refused": Swift throws CancellationError, URLSession reports
        // URLError.cancelled. Storing either would replay it to every caller for a minute — and
        // LLMRetryPolicy classifies CancellationError PERMANENT, so nothing would retry.
        for error in [CancellationError() as any Error, URLError(.cancelled) as any Error] {
            let now = Date()
            let store = try Self.expiredStore(now)
            let spy = CancellingTransport(error: error)
            let coordinator = CodexAuthCoordinator(
                store: store, transport: { try await spy.handle($0) })

            await #expect(throws: (any Error).self) { try await coordinator.validTokens(now: now) }
            // Well inside the cooldown. It must be allowed to try the endpoint again.
            await #expect(throws: (any Error).self) { try await coordinator.validTokens(now: now) }
            #expect(await spy.calls == 2, "\(type(of: error)) armed the cooldown")
        }
    }

    @Test("A real endpoint failure still arms the cooldown — the carve-out is cancellation only")
    func realFailureStillArmsTheCooldown() async throws {
        let now = Date()
        let store = try Self.expiredStore(now)
        let spy = CancellingTransport(error: URLError(.timedOut))
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        await #expect(throws: (any Error).self) { try await coordinator.validTokens(now: now) }
        await #expect(throws: (any Error).self) { try await coordinator.validTokens(now: now) }
        #expect(await spy.calls == 1, """
            widening the cancellation carve-out would reopen the storm it was added to close
            """)
    }

    @Test("A late finisher cannot release a newer refresh's slot")
    func staleReleaseLeavesTheLiveSlotAlone() async throws {
        let now = Date()
        let store = try Self.expiredStore(now)
        let spy = SlowTransport(
            body: ["access_token": try CodexAuthTests.makeJWT(exp: now.addingTimeInterval(3600))])
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        let waiter = Task { _ = try await coordinator.validTokens(now: now) }
        try await Task.sleep(for: .milliseconds(120))
        let live = try #require(await coordinator.inFlightID, "a refresh should be in flight")

        // What a second release site — a timeout abandoning an earlier refresh — would do.
        await coordinator.releaseInFlight(UUID())
        #expect(await coordinator.inFlightID == live, """
            a stale release cleared the live slot, so the next caller would start a second             concurrent refresh while this one is still running
            """)

        await coordinator.releaseInFlight(live)
        #expect(await coordinator.inFlightID == nil, "the owner's own release must still work")
        _ = await waiter.result
    }

    @Test("The refresh settles its own state, so a cancelled initiator still leaves it correct")
    func stateSettlesWithoutTheInitiator() async throws {
        let now = Date()
        let store = try Self.expiredStore(now)
        let fresh = try CodexAuthTests.makeJWT(exp: now.addingTimeInterval(3600))
        let spy = SlowTransport(body: ["access_token": fresh])
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        let initiator = Task { _ = try await coordinator.validTokens(now: now) }
        try await Task.sleep(for: .milliseconds(120))
        initiator.cancel()
        _ = await initiator.result
        // Give the refresh room to finish and commit.
        try await Task.sleep(for: .milliseconds(500))

        // A later caller must be served from what the refresh settled, with no new attempt.
        let tokens = try await coordinator.validTokens(now: now)
        #expect(tokens.accessToken == fresh)
        #expect(await spy.calls == 1, "state the initiator was supposed to record went missing")
    }
}
