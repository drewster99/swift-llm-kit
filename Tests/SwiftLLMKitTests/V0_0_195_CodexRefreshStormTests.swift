import Foundation
import Testing
@testable import SwiftLLMKit

/// Covers the two refresh STORMS: an undateable access token that asked for a refresh on every
/// call, and a FAILING refresh that did the same for an ordinary dateable one.
///
/// These are written against the coordinator rather than against `needsRefresh` alone, because the
/// property that matters is "how many times did we hit the token endpoint", not "what did a
/// predicate return".
@Suite("Codex refresh storms")
struct CodexRefreshStormTests {

    static func makeJWT(exp: Date?) throws -> String {
        try CodexAuthTests.makeJWT(exp: exp)
    }

    static func tempStore() -> CodexAuthStore { CodexAuthTests.tempStore() }

    @Test("An OPAQUE token refreshed moments ago is not refreshed again on the next call")
    func opaqueTokenDoesNotRefreshEveryCall() async throws {
        let store = Self.tempStore()
        let now = Date(timeIntervalSince1970: 1_000_000)
        try store.save(CodexAuthTokens(
            accessToken: "opaque-access-token", refreshToken: "r",
            accountId: "acct", lastRefresh: now))
        let spy = CodexAuthTests.TransportSpy(body: ["access_token": "SHOULD-NOT-BE-USED"])
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        // Five calls inside the undated interval — the shape five concurrent roles produce.
        for offset in [1.0, 10.0, 60.0, 300.0, 599.0] {
            let tokens = try await coordinator.validTokens(now: now.addingTimeInterval(offset))
            #expect(tokens.accessToken == "opaque-access-token", "at +\(offset)s")
        }
        #expect(await spy.calls == 0,
                "an undateable token must not cost one OAuth round trip per LLM call")
    }

    @Test("An OPAQUE token still refreshes once the undated interval has elapsed, and always with no prior refresh")
    func opaqueTokenRefreshesAfterTheInterval() async throws {
        let store = Self.tempStore()
        let now = Date(timeIntervalSince1970: 1_000_000)
        try store.save(CodexAuthTokens(
            accessToken: "opaque-access-token", refreshToken: "r",
            accountId: "acct", lastRefresh: now))
        let spy = CodexAuthTests.TransportSpy(body: ["access_token": "refreshed-opaque"])
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        let tokens = try await coordinator.validTokens(now: now.addingTimeInterval(601))
        #expect(tokens.accessToken == "refreshed-opaque")
        #expect(await spy.calls == 1, "the token is still undateable; it cannot be trusted forever")

        // A credential that has never been refreshed has no clock to read, so it must refresh.
        let virgin = Self.tempStore()
        try CodexAuthTests.write(
            ["tokens": ["access_token": "opaque", "refresh_token": "r", "account_id": "acct"]],
            to: virgin)
        #expect(virgin.load()?.lastRefresh == nil)
        let spy2 = CodexAuthTests.TransportSpy(body: ["access_token": "refreshed-opaque"])
        let coordinator2 = CodexAuthCoordinator(store: virgin, transport: { try await spy2.handle($0) })
        _ = try await coordinator2.validTokens(now: now)
        #expect(await spy2.calls == 1, "no last_refresh means no clock; refresh")
    }

    @Test("A FAILING refresh is attempted once per cooldown, not once per LLM call")
    func failingRefreshIsNotRetriedEveryCall() async throws {
        let store = Self.tempStore()
        let now = Date(timeIntervalSince1970: 1_000_000)
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now), refreshToken: "revoked", accountId: "acct"))
        let spy = CodexAuthTests.TransportSpy(
            body: ["error": "invalid_grant"], status: 400)
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        // Four LLM calls inside one minute. Each must still FAIL — and with the same typed error,
        // so the consumer's transient/permanent classification is unchanged — but only the first
        // may touch the network.
        for offset in [0.0, 1.0, 30.0, 59.0] {
            do {
                _ = try await coordinator.validTokens(now: now.addingTimeInterval(offset))
                Issue.record("expected a refresh failure at +\(offset)s")
            } catch let error as LLMProviderError {
                guard case .httpError(let status, _, _, _) = error else {
                    Issue.record("expected httpError at +\(offset)s, got \(error)")
                    return
                }
                #expect(status == 400, "the replayed error must classify identically")
            }
        }
        #expect(await spy.calls == 1,
                "50 retries x 5 roles x one refresh each is a storm against a failing token endpoint")

        // Past the cooldown, a real attempt is made again — the user is not pinned forever.
        _ = try? await coordinator.validTokens(now: now.addingTimeInterval(61))
        #expect(await spy.calls == 2)
    }

    @Test("`codex login` recovers WITHOUT restarting the app, even mid-cooldown")
    func recoveryNeedsNoRestart() async throws {
        let store = Self.tempStore()
        let now = Date(timeIntervalSince1970: 1_000_000)
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now), refreshToken: "revoked", accountId: "acct"))
        let spy = CodexAuthTests.TransportSpy(body: ["error": "invalid_grant"], status: 400)
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        await #expect(throws: LLMProviderError.self) { try await coordinator.validTokens(now: now) }

        // The user runs `codex login`, which rewrites auth.json in place.
        let fresh = try Self.makeJWT(exp: now.addingTimeInterval(86_400))
        try store.save(CodexAuthTokens(
            accessToken: fresh, refreshToken: "new-r", accountId: "acct", lastRefresh: now))

        // One second later — well inside the failure cooldown — the very next call must succeed.
        let tokens = try await coordinator.validTokens(now: now.addingTimeInterval(1))
        #expect(tokens.accessToken == fresh)
        #expect(await spy.calls == 1, "a good token on disk needs no refresh at all")
    }

    @Test("The undated clock NEVER reaches a dateable token, however recently it was refreshed")
    func aReadableExpiryIsNeverShadowedByARecentRefresh() throws {
        let now = Date(timeIntervalSince1970: 1_000_000)
        // The shape that would be a security hole if `lastRefresh` were consulted first: a token
        // that expired an hour ago, stamped as refreshed one second ago.
        let dead = CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now.addingTimeInterval(-3600)),
            refreshToken: "r", accountId: "a", lastRefresh: now.addingTimeInterval(-1))
        #expect(dead.needsRefresh(within: 300, now: now) == true)
        #expect(dead.needsRefresh(within: 300, undatedAfter: 86_400, now: now) == true,
                "undatedAfter must not reach a token that CAN be dated")

        // And the converse: a live token is not dragged into a refresh by a stale `lastRefresh`.
        let live = CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now.addingTimeInterval(3600)),
            refreshToken: "r", accountId: "a", lastRefresh: now.addingTimeInterval(-86_400))
        #expect(live.needsRefresh(within: 300, now: now) == false)

        // Boundary, both sides, for the undated path only.
        func undated(refreshedAgo seconds: TimeInterval) -> CodexAuthTokens {
            CodexAuthTokens(accessToken: "opaque", refreshToken: "r", accountId: "a",
                            lastRefresh: now.addingTimeInterval(-seconds))
        }
        #expect(undated(refreshedAgo: 599).needsRefresh(within: 300, now: now) == false)
        #expect(undated(refreshedAgo: 600).needsRefresh(within: 300, now: now) == true, "inclusive")
        #expect(undated(refreshedAgo: -5).needsRefresh(within: 300, now: now) == true,
                "a lastRefresh in the future is a clock jump, not a licence to never refresh")
    }

    @Test("A failing refresh shared by five concurrent roles costs ONE attempt, and the cooldown holds after it")
    func concurrentCallersShareOneFailure() async throws {
        let store = Self.tempStore()
        let now = Date(timeIntervalSince1970: 1_000_000)
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now), refreshToken: "revoked", accountId: "acct"))
        let spy = CodexAuthTests.TransportSpy(body: ["error": "invalid_grant"], status: 400)
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<5 {
                group.addTask {
                    do { _ = try await coordinator.validTokens(now: now); return false }
                    catch { return true }
                }
            }
            for await threw in group { #expect(threw, "every caller must see the failure") }
        }
        #expect(await spy.calls == 1, "single-flight")

        // A joiner neither records nor clears the failure, so the cooldown the initiator armed is
        // still in force for the calls that follow.
        await #expect(throws: LLMProviderError.self) {
            try await coordinator.validTokens(now: now.addingTimeInterval(5))
        }
        #expect(await spy.calls == 1, "the cooldown survives having been joined rather than initiated")
    }

    @Test("A backwards clock jump expires the cooldown rather than pinning it")
    func backwardsClockJumpDoesNotPin() async throws {
        let store = Self.tempStore()
        let now = Date(timeIntervalSince1970: 1_000_000)
        // Expired far enough in the past that it is still expired AFTER the backwards jump —
        // otherwise the token simply reads as valid and never reaches the cooldown at all.
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now.addingTimeInterval(-40_000_000)),
            refreshToken: "revoked", accountId: "acct"))
        let spy = CodexAuthTests.TransportSpy(body: ["error": "invalid_grant"], status: 400)
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        await #expect(throws: LLMProviderError.self) { try await coordinator.validTokens(now: now) }
        #expect(await spy.calls == 1)
        // The clock moves a year backwards. Suppressing until it catches up would be a year.
        await #expect(throws: LLMProviderError.self) {
            try await coordinator.validTokens(now: now.addingTimeInterval(-31_536_000))
        }
        #expect(await spy.calls == 2)
    }

    @Test("An UNWRITABLE auth.json costs one refresh, not one per call")
    func unwritableStoreDoesNotRefreshEveryCall() async throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("codex-readonly-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let store = CodexAuthStore(url: dir.appendingPathComponent("auth.json"))
        let now = Date(timeIntervalSince1970: 1_000_000)
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now), refreshToken: "r", accountId: "acct"))
        // A read-only directory: the file still LOADS, but the atomic write cannot land.
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: dir.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path) }

        let fresh = try Self.makeJWT(exp: now.addingTimeInterval(86_400))
        let spy = CodexAuthTests.TransportSpy(body: ["access_token": fresh])
        let coordinator = CodexAuthCoordinator(store: store, transport: { try await spy.handle($0) })

        for offset in [0.0, 1.0, 60.0, 600.0, 3600.0] {
            let tokens = try await coordinator.validTokens(now: now.addingTimeInterval(offset))
            #expect(tokens.accessToken == fresh, "at +\(offset)s")
        }
        #expect(await spy.calls == 1,
                "a write failure must cost ONE redundant refresh, not one per LLM call forever")
        #expect(store.load()?.accessToken != fresh, "the write really did fail")
    }
}
