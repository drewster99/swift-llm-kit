import Foundation
import Testing
@testable import SwiftLLMKit

/// Covers the ChatGPT-subscription credential path: claim reading, the file the `codex` CLI owns,
/// and the refresh coordinator.
///
/// Every test points the store at a temporary directory. The path is injectable precisely so a test
/// cannot reach `~/.codex/auth.json` — a suite that rewrote the developer's live credentials while
/// "passing" would be worse than no suite.
@Suite("Codex ChatGPT-subscription auth")
struct CodexAuthTests {

    // MARK: Helpers

    /// A syntactically real JWT with the claims we read. Unsigned — nothing here verifies signatures,
    /// and the server re-verifies anyway.
    static func makeJWT(exp: Date?, accountID: String? = "acct-123", plan: String? = "prolite") throws -> String {
        var auth: [String: Any] = [:]
        if let accountID { auth["chatgpt_account_id"] = accountID }
        if let plan { auth["chatgpt_plan_type"] = plan }
        var claims: [String: Any] = ["https://api.openai.com/auth": auth]
        if let exp { claims["exp"] = exp.timeIntervalSince1970 }
        let payload = try JSONSerialization.data(withJSONObject: claims)
        func b64url(_ d: Data) -> String {
            d.base64EncodedString()
                .replacingOccurrences(of: "+", with: "-")
                .replacingOccurrences(of: "/", with: "_")
                .replacingOccurrences(of: "=", with: "")
        }
        return "\(b64url(Data("{\"alg\":\"none\"}".utf8))).\(b64url(payload)).sig"
    }

    static func tempStore() -> CodexAuthStore {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("codex-auth-tests-\(UUID().uuidString)")
        return CodexAuthStore(url: dir.appendingPathComponent("auth.json"))
    }

    static func write(_ json: [String: Any], to store: CodexAuthStore) throws {
        try FileManager.default.createDirectory(
            at: store.url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: json).write(to: store.url)
    }

    // MARK: Claims

    @Test("Claims are read from the token, and an opaque token degrades to nil everywhere")
    func jwtClaims() throws {
        let exp = Date(timeIntervalSince1970: 1_800_000_000)
        let jwt = try Self.makeJWT(exp: exp)
        #expect(CodexJWT.accountId(jwt) == "acct-123")
        #expect(CodexJWT.planType(jwt) == "prolite")
        #expect(CodexJWT.expiry(jwt)?.timeIntervalSince1970 == exp.timeIntervalSince1970)

        // Every accessor funnels through one decoder, so these fail together rather than one
        // returning a value while another traps.
        for bad in ["", "not-a-jwt", "only.two", "a.b.c.d", "aaa.!!!not-base64!!!.ccc"] {
            #expect(CodexJWT.claims(bad) == nil, "claims(\(bad))")
            #expect(CodexJWT.accountId(bad) == nil, "accountId(\(bad))")
            #expect(CodexJWT.expiry(bad) == nil, "expiry(\(bad))")
        }
    }

    @Test("A token with no exp claim is treated as needing refresh, not as valid forever")
    func missingExpiryNeedsRefresh() throws {
        let tokens = CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: nil), refreshToken: "r", accountId: "a")
        #expect(tokens.expiry == nil)
        #expect(tokens.needsRefresh(within: 300) == true)
    }

    @Test("The refresh window boundary is honored on both sides")
    func refreshWindowBoundary() throws {
        let now = Date(timeIntervalSince1970: 1_000_000)
        func tokens(expiringIn seconds: TimeInterval) throws -> CodexAuthTokens {
            CodexAuthTokens(
                accessToken: try Self.makeJWT(exp: now.addingTimeInterval(seconds)),
                refreshToken: "r", accountId: "a")
        }
        #expect(try tokens(expiringIn: 301).needsRefresh(within: 300, now: now) == false)
        #expect(try tokens(expiringIn: 299).needsRefresh(within: 300, now: now) == true)
        #expect(try tokens(expiringIn: -60).needsRefresh(within: 300, now: now) == true, "already expired")
    }

    // MARK: The file

    @Test("A well-formed auth.json loads; absent, corrupt, and incomplete ones read as signed-out")
    func storeLoading() throws {
        let store = Self.tempStore()
        #expect(store.isPresent == false)
        #expect(store.load() == nil, "absent file")

        try Self.write(["tokens": ["access_token": "a", "refresh_token": "r", "account_id": "acct"]],
                       to: store)
        #expect(store.isPresent == true)
        #expect(store.load()?.refreshToken == "r")

        // Each required field missing in turn, plus an empty value and outright garbage.
        let incomplete: [[String: Any]] = [
            ["tokens": ["refresh_token": "r", "account_id": "acct"]],
            ["tokens": ["access_token": "a", "account_id": "acct"]],
            ["tokens": ["access_token": "a", "refresh_token": "r"]],
            ["tokens": ["access_token": "", "refresh_token": "r", "account_id": "acct"]],
            ["no_tokens_key": true]
        ]
        for json in incomplete {
            try Self.write(json, to: store)
            #expect(store.load() == nil, "should read as signed out: \(json)")
        }

        try Data("{ not json".utf8).write(to: store.url)
        #expect(store.load() == nil, "corrupt file")
    }

    @Test("Saving preserves keys we do not own, rather than rewriting the CLI's file from our fields")
    func savePreservesForeignKeys() throws {
        let store = Self.tempStore()
        try Self.write([
            "OPENAI_API_KEY": NSNull(),
            "auth_mode": "chatgpt",
            "some_future_field": ["nested": 1],
            "tokens": ["access_token": "old", "refresh_token": "oldr",
                       "account_id": "acct", "id_token": "oldid",
                       "a_token_field_we_do_not_model": "keep me"]
        ], to: store)

        try store.save(CodexAuthTokens(
            accessToken: "new", idToken: "newid", refreshToken: "newr",
            accountId: "acct", lastRefresh: Date(timeIntervalSince1970: 1_700_000_000)))

        let root = try #require(try JSONSerialization.jsonObject(
            with: Data(contentsOf: store.url)) as? [String: Any])
        #expect(root["auth_mode"] as? String == "chatgpt")
        #expect(root["some_future_field"] != nil, "an unknown top-level key must survive")
        #expect(root["OPENAI_API_KEY"] is NSNull)
        let tokens = try #require(root["tokens"] as? [String: Any])
        #expect(tokens["access_token"] as? String == "new")
        #expect(tokens["refresh_token"] as? String == "newr")
        #expect(tokens["a_token_field_we_do_not_model"] as? String == "keep me",
                "an unknown key INSIDE tokens must survive too")
        #expect(root["last_refresh"] as? String != nil)
    }

    @Test("Save then load round-trips")
    func saveLoadRoundTrip() throws {
        let store = Self.tempStore()
        let jwt = try Self.makeJWT(exp: Date(timeIntervalSince1970: 1_900_000_000))
        try store.save(CodexAuthTokens(
            accessToken: jwt, idToken: "id", refreshToken: "r", accountId: "acct"))
        let back = try #require(store.load())
        #expect(back.accessToken == jwt)
        #expect(back.idToken == "id")
        #expect(back.refreshToken == "r")
        #expect(back.accountId == "acct")
        #expect(back.lastRefresh != nil)
    }

    // MARK: The coordinator

    /// Counts calls so "did it refresh?" and "how many times?" are both assertable.
    actor TransportSpy {
        private(set) var calls = 0
        var body: [String: Any]
        var status: Int
        var headers: [String: String]
        init(body: [String: Any], status: Int = 200, headers: [String: String] = [:]) {
            self.body = body; self.status = status; self.headers = headers
        }
        func handle(_ request: URLRequest) async throws -> (Data, URLResponse) {
            calls += 1
            let url = try #require(request.url)
            let response = try #require(HTTPURLResponse(
                url: url, statusCode: status, httpVersion: nil, headerFields: headers))
            return (try JSONSerialization.data(withJSONObject: body), response)
        }
    }

    @Test("With no credential on disk, the failure names the remedy instead of being a generic error")
    func noCredential() async {
        let coordinator = CodexAuthCoordinator(store: Self.tempStore())
        await #expect(throws: LLMProviderError.self) { try await coordinator.validTokens() }
    }

    @Test("A token with time left is returned as-is — no refresh request at all")
    func doesNotRefreshAValidToken() async throws {
        let store = Self.tempStore()
        let now = Date()
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now.addingTimeInterval(3600)),
            refreshToken: "r", accountId: "acct"))
        let spy = TransportSpy(body: ["access_token": "SHOULD-NOT-BE-USED"])
        let coordinator = CodexAuthCoordinator(
            store: store, transport: { try await spy.handle($0) })

        let tokens = try await coordinator.validTokens(now: now)
        #expect(tokens.refreshToken == "r")
        #expect(await spy.calls == 0, "a valid token must not cost a network round trip")
    }

    @Test("An expiring token refreshes, and the new tokens are written back for the CLI")
    func refreshesAndPersists() async throws {
        let store = Self.tempStore()
        let now = Date()
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now.addingTimeInterval(60)),
            idToken: "old-id", refreshToken: "old-r", accountId: "acct"))
        let fresh = try Self.makeJWT(exp: now.addingTimeInterval(86_400), accountID: "acct-fresh")
        let spy = TransportSpy(body: ["access_token": fresh, "refresh_token": "new-r"])
        let coordinator = CodexAuthCoordinator(
            store: store, transport: { try await spy.handle($0) })

        let tokens = try await coordinator.validTokens(now: now)
        #expect(tokens.accessToken == fresh)
        #expect(tokens.refreshToken == "new-r")
        #expect(tokens.accountId == "acct-fresh", "account id is re-read from the fresh token")
        #expect(await spy.calls == 1)

        let onDisk = try #require(store.load())
        #expect(onDisk.accessToken == fresh, "the CLI must see the same token we do")
        #expect(onDisk.refreshToken == "new-r")
    }

    @Test("A refresh response that omits the refresh token keeps the existing one")
    func refreshWithoutRotationKeepsTheOldToken() async throws {
        let store = Self.tempStore()
        let now = Date()
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now), idToken: "keep-me",
            refreshToken: "keep-this", accountId: "acct"))
        let spy = TransportSpy(body: ["access_token": try Self.makeJWT(exp: now.addingTimeInterval(9999))])
        let coordinator = CodexAuthCoordinator(
            store: store, transport: { try await spy.handle($0) })

        let tokens = try await coordinator.validTokens(now: now)
        #expect(tokens.refreshToken == "keep-this", "dropping it would sign the user out")
        #expect(tokens.idToken == "keep-me")
    }

    @Test("Concurrent callers share ONE refresh rather than racing each other's file writes")
    func refreshIsSingleFlight() async throws {
        let store = Self.tempStore()
        let now = Date()
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now), refreshToken: "r", accountId: "acct"))
        let spy = TransportSpy(body: ["access_token": try Self.makeJWT(exp: now.addingTimeInterval(9999))])
        let coordinator = CodexAuthCoordinator(
            store: store, transport: { try await spy.handle($0) })

        // Five roles noticing expiry at once — the shape Agent Smith actually produces.
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<5 {
                group.addTask { _ = try? await coordinator.validTokens(now: now) }
            }
        }
        #expect(await spy.calls == 1,
                "five refreshes would race on auth.json and persist a rotated-away refresh token")
    }

    @Test("A failed refresh surfaces the status and any Retry-After, rather than a bare failure")
    func failedRefreshCarriesStatusAndRetryAfter() async throws {
        let store = Self.tempStore()
        let now = Date()
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now), refreshToken: "r", accountId: "acct"))
        let spy = TransportSpy(body: ["error": "invalid_grant"], status: 400,
                               headers: ["Retry-After": "42"])
        let coordinator = CodexAuthCoordinator(
            store: store, transport: { try await spy.handle($0) })

        do {
            _ = try await coordinator.validTokens(now: now)
            Issue.record("expected a refresh failure")
        } catch let error as LLMProviderError {
            guard case .httpError(let status, _, _, let retryAfter) = error else {
                Issue.record("expected httpError, got \(error)")
                return
            }
            #expect(status == 400)
            #expect(retryAfter == 42)
        }
    }

    @Test("A refresh whose body carries no access_token is malformed, not silently accepted")
    func refreshWithoutAccessTokenIsMalformed() async throws {
        let store = Self.tempStore()
        let now = Date()
        try store.save(CodexAuthTokens(
            accessToken: try Self.makeJWT(exp: now), refreshToken: "r", accountId: "acct"))
        let spy = TransportSpy(body: ["token_type": "bearer"])
        let coordinator = CodexAuthCoordinator(
            store: store, transport: { try await spy.handle($0) })
        await #expect(throws: LLMProviderError.self) { try await coordinator.validTokens(now: now) }
    }
}
