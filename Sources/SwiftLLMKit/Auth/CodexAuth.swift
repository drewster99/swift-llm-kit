import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "CodexAuth")

/// ChatGPT-subscription credentials for the Codex backend, read from the `codex` CLI's own
/// `auth.json` rather than obtained by implementing OAuth ourselves.
///
/// **Why piggyback on the CLI.** The sign-in flow is PKCE against `auth.openai.com` with a browser
/// redirect to a loopback listener. Implementing that means shipping a redirect server, a client id
/// we don't own, and a login UI — for a token the user can mint with one `codex login`. Reading the
/// file it already writes costs none of that, and refreshes are written BACK so the CLI and we never
/// disagree about which token is current.
///
/// **What this is not.** These credentials cannot reach the OpenAI platform API at all: the token is
/// scope-gated (`Missing scopes: model.request` on `/v1/chat/completions`, `api.responses.write` on
/// `/v1/responses`), so it is only good against the Codex backend's Responses endpoint. That is a
/// property of the token, not of the URL, so there is nothing to configure around it.

// MARK: - The token set

/// The credential set as the `codex` CLI stores it.
public struct CodexAuthTokens: Sendable, Equatable {
    public var accessToken: String
    /// Present in practice, but treated as optional: `accountId` falls back to the stored value when
    /// the id token is absent or opaque, so a shape change upstream degrades rather than fails.
    public var idToken: String?
    public var refreshToken: String
    public var accountId: String
    public var lastRefresh: Date?

    public init(
        accessToken: String,
        idToken: String? = nil,
        refreshToken: String,
        accountId: String,
        lastRefresh: Date? = nil
    ) {
        self.accessToken = accessToken
        self.idToken = idToken
        self.refreshToken = refreshToken
        self.accountId = accountId
        self.lastRefresh = lastRefresh
    }

    /// When the access token expires, read from its own `exp` claim. `nil` for an opaque token.
    public var expiry: Date? { CodexJWT.expiry(accessToken) }

    /// Whether the access token is within `window` of expiring (or already has).
    ///
    /// An UNREADABLE expiry answers `true`: a token we cannot date is one we cannot vouch for, and
    /// refreshing a still-good token costs one request while using a dead one fails the call.
    public func needsRefresh(within window: TimeInterval, now: Date = Date()) -> Bool {
        guard let expiry else { return true }
        return expiry.timeIntervalSince(now) < window
    }
}

// MARK: - JWT claims

/// Minimal reader for the claims we need out of the access / id token.
///
/// Deliberately not a JWT library and deliberately not verifying the signature: we are reading a
/// token the CLI already obtained and the server will re-verify. Forging it would only let a user
/// lie to their own client.
public enum CodexJWT {
    /// The account id the Codex backend expects echoed on every request, from the nested
    /// `https://api.openai.com/auth` claim.
    public static func accountId(_ jwt: String) -> String? {
        guard let auth = claims(jwt)?["https://api.openai.com/auth"] as? [String: Any] else { return nil }
        return auth["chatgpt_account_id"] as? String
    }

    /// The subscription tier (`chatgpt_plan_type`), for display only — never for gating.
    public static func planType(_ jwt: String) -> String? {
        guard let auth = claims(jwt)?["https://api.openai.com/auth"] as? [String: Any] else { return nil }
        return auth["chatgpt_plan_type"] as? String
    }

    /// Expiry from the `exp` claim (seconds since epoch). `nil` when absent or unparseable.
    public static func expiry(_ jwt: String) -> Date? {
        guard let exp = claims(jwt)?["exp"] as? Double else { return nil }
        return Date(timeIntervalSince1970: exp)
    }

    /// The decoded payload, or `nil` for anything that is not a three-segment JWT with JSON in the
    /// middle. Every accessor above funnels through this, so an opaque token degrades to `nil`
    /// everywhere rather than trapping in one place and not another.
    public static func claims(_ jwt: String) -> [String: Any]? {
        let parts = jwt.split(separator: ".")
        guard parts.count == 3,
              let data = base64URLDecode(String(parts[1])),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return object
    }

    private static func base64URLDecode(_ segment: String) -> Data? {
        var s = segment.replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        s += String(repeating: "=", count: (4 - s.count % 4) % 4)
        return Data(base64Encoded: s)
    }
}

// MARK: - The file on disk

/// Reads and writes the `codex` CLI's `auth.json`.
///
/// The path is INJECTED rather than hardcoded so tests cannot touch the developer's real
/// credentials — the same reasoning as a persistence layer that takes a testing root: a test that
/// forgets to point somewhere else should fail to find a file, not quietly rewrite the live one.
public struct CodexAuthStore: Sendable {
    public let url: URL

    /// `$CODEX_HOME/auth.json`, or `~/.codex/auth.json` when that is unset — matching where the CLI
    /// looks, including for users who relocate it.
    public static var defaultURL: URL {
        if let home = ProcessInfo.processInfo.environment["CODEX_HOME"], !home.isEmpty {
            return URL(fileURLWithPath: home).appendingPathComponent("auth.json")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".codex/auth.json")
    }

    public init(url: URL = CodexAuthStore.defaultURL) {
        self.url = url
    }

    /// Whether a credential file exists at all — the "is the user signed in?" question, answered
    /// without reading secrets.
    public var isPresent: Bool { FileManager.default.fileExists(atPath: url.path) }

    /// The stored tokens, or `nil` when the file is absent, unreadable, or missing a required field.
    ///
    /// A malformed file reads as "not signed in" rather than throwing: the only remedy either way is
    /// `codex login`, and a decode error phrased as a crash helps nobody.
    public func load() -> CodexAuthTokens? {
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tokens = root["tokens"] as? [String: Any],
              let access = tokens["access_token"] as? String, !access.isEmpty,
              let refresh = tokens["refresh_token"] as? String, !refresh.isEmpty,
              let account = tokens["account_id"] as? String
        else { return nil }
        return CodexAuthTokens(
            accessToken: access,
            idToken: tokens["id_token"] as? String,
            refreshToken: refresh,
            accountId: account,
            lastRefresh: (root["last_refresh"] as? String).flatMap(Self.parseISO8601)
        )
    }

    /// Writes refreshed tokens back, PRESERVING every key we did not set.
    ///
    /// Read-modify-write rather than a fresh document, because this file is the CLI's, not ours: it
    /// carries `OPENAI_API_KEY` and whatever else a future version adds, and rewriting it from our
    /// own field list would silently drop all of it. Preserving by default fails safe; enumerating
    /// fails lossy.
    public func save(_ tokens: CodexAuthTokens) throws {
        var root: [String: Any] = [:]
        if let data = try? Data(contentsOf: url),
           let existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            root = existing
        }
        var stored = root["tokens"] as? [String: Any] ?? [:]
        stored["access_token"] = tokens.accessToken
        stored["refresh_token"] = tokens.refreshToken
        stored["account_id"] = tokens.accountId
        if let id = tokens.idToken { stored["id_token"] = id }
        root["tokens"] = stored
        root["last_refresh"] = Self.formatISO8601(tokens.lastRefresh ?? Date())

        let data = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    static func parseISO8601(_ s: String) -> Date? {
        let withFraction = ISO8601DateFormatter()
        withFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return withFraction.date(from: s) ?? ISO8601DateFormatter().date(from: s)
    }

    static func formatISO8601(_ d: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: d)
    }
}

// MARK: - Refresh

/// Vends a currently-valid token set, refreshing when one is close to expiring.
///
/// An `actor` with a shared in-flight task because the callers are CONCURRENT: Agent Smith runs up
/// to five roles at once, and without single-flighting, five agents noticing expiry in the same
/// instant would fire five refreshes and race each other writing `auth.json`. The loser of that race
/// persists a refresh token the server has already rotated away, which signs the user out.
public actor CodexAuthCoordinator {
    /// Refresh when the access token has less than this left. Matches the Codex CLI's own window.
    public static let refreshWindow: TimeInterval = 5 * 60

    /// The Codex CLI's public PKCE client id, used ONLY for refresh.
    ///
    /// Hardcoded and undocumented, so it can rotate without notice: `codex` 0.154.0 ships this one
    /// alongside a second (`app_69a1d78e929881919bba0dbda1f6436d`). A rotation breaks refresh only —
    /// an unexpired token keeps working — and the remedy is `codex login`, so `refresh` reports the
    /// failure rather than treating it as a transient error worth retrying.
    public static let defaultClientID = "app_EMoamEEZ73f0CkXaXp7hrann"

    public static let defaultTokenURL = "https://auth.openai.com/oauth/token"

    /// Performs the token request. Injected so refresh — including the single-flight behaviour — is
    /// testable without a network or a real credential.
    public typealias Transport = @Sendable (URLRequest) async throws -> (Data, URLResponse)

    private let store: CodexAuthStore
    private let clientID: String
    private let tokenURL: String
    private let transport: Transport

    /// The refresh currently in flight, if any. Concurrent callers await this one rather than
    /// starting their own.
    private var inFlight: Task<CodexAuthTokens, Error>?

    public init(
        store: CodexAuthStore = CodexAuthStore(),
        clientID: String = CodexAuthCoordinator.defaultClientID,
        tokenURL: String = CodexAuthCoordinator.defaultTokenURL,
        transport: @escaping Transport = { try await URLSession.shared.data(for: $0) }
    ) {
        self.store = store
        self.clientID = clientID
        self.tokenURL = tokenURL
        self.transport = transport
    }

    /// A token set good for the next `refreshWindow`, refreshing first if needed.
    ///
    /// - Throws: `LLMProviderError.invalidRequest` when no credential exists (the user has not run
    ///   `codex login`), or the refresh error when refreshing fails.
    public func validTokens(now: Date = Date()) async throws -> CodexAuthTokens {
        guard let current = store.load() else {
            throw LLMProviderError.invalidRequest(
                detail: "No ChatGPT credential found at \(store.url.path). Run `codex login` to sign in.")
        }
        guard current.needsRefresh(within: Self.refreshWindow, now: now) else { return current }

        if let inFlight { return try await inFlight.value }
        let task = Task { try await self.performRefresh(current) }
        inFlight = task
        defer { inFlight = nil }
        return try await task.value
    }

    /// Exchanges the refresh token for a fresh access token and writes the result back to disk.
    private func performRefresh(_ current: CodexAuthTokens) async throws -> CodexAuthTokens {
        // Validated here rather than force-unwrapped at the constant: this package force-unwraps
        // nowhere, and a typo should surface as a typed error naming the bad value, not a crash.
        guard let endpoint = URL(string: tokenURL) else {
            throw LLMProviderError.invalidRequest(
                detail: "Codex token endpoint is not a valid URL: \(tokenURL)")
        }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "grant_type": "refresh_token",
            "refresh_token": current.refreshToken,
            "client_id": clientID,
            "scope": "openid profile email offline_access"
        ])

        let (data, response) = try await transport(request)
        guard let http = response as? HTTPURLResponse else {
            throw LLMProviderError.invalidResponse
        }
        guard http.statusCode == 200 else {
            logger.error("Codex token refresh failed with HTTP \(http.statusCode, privacy: .public) — the user may need to run `codex login` again.")
            throw LLMProviderError.httpError(
                statusCode: http.statusCode,
                body: String(data: data, encoding: .utf8) ?? "",
                url: endpoint,
                retryAfter: LLMProviderError.parseRetryAfter(http.value(forHTTPHeaderField: "Retry-After")))
        }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let access = object["access_token"] as? String, !access.isEmpty
        else {
            throw LLMProviderError.malformedResponse(
                detail: "Codex token refresh returned no access_token")
        }

        var refreshed = CodexAuthTokens(
            accessToken: access,
            // The server omits these when they are unchanged; keeping the current value is what
            // makes a rotation-free refresh a no-op rather than a sign-out.
            idToken: (object["id_token"] as? String) ?? current.idToken,
            refreshToken: (object["refresh_token"] as? String) ?? current.refreshToken,
            accountId: current.accountId,
            lastRefresh: Date()
        )
        if let fresh = CodexJWT.accountId(access) { refreshed.accountId = fresh }

        // A write failure must not fail the CALL: the tokens in hand are valid and the request they
        // were fetched for should proceed. The cost of not persisting is one redundant refresh next
        // time, which is strictly better than failing work over a disk error.
        do { try store.save(refreshed) } catch {
            logger.error("Refreshed Codex tokens could not be written to \(self.store.url.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
        return refreshed
    }
}
