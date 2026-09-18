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
    /// A DATEABLE token answers from its own `exp` claim and from nothing else — `lastRefresh` is
    /// never consulted, so a token that genuinely expired can never be vouched for by a recent
    /// refresh.
    ///
    /// An UNDATEABLE token (opaque, or a JWT with no `exp`) has no expiry to read, so it is dated by
    /// the clock we do have: when we last refreshed it. It refreshes when that clock says
    /// `undatedAfter` has passed, when there is no clock at all, or when the clock reads the future
    /// (a backwards system-clock jump, which must not pin a credential shut).
    ///
    /// Answering a bare `true` for the undateable case — which is what this did until the undated
    /// clock existed — costs ONE OAuth round trip plus one read-modify-write of `auth.json` PER LLM
    /// CALL, forever, because refreshing an opaque token yields an equally opaque one. The trade is
    /// deliberate: an undateable token may be served for up to `undatedAfter` after it actually
    /// died, which fails the call the same way using a dead token always did.
    public func needsRefresh(
        within window: TimeInterval,
        undatedAfter: TimeInterval = CodexAuthCoordinator.undatedRefreshInterval,
        now: Date = Date()
    ) -> Bool {
        if let expiry { return expiry.timeIntervalSince(now) < window }
        guard let lastRefresh else { return true }
        let age = now.timeIntervalSince(lastRefresh)
        return age < 0 || age >= undatedAfter
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

    /// How long an UNDATEABLE access token is trusted before it is refreshed anyway.
    ///
    /// The bound on both failure modes: at worst one refresh per this interval instead of one per
    /// LLM call, and at worst a dead opaque token served for this long. Ten minutes is short next to
    /// any plausible token lifetime (the dateable ones this backend issues run hours) and long next
    /// to a burst of agent turns, which is the ratio that matters.
    public static let undatedRefreshInterval: TimeInterval = 10 * 60

    /// How long a FAILED refresh suppresses the next attempt.
    ///
    /// `lastRefresh` is stamped only on SUCCESS, so nothing else stops a failing refresh from being
    /// re-attempted on every LLM call. The severe case is a 429 or 5xx from the TOKEN endpoint: the
    /// consumer classifies that transient and retries `send` up to 50 times, and each retry
    /// re-enters `validTokens` and fires another refresh POST — fifty requests per LLM call, per
    /// role, at an endpoint already refusing them. (A revoked refresh token is a 400, classified
    /// permanent, so it costs one POST per call rather than fifty.)
    ///
    /// Sized against the consumer's own retry curve (1, 2, 4, 8, 15, 15 … seconds): a minute is
    /// several retries wide, so a blip costs at most one further real attempt per minute rather than
    /// one per retry, while the budget still spans enough minutes to catch a recovery.
    public static let failureCooldown: TimeInterval = 60

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
    ///
    /// Tagged, because the slot is released by the refresh that took it and a LATE finisher must
    /// never release a NEWER refresh's slot — that would let a third caller start yet another
    /// concurrent writer of `auth.json`.
    private var inFlight: (id: UUID, task: Task<CodexAuthTokens, Error>)?

    /// The last refresh FAILURE and when it happened, for the cooldown.
    ///
    /// The error is stored and re-thrown VERBATIM rather than replaced with a marker: the consumer
    /// classifies transient vs permanent off the typed error, so a substitute would change how the
    /// failure is retried and reported. Replaying the real one makes the suppressed calls
    /// indistinguishable from the one that actually went out — which is the point.
    private var lastRefreshFailure: (at: Date, error: any Error)?

    /// The last token set we successfully refreshed, kept in memory.
    ///
    /// Exists for the case where `store.save` FAILS — a read-only `~/.codex`, a full disk. The
    /// refreshed token is perfectly good and the call proceeds, but disk still holds the expired
    /// one, so without this every subsequent call re-reads the stale copy and refreshes AGAIN:
    /// one OAuth round trip per LLM call, forever, visible only in a log line. Disk stays
    /// authoritative whenever it holds a usable token; this is consulted only when it does not.
    private var lastRefreshed: CodexAuthTokens?

    /// How long a failed refresh suppresses the next attempt. Injectable for tests only.
    private let failureCooldown: TimeInterval

    public init(
        store: CodexAuthStore = CodexAuthStore(),
        clientID: String = CodexAuthCoordinator.defaultClientID,
        tokenURL: String = CodexAuthCoordinator.defaultTokenURL,
        failureCooldown: TimeInterval = CodexAuthCoordinator.failureCooldown,
        transport: @escaping Transport = { try await URLSession.shared.data(for: $0) }
    ) {
        self.store = store
        self.clientID = clientID
        self.tokenURL = tokenURL
        self.failureCooldown = failureCooldown
        self.transport = transport
    }

    /// A token set good for the next `refreshWindow`, refreshing first if needed.
    ///
    /// - Throws: `LLMProviderError.invalidRequest` when no credential exists (the user has not run
    ///   `codex login`), or the refresh error when refreshing fails.
    public func validTokens(now: Date = Date()) async throws -> CodexAuthTokens {
        guard var current = store.load() else {
            throw LLMProviderError.invalidRequest(
                detail: "No ChatGPT credential found at \(store.url.path). Run `codex login` to sign in.")
        }
        // Checked BEFORE the cooldown, and that order is load-bearing: it is the whole recovery
        // path. A user who runs `codex login` mid-cooldown puts a good token on disk, which is
        // re-read here on the very next call and returned without consulting — or being blocked
        // by — the stored failure. Nothing pins a user in a failed state.
        guard current.needsRefresh(within: Self.refreshWindow, now: now) else {
            lastRefreshFailure = nil
            return current
        }

        // Disk is stale. If the last refresh we performed is still good, it never reached disk —
        // serve it rather than buying the same token again on every call.
        if let cached = lastRefreshed, !cached.needsRefresh(within: Self.refreshWindow, now: now) {
            lastRefreshFailure = nil
            return cached
        }
        // Both are stale, so refresh from whichever is NEWER: a refresh token the server rotated to
        // us but that never reached disk must not be replaced by the one it superseded.
        if let cached = lastRefreshed,
           (cached.lastRefresh ?? .distantPast) > (current.lastRefresh ?? .distantPast) {
            current = cached
        }

        // Joining a live refresh is always preferable to replaying a stale failure, so this comes
        // before the cooldown: the in-flight attempt is newer information than the stored error.
        if let inFlight { return try await inFlight.task.value }

        if let failure = lastRefreshFailure {
            let age = now.timeIntervalSince(failure.at)
            // `age < 0` is a backwards clock jump. Treated as cooldown EXPIRED, because the
            // alternative is suppressing refreshes until the clock catches up — which for a
            // year-sized jump is a year.
            if age >= 0, age < failureCooldown { throw failure.error }
        }

        // The REFRESH owns the slot and every piece of state it settles — not the caller that
        // happened to start it.
        //
        // Tying either to the caller's scope makes the single-flight depend on that caller reaching
        // its own `defer`. Today it always does, but only because of two language details nothing
        // here pins: awaiting an unstructured task is not a cancellation point, and `Task {}` does
        // not inherit cancellation. If either stopped holding, a cancelled initiator would free the
        // slot with its refresh still running and the next caller would start a SECOND concurrent
        // writer of `auth.json` — the exact race this actor exists to prevent.
        //
        // Doing the writes here also closes a real window: `performRefresh` saves to disk, and if
        // that save FAILS the only record of the new token is `lastRefreshed`. Setting it from the
        // caller left a gap between the slot opening and the value landing, in which an arriving
        // caller saw a stale disk and no cache and refreshed all over again.
        //
        // Deliberately unstructured, so the refresh does NOT inherit the caller's cancellation: it
        // is shared work. Up to five roles may be waiting on it, and abandoning it because the one
        // caller that started it went away would fail all of them.
        let id = UUID()
        let task = Task { () async throws -> CodexAuthTokens in
            defer { self.releaseInFlight(id) }
            do {
                let refreshed = try await self.performRefresh(current, now: now)
                self.lastRefreshFailure = nil
                self.lastRefreshed = refreshed
                return refreshed
            } catch {
                // A cancelled attempt is not the endpoint saying no, and must not arm the cooldown:
                // the stored error is replayed to EVERY caller for a full minute, and the consumer
                // classifies `CancellationError` as PERMANENT — so one cancelled refresh would hard
                // -fail every agent, unretried, for that minute.
                if !Self.isCancellation(error) {
                    self.lastRefreshFailure = (at: now, error: error)
                }
                throw error
            }
        }
        inFlight = (id: id, task: task)
        // One writer above means the caller only propagates. A burst of five roles therefore
        // produces one attempt and one record, with no initiator/joiner asymmetry to get wrong.
        return try await task.value
    }

    /// Releases the single-flight slot, but only if it still belongs to this refresh.
    ///
    /// The identity check is not reachable through the current call order — one release site, and a
    /// successor can only be created after it runs. It is kept, and tested directly, because it is
    /// what makes a SECOND release site safe to add: a timeout that abandons a slow refresh is the
    /// obvious next one, and without this a late finisher would then clear its successor's slot and
    /// hand a third caller its own concurrent writer of `auth.json`.
    ///
    /// Internal rather than private so that guard can be tested at all; the interleaving cannot be
    /// staged through `validTokens`, and an untested guard is one that quietly stops working.
    func releaseInFlight(_ id: UUID) {
        if inFlight?.id == id { inFlight = nil }
    }

    /// The in-flight slot's tag, for the test that covers `releaseInFlight`.
    var inFlightID: UUID? { inFlight?.id }

    /// Whether an error means "this attempt was abandoned" rather than "the endpoint refused".
    ///
    /// Both spellings matter: Swift concurrency throws `CancellationError`, while `URLSession`
    /// reports a cancelled transfer as `URLError.cancelled`. Neither is evidence about the
    /// credential, so neither may arm the failure cooldown.
    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if let urlError = error as? URLError, urlError.code == .cancelled { return true }
        return false
    }

    /// Exchanges the refresh token for a fresh access token and writes the result back to disk.
    ///
    /// `now` comes from the caller rather than being read here so the stamp written to
    /// `last_refresh` — which is the undated token's only clock — agrees with the instant the
    /// decision to refresh was made, and so a test can control both.
    private func performRefresh(_ current: CodexAuthTokens, now: Date = Date()) async throws -> CodexAuthTokens {
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
            lastRefresh: now
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

// MARK: - Process-wide coordinators

extension CodexAuthCoordinator {
    /// The ONE coordinator for a given credential file, shared process-wide.
    ///
    /// Single-flighting is the whole point of this actor, and it only works if every caller that
    /// reads the same `auth.json` holds the same instance: Agent Smith builds one provider PER ROLE,
    /// so five per-provider coordinators would notice expiry in the same instant, fire five
    /// refreshes, and race each other writing the file — the loser persisting a refresh token the
    /// server has already rotated away, which signs the user out. Vending per path rather than one
    /// global instance keeps the injectable store injectable.
    ///
    /// Keyed on the LEXICALLY standardized path, deliberately not `resolvingSymlinksInPath()`:
    /// that resolves nothing for a component that does not exist yet, so the key for
    /// `~/.codex/auth.json` would CHANGE the moment `codex login` created the file — minting a
    /// second coordinator mid-process and reopening the exact race this closes.
    ///
    /// Process-wide, not machine-wide: the `codex` CLI itself and any second instance of the host
    /// app are still separate writers. `CodexAuthStore.save` writes atomically, so the worst case
    /// there is a lost update rather than a corrupt file.
    public static func shared(forStoreAt url: URL) -> CodexAuthCoordinator {
        let key = url.standardizedFileURL
        return coordinators.withLock { registry in
            if let existing = registry[key] { return existing }
            let made = CodexAuthCoordinator(store: CodexAuthStore(url: key))
            registry[key] = made
            return made
        }
    }

    /// The shared coordinator for wherever the `codex` CLI keeps its credentials right now.
    ///
    /// Computed, never a `static let`: `CodexAuthStore.defaultURL` reads `CODEX_HOME` on every
    /// access, and a cached instance would keep serving a file the environment has moved away from.
    public static var sharedForDefaultStore: CodexAuthCoordinator {
        shared(forStoreAt: CodexAuthStore.defaultURL)
    }

    /// Never evicted — the live set is one entry per credential path, which in a real process is
    /// exactly one.
    private static let coordinators =
        OSAllocatedUnfairLock<[URL: CodexAuthCoordinator]>(initialState: [:])
}
