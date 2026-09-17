import Foundation
import os

/// How full the ChatGPT subscription's usage window is, as the backend last reported it.
///
/// Worth having because this arrives on ORDINARY SUCCESSFUL responses — `x-codex-*` headers on
/// every call — not only on the failure that stops work. A ceiling you can watch approaching is a
/// different thing from one you discover by hitting it, and the difference costs nothing to collect.
public struct CodexUsageWindow: Sendable, Equatable {
    /// Which allowance is currently in force (e.g. `premium`).
    public let activeLimit: String?
    /// The named window the percentage belongs to.
    public let limitName: String?
    /// 0–100 of the primary window, when stated.
    public let primaryPercentUsed: Double?
    /// Whether the account's credits are explicitly unmetered. `false` does NOT mean depleted — it
    /// means a balance applies.
    public let creditsUnlimited: Bool?
    /// When this was observed, so a stale reading can be shown as stale rather than as current.
    public let observedAt: Date

    public init(
        activeLimit: String? = nil,
        limitName: String? = nil,
        primaryPercentUsed: Double? = nil,
        creditsUnlimited: Bool? = nil,
        observedAt: Date = Date()
    ) {
        self.activeLimit = activeLimit
        self.limitName = limitName
        self.primaryPercentUsed = primaryPercentUsed
        self.creditsUnlimited = creditsUnlimited
        self.observedAt = observedAt
    }

    /// Whether anything was actually reported. An all-nil reading is not evidence of a healthy
    /// window; it is evidence of no window information, and must not render as "0% used".
    public var hasAnyReading: Bool {
        activeLimit != nil || limitName != nil || primaryPercentUsed != nil || creditsUnlimited != nil
    }

    /// Builds a window from the `x-codex-*` headers of a response, keyed lowercase.
    ///
    /// The percentage header is named `…-primary-over-secondary-limit-percent`, which reads as a
    /// ratio but is reported as a percentage of the primary window; it is surfaced under a name that
    /// says what it is used for rather than reproducing the wire spelling.
    public static func from(headers: [String: String]) -> CodexUsageWindow {
        func number(_ key: String) -> Double? { headers[key].flatMap(Double.init) }
        return CodexUsageWindow(
            activeLimit: headers["x-codex-active-limit"],
            limitName: headers["x-codex-bengalfox-limit-name"],
            primaryPercentUsed: number("x-codex-primary-over-secondary-limit-percent")
                ?? number("x-codex-bengalfox-primary-over-secondary-limit-percent"),
            // The header is a capitalised word, not a JSON bool.
            creditsUnlimited: headers["x-codex-credits-unlimited"].map {
                $0.lowercased() == "true"
            })
    }
}

/// The most recent window reading, shared across the app.
///
/// A plain last-writer-wins box rather than a history: the question it answers is "how much is left
/// right now", and every call refreshes it. Kept out of `LLMResponse` deliberately — threading
/// provider-specific headers through the response type, the agent loop, and every caller in order to
/// reach one settings row would put a Codex detail in everyone's path.
public enum CodexUsageMonitor {
    private static let state = OSAllocatedUnfairLock<CodexUsageWindow?>(initialState: nil)

    /// The latest reading, or `nil` if no Codex call has completed this launch.
    public static var current: CodexUsageWindow? {
        state.withLock { $0 }
    }

    /// Records a reading. Ignores one that carries nothing, so a response without these headers
    /// cannot erase a good reading from the call before it.
    public static func record(_ window: CodexUsageWindow) {
        guard window.hasAnyReading else { return }
        state.withLock { $0 = window }
    }

    /// Forgets the current reading — for sign-out, where the old account's numbers must not linger.
    public static func reset() {
        state.withLock { $0 = nil }
    }
}
