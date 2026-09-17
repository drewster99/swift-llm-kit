import Foundation
import Testing
@testable import SwiftLLMKit

/// The window headers arrive on ordinary successful calls, so these are about not MISREADING a
/// reading — particularly not turning "nothing was reported" into a reassuring number.
@Suite("Codex usage window")
struct CodexUsageWindowTests {

    /// The exact header set a live call returns, lowercased as the provider collects them.
    static let live: [String: String] = [
        "x-codex-active-limit": "premium",
        "x-codex-primary-over-secondary-limit-percent": "0",
        "x-codex-credits-unlimited": "False",
        "x-codex-bengalfox-primary-over-secondary-limit-percent": "0",
        "x-codex-bengalfox-limit-name": "GPT-5.3-Codex-Spark"
    ]

    @Test("A live header set is read into a window")
    func readsLiveHeaders() {
        let window = CodexUsageWindow.from(headers: Self.live)
        #expect(window.activeLimit == "premium")
        #expect(window.limitName == "GPT-5.3-Codex-Spark")
        #expect(window.primaryPercentUsed == 0)
        // The header is a capitalised word, not a JSON bool — parsing it as one yields nil and
        // would read as "unknown" for an account that has a balance.
        #expect(window.creditsUnlimited == false)
        #expect(window.hasAnyReading)
    }

    @Test("No headers is NOT zero percent used")
    func absenceIsNotZero() {
        let window = CodexUsageWindow.from(headers: [:])
        #expect(window.primaryPercentUsed == nil, "absence must not render as a reassuring 0%")
        #expect(window.hasAnyReading == false)
    }

    @Test("A reading without headers cannot erase a good one")
    func emptyReadingDoesNotClobber() {
        CodexUsageMonitor.reset()
        CodexUsageMonitor.record(CodexUsageWindow.from(headers: Self.live))
        // A response from a path that does not carry these headers must leave the last real
        // reading standing, rather than blanking the display to "unknown".
        CodexUsageMonitor.record(CodexUsageWindow.from(headers: [:]))
        #expect(CodexUsageMonitor.current?.limitName == "GPT-5.3-Codex-Spark")
        CodexUsageMonitor.reset()
        #expect(CodexUsageMonitor.current == nil, "sign-out must not leave the old account's numbers")
    }

    @Test("A later reading replaces an earlier one")
    func laterReadingWins() {
        CodexUsageMonitor.reset()
        CodexUsageMonitor.record(CodexUsageWindow(primaryPercentUsed: 10))
        CodexUsageMonitor.record(CodexUsageWindow(primaryPercentUsed: 85))
        #expect(CodexUsageMonitor.current?.primaryPercentUsed == 85)
        CodexUsageMonitor.reset()
    }
}
