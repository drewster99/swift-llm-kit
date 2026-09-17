import Foundation
import Testing
@testable import SwiftLLMKit

/// The whole point of this type is that the three ways of running out are NOT interchangeable, so
/// these tests are mostly about keeping them apart.
@Suite("Codex limit classification")
struct CodexLimitTests {

    @Test("A usage window and a rate limit are different outcomes, not different wording")
    func windowVersusRate() throws {
        let window = try #require(CodexLimit.parse(
            statusCode: 429,
            body: #"{"error":{"type":"usage_limit_reached","resets_in_seconds":3600,"plan_type":"pro"}}"#))
        #expect(window.clearsByWaiting)
        #expect(window.resetsAt != nil, "a window states when it lifts")
        #expect(window.planType == "pro")

        let rate = try #require(CodexLimit.parse(
            statusCode: 429, body: #"{"error":{"type":"rate_limit_exceeded"}}"#))
        #expect(rate.kind == .rateLimited)
        #expect(rate.resetsAt == nil, "a rate limit has no window to wait out")
    }

    @Test("Credit depletion never claims a reset, because waiting may not fix it")
    func creditsHaveNoReset() throws {
        let owner = try #require(CodexLimit.parse(
            statusCode: 429,
            body: #"{"detail":{"rate_limit_reached_type":"workspace_owner_credits_depleted"}}"#))
        #expect(owner.kind == .creditsDepleted(userCanResolve: true))
        #expect(owner.clearsByWaiting == false, """
            scheduling a wake for a reset that does not exist would sleep the task forever
            """)
        #expect(owner.resetsAt == nil)

        // A member cannot top up the workspace, so the remedy — and the message — differ.
        let member = try #require(CodexLimit.parse(
            statusCode: 429,
            body: #"{"detail":{"rate_limit_reached_type":"workspace_member_credits_depleted"}}"#))
        #expect(member.kind == .creditsDepleted(userCanResolve: false))
    }

    @Test("A spend cap outranks the window type it arrives with")
    func spendControlWins() throws {
        // An administrative cap does not lift when the usage window rolls over, so it must not be
        // read as a waitable window just because one is also named.
        let limit = try #require(CodexLimit.parse(
            statusCode: 429,
            body: #"{"detail":{"rate_limit_reached_type":"workspace_owner_usage_limit_reached","spend_control_reached":true}}"#))
        #expect(limit.kind == .spendControlReached)
        #expect(limit.clearsByWaiting == false)
    }

    @Test("The reset instant is read from every spelling in circulation")
    func resetSpellings() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        func reset(_ json: String) -> Date? {
            guard let data = json.data(using: .utf8),
                  let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            return CodexLimit.resetDate(from: CodexLimit.flatten(root), now: now)
        }
        #expect(reset(#"{"resets_in_seconds":120}"#) == now.addingTimeInterval(120))
        #expect(reset(#"{"resets_at":1000600}"#) == Date(timeIntervalSince1970: 1_000_600))
        // Milliseconds are told from seconds by magnitude — a seconds-epoch that large is year
        // 33658, which no reset window reaches.
        #expect(reset(#"{"resets_at":1700000000000}"#) == Date(timeIntervalSince1970: 1_700_000_000))
        #expect(reset(#"{"resets_at":"2026-09-18T12:34:17Z"}"#) != nil)
        #expect(reset(#"{"resets_at":"not a date"}"#) == nil)
        #expect(reset(#"{}"#) == nil)
    }

    @Test("Fields are found wherever the backend nested them")
    func findsNestedFields() throws {
        // Observed under `error`, under `detail`, and at the top level on different paths.
        for body in [
            #"{"type":"usage_limit_reached"}"#,
            #"{"error":{"type":"usage_limit_reached"}}"#,
            #"{"detail":{"inner":{"type":"usage_limit_reached"}}}"#,
            #"{"additional_rate_limits":[{"type":"usage_limit_reached"}]}"#
        ] {
            let limit = try #require(CodexLimit.parse(statusCode: 429, body: body), "\(body)")
            #expect(limit.clearsByWaiting, "\(body)")
            #expect(limit.resetsAt == nil, "no reset was stated in \(body)")
        }
    }

    @Test("A shallower key is not shadowed by a deeper one")
    func shallowKeysWin() throws {
        let limit = try #require(CodexLimit.parse(
            statusCode: 429,
            body: #"{"type":"rate_limit_exceeded","meta":{"nested":{"type":"usage_limit_reached"}}}"#))
        #expect(limit.kind == .rateLimited, "the top-level type describes THIS error")
    }

    @Test("A 429 with nothing recognisable is still a rate limit; a 500 is not a limit at all")
    func fallsBackOnStatusOnly() {
        // The status says this much on its own. Guessing from the message text is the thing this
        // type exists to avoid.
        #expect(CodexLimit.parse(statusCode: 429, body: "not json")?.kind == .rateLimited)
        #expect(CodexLimit.parse(statusCode: 429, body: #"{"error":{"message":"slow down"}}"#)?.kind
                == .rateLimited)
        #expect(CodexLimit.parse(statusCode: 500, body: #"{"error":{"message":"boom"}}"#) == nil)
        #expect(CodexLimit.parse(statusCode: 400, body: #"{"detail":"bad model"}"#) == nil)
    }

    @Test("Display fields are carried through when stated")
    func carriesDisplayFields() throws {
        let limit = try #require(CodexLimit.parse(
            statusCode: 429,
            body: #"{"detail":{"rate_limit_reached_type":"workspace_owner_usage_limit_reached","plan_type":"prolite","limit_name":"GPT-5.3-Codex-Spark","used_percent":100}}"#))
        #expect(limit.planType == "prolite")
        #expect(limit.limitName == "GPT-5.3-Codex-Spark")
        #expect(limit.usedPercent == 100)
    }
}
