import Testing
import Foundation
@testable import SwiftLLMKit

/// The budget-range probe's contract. Every rule here exists because the alternative fabricates a
/// number: an invented ceiling, a rejection read from silence, or a "cap" that is really the point
/// where the endpoint stopped answering.
@Suite("Thinking budget range probe")
struct ThinkingBudgetRangeTests {

    /// A stub whose acceptance is a pure function of the budget, so the search is testable without
    /// a network. Records every budget attempted, which is what proves the convergence rule.
    private final class Recorder: @unchecked Sendable {
        var attempts: [Int] = []
        /// Every (budget, pairedMaxOutputTokens) the probe asked for — Anthropic requires
        /// `max_tokens > budget_tokens`, and a probe that failed to pair them would converge on
        /// the PAIRING boundary and report it as the model's ceiling.
        var pairings: [(budget: Int, pairedMax: Int)] = []
        let accept: (Int) -> Bool?
        init(accept: @escaping (Int) -> Bool?) { self.accept = accept }
    }

    private struct Stub: LLMProvider {
        let budget: Int
        let recorder: Recorder
        func send(messages: [LLMMessage], tools: [LLMToolDefinition],
                  overrides: LLMCallOverrides) async throws -> LLMResponse {
            recorder.attempts.append(budget)
            switch recorder.accept(budget) {
            case true: return LLMResponse(text: "ok")
            case false: throw LLMProviderError.httpError(statusCode: 400, body: "budget too large")
            case nil: throw LLMProviderError.invalidResponse   // classified as "no answer"
            }
        }
    }

    private func probe(maxOutput: Int?, maxContext: Int? = nil,
                       accounting: ThinkingBudgetAccounting? = .drawnFromMaxOutputTokens,
                       accept: @escaping (Int) -> Bool?) async -> (ProbeFinding<Int>, [Int]) {
        let recorder = Recorder(accept: accept)
        let finding = await ModelProber.probeThinkingBudgetRange(
            accounting: accounting, maxOutputTokens: maxOutput, maxContextTokens: maxContext,
            makeProviderWithBudget: { budget, pairedMax in
                recorder.pairings.append((budget, pairedMax))
                return Stub(budget: budget, recorder: recorder)
            })
        return (finding, recorder.attempts)
    }

    @Test("With no known limit it declines rather than inventing a ceiling")
    func noLimitDeclines() async {
        let (finding, attempts) = await probe(maxOutput: nil, maxContext: nil) { _ in true }
        #expect(finding.status == .inconclusive)
        #expect(attempts.isEmpty, "it must not spend a call it cannot interpret")
    }

    /// The ceiling RESERVES room for the pairing when the budget is drawn from the output
    /// allowance: probing at `maxOutputTokens` would need a `max_tokens` above it, so the refusal
    /// would be about the output cap and the search would converge on that boundary every time.
    @Test("The search ceiling reserves pairing room, and accepting it is a real answer")
    func fullAllowanceAccepted() async {
        let expectedCeiling = 64_000 - ThinkingBudget.minimumTokens
        let (finding, attempts) = await probe(maxOutput: 64_000) { _ in true }
        #expect(finding.value == expectedCeiling)
        #expect(attempts == [expectedCeiling], "one call settles it — nothing higher is reachable")
    }

    @Test("A separate allowance is NOT reduced — nothing has to be paired with it")
    func separateAllowanceKeepsFullCeiling() async {
        let (finding, attempts) = await probe(
            maxOutput: nil, maxContext: 200_000, accounting: .separate) { _ in true }
        #expect(finding.value == 200_000)
        #expect(attempts == [200_000])
    }

    @Test("Converges on the real boundary, within the floor's precision")
    func convergesOnBoundary() async {
        // True ceiling 20_000: anything at or below is accepted.
        let (finding, attempts) = await probe(maxOutput: 64_000) { $0 <= 20_000 }
        let found = try? #require(finding.value)
        #expect(finding.status == .established)
        #expect((found ?? 0) <= 20_000, "never reports a budget the endpoint rejected")
        #expect(20_000 - (found ?? 0) <= ThinkingBudget.minimumTokens, "converged to within the floor")
        #expect(attempts.count < 20, "binary search, not a linear walk")
    }

    @Test("A rejected minimum is a finding: there is no usable budget range")
    func minimumRejected() async {
        let (finding, _) = await probe(maxOutput: 64_000) { $0 < ThinkingBudget.minimumTokens }
        #expect(finding.value == 0)
        #expect(finding.status == .established)
    }

    /// Silence is not rejection. Halving the range on a transport error would converge on a number
    /// describing when the endpoint stopped ANSWERING and report it as the model's ceiling.
    @Test("An unanswerable call mid-search stops at the last known-good value")
    func silenceDoesNotNarrow() async {
        // Ceiling rejected and the minimum accepted, so the search starts; then the endpoint goes
        // quiet. The last value it actually accepted is the most that can honestly be claimed.
        let ceiling = 64_000 - ThinkingBudget.minimumTokens
        let (finding, _) = await probe(maxOutput: 64_000) { budget in
            if budget >= ceiling { return false }
            if budget <= ThinkingBudget.minimumTokens { return true }
            return nil
        }
        #expect(finding.value == ThinkingBudget.minimumTokens)
        #expect(finding.evidence?.contains("stopped answering") == true)
    }

    /// Silence at the ceiling is different: nothing was learned at all, so there is no known-good
    /// value to report and the whole probe is inconclusive.
    @Test("An unanswerable ceiling call yields no finding rather than a floor")
    func silenceAtCeilingIsInconclusive() async {
        let (finding, _) = await probe(maxOutput: 64_000) { _ in nil }
        #expect(finding.status == .inconclusive)
        #expect(finding.value == nil)
    }

    /// The probe must hand the factory a `max_tokens` above each budget. Without it, Anthropic
    /// rejects on the PAIRING and the search converges on that boundary instead of the real one.
    @Test("Every attempt pairs a max_tokens above the budget")
    func pairsMaxTokensAboveBudget() async {
        let recorder = Recorder(accept: { $0 <= 20_000 })
        _ = await ModelProber.probeThinkingBudgetRange(
            accounting: .drawnFromMaxOutputTokens, maxOutputTokens: 64_000, maxContextTokens: nil,
            makeProviderWithBudget: { budget, pairedMax in
                recorder.pairings.append((budget, pairedMax))
                return Stub(budget: budget, recorder: recorder)
            })
        #expect(!recorder.pairings.isEmpty)
        #expect(recorder.pairings.allSatisfy { $0.pairedMax > $0.budget },
                "a paired max_tokens at or below the budget makes the refusal about the pairing")
    }

    @Test("Accounting picks which limit bounds the search")
    func accountingPicksTheBound() {
        let drawn = ThinkingBudgetAccounting.drawnFromMaxOutputTokens
        let separate = ThinkingBudgetAccounting.separate
        #expect(drawn.searchCeiling(maxOutputTokens: 8_000, maxContextTokens: 200_000) == 8_000)
        #expect(separate.searchCeiling(maxOutputTokens: 8_000, maxContextTokens: 200_000) == 200_000)
        // Each falls back to the other limit rather than giving up when only one is known.
        #expect(drawn.searchCeiling(maxOutputTokens: nil, maxContextTokens: 200_000) == 200_000)
        #expect(separate.searchCeiling(maxOutputTokens: 8_000, maxContextTokens: nil) == 8_000)
        #expect(drawn.searchCeiling(maxOutputTokens: nil, maxContextTokens: nil) == nil)
    }
}

/// The minimum search. Its sibling above measures a CEILING to a granularity; this one measures a
/// FLOOR to the exact token, because a floor is used verbatim as the value sent.
@Suite("Thinking budget minimum probe")
struct ThinkingBudgetMinimumTests {

    private final class Recorder: @unchecked Sendable {
        var attempts: [Int] = []
        let accept: (Int) -> Bool?
        init(accept: @escaping (Int) -> Bool?) { self.accept = accept }
    }

    private struct Stub: LLMProvider {
        let budget: Int
        let recorder: Recorder
        func send(messages: [LLMMessage], tools: [LLMToolDefinition],
                  overrides: LLMCallOverrides) async throws -> LLMResponse {
            recorder.attempts.append(budget)
            switch recorder.accept(budget) {
            case true: return LLMResponse(text: "ok")
            case false: throw LLMProviderError.httpError(statusCode: 400, body: "budget below minimum")
            case nil: throw LLMProviderError.invalidResponse
            }
        }
    }

    private func probe(knownAccepted: Int = 60_000,
                       cap: Int = ThinkingBudget.minimumSearchCap,
                       accept: @escaping (Int) -> Bool?) async -> (ProbeFinding<Int>, [Int]) {
        let recorder = Recorder(accept: accept)
        let finding = await ModelProber.probeThinkingBudgetMinimum(
            knownAcceptedBudget: knownAccepted, searchCap: cap,
            makeProviderWithBudget: { budget, _ in Stub(budget: budget, recorder: recorder) })
        return (finding, recorder.attempts)
    }

    /// The expected case, and the reason the documented floor is tested as a hypothesis rather than
    /// bisected into: confirming it takes exactly two observations, not fourteen.
    @Test("The documented floor is confirmed in two calls, not bisected into")
    func documentedFloorConfirmedCheaply() async {
        let (finding, attempts) = await probe { $0 >= ThinkingBudget.minimumTokens }
        #expect(finding.value == ThinkingBudget.minimumTokens)
        #expect(finding.status == .established)
        #expect(attempts == [1024, 1023], "accepted at the floor, rejected just below it — that IS the floor")
    }

    @Test("A floor below the documented one is found exactly, not rounded to a granularity")
    func lowerFloorFoundExactly() async {
        let (finding, attempts) = await probe { $0 >= 256 }
        #expect(finding.value == 256)
        #expect(finding.evidence?.contains("255 rejected") == true)
        #expect(attempts.count < 14, "bisection, not a walk")
    }

    /// The case the probe exists for: production floors at 1024 and this model wants more, so every
    /// budgeted request below the real minimum is rejected until the measurement lands.
    @Test("A floor ABOVE the documented one is found — the case this probe exists for")
    func higherFloorIsFound() async {
        let (finding, _) = await probe { $0 >= 4096 }
        #expect(finding.value == 4096)
        #expect(finding.status == .established)
    }

    @Test("No minimum at all is reported as 1, not as the documented floor")
    func noFloorAtAll() async {
        let (finding, _) = await probe { _ in true }
        #expect(finding.value == 1)
    }

    /// The cap is an ASSUMPTION. Recording it as the answer would be indistinguishable from having
    /// measured a model that really demands that much, so it must not be reported as a value.
    @Test("A minimum above the search cap is inconclusive, never the cap itself")
    func aboveCapIsInconclusive() async {
        let (finding, _) = await probe(knownAccepted: 60_000, cap: 8192) { $0 >= 20_000 }
        #expect(finding.status == .inconclusive)
        #expect(finding.value == nil, "the cap is not a measurement")
        #expect(finding.evidence?.contains("8192") == true)
    }

    /// Same rule as the maximum probe: silence is not rejection. Narrowing on a transport error
    /// would converge on where the endpoint stopped ANSWERING and report it as the model's floor.
    @Test("Silence mid-search reports the smallest value actually seen accepted")
    func silenceStopsAtKnownGood() async {
        // 1024 accepted, 1023 accepted, then the endpoint goes quiet inside the search.
        let (finding, _) = await probe { budget in
            if budget >= 1023 { return true }
            return nil
        }
        #expect(finding.value == 1023, "conservative: overstating a floor is safe, understating is a rejection")
        #expect(finding.evidence?.contains("stopped answering") == true)
    }

    @Test("With no usable budget range there is nothing to search below")
    func noAcceptedBudgetMeansNoSearch() async {
        let (finding, attempts) = await probe(knownAccepted: 0) { _ in true }
        #expect(finding.status == .inconclusive)
        #expect(attempts.isEmpty, "it must not spend a call it cannot interpret")
    }

    /// A model whose whole budget allowance is under the documented floor: the hypothesis is capped
    /// to what is actually reachable rather than asking about a value the model can never accept.
    @Test("The hypothesis is capped by the known-accepted budget")
    func hypothesisCappedByKnownAccepted() async {
        let (finding, attempts) = await probe(knownAccepted: 512) { $0 >= 300 }
        #expect(finding.value == 300)
        #expect(attempts.first == 512, "never asks about a budget above the one known to work")
    }
}
