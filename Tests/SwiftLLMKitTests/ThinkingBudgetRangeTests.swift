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
            llm: Stub(budget: 0, recorder: recorder), modelID: "m",
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
        #expect(attempts == [expectedCeiling, ThinkingBudget.minimumTokens],
                "the ceiling settles the maximum; the second call checks the floor we ASSUME")
    }

    @Test("A separate allowance is NOT reduced — nothing has to be paired with it")
    func separateAllowanceKeepsFullCeiling() async {
        let (finding, attempts) = await probe(
            maxOutput: nil, maxContext: 200_000, accounting: .separate) { _ in true }
        #expect(finding.value == 200_000)
        #expect(attempts == [200_000, ThinkingBudget.minimumTokens])
    }

    /// The floor is ASSERTED by `ThinkingBudget.effective`, not searched for, so the one thing this
    /// probe owes is noticing when the assertion is wrong for a model. Nothing reads the evidence
    /// string — this pins that a rejected floor is at least SAID, since it silently would not be
    /// otherwise: the accepted-ceiling path returns before the binary search ever reaches the floor.
    @Test("A rejected floor is reported in the evidence, and does not change the ceiling")
    func rejectedFloorIsSurfaced() async {
        let expectedCeiling = 64_000 - ThinkingBudget.minimumTokens
        let (finding, attempts) = await probe(maxOutput: 64_000) { $0 != ThinkingBudget.minimumTokens }
        #expect(finding.value == expectedCeiling, "a bad floor is not evidence about the ceiling")
        #expect(attempts == [expectedCeiling, ThinkingBudget.minimumTokens])
        #expect(finding.evidence?.contains("REJECTED") == true)
    }

    @Test("An accepted floor is recorded too, so silence never has to be interpreted")
    func acceptedFloorIsSurfaced() async {
        let (finding, _) = await probe(maxOutput: 64_000) { _ in true }
        #expect(finding.evidence?.contains("floor was also accepted") == true)
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
            llm: Stub(budget: 0, recorder: recorder), modelID: "m",
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
