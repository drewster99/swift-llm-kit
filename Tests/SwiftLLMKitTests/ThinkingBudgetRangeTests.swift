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
            makeProviderWithBudget: { Stub(budget: $0, recorder: recorder) })
        return (finding, recorder.attempts)
    }

    @Test("With no known limit it declines rather than inventing a ceiling")
    func noLimitDeclines() async {
        let (finding, attempts) = await probe(maxOutput: nil, maxContext: nil) { _ in true }
        #expect(finding.status == .inconclusive)
        #expect(attempts.isEmpty, "it must not spend a call it cannot interpret")
    }

    @Test("Accepting the full allowance is a real answer, not a search")
    func fullAllowanceAccepted() async {
        let (finding, attempts) = await probe(maxOutput: 64_000) { _ in true }
        #expect(finding.value == 64_000)
        #expect(attempts == [64_000], "one call settles it — nothing above the allowance is reachable")
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
        let (finding, _) = await probe(maxOutput: 64_000) { budget in
            if budget >= 64_000 { return false }
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
