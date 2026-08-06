import Foundation

/// The extended-thinking token budget floor, in ONE place.
///
/// Anthropic rejects a `thinking.budget_tokens` below 1024, and Alibaba Cloud's `thinking_budget`
/// follows the same floor. That number was previously written out at four separate call sites —
/// Anthropic's emission and its `max_tokens` clamp, Alibaba's emission, and the parallel
/// `prepareRequest` path — which is three chances for them to drift apart, and drifting is exactly
/// what a floor must not do: the clamp exists to keep `max_tokens` ABOVE the budget actually sent,
/// so a clamp computed from a different floor than the emission produces the API error it was
/// written to prevent.
///
/// ``minimumTokens`` is the DOCUMENTED floor and is now only a fallback: `ModelProber`
/// `probeThinkingBudgetMinimum` measures the real one per model, and ``effective(_:measuredMaximum:measuredMinimum:)``
/// prefers it whenever it exists. The constant remains because most models are never probed.
public enum ThinkingBudget {
    /// The smallest budget the supporting APIs document. A per-model measurement supersedes it.
    public static let minimumTokens = 1024

    /// The value an editor seeds when the user first overrides the thinking budget: the model's
    /// measured floor (never below the documented one), clamped to the measured ceiling so a
    /// fresh override can never be born already out of range and self-warning.
    public static func overrideSeed(measuredMinimum: Int?, measuredMaximum: Int?) -> Int {
        let floor = max(measuredMinimum ?? minimumTokens, minimumTokens)
        guard let measuredMaximum else { return floor }
        return min(floor, measuredMaximum)
    }

    /// How far above the documented floor the minimum search is willing to look.
    ///
    /// An ASSUMPTION, and the probe reports it as one rather than recording the cap as a result: no
    /// endpoint documents a floor anywhere near this, so a minimum above it is far more likely to
    /// mean the probe is measuring something else (a pairing constraint, a context limit) than that
    /// the model really demands 16K of thinking before it will answer.
    public static let minimumSearchCap = 16_384

    /// The budget that will actually be sent, or `nil` when the model is MEASURED to accept none.
    ///
    /// Callers must use this rather than flooring inline, so the value they reason about is the
    /// value that goes on the wire.
    ///
    /// `measuredMaximum` is the model's own probed ceiling when one exists. Passing it clamps from
    /// ABOVE as well as below, which is the difference between a request that is merely large and
    /// one the endpoint refuses outright.
    /// The `(max_tokens, budget_tokens)` pair to send for a requested budget.
    ///
    /// Anthropic requires `max_tokens` to EXCEED `budget_tokens`, and `max_tokens` must not exceed
    /// the MODEL's output cap. Those pull opposite ways and only one is negotiable: the cap is a
    /// hard limit, the budget is a depth preference. So a tight `requestedMax` is raised to clear
    /// the budget, and past a KNOWN cap it is the budget that gives way.
    ///
    /// `budget == nil` means no legal budget exists — emit no thinking block rather than one the
    /// endpoint will reject.
    ///
    /// Shared by `AnthropicProvider` and `LLMKitManager.prepareRequest` deliberately: they build
    /// the same request from the same configuration, and this rule was already wrong in two
    /// different ways when it lived in one of them and was absent from the other.
    public static func pairing(
        requestedBudget: Int,
        requestedMax: Int,
        modelMaxOutputTokens: Int?,
        measuredMaximum: Int? = nil,
        measuredMinimum: Int? = nil
    ) -> (maxTokens: Int, budget: Int?) {
        guard let wanted = effective(requestedBudget, measuredMaximum: measuredMaximum,
                                     measuredMinimum: measuredMinimum) else {
            return (requestedMax, nil)
        }
        let raised = max(requestedMax, wanted + 1)
        // Never below what the caller already asked for, even if the catalog cap is smaller.
        let maxTokens = modelMaxOutputTokens.map { min(raised, max($0, requestedMax)) } ?? raised
        let room = maxTokens - 1
        // The same floor the budget was raised to. Comparing against the CONSTANT here while
        // `effective` used a measured floor would emit a budget below the model's real minimum
        // whenever the two disagree — the exact rejection this measurement exists to prevent.
        return (maxTokens, room >= floor(measuredMinimum) ? min(wanted, room) : nil)
    }

    /// The floor in force for a model: its measured minimum when one exists, else the documented one.
    ///
    /// A measured floor of 0 is meaningful — it says this endpoint imposes no minimum — so this
    /// takes the measurement at face value rather than treating small values as missing data.
    private static func floor(_ measuredMinimum: Int?) -> Int { measuredMinimum ?? minimumTokens }

    public static func effective(_ requested: Int, measuredMaximum: Int? = nil,
                                 measuredMinimum: Int? = nil) -> Int? {
        let low = floor(measuredMinimum)
        guard let measuredMaximum else { return max(requested, low) }
        // A measured maximum BELOW the floor is a real finding — `probeThinkingBudgetRange`
        // records 0 when even the minimum was rejected — and means no budget is usable at all.
        // Treating it as "unmeasured" and sending the floor anyway ignored the measurement.
        guard measuredMaximum >= low else { return nil }
        return min(max(requested, low), measuredMaximum)
    }
}
