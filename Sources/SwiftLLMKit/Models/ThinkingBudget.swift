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
/// This is a stopgap standing in for a measured fact. The floor is a per-model property we do not
/// currently probe; when `probeThinkingBudgetRange` lands, ``effective(_:)`` becomes the one place
/// that consults the model's own reported range instead of this constant.
public enum ThinkingBudget {
    /// The smallest budget the supporting APIs accept.
    public static let minimumTokens = 1024

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
        measuredMaximum: Int? = nil
    ) -> (maxTokens: Int, budget: Int?) {
        guard let wanted = effective(requestedBudget, measuredMaximum: measuredMaximum) else {
            return (requestedMax, nil)
        }
        let raised = max(requestedMax, wanted + 1)
        // Never below what the caller already asked for, even if the catalog cap is smaller.
        let maxTokens = modelMaxOutputTokens.map { min(raised, max($0, requestedMax)) } ?? raised
        let room = maxTokens - 1
        return (maxTokens, room >= minimumTokens ? min(wanted, room) : nil)
    }

    public static func effective(_ requested: Int, measuredMaximum: Int? = nil) -> Int? {
        guard let measuredMaximum else { return max(requested, minimumTokens) }
        // A measured maximum BELOW the floor is a real finding — `probeThinkingBudgetRange`
        // records 0 when even the minimum was rejected — and means no budget is usable at all.
        // Treating it as "unmeasured" and sending the floor anyway ignored the measurement.
        guard measuredMaximum >= minimumTokens else { return nil }
        return min(max(requested, minimumTokens), measuredMaximum)
    }
}
