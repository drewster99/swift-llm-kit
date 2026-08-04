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
    public static func effective(_ requested: Int, measuredMaximum: Int? = nil) -> Int? {
        guard let measuredMaximum else { return max(requested, minimumTokens) }
        // A measured maximum BELOW the floor is a real finding — `probeThinkingBudgetRange`
        // records 0 when even the minimum was rejected — and means no budget is usable at all.
        // Treating it as "unmeasured" and sending the floor anyway ignored the measurement.
        guard measuredMaximum >= minimumTokens else { return nil }
        return min(max(requested, minimumTokens), measuredMaximum)
    }
}
