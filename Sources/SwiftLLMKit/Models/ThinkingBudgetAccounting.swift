import Foundation

/// Whose token allowance a reasoning budget is drawn from.
///
/// This is NOT derivable from ``ReasoningControl``: that says which KEYS to send, while this says
/// whose budget the thinking tokens come out of. Two providers can share a mechanism and differ
/// here, and getting it wrong silently truncates replies — a caller who sets a 32k budget and a 32k
/// output cap gets no visible answer at all on a `drawnFromMaxOutputTokens` model, with no error.
///
/// Known: Anthropic draws from `max_tokens` and enforces `max_tokens > budget_tokens` (see
/// `AnthropicProvider`'s clamp). Alibaba's `thinking_budget` and Gemini's `thinkingConfig` use
/// separate keys whose accounting is UNVERIFIED — which is why this is a probed/overridable fact
/// rather than a table keyed on provider.
public enum ThinkingBudgetAccounting: String, Sendable, Equatable, Hashable, Codable, CaseIterable {
    /// The budget is spent from the output-token allowance: `max_tokens` must exceed it, and the
    /// usable reply budget is `max_tokens - budget`.
    case drawnFromMaxOutputTokens
    /// The budget is its own allowance, independent of the output cap.
    case separate

    /// Label for the per-model editor picker.
    public var editorTitle: String {
        switch self {
        case .drawnFromMaxOutputTokens: return "Drawn from max output tokens"
        case .separate: return "Separate allowance"
        }
    }

    public var editorDescription: String {
        switch self {
        case .drawnFromMaxOutputTokens:
            return "Thinking tokens are spent from `max_tokens`, which must exceed the budget."
        case .separate:
            return "The thinking budget is independent of the output-token cap."
        }
    }

    /// The largest budget worth attempting for a model with these known limits.
    ///
    /// Derived rather than a constant: when the budget is drawn from the output allowance, a value
    /// above that allowance is unreachable by construction, so searching past it only buys refusals
    /// that say nothing about the budget. When it is separate, the context window is the physical
    /// bound. `nil` means neither limit is known and the caller must fall back to the
    /// absurd-value probe, which reports `inconclusive` rather than inventing a ceiling.
    public func searchCeiling(maxOutputTokens: Int?, maxContextTokens: Int?) -> Int? {
        switch self {
        case .drawnFromMaxOutputTokens: return maxOutputTokens ?? maxContextTokens
        case .separate: return maxContextTokens ?? maxOutputTokens
        }
    }
}
