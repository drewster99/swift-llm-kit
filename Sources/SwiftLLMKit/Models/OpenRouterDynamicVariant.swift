import Foundation

/// A routing suffix OpenRouter accepts on *any* model slug.
///
/// OpenRouter splits variants two ways. **Static** variants (`:free`, `:batch`, `:thinking`,
/// `:extended`) belong to specific models and are enumerated in `/models` as distinct entries,
/// so they arrive through normal decoding. **Dynamic** variants change how a request is routed
/// rather than which model answers it, apply to every model, and appear in `/models` **never** —
/// verified 2026-07-30, where `:nitro`, `:floor`, `:exacto` and `:online` each occurred zero
/// times across 367 entries. Enumeration therefore cannot discover them; the only way they reach
/// a model picker or the capability prober is for us to state them.
///
/// Deliberately only two cases:
/// - `:exacto` is documented ("quality-first provider sorting" for tool-calling reliability) but
///   has never been exercised against the live API here. An untested routing mode is not
///   something to put in front of a user as a working choice.
/// - `:online` is deprecated by OpenRouter in favour of the `openrouter:web_search` server tool.
public enum OpenRouterDynamicVariant: String, CaseIterable, Sendable {
    /// Sort candidate providers by price rather than OpenRouter's default price-weighted
    /// load balancing.
    case floor
    /// Sort candidate providers by throughput.
    case nitro

    /// The suffix appended to a base model slug, e.g. `qwen/qwen3.5-397b-a17b:floor`.
    public var suffix: String { ":\(rawValue)" }

    /// The equivalent `provider.sort` value in a chat-completions request body.
    ///
    /// The suffix and the body field are two spellings of one instruction, and the body field is
    /// strictly more capable — it composes with `quantizations`, `max_price`, `only`/`ignore`, and
    /// it reaches `"latency"`, which has no suffix at all. Callers that would rather not encode a
    /// routing preference into the model id can send `{"provider": {"sort": <this>}}` instead.
    public var providerSortValue: String {
        switch self {
        case .floor: return "price"
        case .nitro: return "throughput"
        }
    }

    /// Appended to the base model's display name, so a picker row explains itself.
    public var displayNameSuffix: String {
        switch self {
        case .floor: return "(floor — cheapest route)"
        case .nitro: return "(nitro — fastest route)"
        }
    }

    /// Prepended to the base model's description. States what the variant changes and, because
    /// the inherited pricing belongs to the default route, that the cost figure is an estimate.
    public var descriptionSentence: String {
        switch self {
        case .floor:
            return "Routing variant: OpenRouter sorts providers by price for this model instead of its default price-weighted load balancing. Pricing shown is the default route's and is an estimate — sorting is what changes which provider, and therefore which price, actually serves the request."
        case .nitro:
            return "Routing variant: OpenRouter sorts providers by throughput for this model instead of its default price-weighted load balancing. Pricing shown is the default route's and is an estimate — a faster provider is frequently a more expensive one."
        }
    }

    /// Whether a model slug may take a dynamic suffix.
    ///
    /// Suffixes do not stack: `foo/model:free:floor` is not a slug OpenRouter accepts, so only
    /// base ids qualify. A colon is exactly the marker of an already-suffixed id.
    public static func acceptsDynamicSuffix(modelID: String) -> Bool {
        !modelID.contains(":")
    }
}
