import Foundation

/// Per-call control over how the model selects from the provided `tools`.
///
/// Each provider has its own native shape:
/// - **Anthropic**: `tool_choice: {"type": "auto"|"any"|"tool"|"none", "name": "..."}`
/// - **OpenAI / OpenAI-compatible**: `tool_choice: "auto"|"required"|"none" |
///   {"type": "function", "function": {"name": "..."}}`
/// - **Gemini**: `toolConfig.functionCallingConfig.mode: "AUTO"|"ANY"|"NONE",
///   allowedFunctionNames: [...]`
///
/// This enum is the unified abstraction; each provider translates to its own
/// wire format. The `nil` value (i.e. callers passing `toolChoice: nil`) means
/// "don't emit the field" — provider's API default applies (usually `auto`).
///
/// **Note:** `LLMToolChoice` is meaningful only when `tools` is non-empty.
/// Setting any value other than `.auto` (or `.textOnly`) without tools
/// produces undefined / provider-rejected behavior; providers may silently
/// ignore the field, return an error, or behave inconsistently. swift-llm-kit
/// does not validate this — the caller is responsible.
public enum LLMToolChoice: Sendable, Equatable {
    /// Model decides whether to call a tool. Equivalent to omitting the field
    /// entirely; emitted explicitly so callers can be unambiguous.
    case auto

    /// Model MUST call exactly one tool (any of the provided tools, but at
    /// least one). Useful for "tool-only" workflows like search or
    /// classification where text-only responses are not desired.
    /// - Anthropic: `{"type": "any"}`
    /// - OpenAI: `"required"`
    /// - Gemini: `mode: "ANY"`
    case required

    /// Model MUST NOT call any tool — respond with text only. Useful for
    /// post-processing turns where you don't want further tool dispatching.
    /// - Anthropic: `{"type": "none"}`
    /// - OpenAI: `"none"`
    /// - Gemini: `mode: "NONE"`
    ///
    /// Named `textOnly` (not `.none`) to avoid silent collision with
    /// `Optional<LLMToolChoice>.none` — Swift's overload resolution makes
    /// `toolChoice: .none` ambiguous and silently picks the Optional case
    /// (i.e. nil), producing the OPPOSITE behavior. `.textOnly` is also
    /// more descriptive of the intent.
    case textOnly

    /// Model MUST call the specific tool named here. Useful for forced
    /// dispatching to a known tool (e.g. always start with `get_user_zipcode`
    /// on a "weather where I am" workflow).
    /// - Anthropic: `{"type": "tool", "name": <name>}`
    /// - OpenAI: `{"type": "function", "function": {"name": <name>}}`
    /// - Gemini: `mode: "ANY", allowedFunctionNames: [<name>]`
    case specific(name: String)
}
