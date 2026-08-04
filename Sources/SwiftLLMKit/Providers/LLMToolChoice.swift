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

public extension LLMToolChoice {
    /// Which capability governs this option. `.auto` rides the general ``ModelCapability/toolChoice``
    /// because it is the endpoint's own default whenever tools are present.
    var requiredCapability: ModelCapability {
        switch self {
        case .auto: return .toolChoice
        case .required: return .toolChoiceRequired
        case .textOnly: return .toolChoiceNone
        case .specific: return .toolChoiceSpecificFunction
        }
    }

    /// The OpenAI-family wire value: an enum string, or an object naming one function.
    var openAIWireValue: AnyCodable {
        switch self {
        case .auto: return .string("auto")
        case .required: return .string("required")
        case .textOnly: return .string("none")
        case .specific(let name):
            return .dictionary(["type": .string("function"),
                                "function": .dictionary(["name": .string(name)])])
        }
    }

    /// Anthropic's wire value: always an object, and its "force some tool" is spelled `any`.
    var anthropicWireValue: AnyCodable {
        switch self {
        case .auto: return .dictionary(["type": .string("auto")])
        case .required: return .dictionary(["type": .string("any")])
        case .textOnly: return .dictionary(["type": .string("none")])
        case .specific(let name):
            return .dictionary(["type": .string("tool"), "name": .string(name)])
        }
    }

    /// The shape THIS provider would actually emit, or `nil` where the field does not exist.
    ///
    /// A probe forcing a raw `tool_choice` must send the shape the provider itself would, or it
    /// measures nothing useful: an OpenAI bare string sent to Anthropic is rejected for being the
    /// wrong SHAPE, and the rejection names `tool_choice`, so it would be recorded as "this model
    /// does not support forcing tools" — flatly wrong for Claude. Gemini has no `tool_choice` at
    /// all (it uses `toolConfig`), so there is nothing to force and `nil` says so.
    func wireValue(for apiType: ProviderAPIType) -> AnyCodable? {
        switch apiType {
        case .anthropic: return anthropicWireValue
        case .gemini: return nil
        default: return openAIWireValue
        }
    }
}
