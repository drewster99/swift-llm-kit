import Foundation

/// A request for structured output — `response_format` on OpenAI-compatible endpoints.
///
/// Two modes, deliberately separate rather than one flag, because a model may accept one and
/// reject the other: DeepSeek documents `json_object` with `json_schema` unconfirmed. They are
/// carried by two independent capabilities (`structuredOutputJSONObject` and the pre-existing
/// `responseSchema`), and this type is what a caller sends once one of them is known true.
///
/// `responseSchema` is reused for the schema mode rather than adding a parallel case: it already
/// exists, is already populated by the Anthropic and OpenRouter decoders, and already means
/// exactly "the model can be held to a JSON schema". A second field for the same fact would be a
/// second source of truth.
public enum LLMResponseFormat: Sendable, Equatable {
    /// `{"type": "json_object"}` — syntactically valid JSON, shape unconstrained. The prompt must
    /// still describe the shape; the endpoint guarantees only that the result parses.
    case jsonObject

    /// `{"type": "json_schema", "json_schema": {...}}` — the response is held to the schema.
    ///
    /// - Parameters:
    ///   - name: Schema name. Required by the wire format; identifies the schema in errors.
    ///   - schema: The JSON Schema itself.
    ///   - strict: Whether the endpoint must refuse to deviate. Endpoints that don't support strict
    ///     mode ignore the key rather than failing, so it is not gated on its own capability.
    case jsonSchema(name: String, schema: [String: AnyCodable], strict: Bool = true)

    /// The wire shape, ONCE. `response_format` is always an object, so `[String: AnyCodable]` is
    /// the honest type; the two representations below are derived from it rather than written
    /// twice. They were, and a probe forcing one while production emitted the other would have
    /// been measuring a shape that never ships.
    public var wireValue: [String: AnyCodable] {
        switch self {
        case .jsonObject:
            return ["type": .string("json_object")]
        case .jsonSchema(let name, let schema, let strict):
            return [
                "type": .string("json_schema"),
                "json_schema": .dictionary([
                    "name": .string(name),
                    "strict": .bool(strict),
                    "schema": .dictionary(schema)
                ])
            ]
        }
    }

    /// Ready to place at `response_format` in a `[String: Any]` request body.
    public var openAIWireValue: [String: Any] { wireValue.mapValues(\.rawValue) }

    /// Ready to force through `extraJSONOverrides`, for probes that must bypass the capability
    /// gate keyed on the very capability they are establishing.
    public var forcedWireValue: AnyCodable { .dictionary(wireValue) }

    /// Whether this provider family has a `response_format` field at all.
    ///
    /// Anthropic and Gemini do not; forcing one there measures nothing and burns a paid call on a
    /// rejection that says nothing about structured-output support.
    public static func isSupportedWireField(for apiType: ProviderAPIType) -> Bool {
        switch apiType {
        case .anthropic, .gemini, .ollama: return false
        default: return true
        }
    }

    /// The capability that must be known-true before this may be sent.
    ///
    /// Sending an unsupported `response_format` is an HTTP 400, so emission fails CLOSED — the same
    /// rule as reasoning effort, for the same reason.
    public var requiredCapability: ModelCapability {
        switch self {
        case .jsonObject: return .structuredOutputJSONObject
        case .jsonSchema: return .responseSchema
        }
    }
}
