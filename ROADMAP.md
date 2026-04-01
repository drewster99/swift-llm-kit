# SwiftLLMKit — Roadmap

## Planned

### Deprecate or reconcile `prepareRequest(for:)` with provider implementations
`LLMKitManager.prepareRequest(for:)` builds a `PreparedRequest` with URL, auth headers, and base body parameters. The provider implementations (`AnthropicProvider`, `GeminiProvider`, etc.) build their own requests from scratch internally. This is duplicated logic with subtle differences (e.g. `prepareRequest()` doesn't override temperature for Anthropic thinking mode). Either deprecate `prepareRequest()` in favor of `makeProvider(for:)`, or have it delegate to the providers internally. Leaving both paths is a maintenance risk.

### ~~Standardize Anthropic endpoint path handling~~ ✅ Completed
Extracted `URL.ensureAnthropicV1()` and `URL.strippingAnthropicV1()` in `ProviderAPIType.swift`, replacing duplicated logic in `AnthropicProvider`, `SwiftLLMKit`, and `ModelFetchService`.

### Add test coverage
Minimal test coverage exists. Add tests for:
- Provider CRUD operations on `LLMKitManager`
- Configuration validation logic
- Persistence round-trips (save/load providers, configurations, model catalog)
- Request preparation for each provider type
- Provider `send()` with mocked URLSession responses
- `EndpointPreset` URL validity
- `ModelConfiguration` convenience accessors

### Add protocol abstraction for networking
`ModelFetchService` and `ModelMetadataService` use `URLSession.shared` directly, making unit testing difficult. Inject a protocol (or closure) for HTTP calls so tests can provide mock responses without hitting real APIs.

### Keychain thread safety
`KeychainService.save()` calls `SecItemUpdate` then falls back to `SecItemAdd`. Between these calls, another thread could insert the item, causing both to fail. Consider using a lock or serial queue for keychain operations.

### `PreparedRequest.baseBody` mutation safety
`PreparedRequest` is `@unchecked Sendable` with a `[String: Any]` dictionary marked as "effectively immutable" by convention. There's no enforcement — callers could mutate it from another isolation domain. Consider making it truly immutable (e.g. copy-on-create, or a sealed wrapper type).

### Verbose logging security warning
`LLMRequestLogger` writes full request/response JSON to disk, which can include API keys in headers/URLs and sensitive model responses. The log methods are public and can be enabled by any code. Consider:
- Filtering sensitive headers before writing
- Adding a prominent warning in the doc comment
- Requiring explicit opt-in (e.g. a configuration flag rather than a static property)

### Streaming support
The `LLMProvider` protocol is synchronous (`async throws -> LLMResponse`). `ModelConfiguration.streaming` is captured but never used by the providers. Add a streaming variant to the protocol (e.g. returning an `AsyncSequence` of partial responses) or document that streaming is not yet implemented.

### Input validation for model IDs and URLs
Provider URLs, model IDs, and API keys are not validated or sanitized. A model ID containing path separators (`/`) could create unexpected URL paths. Add basic validation when constructing requests.

### ~~Gemini tool call ID uniqueness~~ ✅ Completed
`GeminiProvider` now uses `UUID().uuidString` for tool call IDs, matching `OllamaProvider`'s approach. The function name is preserved in the `name` field for `functionResponse` matching.

### Surface server error details in model fetch failures
`ModelFetchError.httpError(statusCode:)` only reports the HTTP status code. The response body often contains specific error messages (e.g. `{"error":{"type":"authentication_error","message":"invalid x-api-key"}}`). Parse and surface these details so UI consumers can show actionable error messages.

### Shared message preprocessing utilities
System message extraction and consecutive same-role message merging are implemented independently in each provider. Consider extracting common patterns into shared utilities that providers can compose, reducing duplication while preserving provider-specific differences.

### LiteLLM prefix coverage
`ProviderAPIType.liteLLMPrefix` returns `nil` for HuggingFace, LM Studio, and z.ai, which means no pricing/capability enrichment from LiteLLM. Document this gap and investigate whether LiteLLM has entries for these providers under different prefixes.

### Typed error semantics for `LLMProvider`
The `LLMProvider.send()` method throws untyped errors. Callers cannot distinguish transient failures (network timeout, rate limit) from permanent ones (auth failure, model not found). Consider a typed error enum or `Result`-based return so callers can implement appropriate retry logic.

### Distinguish empty vs absent response content in `LLMResponse`
`LLMResponse.text` is `nil` both when the model produced no text and when parsing failed. Consider adding a status field or richer return type so consumers can tell "model chose not to respond with text" from "something went wrong during parsing."

### `ModelMetadataService` Sendable conformance for `UserDefaults`
`ModelMetadataService` is an actor that holds a `UserDefaults` instance, which is not `Sendable`. In practice it's safe because all access is within the actor, but the type system can't verify this. Consider wrapping it in a `@unchecked Sendable` newtype or using `nonisolated(unsafe)` with a comment, so the intent is explicit.

### Replace `[String: Any]` dictionary construction with Codable structs
All providers build API request bodies as `[String: Any]` dictionaries. This has no compile-time checking that keys or value types match the API schema. Consider defining Codable request/response structs per provider for type safety and easier maintenance.

### Unit tests for `OllamaProvider.normalizeMessages`
`OllamaProvider.normalizeMessages` is ~60 lines of complex state management handling role alternation, tool call merging, and edge cases. It works but is hard to verify by inspection. Add targeted unit tests covering the key scenarios (consecutive same-role messages, interleaved tool calls, empty messages).
