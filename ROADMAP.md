# SwiftLLMKit — Roadmap

## Planned

### Model metadata: what each vendor actually publishes, and the shape that holds it (agreed design, 2026-07-17)

Investigation triggered by "why is our vision data wrong?". The answer turned out not to be bad data but a wrong premise: **we ask LiteLLM for facts the vendors already publish, and hand-author the rest until it rots.** Four live bugs came out of it (below). This section records the evidence and the agreed shape so the reasoning survives.

**What each provider's `/models` actually returns** (measured, not assumed — logs in `$TMPDIR`):

- **Anthropic** — the richest by far, and we discard nearly all of it. Per model: `max_input_tokens` (1,000,000 / 200,000), `max_tokens`, and a nested `capabilities` block carrying `image_input`, `pdf_input`, `thinking.{supported, types.{adaptive, enabled}}`, `effort.{low,medium,high,max,xhigh}` (each `{supported: Bool}`), `structured_outputs`, `code_execution`, `batch`, `citations`, `context_management`. **`decodeAnthropicModels` keeps five scalars and throws the whole `capabilities` block away** — including `image_input` and `pdf_input`, the only two capability flags that drive behavior in AgentSmith.
- **OpenAI** — nothing. Four keys, on all ~129 entries: `id`, `object`, `created`, `owned_by`. No limits, no capabilities, no reasoning. Everything we know about OpenAI models is hand-authored or inferred.
- **Gemini** — flat but useful, also discarded: `inputTokenLimit`, `outputTokenLimit`, **`thinking: Bool`**, `temperature`, `maxTemperature`, `topP`, `topK`, `supportedGenerationMethods` (which answers "is it chat?" far better than the `mode`/`supported_endpoints` guessing).
- **Ollama** — `capabilities` array; already decoded (`contains("tools") → toolUse`). The one decoder that reads provider capabilities at all.

**The organising principle: classify by _who knows the fact_**, not by "capability vs behavior". Of the 8 `BehaviorFlags`, only 2 are genuinely behavior:

| Flag | Who knows it |
|---|---|
| `requiresAdaptiveThinking` | **Anthropic publishes it** (`thinking.types.enabled.supported == false`) |
| `supportsReasoningEffort` | **Anthropic publishes it** (`effort.supported`, and which levels) |
| `mustNeverSendTemperatureParam`, `useMaxCompletionTokens`, `supportsDeveloperRole`, `replayReasoningContent` | model facts **nobody** publishes — hand-author or probe |
| `glmTemplateSalvage`, `disableParallelToolCalls` | genuinely **our** workaround/policy |

Rule: **decode what the vendor publishes; hand-author only what nobody does; probe what we can get neither way.** All four bugs below are "a vendor-published fact we chose to hand-type instead."

**Agreed shape.** `ModelInfo` is authoritative (what the model *is*); `ModelConfiguration` is what we *send*; resolution is `config.x ?? modelInfo.x`. The optionality **inverts** from today, where the guess is mandatory and the fact is optional:

```
ModelInfo           maxContextTokens: Int      (renamed from maxInputTokens — our vocabulary, not the vendors')
                    maxOutputTokens:  Int?
                    capabilities:     ModelCapabilities   // Bool? so "unknown" is sayable
                    validEffortLevels: [String]            // rank-ordered; [] == no effort knob
                    isEffortConfigurationByNamedLevelSupported: Bool  { !validEffortLevels.isEmpty }
                    validReasoningTokenBudgetRange: Range<Int>?       // half-open: budget < max_tokens
                    isReasoningTokenBudgetSupported: Bool { range != nil }
                    providerData: [String: AnyCodable]     // everything the vendor said that we don't model

ModelConfiguration  maxContextTokens: Int?     (was Int = 128_000)   nil = use the model's
                    maxOutputTokens:  Int?
                    reasoningEffort:  String?  (was thinkingEffort)  nil = the vendor's default
                    reasoningTokenBudget: Int? (was thinkingBudget)
```

`validEffortLevels`, **not** `validReasoningEffortLevels`: Anthropic's own name is `effort`, and their docs are explicit that it *"doesn't require thinking to be enabled"* and governs text and tool calls. "Reasoning" is right for OpenAI and wrong for Anthropic; the general name holds both. `requiresAdaptiveThinking` keeps its name — "adaptive thinking" is Anthropic's term of art and the literal wire value — but should stop being hand-authored: it is derivable as `capabilities.reasoning == true && !isReasoningTokenBudgetSupported`.

**Effort rank table** — a constant, derived from the name at sort time, never persisted, so renumbering is free. Spaced by 100 to leave room:

```
none -100 · minimal 0 · low 100 · medium 200 · high 300 · xhigh 400 · max 500
```

Shared across vendors — both order `xhigh < max`. Only *membership* is per-model (`validEffortLevels`).

**Established about effort** (from Anthropic's docs + the payload, after an inference of ours was proven wrong):

- Order is `low < medium < high < xhigh < max`. An earlier inference from the payload's nesting concluded `max < xhigh` and was **wrong**: support is *not* nested. Sonnet 4.6 and Opus 4.6 have `max` but **not** `xhigh`, because `xhigh` is a newer, specialised level ("long-horizon work… tasks over 30 minutes"), not one notch below max. Availability tracks recency and purpose, not depth.
- Effort is **not a thinking knob and not a budget** — *"Effort is a behavioral signal, not a strict token budget"*; it *"affects all tokens… including tool calls"* and works with thinking off. Lower effort makes the model **make fewer tool calls** — a behavioral change to an agent, not a cost dial.
- Effort **cannot be turned off** on Anthropic: no `none`, floor is `low`, and omitting it *is* `high`. OpenAI *does* have `none`, where it doubles as the reasoning off-switch. So enablement and levels must stay separate fields — `none` can't be normalised into a shared scale.
- `nil` is not redundant with `"high"`: it means "the vendor's default", which is `high` for Anthropic but **`medium` for gpt-5.5**. UI should render `nil` as "Default", not "High".

**What deliberately cannot be decoded.** Fable 5, Sonnet 5, Opus 4.8 and Opus 4.7 report **byte-identical** signatures (`adaptive: true, enabled: false`) and behave three different ways: Fable 5 is always on and *rejects* `thinking: {type: "disabled"}`; Sonnet 5 is on by default and accepts it; **Opus 4.8 is OFF unless you explicitly send `thinking: {type: "adaptive"}`**. Default state and disable-ability exist only in prose. Hand-author from docs, or probe — "does `disabled` 400?" is exactly a probe question.

**Open decisions:**

1. `maxContextTokens: Int` non-optional on `ModelInfo` means inventing a number when nobody reports one (OpenAI reports nothing). Once it's a plain `Int`, "Anthropic said 1,000,000" is indistinguishable from "we assumed 128,000" — the confusion `Bool?` exists to kill.
2. **Migration.** Every stored config has an explicit `"maxContextTokens": 128000`; flipping to `Int?` decodes those as `.some(128000)` = "deliberate", so the fix reaches nobody. It's user data — unlike the model catalog, it can't just be discarded. The same trap applies to `Bool?` capabilities and the catalog cache (which *can* be discarded: `refreshIfNeeded` already rebuilds when `models.isEmpty`, so a schema-version bump self-heals).
3. **Providers cannot see `ModelInfo`** — they get `(ModelConfiguration, ModelProvider, BehaviorFlags)` only, which is also the property the capability probe relies on for its can't-cheat guarantee. Deriving `requiresAdaptiveThinking` therefore means `makeProvider` populating flags from the catalog rather than from JSON: same signature, no probe impact.

### Bugs found by the metadata investigation (2026-07-17)

**Decode-side progress (0.0.60):** the Anthropic decoder now reads the whole capabilities block — vision/pdf/reasoning/code-execution/structured-outputs onto `capabilities`, per-model effort levels onto the new `ModelInfo.validEffortLevels`, and `requiresAdaptiveThinking` + `mustNeverSendTemperatureParam` **derived** from `thinking.types.enabled == false` rather than hand-listed (bundled entries remain as cold-start gap-fill). Verified end-to-end: sonnet-4-6 shows `[low, medium, high, max]` in the live catalog. A correction to the survey above: the Gemini decoder was already keeping `inputTokenLimit`/`outputTokenLimit`/`thinking` — the earlier claim that it discarded them was wrong; Anthropic was the discard case.

1. **~~`claude-fable-5` rejected every request~~ ✅ Fixed in 0.0.56.** `api.anthropic.com` answers `temperature` with `400 "deprecated for this model"`; no Anthropic model carried `mustNeverSendTemperatureParam`; `ModelConfiguration` defaults temperature to 0.7 and `AnthropicProvider` sends it unless flagged. The model was unusable. Found because the capability probe recorded "claude-fable-5 cannot call tools" — its own malformed request read as the model's answer.
2. **~~`claude-sonnet-5` and `claude-fable-5` are missing `requiresAdaptiveThinking`~~ ✅ Fixed in 0.0.59 — and it was bigger than a missing flag.** The capability probe found sonnet-5 *also* rejecting `temperature` with a 400, exactly as fable-5 did. A targeted probe sweep proved the pattern: Anthropic's adaptive-ONLY models (`thinking.types.enabled.supported == false`) BOTH require adaptive thinking AND reject temperature; dual-mode models (`enabled == true`) accept it. opus-4-8 rejects; sonnet-4-6 and opus-4-5 accept. One predictor, and a fact the payload publishes. So `{fable-5, sonnet-5, opus-4-8, opus-4-7}` now carry both flags. These should eventually be **derived** from the payload's `enabled` flag rather than hand-listed — that's the standing task; the flags are the interim fix. (OpenRouter variants get `requiresAdaptiveThinking` but not the temperature flag: that was proven only against `api.anthropic.com` directly, and whether OpenRouter forwards or strips temperature is untested.)
3. **`ModelConfiguration.maxContextTokens` is never wired from the model's real limit.** See the AgentSmith roadmap — it costs ~8× context on live configs.
4. **`GeminiProvider` implements no thinking at all.** Gemini publishes `thinking: true` in `/models`, and setting `thinkingBudget` on a Gemini config is silently dropped. `GeminiProvider` also takes no `behaviorFlags` — alone among the providers — so it cannot honor `mustNeverSendTemperatureParam` or any other knob.
5. **~~The effort allowlist is a vendor-blind union missing `none`~~ ✅ Partially fixed in 0.0.60.** `ModelInfo.validEffortLevels` now carries each model's own list (decoded from Anthropic's payload, rank-ordered), and validation checks a chosen effort against it when known — which catches `minimal` on Claude and knows sonnet-4-6 takes `max` but not `xhigh`. The union allowlist still runs first as a spell-check and still lacks `none`; retire it once OpenAI models also carry per-model lists (hand-authored table, since OpenAI publishes nothing).
6. **`LLMRequestLogger.logResponse` rewrites the evidence.** It re-serialises with `.sortedKeys`, so a logged response no longer shows what the server actually sent — key order is destroyed before anyone reads it. Deterministic serialisation is correct for *request* bodies (there's a test pinning it) and wrong for responses.

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

### ~~LiteLLM prefix coverage~~ ✅ Completed (0.0.47–0.0.52)
The premise was wrong, not just the coverage. `ProviderAPIType.liteLLMPrefix` guessed a **key prefix** per apiType, but a LiteLLM key's first segment is frequently not a provider at all: `1024-x-1024/dall-e-2` is `openai`, `high/1536-x-1024/gpt-image-1.5` is a quality tier, and `global.anthropic.claude-fable-5` is `bedrock_converse`, not Anthropic. Anthropic has **zero** `anthropic/`-prefixed keys — all 23 entries are bare — so the prefix step always missed and only a fallback was carrying it.

Every entry declares its owner in a `litellm_provider` **field**, and LiteLLM's own resolver treats the key as a search term that the field then vetoes. Matching now uses that field: `ModelProvider.liteLLMProviderName` is set explicitly on all 13 built-ins (`nil` for Hugging Face and LM Studio, which LiteLLM genuinely does not catalogue), and the index is keyed `litellm_provider → model name`, where the name drops only the entry's **own** provider prefix — which is what makes the junk keys self-reject. All fallbacks were removed rather than kept alongside, so a miss is now a reportable "LiteLLM does not catalogue this model for this provider" instead of a wrong answer from a fuzzy guess. `liteLLMPrefix` is deleted; its table was wrong anyway (`.anthropic` matched nothing, `.metaLlama` used a hyphen where upstream uses `meta_llama`, `.zAI`/`.alibabaCloud` returned `nil` where `zai`/`dashscope` exist).

Measured against the real 845-model catalog: Anthropic 10/10, OpenAI 123/129, 317 total. The residual misses are LiteLLM's own coverage ceiling, not a matching bug — surfaced in AgentSmith's Settings → Metadata tab rather than papered over.

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
