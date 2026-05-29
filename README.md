# SwiftLLMKit

A Swift package for driving LLM APIs from Apple platforms. One `LLMProvider`
protocol, a shared `LLMKitManager` for settings and model discovery, and
built-in drivers for the most common APIs — with API keys stored safely in
Keychain.

## Supported providers

- **Anthropic** (Claude)
- **OpenAI** (and OpenAI-compatible: DeepSeek, LM Studio, etc.)
- **Google Gemini**
- **xAI** (Grok)
- **z.ai**
- **Meta Llama**
- **Mistral**
- **Alibaba Cloud** (DashScope)
- **Hugging Face**
- **OpenRouter**
- **Ollama** (local or cloud)

## Features

- **One `LLMProvider` interface.** Every backend implements `send(messages:tools:) async throws -> LLMResponse`, so the rest of your app doesn't care which vendor is answering.
- **Tool calling.** Native `LLMToolDefinition` / `LLMToolCall` / tool-result messages across every provider that supports them.
- **Image input.** Multimodal messages carry base64-encoded images to providers that accept them.
- **Built-in provider presets.** Stable IDs (`BuiltInProviders.ID.openai`, `.anthropic`, …) so app code can branch on specific providers without string-matching hostnames.
- **Keychain-backed API keys.** `LLMKitManager` stores user-supplied keys in Keychain with a configurable service prefix; nothing lives in UserDefaults or plain files.
- **Observable settings.** `LLMKitManager` is an `@Observable @MainActor` class — providers, models, and configurations are SwiftUI-ready.
- **Model catalog refresh.** Fetches each provider's current model list, layered with bundled pricing/capability metadata and optional user overrides.
- **Prompt caching.** Built-in support for Anthropic's ephemeral prompt cache and for OpenAI/xAI cached-input reporting.
- **Per-request bodies you own.** `prepareRequest(for:)` returns a configured `URLRequest` + base body; your app appends `messages`/`tools` and sends it. Useful when you need to stream or handle the transport yourself.

## Requirements

- macOS 14, iOS 17, visionOS 1, or newer
- Swift 6.0 toolchain

## Installation

```swift
.package(url: "https://github.com/drewster99/swift-llm-kit", from: "0.1.0")
```

Add the product to your target:

```swift
.product(name: "SwiftLLMKit", package: "swift-llm-kit")
```

## Quick start

### Direct provider use

```swift
import SwiftLLMKit

let provider = AnthropicProvider(
    configuration: ModelConfiguration(
        id: UUID(),
        providerID: BuiltInProviders.ID.anthropic,
        model: "claude-sonnet-4-6",
        maxTokens: 4096,
        temperature: 1.0,
        useDefaultTemperature: false
    ),
    readAPIKey: { ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"] ?? "" }
)

let response = try await provider.send(
    messages: [LLMMessage(role: .user, content: .text("Hello!"))],
    tools: []
)
print(response.text ?? "")
```

### Managed settings + Keychain

```swift
import SwiftLLMKit

@MainActor
let kit = LLMKitManager(
    appIdentifier: Bundle.main.bundleIdentifier ?? "com.example.app",
    keychainServicePrefix: "com.example.SwiftLLMKit"
)
kit.load()

// First-time setup: seed a built-in provider and store its API key in Keychain.
kit.setBuiltInProviderAPIKey(id: BuiltInProviders.ID.openai, apiKey: "sk-...")

await kit.refreshIfNeeded()   // pulls each provider's model list
let configID = kit.configurations.first!.id

// Build an authenticated URLRequest for your own send loop, or:
let provider = try kit.makeProvider(for: configID)
let response = try await provider.send(messages: [.init(role: .user, content: .text("Hi"))], tools: [])
```

## Changelog

### 0.0.30 — `LLMToolChoice` per-call control

Adds per-call control over how the model selects from the provided `tools`.
New `LLMToolChoice` enum is the unified abstraction; each provider translates
to its own wire format.

#### New

- **`LLMToolChoice`** enum: `.auto`, `.required`, `.textOnly`, `.specific(name:)`.
  Note: the "must not call any tool" case is named `.textOnly` (not `.none`)
  because `LLMToolChoice.none` would silently collide with
  `Optional<LLMToolChoice>.none` — Swift's overload resolution picks the
  Optional case at the call site, producing the OPPOSITE behavior.
- **`LLMProvider.send`** gains an optional `toolChoice:` parameter (default
  nil → omit field → provider default applies). Two backward-compatible
  protocol extension overloads preserve the original 2- and 3-argument call
  shapes; existing callers compile unchanged.

#### Wire translation per provider

| Choice | Anthropic | OpenAI | Gemini | Ollama |
|---|---|---|---|---|
| `.auto` | `{type: "auto"}` | `"auto"` | `mode: "AUTO"` | `"auto"` |
| `.required` | `{type: "any"}` | `"required"` | `mode: "ANY"` | `"required"` |
| `.textOnly` | `{type: "none"}` | `"none"` | `mode: "NONE"` | `"none"` |
| `.specific("foo")` | `{type: "tool", name: "foo"}` | `{type: "function", function: {name: "foo"}}` | `mode: "ANY", allowedFunctionNames: ["foo"]` | `{type: "function", function: {name: "foo"}}` |

`toolChoice` is meaningful only when `tools` is non-empty — providers gate
emission on `!tools.isEmpty`, so setting it without tools is a no-op.

23 new tests in `V0_0_30_ToolChoiceTests` covering every (provider × choice)
combination plus the nil + empty-tools cases. 194 tests total (was 171).

### 0.0.29 — fix prepareRequest stale + duplicateConfiguration field drop

Two real bugs surfaced by a comprehensive code review:

- **`LLMKitManager.prepareRequest` was stale relative to 0.0.27/0.0.28** —
  emitted legacy `thinking: {type: "enabled", budget_tokens: N}` unconditionally,
  never emitted `output_config.effort` or `reasoning_effort`. Any caller using
  this lower-level prep API against Opus 4.7/4.8 got guaranteed HTTP 400.
  Fixed to mirror the dispatch in AnthropicProvider / OpenAICompatibleProvider.

- **`LLMKitManager.duplicateConfiguration` silently dropped 3 fields** —
  `thinkingEffort`, `extendedCacheTTL`, `extraJSONOverrides`. Was using a
  rebuild-from-scratch named-init pattern that's blind to field additions.
  Switched to `var newConfig = original; newConfig.id = UUID(); ...` mutate-
  from-existing pattern — every future field is preserved automatically.

Plus two important polish items:

- **`thinkingEffort` validation** in `validateConfigurations`: rejects unknown
  enum values; rejects when the routed provider doesn't consume the field.
- **OpenRouter `providerEntries`** for every model with a per-apiType
  behavior flag — `builtin.openrouter/anthropic/claude-opus-4-{7,8}` get
  `requiresAdaptiveThinking`, `builtin.openrouter/openai/{o-series, gpt-5*}`
  get `supportsReasoningEffort`. Users routing via OpenRouter get the right
  wire shape without manual override.

### 0.0.28 — OpenAI `reasoning_effort` emission

Extends 0.0.27's `ModelConfiguration.thinkingEffort` field to flow through the
OpenAI-compatible wire path. Set once on the config; swift-llm-kit emits the
right wire shape per provider — `output_config.effort` for Anthropic,
`reasoning_effort` for OpenAI.

#### New

- **`BehaviorFlags.supportsReasoningEffort: Bool`** (default false). Gates
  emission of `reasoning_effort` on OpenAI-compatible providers. Non-reasoning
  models (GPT-4o, GPT-3.5-turbo, DeepSeek-V*, etc.) reject the field with
  HTTP 400, so we gate per-model rather than emit unconditionally.
- **Bundled metadata entries** for OpenAI reasoning models with
  `supportsReasoningEffort: true`: `o1`, `o1-preview`, `o1-mini`, `o3`,
  `o3-mini`, `o4-mini`, `gpt-5`, `gpt-5-mini`, `gpt-5-pro`. Existing configs
  pointing at these model IDs that set `thinkingEffort` will auto-emit
  `reasoning_effort` on the wire.

#### Behavior

- `configuration.thinkingEffort = "high"` on `o3-mini` → emits
  `reasoning_effort: "high"` at the top level of the chat-completions request.
- Same field on `gpt-4o` → not emitted (no flag → silent skip → request goes
  through without the field, which gpt-4o requires anyway).
- Custom-tuned reasoning models can opt in via
  `BehaviorFlagsOverride.supportsReasoningEffort = true`.

#### Cross-provider effort enum compatibility

OpenAI and Anthropic share `low`, `medium`, `high`. OpenAI adds `minimal`
(GPT-5 only); Anthropic adds `xhigh` (Opus 4.7+) and `max`. swift-llm-kit
doesn't validate effort strings — pass-through means a value not supported
by the routed model produces a clear API 400 rather than a silent drop.

Use OpenAI's enum values if your config might route to OpenAI; use
Anthropic's if it routes only to Anthropic.

#### Not yet covered

- **Gemini effort mapping**: Gemini doesn't have an effort enum — its
  `thinkingConfig.thinkingBudget` is a token count. Arbitrary mapping
  (`low` → 1024, `high` → 8192, etc.) would be confusing. Deferred until
  someone needs it; will likely surface as a separate
  `ModelConfiguration.thinkingBudgetGemini` field or a per-effort lookup
  table.
- The `display: "summarized" | "omitted"` field on Anthropic adaptive
  thinking is not yet typed; use `extraJSONOverrides` for now (see 0.0.27
  notes).

9 new tests in `V0_0_28_Tests`. 170 total (was 161).

### 0.0.27 — Anthropic adaptive thinking + output_config.effort

Adds support for Anthropic's new thinking API. Claude Opus 4.7 and 4.8 require
`thinking: {type: "adaptive"}` and reject the legacy `thinking: {type: "enabled",
budget_tokens: N}` format with HTTP 400. swift-llm-kit now detects required-
adaptive models via `BehaviorFlags.requiresAdaptiveThinking` (auto-set in the
bundled metadata for known model IDs) and emits the correct wire shape. Plus
typed `thinkingEffort` field for the new `output_config.effort` knob.

#### New

- **`BehaviorFlags.requiresAdaptiveThinking: Bool`** (default false). When
  true, `AnthropicProvider` emits `thinking: {type: "adaptive"}` instead of
  the legacy manual format. The `thinkingBudget` config field acts as a
  boolean signal on these models: > 0 means "thinking on" (depth decided by
  the model and steered via effort), 0 / nil means "thinking off."
- **`ModelConfiguration.thinkingEffort: String?`** (Optional). Valid values:
  `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"` (xhigh only on Opus 4.7,
  4.8). When non-nil, `AnthropicProvider` emits a top-level
  `output_config: {effort: <value>}` field independent of the thinking mode.
  Backward-compatible Codable (legacy configs without the key decode cleanly
  with `thinkingEffort = nil`).
- **Bundled metadata entries** for `claude-opus-4-7` and `claude-opus-4-8`
  with `requiresAdaptiveThinking: true`. Existing configs pointing at these
  modelIDs auto-flip to adaptive on first request — no user action needed.
  Configs with a custom Opus 4.7/4.8 model ID can opt in via
  `BehaviorFlagsOverride.requiresAdaptiveThinking = true`.
- `AnthropicProvider.init` now takes an optional `behaviorFlags: BehaviorFlags`
  parameter (default `BehaviorFlags()`). `LLMKitManager.makeProvider(for:)`
  passes the resolved flags through automatically.

#### Migration matrix

| Model | Existing behavior (0.0.26) | New behavior (0.0.27) |
|---|---|---|
| **Opus 4.5 and earlier** | Manual `{type: "enabled", budget_tokens: N}` | Unchanged |
| **Opus 4.6, Sonnet 4.6** | Manual works (deprecated by Anthropic) | Unchanged unless you set the flag yourself |
| **Opus 4.7, Opus 4.8** | HTTP 400 on every thinking request | Auto-adaptive via registry; works |
| Custom modelIDs | n/a | Set `requiresAdaptiveThinking` via override |

#### Notes

- The `output_config.effort` field is emitted unconditionally when
  `thinkingEffort` is set — older models that don't support it will return
  HTTP 400. Set only on models that support it (Opus 4.5+, Sonnet 4.6+).
- Adaptive thinking does NOT have the manual `max_tokens > budget_tokens + 1`
  constraint — `AnthropicProvider` skips that floor on adaptive paths.
- `temperature = 1.0` is still forced when thinking is on (both manual and
  adaptive paths), preserving the existing Anthropic-thinking convention.
- `display: "summarized" | "omitted"` is not yet exposed as a typed field.
  Opus 4.7/4.8 default to `"omitted"` — signatures still flow for
  multi-turn continuity (thinking continuity unchanged), but visible
  reasoning text in `LLMResponse.reasoning` will be empty on these models
  unless you set `display: "summarized"` via `extraJSONOverrides`.

15 new tests in `V0_0_27_Tests`. 160 total (was 145).

### 0.0.26 — Gemini parts redesign + Anthropic empty-content fix

Resolves both 0.0.25 known limitations.

#### Fixed

- **Gemini per-part `thoughtSignature` index keying was shape-fragile.** The
  0.0.24 design stored signatures as `[String: String]` keyed by part-array
  index. When `.assistant(from: response)` collapsed a multi-part response
  into a single `.text` / `.toolCalls` / `.mixed` content (e.g. `[thought,
  fcA, fcB]` collapses to `.toolCalls([fcA, fcB])`), the replay re-attached
  signatures by emitted-part index — misaligning them onto the wrong parts
  and dropping the last sig entirely.

  Replaced with `ProviderContinuation.geminiResponseParts: [GeminiResponsePart]?`
  — the full parts structure captured verbatim at parse time. The Gemini
  encoder emits these parts verbatim on replay, bypassing the content-shape
  collapse entirely. Faithful 1:1 round-trip regardless of how the factory
  represents the content.

  `geminiThoughtSignatures` is kept for backward-compatible decoding of
  0.0.24/0.0.25 saved conversations (and is auto-derived from the new parts
  on parse so code reading the old field still sees signatures) but is now
  marked `@available(*, deprecated)`. Migrate to `geminiResponseParts` for
  new code paths.

- **AnthropicProvider rejected `content: ""` with HTTP 400.**
  `.assistant(from: LLMResponse(text: nil, toolCalls: []))` produces
  `.text("")` (the "model returned nothing" degenerate shape — rare but
  reachable). AnthropicProvider was emitting either `"content": ""` (plain-
  text path) or `[{type:"text",text:""}]` (content-array path), both
  rejected by the API.

  Fix is localized to AnthropicProvider's encoder:
  - Plain-text path: empty text becomes a single space `" "` (the
    documented Anthropic workaround for "empty assistant turn").
  - Content-array path (with thinking blocks or images): empty
    `{type:"text",text:""}` entries are silently skipped — the other
    blocks alone satisfy the "non-empty content" requirement.
  - `.mixed` with empty text + tool calls: the empty text block is skipped,
    leaving only the tool_use blocks.

  `LLMMessage.content` is left honest — the factory still produces
  `.text("")` for an empty response, accurately reflecting what the model
  returned. The wire-level workaround stays in the one provider that needs
  it.

#### Tests

11 new tests in `V0_0_26_Tests`:
- Gemini parses full parts array into the new field (including text +
  functionCall + thoughtSignature + thought-flag per part)
- Gemini still auto-populates the legacy `geminiThoughtSignatures` field
  for backward compat
- Gemini encoder uses saved parts verbatim when available — including the
  worst-case shape `[empty-thought-text, fcA, fcB]` collapsed by the
  factory into `.toolCalls([fcA, fcB])`, where the wire correctly emits
  3 parts with sigs intact
- Gemini parse → replay round-trip is byte-faithful
- Gemini encoder falls back to legacy position-keyed sigs when only the
  legacy field is present (covers 0.0.24/0.0.25 saved conversations)
- ProviderContinuation Codable round-trips parts and falls back cleanly on
  legacy-only JSON
- Anthropic substitutes space for empty plain text content
- Anthropic skips empty text blocks in content-array and `.mixed` paths
- Anthropic still emits non-empty text unchanged

144 tests total (was 133).

#### Migration

Existing consumers reading `continuation.geminiThoughtSignatures` continue
to work without code changes — the field is deprecated but still populated.
The deprecation warning points at the new field. Migration is mechanical:

```swift
// BEFORE — still works, emits deprecation warning
if let sigs = response.continuation?.geminiThoughtSignatures {
    // sigs["0"] etc.
}

// AFTER — structurally faithful
if let parts = response.continuation?.geminiResponseParts {
    for part in parts {
        // part.text, part.functionCall, part.thoughtSignature, part.thought
    }
}
```

### 0.0.25 — Ollama developer-role fold

Fixes a 0.0.24 regression: `OllamaProvider` was passing
`LLMMessage.Role.developer` straight through to the wire as
`"role": "developer"`, which Ollama's chat templates (e.g. gemma3) reject
with HTTP 400. Now developer turns are folded into the consolidated system
message at `extractSystemMessages`, matching the documented contract that
non-OpenAI backends translate `.developer` to `system`. AnthropicProvider,
GeminiProvider, and OpenAICompatibleProvider already did the right thing
in 0.0.24 — only the Ollama path was missed. New regression test
`ollama_developerFoldsIntoSystem` locks the behavior.

Known limitations (not fixed; flagged for a future release):

- **Gemini per-part thoughtSignature index keying is shape-fragile.**
  Signatures are keyed by the response's part-array index. When
  `LLMMessage.assistant(from: response)` collapses a parsed response into
  `.text` / `.toolCalls` / `.mixed`, the replay's part-array indices may
  not align 1:1 with the original (e.g. a response with `[thinkingText,
  fc0, fc1]` collapses to `.toolCalls([fc0, fc1])` and the sig at original
  index 0 attaches to fc0). The minimum-disruption use case (single-shape
  response replay) works; pathological multi-thinking-part shapes need a
  redesign keyed by content type rather than position.
- **`LLMMessage.assistant(from: response)` with `text=nil` and empty
  `toolCalls` produces `.text("")`.** AnthropicProvider then emits
  `{"role":"assistant","content":""}` which the API rejects. Callers
  should special-case "empty model response" before constructing the
  message; no caller in the current ecosystem hits this path, but the
  factory's behavior is unsafe for blind use.

### 0.0.24 — Thinking continuity + role-tagged factories + developer role

This release plumbs the provider-specific "thinking continuation" data that
Anthropic and Gemini 2.5 require for multi-turn / tool-use conversations, adds
misuse-resistant message factories, introduces the `developer` role, and
re-applies the 0.0.22 `functionResponse.name` fix on top of the new
signature-replay path so it no longer regresses Gemini tool use.

#### New

- **`ProviderContinuation`** — new value type carrying provider-specific
  thinking-continuity blobs. Captured automatically on `LLMResponse` and
  replayed automatically when you build the next assistant turn via the new
  `.assistant(from:)` factory. Fields:
  - `anthropicThinkingBlocks: [AnthropicThinkingBlock]?` — captured from
    Anthropic `thinking` content blocks (text + opaque signature). Replayed
    verbatim at the start of the assistant turn — required for thinking
    continuity with `thinkingBudget` > 0.
  - `geminiThoughtSignatures: [String: String]?` — captured per response part
    (keyed by stringified part index). Re-attached to the matching outgoing
    parts. Required for Gemini 2.5 (thinking on by default in Pro) — without
    these the model silently drops tool results in multi-turn flows.
- **`LLMMessage.continuation: ProviderContinuation?`** — new field, optional,
  backward-compatible Codable.
- **`LLMResponse.continuation: ProviderContinuation?`** — new field carrying
  per-provider continuation data the model returned.
- **`LLMMessage.Role.developer`** — new role case for OpenAI's `developer`
  role (o-series / GPT-5). Providers without native support fold it into
  their system field; OpenAI-compatible providers emit it on the wire only
  when `BehaviorFlags.supportsDeveloperRole = true`.
- **`BehaviorFlags.supportsDeveloperRole: Bool`** — new flag. Default false
  (downgrade developer→system). Set true for OpenAI o-series / GPT-5 models.
- **Static factories on `LLMMessage`** — the new misuse-resistant API:
  - `.user(_:)` / `.user(_:images:)`
  - `.system(_:)`
  - `.developer(_:)`
  - `.assistant(from: LLMResponse)` ← **the load-bearing one** —
    automatically carries `reasoning` AND `continuation`. Use this whenever
    you record an actual model response in conversation history.
  - `.toolResult(_:callID:)`
  - `.assistant(text:reasoning:)` and `.assistant(toolCalls:text:reasoning:)`
    — synthetic, for tests / saved-conversation reload only. Marked
    deprecated to nudge callers toward `.assistant(from:)`.

#### Fixed

- **Anthropic thinking continuity in multi-turn / tool-use flows.** Prior
  releases parsed thinking blocks into `LLMResponse.reasoning` (display only)
  but discarded the signatures. With `thinkingBudget` > 0, Anthropic requires
  the signed blocks to be replayed unchanged or the model loses thinking
  state across turns. `AnthropicProvider` now extracts them into
  `ProviderContinuation` on parse and prepends them to assistant content
  blocks on encode.
- **Gemini 2.5 thinking continuity in multi-turn / tool-use flows.** Gemini
  returns `thoughtSignature` on every response part when thinking is enabled
  (default on Pro). `GeminiProvider` now captures them per part index and
  re-attaches them to the matching outgoing parts.
- **Gemini `functionResponse.name` correctness (re-do of 0.0.22).** With
  thoughtSignature replay now in place, Gemini's strict validation path no
  longer drops tool results when the function name is correct.
  `GeminiProvider` builds a `[toolCallID → functionName]` lookup from prior
  assistant turns and uses it to populate `functionResponse.name`. Parallel
  tool calls now pair correctly (previously they relied on positional
  fallback, which silently mismatched under parallelism).

#### Deprecated (not removed)

- `LLMMessage.init(role:content:images:reasoning:continuation:)` — generic
  init. Prefer the role-specific factories above. The generic init is still
  callable but emits a deprecation warning whose message points at the
  misuse: manual construction from an `LLMResponse` silently drops
  `reasoning` and `continuation`, breaking multi-turn thinking and tool-use.
- `LLMMessage.init(role:text:images:reasoning:)` — text convenience init.
  Same reason.
- `LLMMessage.assistant(text:reasoning:)` and
  `LLMMessage.assistant(toolCalls:text:reasoning:)` — synthetic factories.
  Useful for tests and saved-conversation reload; **never** for real
  responses (they don't carry continuation).

#### Migration guide

The deprecations are advisory — existing code keeps working. To get the
benefits (thinking continuity, no more silent multi-turn breakage), migrate
the **load-bearing site** in your agent loop:

```swift
// BEFORE — drops continuation; multi-turn thinking breaks
let response = try await provider.send(messages: history, tools: tools)
history.append(LLMMessage(
    role: .assistant,
    content: .text(response.text ?? "")
))

// AFTER — preserves reasoning + continuation automatically
let response = try await provider.send(messages: history, tools: tools)
history.append(.assistant(from: response))
```

If you persist conversations, your storage type's Codable shape now
automatically round-trips `continuation` (new key, optional, backward
compatible — legacy stored conversations decode cleanly with continuation
= nil).

### 0.0.23 — Revert 0.0.22 (Gemini functionResponse.name fix)

- 0.0.22 changed `GeminiProvider`'s `functionResponse.name` to use the
  actual function name (correct per Gemini's spec) instead of the
  toolCallID. Empirical testing showed this **regressed Gemini 2.5 Pro
  tool-using conversations** — Gemini ignored tool results entirely.
- Root cause: Gemini 2.5 Pro has thinking enabled by default and returns
  `thoughtSignature` blobs that must be echoed back on subsequent turns
  for thinking-continuity. We don't preserve them. Pre-0.0.22, the
  wrong-name behavior triggered Gemini's positional-pairing fallback,
  which bypassed thinking-continuity validation. The "correct" name in
  0.0.22 triggered stricter validation that depends on the missing
  signature — Gemini treated tool results as unconnected to its prior
  thought and ignored them.
- The proper fix is to plumb `thoughtSignature` through `LLMResponse` /
  `LLMMessage` and echo it back on subsequent assistant turns. Once that
  lands, the name fix can be re-applied safely.
- Until then, the pre-0.0.22 behavior (toolCallID in `functionResponse.name`)
  stays — Gemini's positional pairing handles serial tool use correctly.
  Parallel tool calls remain a known latent issue.

### 0.0.21 — **BREAKING for `useDefaultTemperature` consumers**

- `ModelConfiguration.temperature` is now `Double?` (was `Double`). **`nil`
  means "omit the field entirely"** — providers send no `temperature` key
  on the wire, letting the model use its default. Required for models that
  reject explicit `temperature` (Claude Opus 4.7, GPT-5).
- `ModelConfiguration.useDefaultTemperature` (the previous flag-based opt-out)
  is **removed as a stored property**. A deprecated computed bridge maps
  reads to `temperature == nil`; on write, `true` clears temperature to nil
  and `false` is a no-op (set `temperature` to a value to assert a specific one).
- **Migration:** GUI consumers that previously did
  `config.useDefaultTemperature = true` should switch to
  `config.temperature = nil`. Legacy persisted JSON with
  `useDefaultTemperature: true` migrates correctly — the decoder sets
  `temperature = nil` regardless of any stored temperature value.
- Removes the silent-leak risk: previously, ConfiguredHydra (project-hydra
  CLI) passed `temperature: 1.0` as a placeholder when nil was meant. The
  flag-based gate hid it, but a tightened per-provider temperature range
  in a future swift-llm-kit could have rejected the placeholder. No more
  placeholder.

### 0.0.20
- **Deep-merge for `ModelConfiguration.extraJSONOverrides`.** Previously a flat
  top-level merge: setting `extraJSONOverrides["generationConfig"]` on Gemini
  (or `["system"]` / `["tools"]` on Anthropic) wiped sibling sub-keys the
  provider had built (defaults under `generationConfig`, the `cache_control`
  breakpoint on the system block, etc.). Now dict-valued overrides recurse so
  only the keys you specify replace; siblings survive. Arrays and scalars still
  replace outright. See `mergeJSONOverrides` in `Providers/JSONMerge.swift`.

### 0.0.19 — **BREAKING for Anthropic consumers**
- `TokenUsage.inputTokens` is now normalized across providers to represent the
  **full prompt input** (uncached + cache_read + cache_write). Previously
  Anthropic surfaced the wire `input_tokens` field, which is only the
  uncached portion. After 0.0.19 you can compute `cacheReadTokens / inputTokens`
  as a uniform hit-rate across all providers.
- **Migration:** if you display `inputTokens` for Anthropic responses, the
  value will be larger when caching is active. That's the correct total.
  If you summed `inputTokens + cacheReadTokens` to estimate billable input,
  you now double-count for Anthropic — switch to `inputTokens` alone.
- The full raw provider usage object is still preserved as a JSON string on
  `TokenUsage.rawUsage` if you need the old per-field semantics.
- Also clarified the stale comment about top-level `cache_control` —
  Anthropic supports it as a stable "automatic caching" feature.

### 0.0.18
- Fix Keychain regression introduced in 0.0.16: `apiKey(...)` migration
  path called the public `save()` which under the new fallback could write
  back to the legacy keychain, then the migration code deleted the legacy
  entry — losing the only copy. Migration now bypasses the fallback when
  trying DPK, so unentitled CLI binaries keep their keys persistent.

### 0.0.17
- `TokenUsage.reasoningTokens: Int` for OpenAI
  (`completion_tokens_details.reasoning_tokens`) and Gemini
  (`usageMetadata.thoughtsTokenCount`). Anthropic folds thinking into
  `output_tokens` and stays at 0. Codable backward-compat preserved.

### 0.0.16
- `KeychainService.save` falls back to the legacy (login) keychain when the
  Data Protection Keychain returns `errSecMissingEntitlement`. Lets unsigned
  CLI binaries store API keys without per-build codesigning. GUI consumers
  with proper entitlements are unaffected.

### 0.0.15
- `ModelConfiguration.extraJSONOverrides: [String: AnyCodable]?` for
  per-LLM provider-specific knobs (Anthropic `thinking`, OpenAI
  `reasoning_effort`, Gemini `safetySettings`, etc.).

### 0.0.14
- All providers serialize outbound request bodies with `[.sortedKeys]` so
  dictionary key order doesn't perturb the wire bytes — required for
  upstream prompt caches (Anthropic, OpenAI auto-prefix) to keep hitting.

## License

MIT. See [LICENSE](LICENSE).
