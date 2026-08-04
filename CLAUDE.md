# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

SwiftLLMKit is a Swift Package (library, no executable) targeting macOS 15+ with Swift 6 strict concurrency. It manages LLM providers, model catalogs, configurations, and authenticated request preparation for multi-provider chat/tool-use applications. The package is consumed by host apps via SPM.

## Commands

This is a Swift Package that ships compiled resources (`Resources/*.json`). Per the parent CLAUDE.md, **always build via drews-xcode-mcp, never `swift build` or `xcodebuild` directly.**

```
# Build / test the package
mcp__drews-xcode-mcp__get_project_schemes --project_path /Users/andrew/cursor/swift-llm-kit
mcp__drews-xcode-mcp__build_project        --project_path /Users/andrew/cursor/swift-llm-kit
mcp__drews-xcode-mcp__run_project_tests    --project_path /Users/andrew/cursor/swift-llm-kit
```

Tests use the Swift Testing framework (`import Testing`, `@Test` / `#expect`), not XCTest. The test suite is currently minimal — see ROADMAP.md for the planned test coverage list.

## Architecture

### Entry point: `LLMKitManager` (`SwiftLLMKit.swift`)

`@Observable @MainActor` central coordinator. A host app creates one, calls `load()` then `await refreshIfNeeded()`. It owns three persisted datasets and exposes CRUD over them:

1. **Providers** (`[ModelProvider]`) — endpoint + apiType + display name. API keys live in Keychain, *not* in the provider record.
2. **Configurations** (`[ModelConfiguration]`) — user-named (provider, model, temp, max_tokens, thinking budget, …) tuples referenced by UUID.
3. **Models** (`[ModelInfo]`) — the cached, enriched per-provider model catalog.

To send a request, the host either:
- Calls `makeProvider(for: configurationID)` to get a fully-built `LLMProvider` and calls `.send(messages:tools:)`, **or**
- Calls `prepareRequest(for: configurationID)` to get a `PreparedRequest` (URL + auth headers + base body dict) and finishes constructing the body itself.

Note the duplication between these two paths is a known issue tracked in ROADMAP.md ("Deprecate or reconcile `prepareRequest(for:)`").

### Provider abstraction (`Providers/`)

`LLMProvider` is the per-call protocol: `send(messages:tools:) async throws -> LLMResponse`. Concrete adapters: `AnthropicProvider`, `OpenAICompatibleProvider`, `GeminiProvider`, `OllamaProvider`. `OpenAICompatibleProvider` is reused for many `ProviderAPIType` cases (mistral, xAI, z.ai, huggingFace, lmStudio, metaLlama, alibabaCloud, openRouter) — when adding a new provider type, **always update every `switch` over `ProviderAPIType`** (in `SwiftLLMKit.swift` `prepareRequest` + `makeProvider`, `ProviderAPIType.swift`, `ModelFetchService.swift`).

`LLMMessage.Content` is a tagged enum (`.text`, `.toolCalls`, `.mixed`, `.toolResult`) with custom Codable. `LLMResponse` carries text + toolCalls + reasoning + `TokenUsage` (which preserves `rawUsage` JSON verbatim for fields not yet parsed).

All providers share `llmURLSession` (10-min request / 15-min resource timeout) — local Ollama generations can run for minutes, and the URLSession default of 7 days hangs forever after sleep/wake.

### `ProviderAPIType` (`Models/ProviderAPIType.swift`)

The 12-case enum that drives every per-provider branch. Carries `temperatureRange`, `endpointPresets`, `liteLLMPrefix`, and `displayName`. The `URL.ensureAnthropicV1()` / `strippingAnthropicV1()` helpers are defined here — Anthropic's base endpoint is sometimes used with `/v1` and sometimes without, depending on the operation.

### Built-in providers

`Resources/built_in_providers.json` is bundled into the package via SPM `.process("Resources")` and loaded by `BuiltInProviders.all`. Built-in IDs use the `builtin.<name>` namespace and are *immutable except for API key* — `updateProvider` clamps name/apiType/endpoint back to the preset, and `deleteProvider` refuses them.

`seedBuiltInProviders()` runs on every `load()` and is **idempotent**:
1. If the built-in ID already exists, refresh fixed fields from the preset.
2. Otherwise, look for a unique user-created provider whose `apiType + endpoint` match the preset and **adopt** it (rewrite ID, migrate Keychain entry, rewrite all `ModelConfiguration.providerID` references).
3. Fall back to creating an empty built-in row.

### Model metadata enrichment

`fetchAndEnrich()` layers four sources in this priority order — **higher priority wins**:

1. **User overrides** (`setUserOverrides`) — force-replace, user always wins.
2. **Provider API** — base data from listing the provider's `/models` endpoint.
3. **App-bundled** (`Resources/bundled_model_metadata.json` via `BundledModelMetadataRegistry`) — gap-fill.
4. **LiteLLM** (`ModelMetadataService` actor, fetched from `BerriAI/litellm` on GitHub with ETag/Last-Modified caching) — gap-fill, lowest priority.

Refresh is gated by a YYYYMMDD key in `UserDefaults` *plus* a "straggler" pass: after the daily refresh, providers whose cache slice is empty *and* are refreshable (have a key in Keychain, or are local no-auth like ollama/lmStudio) get refetched anyway. Without the straggler pass, providers added in a later app version would stay empty forever.

### Effort: two constructs, never one (2026-08-03)

"Effort" is two different wire parameters, and conflating them was a live bug:

- **General effort** — Anthropic's top-level `output_config.effort`. Steers overall work and
  applies **even when reasoning is disabled**. Not a thinking knob.
- **Reasoning effort** — `reasoning_effort` on OpenAI-compatible endpoints (OpenAI o-series/GPT-5,
  Moonshot). Exists only for reasoning models; everything else rejects it with HTTP 400.

Both are `EffortSupport?` on `ModelFacts`/`ModelInfo` (`generalEffort`, `reasoningEffort`), and both
are separate on the config (`ModelConfiguration.effort` / `.reasoningEffort`, same on
`LLMCallOverrides` and `ModelConfigurationOverride`).

**`EffortSupport` is one value, not a ladder plus a flag.** Four states are genuinely needed —
`nil` (nothing known), `.unsupported`, `.supportedLevelsUnknown`, `.levels([...])` — and a
ladder+flag pair could also represent CONTRADICTIONS. One was live: a probe that attempted the
complete ladder and had every level rejected wrote `[]`, a forced override set the flag `true`, the
provider emitted `reasoning_effort` on every request, and validation couldn't pre-flight it because
it keyed on `!isEmpty`. `init(levels:)` normalizes `[]` to `.unsupported` so the contradiction is
unconstructable. `.supportedLevelsUnknown` is not a placeholder — it is OpenAI's normal state, since
OpenAI publishes no ladder at all.

**Emission is asymmetric on purpose, and the direction is per-knob.** General effort fails OPEN (sent
unless the model is KNOWN not to take it — a clear API error beats a silently dropped knob).
Reasoning effort, structured output and `strict` fail CLOSED (sent only when known-supported,
because each is an HTTP 400 otherwise). `tool_choice` fails OPEN despite being gated per option: it
predates this gating, and silently dropping a caller's explicit choice on every model no source has
described would be a behaviour change, not a safety measure.

**A RECORDED mechanism is gated strictly; the legacy `apiType` fallback is not.** That distinction is
the whole meaning of "nil keeps the legacy behaviour" — no Alibaba model has its capabilities
recorded, so applying the strict gates to the fallback would stop sending `thinking_budget` to every
one of them. Both read per-model catalog data
injected at provider construction, the same path `behaviorFlags` travels — never an `apiType` branch.

`BehaviorFlags.supportsReasoningEffort` is RETIRED; its 18 bundled entries migrated to
`reasoningEffort: .supportedLevelsUnknown`, which is what the flag actually meant.

### `ReasoningControl`: how reasoning is switched, as data

`provider.apiType == .alibabaCloud` stopped working the moment Moonshot and DeepSeek arrived: both
are `openAICompatible` alongside OpenAI, and all three want different keys. `ReasoningControl` makes
the mechanism per-model data (`unsupported` / `reasoningEffortOnly` / `thinkingBlock` /
`enableThinkingFlag` / `anthropicThinking` / `geminiThinkingConfig`).

An enum because the mechanisms are mutually exclusive — as booleans,
`usesThinkingBlock && usesEnableThinkingFlag` would describe a model that cannot exist. The "no
control" case is spelled **`unsupported`, never `none`**: the type is nearly always
`ReasoningControl?`, where `.none` binds to `Optional.none`, so "has no reasoning knob" and "nobody
has said" would be written identically and mean opposite things.

**`nil` keeps the legacy `apiType` behaviour rather than emitting nothing.** Silently disabling
reasoning on every not-yet-recorded model is a regression dressed up as caution.

Whether reasoning can be turned on and whether it can be turned OFF are SEPARATE capabilities
(`reasoningCanBeEnabled` / `reasoningCanBeDisabled`) — Kimi documents models supporting only one
direction.

### Capabilities are vendor facts; every new one needs five edits and a UI slot

`ModelCapabilities` describes **what the model can do**, not what SwiftLLMKit can send. Several
capabilities have no typed send path (`audioInput`, `videoInput`, `computerUse`, …) and that is
deliberate — it is a catalog library, and discarding a stated vendor fact means re-fetching it later.

Adding a capability is NOT free. Each needs: a `ModelCapability` case, its three exhaustive switch
arms (`editorTitle`, short label, `editorDescription`), a `ModelCapabilities` accessor, a
`ModelCapabilitiesOverride` field + subscript arms + `apply`, and a `ModelFactsFieldTable` row. The
field-table completeness test fails if the last one is missed.

**Anything added to `ModelMetadataOverride` must be reachable from a UI editor**, or users cannot
correct a wrong value. The override sheets start from the EXISTING override and mutate only what
they own — rebuilding field-by-field made every field a given sheet didn't know about vanish on save.

### Probes: force the parameter, and grade the right thing

Probes never need production emission code. They force the raw field through
`ModelConfiguration.extraJSONOverrides` (or `LLMCallOverrides`) and grade with
`probeParameterAcceptance`; `mergeJSONOverrides` deep-merges dicts and replaces arrays wholesale, so
even `strict` inside `tools` is reachable by supplying a full replacement array.

**Acceptance is not always the right grade.** `probeParameterAcceptance` discards the response body,
so for `response_format` it would record every endpoint that IGNORES the field as supporting it —
`probeStructuredOutput` parses the response and checks the ASKED-FOR shape, not merely that some
JSON came back. `tool_choice` stays acceptance-graded on purpose: grading "did it really force a
call" produces false negatives on well-behaved models.

**A probe must never ask through the knob it is establishing.** `probeStructuredOutput` originally
went through `LLMCallOverrides.responseFormat`, whose emission is gated on the very capability being
probed — so on an unknown it sent no `response_format` at all and then graded a model that merely
followed the prompt's wording. Force the raw field; the production gate is for production.

**Pair the parameters a probe's target constrains.** Anthropic requires `max_tokens > budget_tokens`,
so `probeThinkingBudgetRange` hands its factory a paired max. Without it the search converges on the
PAIRING boundary and records that as the model's intrinsic ceiling.

**A probe that measures a flag-gated parameter must say so.** `probeEffortLevel` sends the config's
general effort, which a gated endpoint silently drops — turning "no error" into a false positive.
`probe(...)` refuses to run it unless the caller passes
`supportsUnconditionalGeneralEffortEmission`. That guard used to live in the single caller, where it
could be forgotten.

### Capability names say what is TRUE of the model, not what the parameter is

Two grammatical families, and mixing them is what made the first batch unreadable:

- **"the model HAS X"** — `vision`, `toolUse`, `pdfInput`, `parallelToolCalls`. The noun IS the
  feature and reads correctly.
- **"the model ACCEPTS value v of parameter p"** — these must SAY so. Written as
  `<parameter><Value>` they parse as an adjective phrase about the parameter and invert:
  `toolChoiceRequired` reads "a tool_choice is required", the opposite of "the tool_choice
  parameter accepts the value `required`". Hence `toolChoiceSupportsValueRequired`,
  `structuredOutputSupportsJSONObject`, `thinkingSupportsKeepAll`, `toolDefinitionsSupportStrict`.

A capability must also not share a spelling with a request KNOB. `thinkingBudgetTokens` was a Bool
capability and an `Int?` on `LLMCallOverrides`, twenty lines apart in the same file; it is
`thinkingSupportsTokenBudget` now.

**`ModelCapability` rawValues are the persisted keys and are PINNED explicitly**, with a guard test
asserting them against an independent table plus completeness. They were implicit, so a rename
silently rewrote the key and orphaned every record using it — `vision` alone is in 1,414. Renaming a
case is free; changing a wire string fails the build. When a wire string genuinely must change, it
is a one-time script (`scripts/migrate_capability_wire_names.py`) run with the app quit, landing in
the SAME commit as the pin change — the app reads exactly one spelling at a time. A record carrying
an unmigrated spelling decodes with that capability ABSENT, never wrong, which is silent; that is
why the script verifies zero survivors rather than trusting the pass.

### One source per wire shape, and per gate rule

Both of these were duplicated, both drifted-in-waiting, and both cost a real bug before they were
consolidated. The rule now: **the shape and the gate live on the type, and every caller derives.**

- **`LLMToolChoice`** owns `openAIWireValue`, `anthropicWireValue`, `requiredCapability`, and
  `wireValue(for apiType:)`. There were THREE hand-written encoders (OpenAI-compatible, an
  Ollama copy whose own comment admitted it mirrored the first, and a third in Agent Smith's eval
  runner). `wireValue(for:)` returns `nil` where the field does not exist — Gemini has no
  `tool_choice` — so a probe skips instead of spending a call.
- **`LLMResponseFormat`** owns `wireValue`; `openAIWireValue` and `forcedWireValue` derive from it.
- **`ModelCapabilities.permitsToolChoice(_:)`** is the ONE gate rule, called by all three providers
  AND by `CapabilityProbe`. Two facts ride in it: `.toolChoice` is "the endpoint accepts the
  PARAMETER" (what the decoders actually write), so it is a PRECONDITION over every option, and the
  per-option capability is that option's own veto.
- **`ThinkingBudget.pairing`** is the one `(max_tokens, budget_tokens)` rule, shared by
  `AnthropicProvider` and `prepareRequest`.

**Why this is not tidiness.** A probe that forces a different shape than production emits measures
something that never ships, and the finding EXPORTS as data the emission gates then act on — so a
drifted encoder ships a wrong capability table, which then suppresses a working field or sends a
400-producing one. Concretely: forcing OpenAI's bare `"required"` at Anthropic (which needs
`{"type": "any"}`) is rejected for the SHAPE, the rejection names `tool_choice`, and it records
"this model cannot force a tool call" — flatly wrong for Claude. Likewise `CapabilityProbe` must not
claim it FORCED a call through a field the provider suppressed, which is why it shares the gate
rather than re-deriving it.

**Switch exhaustiveness does not protect any of this.** It covers CASES, not wire strings, key names
or nesting. The worst instance was a capability mapping written as an ARRAY literal in the eval
runner: adding an `LLMToolChoice` case breaks every switch loudly and leaves that array silently one
probe short.

### `ThinkingBudgetAccounting`: whose allowance the thinking tokens come from

Not derivable from ``ReasoningControl`` — that says which KEYS to send, this says whose budget is
spent. Anthropic draws from `max_tokens` and enforces `max_tokens > budget_tokens`; the others use
separate keys whose accounting is UNVERIFIED, which is why it is probed and overridable rather than
tabulated per provider. Getting it wrong does not error: it silently truncates, because a budget
equal to the output cap leaves nothing for the answer.

Two consumers: it bounds `probeThinkingBudgetRange`'s search ceiling, and it decides whether
`OpenAICompatibleProvider`'s `thinkingBlock` branch pairs the emitted budget against `max_tokens`.
Until the second existed, the type documented a truncation nothing anywhere prevented.

### Probe records are schema-versioned and migrated, never soft-decoded

`ProbeRecord.currentSchemaVersion` is **3** (v2 split `effortLevels` into `generalEffortLevels` +
`reasoningEffortLevels`; v3 added `capabilityFindings` and `maxThinkingBudgetTokens`). There is NO
runtime migration — the script is the only path, deliberately, so a half-migrated corpus is a loud
failure rather than a silent partial decode. The store decodes with `try?` and SKIPS failures silently, so an
unmigrated record does not error — it VANISHES, taking every other finding in it with it. Migrations
must therefore rewrite **every** record, not just the ones with interesting data.
`scripts/migrate_effort_split.py` is the pattern: refuses to run while the host app is live, backs
up first, verifies after. The rule it applies (Anthropic → general, everyone else → reasoning) is
exhaustive because those are the only providers that ever recorded a ladder.

### Persistence layout

`StorageManager` writes to `~/Library/Application Support/SwiftLLMKit/<appBundleID>/`:
- `providers.json`, `model_configurations.json`, `model_catalog.json` (catalog is a cache, regenerated on refresh)
- LiteLLM cache (`litellm_metadata.json`, `litellm_headers.json`) lives in the same directory but is owned by `ModelMetadataService`.

Both `LLMKitManager` and `StorageManager` track `providersLoadedOK` / `configurationsLoadedOK` flags — **if the initial decode fails (e.g. schema change), the in-memory state is empty but `save()` is suppressed** so the on-disk file isn't overwritten with empty data. Don't bypass this guard.

API keys live in Keychain via `KeychainService`, keyed by provider ID. `apiKeyChangeCounter` is bumped on every key write so SwiftUI views can observe a change to a non-observable Keychain read.

### Logging

`LLMRequestLogger` writes full request/response JSON to `$TMPDIR/SwiftLLMKit-Logs/`. Enabled per-`LLMKitManager` via `verboseLogging = true`. **Logs include API keys in headers and full prompt/response content** — there's a planned ROADMAP item to filter sensitive fields. `os.Logger` is used for everything else (subsystem `"SwiftLLMKit"`, per-file category).

## Project conventions

- ROADMAP.md is the source of truth for planned work and known issues. Per the user's global rules, **completed items stay in the file** (marked complete with `~~strikethrough~~ ✅ Completed` and a brief note on what changed); they are not deleted.
- The library deliberately exposes provider-specific quirks (Anthropic's `temperature = 1` requirement when `thinking` is enabled, Mistral's parallel-tool-call quirk in `makeProvider`, Gemini's per-request URL-keyed auth in `prepareRequest` vs. `x-goog-api-key` header in `ModelFetchService`). When changing one provider, search for the others — request preparation logic is duplicated across `prepareRequest` and the individual `*Provider` adapters.
- `ModelConfiguration.useDefaultTemperature` exists for models (e.g. Alibaba QVQ) that reject any explicit temperature; `extendedCacheTTL` toggles Anthropic's 1h vs 5min ephemeral cache.
- Backward-compatible Codable: new optional fields on persisted types should use `decodeIfPresent` with a default (see `ModelConfiguration.init(from:)` for the pattern). Don't break old on-disk files.
