# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

SwiftLLMKit is a Swift Package (library, no executable) targeting macOS 15+ with Swift 6 strict concurrency. It manages LLM providers, model catalogs, configurations, and authenticated request preparation for multi-provider chat/tool-use applications. The package is consumed by host apps via SPM.

## Commands

This is a Swift Package that ships compiled resources (`Resources/*.json`). Per the parent CLAUDE.md, **always build via xcode-mcp-server, never `swift build` or `xcodebuild` directly.**

```
# Build / test the package
mcp__xcode-mcp-server__get_project_schemes --project_path /Users/andrew/cursor/swift-llm-kit
mcp__xcode-mcp-server__build_project        --project_path /Users/andrew/cursor/swift-llm-kit
mcp__xcode-mcp-server__run_project_tests    --project_path /Users/andrew/cursor/swift-llm-kit
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
