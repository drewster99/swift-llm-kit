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

### 0.0.22 — Gemini functionResponse.name correctness fix

- `GeminiProvider` was previously setting `functionResponse.name` to the
  `toolCallID` when encoding `.toolResult` content. Gemini's API expects
  the actual **function name** (matching the prior `functionCall.name`)
  for parallel-call pairing — toolCallIDs aren't on Gemini's wire schema.
- Effect of the old bug: **serial** tool calls worked because Gemini
  falls back to positional pairing. **Parallel** tool calls (multiple
  `functionCall` parts in one model turn) silently failed because name
  was the only signal — and every response had the wrong name.
- Fix: walk the conversation once at request-build time to construct
  `[toolCallID → functionName]` from prior `.toolCalls` / `.mixed`
  assistant turns. Resolve each `.toolResult`'s name at encode time.
  Orphan tool results (no matching prior call — malformed conversation)
  fall back to the toolCallID + a logged warning to preserve back-compat.
- No public API change. Existing consumers' working code keeps working;
  parallel-tool-call setups that were silently broken now work correctly.

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
