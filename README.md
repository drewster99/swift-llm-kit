# SwiftLLMKit

[![CI](https://github.com/drewster99/swift-llm-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/drewster99/swift-llm-kit/actions/workflows/ci.yml)
[![Latest tag](https://img.shields.io/github/v/tag/drewster99/swift-llm-kit?label=version&sort=semver)](https://github.com/drewster99/swift-llm-kit/tags)
![Swift 6](https://img.shields.io/badge/Swift-6-F05138?logo=swift&logoColor=white)
![Platform](https://img.shields.io/badge/platform-macOS%2015%2B-lightgrey)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Good first issues](https://img.shields.io/github/issues/drewster99/swift-llm-kit/good%20first%20issue?label=good%20first%20issues&color=7057ff)](https://github.com/drewster99/swift-llm-kit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

A Swift package for talking to many LLM providers through one interface. It
handles tool calling, reasoning and effort controls, structured output, and
image and PDF input, and it keeps a model catalog that knows which of those
each model actually supports. API keys are stored in the Keychain.

## Supported providers

Each provider below ships as a built-in preset with a stable ID and a fixed
endpoint. Any other OpenAI-compatible backend can be added as a custom provider.

| Provider | Built-in ID | Notes |
|---|---|---|
| Anthropic (Claude) | `builtin.anthropic` | Native Messages API, prompt caching, adaptive and extended thinking |
| OpenAI | `builtin.openai` | |
| ChatGPT subscription (Codex) | `builtin.codex-chatgpt` | Signs in with the Codex CLI's `~/.codex/auth.json`, not an API key |
| Google Gemini | `builtin.gemini` | Native `generateContent` API |
| xAI (Grok) | `builtin.xai` | |
| DeepSeek | `builtin.deepseek` | |
| Moonshot AI (Kimi) | `builtin.moonshot` | |
| z.ai | `builtin.zai`, `builtin.zai-coding` | General and coding endpoints |
| Meta Model API | `builtin.meta-model-api` | |
| Mistral | `builtin.mistral` | |
| Alibaba Cloud (DashScope) | `builtin.alibabacloud`, `-beijing`, `-singapore` | US, Beijing, Singapore regions |
| Hugging Face | `builtin.huggingface` | Inference router |
| OpenRouter | `builtin.openrouter` | |
| Ollama | `builtin.ollama`, `builtin.ollama-cloud` | Local or cloud |
| LM Studio | `builtin.lmstudio` | Local |

## Features

- **One `LLMProvider` interface.** Every backend implements `send(messages:tools:)`, so your app doesn't need to know which vendor is answering.
- **Tool calling.** Tools, tool calls, and tool results work the same way on every provider, with per-call `LLMToolChoice` (`auto`, `required`, `textOnly`, or one specific tool) and optional `strict` schemas.
- **Reasoning and effort, modeled correctly.** Anthropic's general `effort` and OpenAI's `reasoning_effort` are kept as separate settings, and so are thinking budgets and on/off switches. Each is sent only to models known to accept it, so an unsupported parameter never causes an HTTP 400.
- **Structured output.** `LLMResponseFormat` covers JSON-object and JSON-schema modes.
- **Multimodal input.** Images and PDF documents are attached to messages.
- **A model catalog that knows capabilities.** It combines each provider's `/models` listing, bundled metadata, [LiteLLM](https://github.com/BerriAI/litellm) data, and your own overrides, in a fixed priority order.
- **Capability probing.** `ModelProber` measures what an endpoint really accepts (thinking-budget ranges, effort levels, tool choice, structured output) instead of trusting documentation.
- **Keychain-backed API keys.** Keys never live in UserDefaults or plain files.
- **SwiftUI-ready settings.** `LLMKitManager` is `@Observable @MainActor`, so providers, models, and configurations bind directly to views.
- **Prompt caching.** Supports Anthropic's ephemeral cache (5 min or 1 h) and reports cached-input tokens from OpenAI and xAI. Request bodies are serialized deterministically, so provider-side caches keep hitting.

## Requirements

- macOS 15 or newer
- Swift 6.1 or newer (Xcode 16.4+)

iOS and visionOS support is [on the wish list](https://github.com/drewster99/swift-llm-kit/issues?q=is%3Aissue+is%3Aopen+iOS), and contributions are welcome.

## Installation

```swift
.package(url: "https://github.com/drewster99/swift-llm-kit", from: "0.0.210")
```

Then add the product to your target:

```swift
.product(name: "SwiftLLMKit", package: "swift-llm-kit")
```

## Quick start

```swift
import SwiftLLMKit

@MainActor
func hello(apiKey: String) async throws {
    let kit = LLMKitManager(
        appIdentifier: "com.example.MyApp",
        keychainServicePrefix: "com.example.MyApp"
    )
    kit.load()

    // Stored in the Keychain — never in UserDefaults or a plain file.
    try kit.setBuiltInProviderAPIKey(id: BuiltInProviders.ID.anthropic, apiKey: apiKey)

    await kit.refreshIfNeeded()   // fetches each provider's current model list

    let configuration = ModelConfiguration(
        name: "Claude Sonnet",
        providerID: BuiltInProviders.ID.anthropic,
        modelID: "claude-sonnet-5"
    )
    kit.addConfiguration(configuration)

    let provider = try kit.makeProvider(for: configuration.id)
    let response = try await provider.send(messages: [.user("Hello!")], tools: [])
    print(response.text ?? "")
}
```

### Tool calling

```swift
let weather = LLMToolDefinition(
    name: "get_weather",
    description: "Current weather for a city",
    parameters: [
        "type": .string("object"),
        "properties": .dictionary([
            "city": .dictionary(["type": .string("string")])
        ]),
        "required": .array([.string("city")])
    ]
)

var messages: [LLMMessage] = [.user("What's the weather in Oslo?")]
let response = try await provider.send(messages: messages, tools: [weather])
messages.append(.assistant(from: response))

for call in response.toolCalls {
    // call.arguments is the model's JSON argument string, e.g. {"city":"Oslo"}
    messages.append(.toolResult("12°C, light rain", callID: call.id))
}
let answer = try await provider.send(messages: messages, tools: [weather])
print(answer.text ?? "")
```

`.assistant(from:)` carries the provider's reasoning continuation (Anthropic
thinking blocks, Gemini thought signatures, Codex reasoning items), so
multi-turn tool use keeps working on reasoning models.

## How it fits together

- **`LLMKitManager`** is the central coordinator. It owns providers, model configurations, and the model catalog, and it saves them under Application Support.
- **`LLMProvider`** is the per-request protocol, implemented by `AnthropicProvider`, `OpenAICompatibleProvider`, `GeminiProvider`, and `OllamaProvider`, plus an internal Codex Responses adapter that `makeProvider(for:)` returns for the ChatGPT-subscription provider.
- **`ModelInfo` / `ModelCapabilities`** record what a model can do. The providers read them to decide which request fields to send.
- **`ModelProber`** measures capabilities that no vendor publishes.

[`CLAUDE.md`](CLAUDE.md) explains the design decisions in depth: why effort is
two settings, why some knobs fail open and others fail closed, and how probe
records are versioned.

## Contributing

Contributions are very welcome. Good places to start:

- [Good first issues](https://github.com/drewster99/swift-llm-kit/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are small and self-contained, with pointers to the exact files involved.
- [Help wanted](https://github.com/drewster99/swift-llm-kit/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) issues are larger features and improvements.
- [ROADMAP.md](ROADMAP.md) has the longer-term plans and the reasoning behind them.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to build, test, and submit a change.

## Changelog

See [GitHub Releases](https://github.com/drewster99/swift-llm-kit/releases), plus
[CHANGELOG.md](CHANGELOG.md) for the historical notes.

## License

MIT. See [LICENSE](LICENSE).
