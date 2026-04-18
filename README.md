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

## License

MIT. See [LICENSE](LICENSE).
