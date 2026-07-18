import Testing
import Foundation
@testable import SwiftLLMKit

// MARK: - Model-name derivation

/// LiteLLM entries are matched on the `litellm_provider` FIELD plus a model name derived from the
/// key. These cover the derivation rule, whose whole job is to reduce every key shape LiteLLM uses
/// down to exactly what a provider's `/models` endpoint reports — and to refuse to do so when the
/// key's first segment isn't actually the provider.
@Suite("LiteLLM model-name derivation")
struct LiteLLMModelNameTests {

    @Test("A bare key passes through untouched")
    func bareKeyPassesThrough() {
        // All 23 real Anthropic entries are bare — there are zero `anthropic/` keys upstream.
        #expect(ModelMetadataService.modelName(fromKey: "claude-fable-5", provider: "anthropic") == "claude-fable-5")
        #expect(ModelMetadataService.modelName(fromKey: "gpt-4o", provider: "openai") == "gpt-4o")
    }

    @Test("The entry's own provider prefix is stripped")
    func ownPrefixIsStripped() {
        #expect(ModelMetadataService.modelName(fromKey: "mistral/codestral-latest", provider: "mistral") == "codestral-latest")
        #expect(ModelMetadataService.modelName(fromKey: "zai/glm-4.6", provider: "zai") == "glm-4.6")
        #expect(ModelMetadataService.modelName(fromKey: "ollama/codegemma", provider: "ollama") == "codegemma")
    }

    @Test("Only the FIRST segment goes — an author segment survives, matching what the API reports")
    func authorSegmentSurvives() {
        // OpenRouter's API reports "anthropic/claude-3-haiku", so that is what must remain.
        #expect(ModelMetadataService.modelName(fromKey: "openrouter/anthropic/claude-3-haiku", provider: "openrouter")
                == "anthropic/claude-3-haiku")
        #expect(ModelMetadataService.modelName(fromKey: "deepinfra/meta-llama/Llama-3.2-11B-Vision-Instruct", provider: "deepinfra")
                == "meta-llama/llama-3.2-11b-vision-instruct")
    }

    /// The safety property that makes a fallback-free matcher viable: LiteLLM keys frequently lead
    /// with something that is NOT a provider (an image size, a quality tier). Because only the
    /// entry's own provider prefix is eligible for stripping, those keys keep their junk segment
    /// and are therefore unreachable by a query for the bare model name.
    @Test("A first segment that isn't the provider is NOT stripped, so junk keys self-reject")
    func nonProviderPrefixIsNotStripped() {
        // Real upstream entries: both are litellm_provider "openai".
        #expect(ModelMetadataService.modelName(fromKey: "1024-x-1024/dall-e-2", provider: "openai") == "1024-x-1024/dall-e-2")
        #expect(ModelMetadataService.modelName(fromKey: "high/1536-x-1024/gpt-image-1.5", provider: "openai")
                == "high/1536-x-1024/gpt-image-1.5")
        // Crucially, neither can ever satisfy a lookup for the plain model name.
        #expect(ModelMetadataService.modelName(fromKey: "1024-x-1024/dall-e-2", provider: "openai")
                != ModelMetadataService.normalize("dall-e-2"))
    }

    /// Bedrock model IDs embed the model VENDOR ("anthropic") in the key while carrying
    /// litellm_provider "bedrock"/"bedrock_converse". They must never be reachable as Anthropic.
    @Test("Bedrock inference-profile keys keep their vendor segment and stay under Bedrock")
    func bedrockProfileKeysAreNotAnthropic() {
        for key in ["anthropic.claude-fable-5",
                    "global.anthropic.claude-fable-5",
                    "us.anthropic.claude-fable-5",
                    "eu.anthropic.claude-fable-5"] {
            let name = ModelMetadataService.modelName(fromKey: key, provider: "bedrock_converse")
            // Dotted vendor prefixes aren't slash-separated, so nothing is stripped at all.
            #expect(name == ModelMetadataService.normalize(key))
            // And they never collapse onto the real Anthropic model name.
            #expect(name != ModelMetadataService.normalize("claude-fable-5"))
        }
    }

    @Test("Names are case-folded on both sides")
    func namesAreCaseFolded() {
        #expect(ModelMetadataService.normalize("Llama-3.2-11B-Vision-Instruct") == "llama-3.2-11b-vision-instruct")
        #expect(ModelMetadataService.modelName(fromKey: "crusoe/Qwen/Qwen3-235B-A22B-Instruct-2507", provider: "crusoe")
                == "qwen/qwen3-235b-a22b-instruct-2507")
    }
}

// MARK: - Lookup + resolution

@Suite("LiteLLM lookup and resolution")
struct LiteLLMResolutionTests {

    /// Mirrors the upstream shapes: a bare Anthropic entry, a prefixed Mistral entry, and a
    /// Bedrock entry whose key mentions "anthropic".
    private static let sampleJSON = """
    {
      "sample_spec": { "note": "not a model" },
      "claude-fable-5": { "litellm_provider": "anthropic", "max_input_tokens": 200000, "supports_vision": true },
      "mistral/codestral-latest": { "litellm_provider": "mistral", "max_input_tokens": 32000 },
      "global.anthropic.claude-fable-5": { "litellm_provider": "bedrock_converse", "max_input_tokens": 200000 },
      "1024-x-1024/dall-e-2": { "litellm_provider": "openai" },
      "no-provider-model": { "max_input_tokens": 123 }
    }
    """

    private func makeService() async -> ModelMetadataService {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("litellm-tests-\(UUID().uuidString)")
        let service = ModelMetadataService(storageDirectory: dir, userDefaultsSuiteName: "litellm-tests-\(UUID().uuidString)")
        await service.ingestForTesting(Data(Self.sampleJSON.utf8))
        return service
    }

    @Test("A bare-keyed Anthropic model resolves via the litellm_provider field")
    func anthropicResolves() async {
        let service = await makeService()
        let entry = await service.metadata(for: "claude-fable-5", liteLLMProviderName: "anthropic")
        #expect(entry?.maxInputTokens == 200000)
        #expect(entry?.supportsVision == true)
        #expect(await service.resolution(forModelID: "claude-fable-5", liteLLMProviderName: "anthropic") == .resolved)
    }

    @Test("A prefixed model resolves by its bare name, as the provider API reports it")
    func mistralResolves() async {
        let service = await makeService()
        #expect(await service.metadata(for: "codestral-latest", liteLLMProviderName: "mistral")?.maxInputTokens == 32000)
        // Case-insensitively, too.
        #expect(await service.metadata(for: "CODESTRAL-LATEST", liteLLMProviderName: "mistral") != nil)
    }

    @Test("Bedrock's anthropic-named entry is NOT reachable as Anthropic")
    func bedrockEntryIsNotAnthropic() async {
        let service = await makeService()
        // The Bedrock key mentions "anthropic" but belongs to bedrock_converse.
        #expect(await service.metadata(for: "global.anthropic.claude-fable-5", liteLLMProviderName: "anthropic") == nil)
        #expect(await service.metadata(for: "global.anthropic.claude-fable-5", liteLLMProviderName: "bedrock_converse") != nil)
    }

    @Test("An unmapped provider is reported distinctly from a missing one")
    func resolutionDistinguishesFailureLevels() async {
        let service = await makeService()
        // No mapping at all (e.g. LM Studio / Hugging Face).
        #expect(await service.resolution(forModelID: "anything", liteLLMProviderName: nil) == .providerNotMapped)
        // Mapped to a name LiteLLM doesn't know.
        #expect(await service.resolution(forModelID: "anything", liteLLMProviderName: "not-a-real-provider") == .providerNotFound)
        // Provider known, model absent.
        #expect(await service.resolution(forModelID: "no-such-model", liteLLMProviderName: "anthropic") == .modelNotFound)
    }

    @Test("There are no fallbacks — a wrong provider mapping simply misses")
    func noFallbacks() async {
        let service = await makeService()
        // Previously the bare-ID fallback would have found this; now the mapping must be right.
        #expect(await service.metadata(for: "claude-fable-5", liteLLMProviderName: "mistral") == nil)
        // And the junk image-size key never matches the plain model name.
        #expect(await service.metadata(for: "dall-e-2", liteLLMProviderName: "openai") == nil)
    }

    @Test("Entries with no litellm_provider are dropped as unroutable")
    func entriesWithoutProviderAreDropped() async {
        let service = await makeService()
        let names = await service.allLiteLLMProviderNames().map(\.name)
        #expect(names == ["anthropic", "bedrock_converse", "mistral", "openai"])
        // sample_spec must not become a provider.
        #expect(!names.contains("sample_spec"))
    }

    @Test("Discovery lists every provider carrying a given model name")
    func discoveryFindsProvidersForModel() async {
        let service = await makeService()
        #expect(await service.liteLLMProviderNames(matchingModelID: "claude-fable-5") == ["anthropic"])
        #expect(await service.liteLLMProviderNames(matchingModelID: "codestral-latest") == ["mistral"])
        #expect(await service.liteLLMProviderNames(matchingModelID: "nothing-here").isEmpty)
    }
}

// MARK: - Built-in preset mapping

/// `BuiltInProviders.loadFromBundle()` fails SILENTLY — a decode error logs and yields an empty
/// list, which would erase every built-in provider. These assert the bundled JSON still decodes
/// and that each preset carries the `litellm_provider` value its models are actually catalogued
/// under upstream.
@Suite("Built-in provider LiteLLM mapping")
struct BuiltInProviderMappingTests {

    @Test("The bundled provider JSON decodes (guards the silent-empty failure mode)")
    func bundledJSONDecodes() {
        #expect(BuiltInProviders.all.count == 13)
    }

    @Test("Every built-in maps to the litellm_provider value its models actually use")
    func mappingsMatchUpstream() {
        let expected: [String: String?] = [
            "builtin.anthropic": "anthropic",       // 23 bare claude-* entries; zero "anthropic/" keys
            "builtin.openai": "openai",
            "builtin.gemini": "gemini",
            "builtin.xai": "xai",
            "builtin.mistral": "mistral",
            "builtin.zai": "zai",
            "builtin.ollama": "ollama",
            "builtin.ollama-cloud": "ollama",
            "builtin.openrouter": "openrouter",
            "builtin.alibabacloud": "dashscope",    // Alibaba's API is DashScope
            "builtin.metallama": "meta_llama",      // underscore is the upstream wire value
            "builtin.huggingface": nil,             // LiteLLM catalogues none
            "builtin.lmstudio": nil                 // local, arbitrary models
        ]
        for (id, want) in expected {
            let preset = BuiltInProviders.preset(id: id)
            #expect(preset != nil, "missing preset \(id)")
            #expect(preset?.liteLLMProviderName == want, "\(id) mapped to \(preset?.liteLLMProviderName ?? "nil")")
        }
    }

    @Test("The mapping survives into the constructed ModelProvider")
    func mappingReachesModelProvider() throws {
        let preset = try #require(BuiltInProviders.preset(id: "builtin.anthropic"))
        #expect(ModelProvider(builtIn: preset).liteLLMProviderName == "anthropic")
    }

    /// Old providers.json predates the field; a missing key must decode to nil rather than throw
    /// (which would take the whole provider list with it).
    @Test("A stored provider without the field decodes to nil")
    func legacyProviderDecodes() throws {
        let json = #"{"id":"custom-1","name":"Mine","apiType":"openAICompatible","endpoint":"https://x.test/v1"}"#
        let provider = try JSONDecoder().decode(ModelProvider.self, from: Data(json.utf8))
        #expect(provider.liteLLMProviderName == nil)
    }
}

// MARK: - Seeding preserves user mappings

/// Seeding forces built-in rows back onto their preset values on every launch. `liteLLMProviderName`
/// is the one field a user may edit on a built-in, so it must survive that — otherwise the restart
/// the mapping editor asks for would itself revert the edit.
@Suite("Built-in seeding preserves the LiteLLM mapping")
struct BuiltInSeedingTests {

    private var anthropicPreset: BuiltInProviderPreset {
        get throws { try #require(BuiltInProviders.preset(id: "builtin.anthropic")) }
    }

    @Test("A user's override survives seeding")
    func userOverrideSurvives() throws {
        var stored = ModelProvider(builtIn: try anthropicPreset)
        stored.liteLLMProviderName = "bedrock_converse"   // user re-pointed it
        let canonical = LLMKitManager.canonicalBuiltIn(preset: try anthropicPreset, existing: stored)
        #expect(canonical.liteLLMProviderName == "bedrock_converse")
    }

    @Test("A row predating the field is seeded from the preset (existing installs self-heal)")
    func nilIsSeededFromPreset() throws {
        let legacy = ModelProvider(id: "builtin.anthropic", name: "Anthropic",
                                   apiType: .anthropic, endpoint: URL(string: "https://api.anthropic.com")!,
                                   liteLLMProviderName: nil)
        let canonical = LLMKitManager.canonicalBuiltIn(preset: try anthropicPreset, existing: legacy)
        #expect(canonical.liteLLMProviderName == "anthropic")
    }

    @Test("A brand-new row takes the preset value")
    func newRowTakesPreset() throws {
        #expect(LLMKitManager.canonicalBuiltIn(preset: try anthropicPreset, existing: nil).liteLLMProviderName == "anthropic")
    }

    @Test("The fixed fields are still forced back to the preset")
    func fixedFieldsStillReset() throws {
        var tampered = ModelProvider(builtIn: try anthropicPreset)
        tampered.name = "Renamed"
        tampered.endpoint = URL(string: "https://evil.test")!
        let canonical = LLMKitManager.canonicalBuiltIn(preset: try anthropicPreset, existing: tampered)
        #expect(canonical.name == "Anthropic")
        #expect(canonical.endpoint == URL(string: "https://api.anthropic.com"))
    }
}

// MARK: - Chat-endpoint reachability

/// `supportsChatCompletions` answers exactly one question — will `/v1/chat/completions` accept
/// this model — and deliberately does not try to judge whether the model makes a good agent.
@Suite("Chat endpoint reachability")
struct ChatEndpointTests {

    private func entry(mode: String?, endpoints: [String]?) -> LiteLLMEntry {
        LiteLLMEntry(
            maxInputTokens: nil, maxOutputTokens: nil, pricing: nil,
            supportsToolUse: false, supportsVision: false, supportsReasoning: false,
            supportsPromptCaching: false, supportsComputerUse: false, supportsAudioInput: false,
            supportsAudioOutput: false, supportsVideoInput: false, supportsResponseSchema: false,
            supportsParallelToolCalls: false, supportsPdfInput: false, supportsWebSearch: false,
            supportsSystemMessages: false, supportsAssistantPrefill: false, supportsToolChoice: false,
            mode: mode, supportedEndpoints: endpoints
        )
    }

    /// The trap: reading `mode` before `supported_endpoints` disqualifies this real model, which
    /// is mode "responses" yet explicitly advertises the chat endpoint.
    @Test("An explicit endpoint list outranks mode")
    func endpointsOutrankMode() {
        let deepResearch = entry(mode: "responses",
                                 endpoints: ["/v1/chat/completions", "/v1/batch", "/v1/responses"])
        #expect(deepResearch.supportsChatCompletions)
        // responses-only, no chat endpoint -> genuinely unreachable for us
        #expect(!entry(mode: "responses", endpoints: ["/v1/responses", "/v1/batch"]).supportsChatCompletions)
    }

    @Test("mode decides when no endpoint list is given (the ~80% case)")
    func modeIsTheFallback() {
        #expect(entry(mode: "chat", endpoints: nil).supportsChatCompletions)
        #expect(!entry(mode: "embedding", endpoints: nil).supportsChatCompletions)
        #expect(!entry(mode: "moderation", endpoints: nil).supportsChatCompletions)
    }

    @Test("Unknown on both counts fails open — an unrecognised model stays usable")
    func unknownFailsOpen() {
        #expect(entry(mode: nil, endpoints: nil).supportsChatCompletions)
    }

    /// The kinds one might reach for a denylist are, bar one, unreachable anyway: upstream, ZERO
    /// embedding / audio_* / rerank / video_generation / ocr / moderation entries advertise the
    /// chat endpoint. This check already excludes every one, so a mode denylist would add nothing.
    @Test("Non-chat kinds are excluded by the endpoint question alone")
    func nonChatKindsAreAlreadyExcluded() {
        for kind in ["embedding", "audio_transcription", "audio_speech", "rerank",
                     "video_generation", "ocr", "moderation"] {
            #expect(!entry(mode: kind, endpoints: nil).supportsChatCompletions, "\(kind) should be excluded")
        }
    }

    /// The one exception, and the reason no mode denylist exists: `gemini-2.5-flash-image` is
    /// `mode: "image_generation"` yet claims function calling, tool_choice, parallel tools and
    /// system messages over the chat endpoint. Disqualifying it on its routing label would
    /// discard a model that may well drive an agent. Whether it actually can is a question for
    /// `toolUse` to answer from evidence, not for this one to guess from a kind.
    @Test("A chat-reachable image model is NOT disqualified here")
    func chatReachableImageModelIsNotDisqualified() {
        let geminiImage = entry(mode: "image_generation",
                                endpoints: ["/v1/chat/completions", "/v1/completions", "/v1/batch"])
        #expect(geminiImage.supportsChatCompletions)
    }

    @Test("mode survives a Codable round-trip and is absent on legacy records")
    func modeRoundTrips() throws {
        let info = ModelInfo(providerID: "p", modelID: "m", mode: "image_generation")
        let back = try JSONDecoder().decode(ModelInfo.self, from: JSONEncoder().encode(info))
        #expect(back.mode == "image_generation")
        // A record written before the field existed decodes to unknown, not a bogus kind.
        let legacy = #"{"providerID":"p","modelID":"m"}"#
        #expect(try JSONDecoder().decode(ModelInfo.self, from: Data(legacy.utf8)).mode == nil)
    }
}

/// The two coverage-screen bugs found live: sample_spec minting a phantom provider, and Ollama
/// Cloud's bare model IDs missing LiteLLM's `-cloud`-suffixed entries.
@Suite("LiteLLM sample_spec and -cloud alias")
struct LiteLLMSampleSpecAndCloudAliasTests {
    private func makeService(ingesting json: String) async -> ModelMetadataService {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("litellm-tests-\(UUID().uuidString)", isDirectory: true)
        let service = ModelMetadataService(storageDirectory: dir, userDefaultsSuiteName: "litellm-tests-\(UUID().uuidString)")
        await service.ingestForTesting(Data(json.utf8))
        return service
    }

    @Test("sample_spec never becomes a provider or model")
    func sampleSpecSkipped() async {
        let json = #"""
        {"sample_spec": {"litellm_provider": "one of https://docs.litellm.ai/docs/providers",
                         "max_input_tokens": "set to max input tokens"},
         "ollama/llama2": {"litellm_provider": "ollama", "max_input_tokens": 4096}}
        """#
        let service = await makeService(ingesting: json)
        let phantom = await service.allLiteLLMProviderNames().first { $0.name.contains("one of") }
        #expect(phantom == nil, "the schema-doc entry must not mint a phantom provider")
        #expect(await service.metadata(for: "llama2", liteLLMProviderName: "ollama") != nil)
    }

    @Test("A bare Ollama Cloud model ID matches LiteLLM's -cloud-suffixed key")
    func cloudSuffixAlias() async {
        let json = #"""
        {"ollama/gpt-oss:120b-cloud": {"litellm_provider": "ollama", "max_input_tokens": 131072,
                                       "supports_function_calling": true}}
        """#
        let service = await makeService(ingesting: json)
        let entry = await service.metadata(for: "gpt-oss:120b", liteLLMProviderName: "ollama")
        #expect(entry?.maxInputTokens == 131072)
        #expect(entry?.supportsToolUse == true)
        #expect(await service.resolution(forModelID: "gpt-oss:120b", liteLLMProviderName: "ollama") == .resolved)
        // An exact key still wins over the alias, and a genuinely absent model still misses.
        #expect(await service.resolution(forModelID: "glm-5.2", liteLLMProviderName: "ollama") == .modelNotFound)
    }
}
