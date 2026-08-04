import Testing
import Foundation
@testable import SwiftLLMKit

/// The knobs added alongside the effort split: reasoning on/off by MECHANISM, structured output,
/// and `strict` tool definitions. Every one of them is an HTTP 400 when sent to a model that
/// doesn't take it, so all of them fail CLOSED — emitted only on a KNOWN-true capability.
@Suite("Reasoning control, structured output, strict tools")
struct ReasoningControlEmissionTests {

    private func provider(
        control: ReasoningControl? = nil,
        capabilities: ModelCapabilities = ModelCapabilities(),
        apiType: ProviderAPIType = .openAICompatible,
        thinkingBudget: Int? = nil
    ) throws -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m",
                                              thinkingBudget: thinkingBudget),
            provider: ModelProvider(id: "p", name: "p", apiType: apiType,
                                    endpoint: try #require(URL(string: "https://x.test/v1"))),
            readAPIKey: { "" },
            reasoningControl: control,
            modelCapabilities: capabilities
        )
    }

    // MARK: Reasoning on/off

    @Test("thinkingBlock: enabling is emitted only when the model can be switched ON")
    func thinkingBlockEnable() throws {
        let allowed = try provider(control: .thinkingBlock,
                                   capabilities: ModelCapabilities([.reasoningEnableable]))
        let body = allowed.buildRequestBody(messages: [.user("hi")], tools: [],
                                            overrides: LLMCallOverrides(reasoningEnabled: true))
        #expect((body["thinking"] as? [String: Any])?["type"] as? String == "enabled")

        // Kimi documents models that can be switched on but NOT off — so the two directions are
        // gated independently, and an unknown one stays silent rather than guessing.
        let unknown = try provider(control: .thinkingBlock)
        let silent = unknown.buildRequestBody(messages: [.user("hi")], tools: [],
                                              overrides: LLMCallOverrides(reasoningEnabled: true))
        #expect(silent["thinking"] == nil)
    }

    @Test("thinkingBlock: disabling requires the DISABLE capability, not the enable one")
    func thinkingBlockDisableIsSeparate() throws {
        let onlyEnableable = try provider(control: .thinkingBlock,
                                          capabilities: ModelCapabilities([.reasoningEnableable]))
        let body = onlyEnableable.buildRequestBody(messages: [.user("hi")], tools: [],
                                                   overrides: LLMCallOverrides(reasoningEnabled: false))
        #expect(body["thinking"] == nil, "enable-able must not imply disable-able")

        let disableable = try provider(control: .thinkingBlock,
                                       capabilities: ModelCapabilities([.reasoningDisableable]))
        let ok = disableable.buildRequestBody(messages: [.user("hi")], tools: [],
                                              overrides: LLMCallOverrides(reasoningEnabled: false))
        #expect((ok["thinking"] as? [String: Any])?["type"] as? String == "disabled")
    }

    @Test("thinkingBlock: keep and budget ride the same object, each on its own capability")
    func thinkingBlockKeepAndBudget() throws {
        let caps = ModelCapabilities([.reasoningEnableable, .thinkingKeepAll, .thinkingBudgetTokens])
        let p = try provider(control: .thinkingBlock, capabilities: caps)
        let body = p.buildRequestBody(messages: [.user("hi")], tools: [],
                                      overrides: LLMCallOverrides(reasoningEnabled: true,
                                                                  thinkingBudgetTokens: 2048,
                                                                  keepThinking: true))
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["keep"] as? String == "all")
        #expect(thinking["budget_tokens"] as? Int == 2048)

        // Without the keep capability the rest still goes; only the unsupported key is dropped.
        let noKeep = try provider(control: .thinkingBlock,
                                  capabilities: ModelCapabilities([.reasoningEnableable]))
        let partial = noKeep.buildRequestBody(messages: [.user("hi")], tools: [],
                                              overrides: LLMCallOverrides(reasoningEnabled: true, keepThinking: true))
        #expect((partial["thinking"] as? [String: Any])?["keep"] == nil)
        #expect((partial["thinking"] as? [String: Any])?["type"] as? String == "enabled")
    }

    /// An unknown mechanism must not silently disable reasoning on every model whose control has
    /// not been recorded yet — Alibaba's legacy apiType behaviour is preserved.
    @Test("An unknown control keeps the legacy apiType behaviour for Alibaba")
    func unknownControlKeepsLegacyAlibaba() throws {
        let p = try provider(control: nil, apiType: .alibabaCloud, thinkingBudget: 4096)
        let body = p.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["enable_thinking"] as? Bool == true)
        #expect(body["thinking_budget"] as? Int == 4096)
    }

    @Test("reasoningEffortOnly and unsupported emit no on/off object at all")
    func noOnOffMechanisms() throws {
        for control in [ReasoningControl.reasoningEffortOnly, .unsupported] {
            let p = try provider(control: control,
                                 capabilities: ModelCapabilities([.reasoningEnableable]))
            let body = p.buildRequestBody(messages: [.user("hi")], tools: [],
                                          overrides: LLMCallOverrides(reasoningEnabled: true))
            #expect(body["thinking"] == nil, "\(control) has no on/off knob")
            #expect(body["enable_thinking"] == nil)
        }
    }

    // MARK: Structured output

    @Test("response_format is emitted only for the mode the model actually supports")
    func structuredOutputGating() throws {
        let objectOnly = try provider(capabilities: ModelCapabilities([.structuredOutputJSONObject]))
        let ok = objectOnly.buildRequestBody(messages: [.user("hi")], tools: [],
                                             overrides: LLMCallOverrides(responseFormat: .jsonObject))
        #expect((ok["response_format"] as? [String: Any])?["type"] as? String == "json_object")

        // The two modes are separate capabilities because a model may take one and reject the
        // other — DeepSeek documents json_object with json_schema unconfirmed.
        let schema = objectOnly.buildRequestBody(
            messages: [.user("hi")], tools: [],
            overrides: LLMCallOverrides(responseFormat: .jsonSchema(name: "s", schema: [:])))
        #expect(schema["response_format"] == nil, "json_object support must not imply json_schema")

        let unknown = try provider()
        let none = unknown.buildRequestBody(messages: [.user("hi")], tools: [],
                                            overrides: LLMCallOverrides(responseFormat: .jsonObject))
        #expect(none["response_format"] == nil, "fails closed on unknown")
    }

    @Test("json_schema carries name, strict and the schema at the documented path")
    func schemaWireShape() throws {
        let p = try provider(capabilities: ModelCapabilities([.responseSchema]))
        let body = p.buildRequestBody(
            messages: [.user("hi")], tools: [],
            overrides: LLMCallOverrides(responseFormat: .jsonSchema(
                name: "answer", schema: ["type": AnyCodable.string("object")], strict: true)))
        let format = try #require(body["response_format"] as? [String: Any])
        #expect(format["type"] as? String == "json_schema")
        let schema = try #require(format["json_schema"] as? [String: Any])
        #expect(schema["name"] as? String == "answer")
        #expect(schema["strict"] as? Bool == true)
        #expect((schema["schema"] as? [String: Any])?["type"] as? String == "object")
    }

    // MARK: strict tool definitions

    @Test("strict rides the function definition only when the model is known to take it")
    func strictToolGating() throws {
        let tool = LLMToolDefinition(name: "t", description: "d", parameters: [:], strict: true)

        let supported = try provider(capabilities: ModelCapabilities([.strictToolDefinitions]))
        let body = supported.buildRequestBody(messages: [.user("hi")], tools: [tool])
        let fn = try #require((body["tools"] as? [[String: Any]])?.first?["function"] as? [String: Any])
        #expect(fn["strict"] as? Bool == true)

        let unknown = try provider()
        let plain = unknown.buildRequestBody(messages: [.user("hi")], tools: [tool])
        let plainFn = try #require((plain["tools"] as? [[String: Any]])?.first?["function"] as? [String: Any])
        #expect(plainFn["strict"] == nil, "omitted on unknown — endpoints vary between ignore and reject")
        #expect(plainFn["name"] as? String == "t", "the tool itself still goes")
    }
}

/// The `supportsReasoningEffort` flag was retired and its 18 bundled entries rewritten as
/// `reasoningEffort: .supportedLevelsUnknown`. Those entries are the ONLY record that OpenAI's
/// reasoning models accept the parameter — OpenAI's `/models` publishes nothing — and reasoning
/// effort now fails CLOSED, so losing them silently stops sending `reasoning_effort` to every
/// o-series and GPT-5 model with no error anywhere.
@Suite("Bundled reasoning-effort entries survive the flag retirement")
struct BundledReasoningEffortMigrationTests {

    @Test("The bundled registry still states reasoning-effort support for OpenAI models")
    func bundledEntriesStillStateSupport() throws {
        let registry = BundledModelMetadataRegistry.load()
        // A model from each keying axis the migration touched.
        let byAPIType = try #require(registry.override(providerAPIType: "openAICompatible", modelID: "gpt-5"))
        #expect(byAPIType.reasoningEffort == EffortSupport.supportedLevelsUnknown)

        let byProvider = try #require(
            registry.override(providerID: "builtin.openrouter", modelID: "openai/gpt-5"))
        #expect(byProvider.reasoningEffort == EffortSupport.supportedLevelsUnknown)
    }

    /// Reaching `ModelFacts` is the part that matters — an override field the merge never reads is
    /// the same as no override at all.
    @Test("The stated support reaches ModelFacts, so the merge can act on it")
    func supportReachesFacts() throws {
        let registry = BundledModelMetadataRegistry.load()
        let entry = try #require(registry.override(providerAPIType: "openAICompatible", modelID: "o3"))
        #expect(entry.asFacts.reasoningEffort == EffortSupport.supportedLevelsUnknown)
    }
}

/// `ModelMetadataOverride` hand-writes its `Codable`, so a stored property added without a matching
/// CodingKeys case is silently never persisted AND never decoded. That is not hypothetical: five
/// fields shipped that way and the bundled `reasoningEffort` entries stopped decoding, which under
/// fail-closed emission silently stopped sending `reasoning_effort` to every OpenAI reasoning model
/// — no error anywhere. `ModelProfile` has had this guard for exactly the same reason.
@Suite("ModelMetadataOverride CodingKeys coverage")
struct ModelMetadataOverrideCodingKeyCoverageTests {

    /// Reflection, not a round trip: an uncovered property decodes back to its default and compares
    /// equal to any fixture that also left it defaulted, so a round-trip test stays green.
    @Test("Every stored property is encoded")
    func everyStoredPropertyIsEncoded() throws {
        let populated = ModelMetadataOverride(
            displayName: "M", maxInputTokens: 1000, maxOutputTokens: 100, sizeLabel: "7B",
            capabilities: ModelCapabilitiesOverride(vision: true),
            pricing: ModelPricing(base: PricingTier(input: 1, output: 2)),
            behaviorFlags: BehaviorFlagsOverride(requiresAdaptiveThinking: true),
            generalEffort: .levels(["low"]), reasoningEffort: .supportedLevelsUnknown,
            reasoningControl: .thinkingBlock, thinkingBudgetAccounting: .separate,
            maxThinkingBudgetTokens: 32_000,
            hidden: true, isAvailable: true, isAccessDenied: false)
        let object = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(populated)) as? [String: Any])
        // `supportsChatCompletions` is computed (it rides inside capabilities), so it is not stored.
        let expected = Set(Mirror(reflecting: populated).children.compactMap(\.label))
        let missing = expected.subtracting(object.keys).sorted()
        #expect(missing.isEmpty, """
            \(missing.joined(separator: ", ")) has no CodingKeys case, so it is silently never \
            persisted or decoded. Add it to ModelMetadataOverride.CodingKeys.
            """)
    }

    @Test("A full round trip preserves every field")
    func roundTrips() throws {
        let original = ModelMetadataOverride(
            generalEffort: .levels(["low", "max"]), reasoningEffort: .unsupported,
            reasoningControl: .enableThinkingFlag, thinkingBudgetAccounting: .drawnFromMaxOutputTokens,
            maxThinkingBudgetTokens: 8192)
        let restored = try JSONDecoder().decode(
            ModelMetadataOverride.self, from: JSONEncoder().encode(original))
        #expect(restored == original)
    }
}

/// The same reflection guard for the other two hand-written `Codable` types that gained fields.
///
/// `ModelMetadataOverride` shipped five silently-dropped fields before this pattern was caught;
/// these types are written the same way, so they get the same check rather than the same incident.
@Suite("Hand-written Codable coverage: ModelInfo and ModelConfiguration")
struct HandWrittenCodableCoverageTests {

    @Test("Every ModelInfo stored property is encoded")
    func modelInfoCoverage() throws {
        let info = ModelInfo(
            providerID: "p", modelID: "m", displayName: "M", createdAt: Date(timeIntervalSince1970: 0),
            maxInputTokens: 1000, maxOutputTokens: 100,
            capabilities: ModelCapabilities([.vision]), sizeLabel: "7B", quantizationLabel: "q4",
            pricing: ModelPricing(base: PricingTier(input: 1, output: 2)), mode: "chat",
            generalEffort: .levels(["low"]), reasoningEffort: .supportedLevelsUnknown,
            reasoningControl: .thinkingBlock, thinkingBudgetAccounting: .separate,
            maxThinkingBudgetTokens: 32_000,
            behaviorFlags: BehaviorFlags(requiresAdaptiveThinking: true),
            deprecatedOn: Date(timeIntervalSince1970: 1), deprecationReplacement: "m2",
            maxTemperature: 2, modelDescription: "d",
            samplingDefaults: SamplingDefaults(temperature: 0.5), isFree: false,
            benchmarks: nil, huggingFaceID: "hf/m", hidden: false,
            isAvailable: true, isAccessDenied: false, outputBoundedByContext: true,
            fetchedAt: Date(timeIntervalSince1970: 2), lastProbedAt: Date(timeIntervalSince1970: 3))
        // Every field must be NON-DEFAULT: several encode conditionally, so a defaulted fixture
        // would report them missing and a genuinely uncovered one would hide among the noise.
        try expectEveryStoredPropertyEncoded(info, ignoring: ["benchmarks"])
    }

    @Test("Every ModelConfiguration stored property is encoded")
    func modelConfigurationCoverage() throws {
        let config = ModelConfiguration(
            name: "c", providerID: "p", modelID: "m", temperature: 0.5,
            maxOutputTokens: 100, maxContextTokens: 1000, thinkingBudget: 2048,
            effort: "high", reasoningEffort: "low", extendedCacheTTL: true, streaming: false,
            extraJSONOverrides: ["k": .string("v")])
        // `isValid`/`validationError` are runtime state, not persisted configuration.
        try expectEveryStoredPropertyEncoded(config, ignoring: ["isValid", "validationError"])
    }

    /// Reflection rather than a round trip, for the reason documented on the override guard: an
    /// uncovered property decodes back to its default and compares equal to a defaulted fixture.
    private func expectEveryStoredPropertyEncoded<T: Encodable>(
        _ value: T, ignoring: Set<String>, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let object = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any],
            sourceLocation: sourceLocation)
        let stored = Set(Mirror(reflecting: value).children.compactMap(\.label)).subtracting(ignoring)
        let missing = stored.subtracting(object.keys).sorted()
        #expect(missing.isEmpty, """
            \(type(of: value)): \(missing.joined(separator: ", ")) is stored but never encoded — \
            add it to the hand-written CodingKeys/encode.
            """, sourceLocation: sourceLocation)
    }
}

/// Anthropic and Gemini ignored the per-call reasoning overrides entirely, and Gemini never emitted
/// the thinking budget its own source comment had documented since the file was written.
@Suite("Anthropic and Gemini honour the per-call reasoning overrides")
struct NonOpenAIReasoningOverrideTests {

    private func anthropic(thinkingBudget: Int?) throws -> AnthropicProvider {
        AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m",
                                              thinkingBudget: thinkingBudget),
            provider: ModelProvider(id: "p", name: "p", apiType: .anthropic,
                                    endpoint: try #require(URL(string: "https://api.anthropic.com"))),
            readAPIKey: { "" })
    }

    private func gemini(thinkingBudget: Int?, capabilities: ModelCapabilities = ModelCapabilities()) throws -> GeminiProvider {
        GeminiProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m",
                                              thinkingBudget: thinkingBudget),
            provider: ModelProvider(id: "p", name: "p", apiType: .gemini,
                                    endpoint: try #require(URL(string: "https://g.test/v1beta"))),
            readAPIKey: { "" }, modelCapabilities: capabilities)
    }

    @Test("Anthropic: a per-call budget overrides the configured one")
    func anthropicPerCallBudget() throws {
        let body = try anthropic(thinkingBudget: 2048).buildRequestBody(
            messages: [.user("hi")], tools: [],
            overrides: LLMCallOverrides(thinkingBudgetTokens: 8192))
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["budget_tokens"] as? Int == 8192, "the per-call value must win")
    }

    @Test("Anthropic: reasoningEnabled false turns thinking off despite a configured budget")
    func anthropicPerCallDisable() throws {
        let body = try anthropic(thinkingBudget: 4096).buildRequestBody(
            messages: [.user("hi")], tools: [],
            overrides: LLMCallOverrides(reasoningEnabled: false))
        #expect(body["thinking"] == nil, "an explicit off must beat a configured budget")
    }

    /// The field the provider documented and never sent — now emitted, but only on a KNOWN-true
    /// capability.
    @Test("Gemini emits thinkingConfig.thinkingBudget once the model is known to take it")
    func geminiEmitsBudget() throws {
        let body = try gemini(thinkingBudget: 4096, capabilities: ModelCapabilities([.thinkingBudgetTokens]))
            .buildRequestBody(messages: [.user("hi")], tools: [])
        let config = try #require(body["generationConfig"] as? [String: Any])
        let thinking = try #require(config["thinkingConfig"] as? [String: Any])
        #expect(thinking["thinkingBudget"] as? Int == 4096)
    }

    /// Fails CLOSED, and that asymmetry with Alibaba's `enable_thinking` fallback is deliberate:
    /// that one preserves behaviour the library already had, whereas this field was NEVER emitted.
    /// Failing open would start sending `thinkingConfig` to the non-thinking Gemini models that
    /// reject it — breaking requests that work today, to no benefit.
    @Test("Gemini sends nothing while support is UNKNOWN, not just when it is denied")
    func geminiFailsClosedOnUnknown() throws {
        for capabilities in [ModelCapabilities(), {
            var denied = ModelCapabilities(); denied[.thinkingBudgetTokens] = false; return denied
        }()] {
            let body = try gemini(thinkingBudget: 4096, capabilities: capabilities)
                .buildRequestBody(messages: [.user("hi")], tools: [])
            let config = try #require(body["generationConfig"] as? [String: Any])
            #expect(config["thinkingConfig"] == nil)
        }
    }

    @Test("Gemini: an explicit off sends budget 0 on a supporting model")
    func geminiExplicitOff() throws {
        let off = try gemini(thinkingBudget: 4096, capabilities: ModelCapabilities([.thinkingBudgetTokens]))
            .buildRequestBody(messages: [.user("hi")], tools: [],
                              overrides: LLMCallOverrides(reasoningEnabled: false))
        let config = try #require(off["generationConfig"] as? [String: Any])
        #expect((config["thinkingConfig"] as? [String: Any])?["thinkingBudget"] as? Int == 0,
                "Gemini turns thinking off with a zero budget")
    }
}


/// Anthropic requires `max_tokens > budget_tokens`, and `max_tokens` must not exceed the output
/// ceiling the caller sanctioned. Both were violated in turn: first by clamping only when a
/// per-call max override existed, then by raising `max_tokens` past the ceiling to satisfy the
/// pairing. The budget is the negotiable side.
@Suite("Anthropic thinking budget never produces an invalid pairing")
struct AnthropicBudgetPairingTests {

    private func body(maxTokens: Int, budget: Int?, overrideMax: Int? = nil,
                      perCallBudget: Int? = nil, modelCap: Int? = nil) throws -> [String: Any] {
        let provider = AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m",
                                              maxOutputTokens: maxTokens, thinkingBudget: budget),
            provider: ModelProvider(id: "p", name: "p", apiType: .anthropic,
                                    endpoint: try #require(URL(string: "https://api.anthropic.com"))),
            readAPIKey: { "" }, modelMaxOutputTokens: modelCap)
        return try provider.buildRequestBody(
            messages: [.user("hi")], tools: [],
            overrides: LLMCallOverrides(reasoningEffort: nil, thinkingBudgetTokens: perCallBudget,
                                        maxOutputTokens: overrideMax))
    }

    private func pairing(_ body: [String: Any]) -> (maxTokens: Int, budget: Int)? {
        guard let maxTokens = body["max_tokens"] as? Int,
              let thinking = body["thinking"] as? [String: Any],
              let budget = thinking["budget_tokens"] as? Int else { return nil }
        return (maxTokens, budget)
    }

    /// With the model's cap KNOWN, `max_tokens` may not be raised past it — so the budget is what
    /// gives way. The configured value is only a preference; the cap is the hard limit.
    @Test("A budget beyond the MODEL's cap degrades the budget, not the ceiling")
    func oversizedBudgetDegrades() throws {
        let sent = try body(maxTokens: 4096, budget: nil, perCallBudget: 8192, modelCap: 4096)
        let pair = try #require(pairing(sent))
        #expect(pair.maxTokens <= 4096, "max_tokens must not exceed the model's own cap")
        #expect(pair.budget < pair.maxTokens, "the pairing must still be legal")
    }

    /// With the cap UNKNOWN, raising is the long-standing behaviour and stays — refusing to raise
    /// would break thinking on every model whose cap has not been catalogued.
    @Test("An unknown model cap still raises max_tokens to clear the budget")
    func unknownCapStillRaises() throws {
        let sent = try body(maxTokens: 4096, budget: 4096)
        let pair = try #require(pairing(sent))
        #expect(pair.budget == 4096, "the configured depth survives")
        #expect(pair.maxTokens == 4097, "raised to clear it — the old code emitted 4096/4096, which is invalid")
    }

    @Test("A tight per-call cap is raised back toward the configured ceiling — the documented case")
    func tightPerCallCapIsRaised() throws {
        let sent = try body(maxTokens: 64_000, budget: 8192, overrideMax: 512)
        let pair = try #require(pairing(sent))
        #expect(pair.budget == 8192, "the configured depth survives a tight per-call cap")
        #expect(pair.maxTokens > pair.budget)
        #expect(pair.maxTokens <= 64_000)
    }

    @Test("A model cap too small for any legal budget emits no thinking block")
    func noRoomEmitsNothing() throws {
        // Cap 512 leaves 511 tokens of room — below the 1024 floor, so no legal budget exists.
        let sent = try body(maxTokens: 512, budget: 8192, modelCap: 512)
        #expect(sent["thinking"] == nil, "a rejected request is worse than no thinking")
    }
}
