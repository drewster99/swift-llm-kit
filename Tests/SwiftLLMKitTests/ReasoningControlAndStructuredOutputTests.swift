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
                                   capabilities: ModelCapabilities([.reasoningCanBeEnabled]))
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
                                          capabilities: ModelCapabilities([.reasoningCanBeEnabled]))
        let body = onlyEnableable.buildRequestBody(messages: [.user("hi")], tools: [],
                                                   overrides: LLMCallOverrides(reasoningEnabled: false))
        #expect(body["thinking"] == nil, "enable-able must not imply disable-able")

        let disableable = try provider(control: .thinkingBlock,
                                       capabilities: ModelCapabilities([.reasoningCanBeDisabled]))
        let ok = disableable.buildRequestBody(messages: [.user("hi")], tools: [],
                                              overrides: LLMCallOverrides(reasoningEnabled: false))
        #expect((ok["thinking"] as? [String: Any])?["type"] as? String == "disabled")
    }

    @Test("thinkingBlock: keep and budget ride the same object, each on its own capability")
    func thinkingBlockKeepAndBudget() throws {
        let caps = ModelCapabilities([.reasoningCanBeEnabled, .thinkingSupportsKeepAll, .thinkingSupportsTokenBudget])
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
                                  capabilities: ModelCapabilities([.reasoningCanBeEnabled]))
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
                                 capabilities: ModelCapabilities([.reasoningCanBeEnabled]))
            let body = p.buildRequestBody(messages: [.user("hi")], tools: [],
                                          overrides: LLMCallOverrides(reasoningEnabled: true))
            #expect(body["thinking"] == nil, "\(control) has no on/off knob")
            #expect(body["enable_thinking"] == nil)
        }
    }

    // MARK: Structured output

    @Test("response_format is emitted only for the mode the model actually supports")
    func structuredOutputGating() throws {
        let objectOnly = try provider(capabilities: ModelCapabilities([.structuredOutputSupportsJSONObject]))
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
        let p = try provider(capabilities: ModelCapabilities([.structuredOutputSupportsJSONSchema]))
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

        let supported = try provider(capabilities: ModelCapabilities([.toolDefinitionsSupportStrict]))
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
            minThinkingBudgetTokens: 1_024,
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
            minThinkingBudgetTokens: 1_024,
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
        let body = try gemini(thinkingBudget: 4096, capabilities: ModelCapabilities([.thinkingSupportsTokenBudget]))
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
            var denied = ModelCapabilities(); denied[.thinkingSupportsTokenBudget] = false; return denied
        }()] {
            let body = try gemini(thinkingBudget: 4096, capabilities: capabilities)
                .buildRequestBody(messages: [.user("hi")], tools: [])
            let config = try #require(body["generationConfig"] as? [String: Any])
            #expect(config["thinkingConfig"] == nil)
        }
    }

    @Test("Gemini: an explicit off sends budget 0 on a supporting model")
    func geminiExplicitOff() throws {
        let off = try gemini(thinkingBudget: 4096, capabilities: ModelCapabilities([.thinkingSupportsTokenBudget]))
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

/// `prepareRequest` builds the same request as `AnthropicProvider` for callers that finish the body
/// themselves, so the two must agree. They did not: this path had NO pairing at all — it could emit
/// `max_tokens == budget_tokens`, which Anthropic rejects — and once `ThinkingBudget.effective`
/// became optional it was putting an `Int?` into `[String: Any]`, serializing `budget_tokens: null`.
@Suite("ThinkingBudget.pairing is the one rule both Anthropic paths use")
struct ThinkingBudgetPairingTests {

    @Test("The pair is always legal: max_tokens strictly exceeds the budget")
    func pairIsAlwaysLegal() {
        let cases: [(budget: Int, max: Int, cap: Int?)] = [
            (4096, 4096, nil), (4096, 4096, 4096), (8192, 4096, 4096),
            (1024, 64_000, 64_000), (32_000, 2048, nil), (100, 4096, nil)
        ]
        for c in cases {
            let pair = ThinkingBudget.pairing(requestedBudget: c.budget, requestedMax: c.max,
                                              modelMaxOutputTokens: c.cap)
            if let budget = pair.budget {
                #expect(pair.maxTokens > budget,
                        "illegal pairing for \(c): max_tokens=\(pair.maxTokens) budget=\(budget)")
                #expect(budget >= ThinkingBudget.minimumTokens, "below Anthropic's floor for \(c)")
            }
            if let cap = c.cap {
                #expect(pair.maxTokens <= max(cap, c.max), "exceeded the model cap for \(c)")
            }
        }
    }

    @Test("A known cap degrades the budget; an unknown cap raises max_tokens")
    func capDecidesWhichSideGives() {
        let known = ThinkingBudget.pairing(requestedBudget: 8192, requestedMax: 4096,
                                           modelMaxOutputTokens: 4096)
        #expect(known.maxTokens == 4096)
        #expect(known.budget == 4095, "the budget gives way under a known cap")

        let unknown = ThinkingBudget.pairing(requestedBudget: 8192, requestedMax: 4096,
                                             modelMaxOutputTokens: nil)
        #expect(unknown.maxTokens == 8193, "raised, because refusing would break uncatalogued models")
        #expect(unknown.budget == 8192)
    }

    @Test("No room for a legal budget yields nil rather than an illegal one")
    func noRoomYieldsNil() {
        let pair = ThinkingBudget.pairing(requestedBudget: 8192, requestedMax: 512,
                                          modelMaxOutputTokens: 512)
        #expect(pair.budget == nil, "511 tokens of room is below the 1024 floor")
    }

    @Test("A measured ceiling of zero means no budget at all")
    func measuredZeroMeansNone() {
        let pair = ThinkingBudget.pairing(requestedBudget: 8192, requestedMax: 64_000,
                                          modelMaxOutputTokens: 64_000, measuredMaximum: 0)
        #expect(pair.budget == nil, "a probed 'nothing works' must not be overridden by the floor")
    }
}

/// The wire shapes are derived from ONE source per type now. They were written twice — a probe
/// forcing one representation while production emitted the other would measure a shape that never
/// ships, and the finding exports as data the emission gates then act on.
@Suite("Wire shapes have one source")
struct WireShapeConsolidationTests {

    @Test("LLMResponseFormat's two representations are the same shape")
    func responseFormatRepresentationsAgree() throws {
        for mode: LLMResponseFormat in [.jsonObject,
                                        .jsonSchema(name: "s", schema: ["type": .string("object")])] {
            let forced = try #require(mode.forcedWireValue.rawValue as? [String: Any])
            let production = mode.openAIWireValue
            #expect(NSDictionary(dictionary: forced) == NSDictionary(dictionary: production),
                    "\(mode) encodes differently for the probe than for production")
        }
    }

    /// Anthropic's `tool_choice` is an OBJECT and spells "force some tool" as `any`. Forcing
    /// OpenAI's bare `"required"` there is rejected for the SHAPE, and the rejection names
    /// `tool_choice` — so it would be recorded as "cannot force a tool call", which is wrong.
    @Test("Each provider family gets its own tool_choice shape")
    func toolChoiceShapeIsPerProvider() throws {
        let openAI = try #require(LLMToolChoice.required.wireValue(for: .openAICompatible))
        #expect(openAI.rawValue as? String == "required")

        let anthropic = try #require(LLMToolChoice.required.wireValue(for: .anthropic))
        let object = try #require(anthropic.rawValue as? [String: Any])
        #expect(object["type"] as? String == "any", "Anthropic spells it `any`, and as an object")

        #expect(LLMToolChoice.required.wireValue(for: .gemini) == nil,
                "Gemini has no tool_choice field at all — nothing to force")
    }

    @Test("Every option maps to its own capability, and .auto rides the general one")
    func capabilityPerOption() {
        #expect(LLMToolChoice.auto.requiredCapability == .toolChoiceSupported)
        #expect(LLMToolChoice.required.requiredCapability == .toolChoiceSupportsValueRequired)
        #expect(LLMToolChoice.textOnly.requiredCapability == .toolChoiceSupportsValueNone)
        #expect(LLMToolChoice.specific(name: "t").requiredCapability == .toolChoiceSupportsNamedFunction)
    }

    /// `.toolChoiceSupported` is written by the decoders as "accepts the PARAMETER", so a stated NO must
    /// suppress every option — not just `auto`.
    @Test("A stated NO on the tool_choice parameter suppresses all four options")
    func parameterLevelNoSuppressesEverything() throws {
        var caps = ModelCapabilities()
        caps[.toolChoiceSupported] = false
        caps[.toolChoiceSupportsValueRequired] = true          // option says yes, parameter says no
        let provider = OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(id: "p", name: "p", apiType: .openAICompatible,
                                    endpoint: try #require(URL(string: "https://x.test/v1"))),
            readAPIKey: { "" }, modelCapabilities: caps)
        let body = provider.buildRequestBody(
            messages: [.user("hi")], tools: [CapabilityProbe.makeProbeTool()],
            overrides: LLMCallOverrides(toolChoice: .required))
        #expect(body["tool_choice"] == nil)
    }
}

/// Anthropic and Ollama took no `modelCapabilities` at all, so four gates the rest of the library
/// applies were structurally unreachable on those routes — a fail-closed doctrine with two holes.
@Suite("Every provider route applies the capability gates")
struct UniformGatingTests {

    private func anthropic(_ caps: ModelCapabilities, thinkingBudget: Int? = nil) throws -> AnthropicProvider {
        AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m",
                                              thinkingBudget: thinkingBudget),
            provider: ModelProvider(id: "p", name: "p", apiType: .anthropic,
                                    endpoint: try #require(URL(string: "https://api.anthropic.com"))),
            readAPIKey: { "" }, modelCapabilities: caps)
    }

    private func ollama(_ caps: ModelCapabilities) throws -> OllamaProvider {
        OllamaProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(id: "p", name: "p", apiType: .ollama,
                                    endpoint: try #require(URL(string: "http://localhost:11434/api"))),
            readAPIKey: { "" }, modelCapabilities: caps)
    }

    private func denying(_ capability: ModelCapability) -> ModelCapabilities {
        var caps = ModelCapabilities(); caps[capability] = false; return caps
    }

    @Test("Anthropic honours a stated NO on a tool_choice option")
    func anthropicGatesToolChoice() throws {
        let tool = CapabilityProbe.makeProbeTool()
        let denied = try anthropic(denying(.toolChoiceSupportsValueRequired)).buildRequestBody(
            messages: [.user("hi")], tools: [tool], toolChoice: .required)
        #expect(denied["tool_choice"] == nil)

        // Unknown still emits — the gate fails open, as it does everywhere else.
        let unknown = try anthropic(ModelCapabilities()).buildRequestBody(
            messages: [.user("hi")], tools: [tool], toolChoice: .required)
        let shape = try #require(unknown["tool_choice"] as? [String: Any])
        #expect(shape["type"] as? String == "any", "and in Anthropic's own shape")
    }

    @Test("Ollama honours a stated NO on a tool_choice option")
    func ollamaGatesToolChoice() throws {
        let tool = CapabilityProbe.makeProbeTool()
        let denied = try ollama(denying(.toolChoiceSupportsValueRequired)).buildRequestBody(
            messages: [.user("hi")], tools: [tool], toolChoice: .required)
        #expect(denied["tool_choice"] == nil)

        let unknown = try ollama(ModelCapabilities()).buildRequestBody(
            messages: [.user("hi")], tools: [tool], toolChoice: .required)
        #expect(unknown["tool_choice"] as? String == "required")
    }

    @Test("Anthropic honours a stated NO on the thinking budget")
    func anthropicGatesThinkingBudget() throws {
        let denied = try anthropic(denying(.thinkingSupportsTokenBudget), thinkingBudget: 4096)
            .buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(denied["thinking"] == nil)

        // Unknown keeps emitting: no Anthropic model has this recorded, so requiring known-true
        // would switch manual thinking off everywhere.
        let unknown = try anthropic(ModelCapabilities(), thinkingBudget: 4096)
            .buildRequestBody(messages: [.user("hi")], tools: [])
        #expect((unknown["thinking"] as? [String: Any])?["budget_tokens"] as? Int == 4096)
    }

    /// The parameter-level NO is a precondition over every option, on every route.
    @Test("A stated NO on the parameter suppresses all four options everywhere")
    func parameterLevelNoIsUniform() throws {
        let caps = denying(.toolChoiceSupported)
        let tool = CapabilityProbe.makeProbeTool()
        for choice: LLMToolChoice in [.auto, .required, .textOnly, .specific(name: "x")] {
            #expect(!caps.permitsToolChoice(choice), "\(choice) should be suppressed")
        }
        #expect(try anthropic(caps).buildRequestBody(messages: [.user("hi")], tools: [tool],
                                                     toolChoice: .required)["tool_choice"] == nil)
        #expect(try ollama(caps).buildRequestBody(messages: [.user("hi")], tools: [tool],
                                                  toolChoice: .required)["tool_choice"] == nil)
    }

    /// The probe must not claim it FORCED a call through a field the provider suppressed.
    @Test("permitsToolChoice is the one rule the probe and the providers share")
    func probeAndProvidersShareTheRule() {
        #expect(ModelCapabilities().permitsToolChoice(.required), "unknown permits")
        #expect(!denying(.toolChoiceSupportsValueRequired).permitsToolChoice(.required))
        #expect(!denying(.toolChoiceSupported).permitsToolChoice(.required), "the parameter-level veto wins")
    }
}

/// The per-provider `tool_choice` encoders were deleted when the shape moved onto `LLMToolChoice`.
/// These pin the exact bytes each family expects, so that consolidation cannot have quietly changed
/// what goes on the wire — and so a future edit to the enum cannot either.
@Suite("tool_choice wire bytes, per provider family")
struct ToolChoiceWireByteTests {

    @Test("Anthropic's shapes are objects, and 'force some tool' is `any`")
    func anthropicBytes() throws {
        let expected: [(LLMToolChoice, [String: String])] = [
            (.auto, ["type": "auto"]),
            (.required, ["type": "any"]),
            (.textOnly, ["type": "none"]),
            (.specific(name: "get_x"), ["type": "tool", "name": "get_x"])
        ]
        for (choice, want) in expected {
            let got = try #require(choice.anthropicWireValue.rawValue as? [String: Any])
            #expect(NSDictionary(dictionary: got) == NSDictionary(dictionary: want),
                    "\(choice) encoded as \(got), expected \(want)")
        }
    }

    @Test("The OpenAI family uses enum strings, and an object only for a named function")
    func openAIBytes() throws {
        #expect(LLMToolChoice.auto.openAIWireValue.rawValue as? String == "auto")
        #expect(LLMToolChoice.required.openAIWireValue.rawValue as? String == "required")
        #expect(LLMToolChoice.textOnly.openAIWireValue.rawValue as? String == "none",
                "OpenAI spells 'no tools' as `none`, not `textOnly`")
        let specific = try #require(
            LLMToolChoice.specific(name: "get_x").openAIWireValue.rawValue as? [String: Any])
        #expect(specific["type"] as? String == "function")
        #expect((specific["function"] as? [String: Any])?["name"] as? String == "get_x")
    }

    /// Ollama follows the OpenAI shape — its deleted encoder was byte-identical, and its own
    /// comment said so. Pinned because "identical today" is not the same as "identical tomorrow".
    @Test("Ollama resolves to the OpenAI shape, not a copy of it")
    func ollamaMatchesOpenAI() throws {
        for choice: LLMToolChoice in [.auto, .required, .textOnly, .specific(name: "get_x")] {
            let ollama = try #require(choice.wireValue(for: .ollama))
            #expect(ollama == choice.openAIWireValue, "\(choice) diverged from the OpenAI shape")
        }
    }
}

/// A measured floor has to reach the wire, or measuring it was decoration. `ThinkingBudget
/// .minimumTokens` is the DOCUMENTED floor and now only a fallback — these pin that a per-model
/// measurement supersedes it in both directions, which is the whole point of probing for one.
@Suite("A measured minimum budget supersedes the documented floor")
struct MeasuredMinimumBudgetTests {

    @Test("A higher measured floor raises a request that the constant would have under-sent")
    func higherFloorRaisesTheBudget() {
        // The case the probe exists for: production floors at 1024, this model rejects below 4096.
        #expect(ThinkingBudget.effective(500, measuredMinimum: 4096) == 4096)
        #expect(ThinkingBudget.effective(500) == ThinkingBudget.minimumTokens,
                "without a measurement the documented floor still governs")
    }

    @Test("A lower measured floor permits a smaller budget than the constant would")
    func lowerFloorPermitsSmallerBudgets() {
        #expect(ThinkingBudget.effective(256, measuredMinimum: 128) == 256)
        #expect(ThinkingBudget.effective(64, measuredMinimum: 128) == 128)
    }

    /// A measured floor of 0 says this endpoint imposes no minimum. Treating small values as
    /// missing data would silently reinstate the constant on exactly the models that disproved it.
    @Test("A measured floor of zero is a measurement, not missing data")
    func zeroFloorIsHonoured() {
        #expect(ThinkingBudget.effective(1, measuredMinimum: 0) == 1)
    }

    @Test("A maximum below the measured floor leaves no usable budget")
    func maximumBelowFloorYieldsNil() {
        #expect(ThinkingBudget.effective(8192, measuredMaximum: 2048, measuredMinimum: 4096) == nil)
        #expect(ThinkingBudget.effective(8192, measuredMaximum: 2048) == 2048,
                "the same maximum is fine against the documented floor")
    }

    /// `pairing` gates on having room for a legal budget. Comparing that room against the CONSTANT
    /// while `effective` raised to a measured floor would emit a budget below the model's real
    /// minimum whenever the two disagree — the exact rejection the measurement prevents.
    @Test("The pairing's room check uses the same floor the budget was raised to")
    func pairingRoomUsesTheMeasuredFloor() {
        // Room is 2047, which clears the documented 1024 but not this model's measured 4096.
        let pair = ThinkingBudget.pairing(requestedBudget: 8192, requestedMax: 2048,
                                          modelMaxOutputTokens: 2048, measuredMinimum: 4096)
        #expect(pair.budget == nil, "no legal budget fits, so none is sent")

        let withoutMeasurement = ThinkingBudget.pairing(requestedBudget: 8192, requestedMax: 2048,
                                                        modelMaxOutputTokens: 2048)
        #expect(withoutMeasurement.budget == 2047)
    }

    @Test("A paired max_tokens still clears a raised measured floor")
    func pairingRaisesMaxAboveTheMeasuredFloor() {
        let pair = ThinkingBudget.pairing(requestedBudget: 100, requestedMax: 512,
                                          modelMaxOutputTokens: nil, measuredMinimum: 4096)
        #expect(pair.budget == 4096)
        #expect(pair.maxTokens > 4096, "max_tokens must still exceed the budget actually sent")
    }
}

/// The two reasoning switches are probed separately and neither implies the other; this is what
/// they establish TOGETHER. Pure inference, so every branch is reachable without a network — which
/// matters because this is the part that can record a wrong answer about a model.
@Suite("What the two reasoning switch directions conclude")
struct ReasoningConclusionTests {

    private func observation(accepted: Bool?, emitted: Bool?, tokens: Int = 0)
        -> ModelProber.ReasoningToggleObservation {
        let finding: ProbeFinding<Bool>
        switch accepted {
        case true?:  finding = .established(true, "accepted")
        case false?: finding = .established(false, "refused")
        case nil:    finding = .inconclusive("no answer")
        }
        return .init(finding: finding, reasoningEmitted: emitted, reasoningTokens: tokens)
    }

    @Test("Observing reasoning establishes that the model reasons")
    func observedReasoningEstablishesIt() {
        let c = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 240),
            off: observation(accepted: true, emitted: false))
        #expect(c.reasons.value == true)
        #expect(c.reasons.evidence?.contains("240 thinking tokens") == true)
    }

    /// Anthropic folds thinking into `outputTokens` and reports `reasoningTokens` as 0 by design.
    /// Reading the token count alone would record every Anthropic model as not reasoning.
    @Test("Reasoning TEXT with zero billed tokens still establishes it")
    func reasoningTextCountsWithoutTokens() {
        let c = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 0),
            off: observation(accepted: true, emitted: false))
        #expect(c.reasons.value == true)
        #expect(c.reasons.evidence?.contains("reasoning content returned") == true,
                "must not claim 0 thinking tokens as the evidence")
    }

    /// A model that thinks even when told not to still demonstrably thinks.
    @Test("Reasoning seen in EITHER direction establishes it")
    func eitherDirectionCounts() {
        let c = ModelProber.concludeReasoning(
            on: observation(accepted: false, emitted: nil),
            off: observation(accepted: true, emitted: true, tokens: 10))
        #expect(c.reasons.value == true)
    }

    /// Silence has three causes and this probe cannot tell them apart, so it must not pick one.
    @Test("Never establishes that a model does NOT reason")
    func absenceIsNeverAProof() {
        let c = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: false),
            off: observation(accepted: true, emitted: false))
        #expect(c.reasons.status == .inconclusive)
        #expect(c.reasons.value == nil)
    }

    /// The defect this work exists to catch: `thinking` is an unknown key to most OpenAI-compatible
    /// endpoints, and unknown keys are IGNORED rather than refused — so acceptance-grading recorded
    /// "reasoning can be turned off" for a model that carried right on thinking.
    @Test("A switch that is accepted and then ignored is established FALSE")
    func acceptedButIgnoredIsFalse() {
        let c = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 500),
            off: observation(accepted: true, emitted: true, tokens: 480))
        #expect(c.canBeDisabled.value == false)
        #expect(c.canBeDisabled.evidence?.contains("ignored") == true)
    }

    @Test("A switch that demonstrably stopped the reasoning is established TRUE")
    func reasoningStoppedIsTrue() {
        let c = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 500),
            off: observation(accepted: true, emitted: false))
        #expect(c.canBeDisabled.value == true)
    }

    /// REAL DATA, and the case a test against zero got wrong. kimi-k2.6 answers the trivial probe
    /// prompt with 32 thinking tokens enabled and 1 disabled — the switch plainly worked, but
    /// `tokens > 0` read that residual token as "still thinking" and recorded canBeDisabled=false.
    @Test("A residual token against a real baseline is a working switch, not an ignored one")
    func residualTokenIsNotStillThinking() {
        let c = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 32),
            off: observation(accepted: true, emitted: true, tokens: 1))   // kimi-k2.6, 2026-08-03
        #expect(c.canBeDisabled.value == true, "1 of 32 tokens is a switch that worked")
        #expect(c.canBeDisabled.evidence?.contains("collapsed") == true)
    }

    /// A partial reduction is a switch that WORKED. It reads as `inconclusive` only until the
    /// verdict is traced into production: an unestablished capability does not project, the
    /// provider gates emission on a known true, and the field would then not be sent at all — so a
    /// model whose thinking the switch measurably reduced would go back to thinking in full, for
    /// the user who explicitly asked for reasoning off. That is the worse of the two errors.
    @Test("A switch that only reduced the thinking still counts as working")
    func partialReductionStillCountsAsWorking() {
        let c = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 100),
            off: observation(accepted: true, emitted: true, tokens: 30))
        #expect(c.canBeDisabled.value == true)
        #expect(c.canBeDisabled.evidence?.contains("reduced but not eliminated") == true,
                "the ambiguity belongs in the evidence, not in the verdict")
    }

    /// The boundary itself, both sides — the one number that decides ignored-versus-worked.
    @Test("Half the baseline is the line between worked and ignored")
    func theThresholdHoldsOnBothSides() {
        let atHalf = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 100),
            off: observation(accepted: true, emitted: true, tokens: 50))
        #expect(atHalf.canBeDisabled.value == false, "at the line, the switch changed nothing worth counting")

        let justUnder = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 100),
            off: observation(accepted: true, emitted: true, tokens: 49))
        #expect(justUnder.canBeDisabled.value == true)
    }

    /// Anthropic reports `reasoningTokens` as 0 by design (thinking is folded into outputTokens),
    /// so there is no baseline to ratio against and the signal is binary.
    @Test("With no token baseline the thinking TEXT decides it, both ways")
    func textOnlyProvidersAreGradedBinary() {
        let stillThinking = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 0),
            off: observation(accepted: true, emitted: true, tokens: 0))
        #expect(stillThinking.canBeDisabled.value == false)

        let stopped = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 0),
            off: observation(accepted: true, emitted: false, tokens: 0))
        #expect(stopped.canBeDisabled.value == true)
    }

    /// Visible thinking with nothing billed is the endpoint ignoring the switch, not the thinking
    /// collapsing — a ratio of 0/baseline would otherwise read it as a clean success.
    @Test("Thinking text with zero billed tokens is not ratio'd away")
    func textWithoutTokensIsNotACollapse() {
        let c = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 200),
            off: observation(accepted: true, emitted: true, tokens: 0))
        #expect(c.canBeDisabled.value == false)
        #expect(c.canBeDisabled.evidence?.contains("no thinking tokens were billed") == true)
    }

    /// With no reasoning visible when ON there is no baseline, so its absence when OFF says nothing
    /// about the switch. The acceptance answer has to stand rather than be upgraded on no evidence.
    @Test("Without a baseline the acceptance answer is passed through untouched")
    func noBaselineKeepsTheAcceptanceAnswer() {
        let refused = observation(accepted: false, emitted: nil)
        let c = ModelProber.concludeReasoning(on: observation(accepted: true, emitted: false), off: refused)
        #expect(c.canBeDisabled.value == false)
        #expect(c.canBeDisabled.evidence == "refused", "passed through, not re-derived")

        let unanswered = observation(accepted: nil, emitted: nil)
        let c2 = ModelProber.concludeReasoning(on: observation(accepted: true, emitted: false), off: unanswered)
        #expect(c2.canBeDisabled.status == .inconclusive)
    }

    /// An OFF direction that was never accepted cannot be upgraded to a true by the absence of
    /// reasoning — nothing was disabled, the call was refused.
    @Test("A refused OFF switch is not upgraded by a quiet reply")
    func refusedSwitchIsNotUpgraded() {
        let c = ModelProber.concludeReasoning(
            on: observation(accepted: true, emitted: true, tokens: 100),
            off: observation(accepted: false, emitted: nil))
        #expect(c.canBeDisabled.value == false)
        #expect(c.canBeDisabled.evidence == "refused")
    }
}

/// Reasoning is switched a different way at every vendor, and `apiType` cannot decide which —
/// OpenAI, Moonshot and DeepSeek are all `.openAICompatible` and do not agree. So the mechanism is
/// discovered. Asking everyone for a `thinking` block earned OpenAI's "Unknown parameter" and
/// recorded 60+ reasoning models as unable to reason.
@Suite("Discovering how an endpoint switches reasoning")
struct ReasoningMechanismDiscoveryTests {

    /// Answers each forced body the way a named vendor would, so the search is exercised end to end.
    private struct Endpoint: LLMProvider, @unchecked Sendable {
        let forced: [String: AnyCodable]
        let accepts: @Sendable (String) -> Bool          // which top-level key this vendor takes
        let emitsReasoning: @Sendable (String) -> Bool   // and which one actually makes it think
        let seen: Recorder
        final class Recorder: @unchecked Sendable { var keys: [String] = [] }

        func send(messages: [LLMMessage], tools: [LLMToolDefinition],
                  overrides: LLMCallOverrides) async throws -> LLMResponse {
            let key = forced.keys.sorted().first { $0 != "max_tokens" } ?? ""
            seen.keys.append(key)
            guard accepts(key) else {
                throw LLMProviderError.httpError(statusCode: 400, body: "Unknown parameter: '\(key)'.")
            }
            let thinking = emitsReasoning(key)
            return LLMResponse(text: "ok", reasoning: thinking ? "thought about it" : nil,
                               usage: TokenUsage(inputTokens: 5, outputTokens: 2,
                                                 reasoningTokens: thinking ? 64 : 0,
                                                 cacheReadTokens: 0, cacheWriteTokens: 0))
        }
    }

    private func discover(apiType: ProviderAPIType, accepts: @escaping @Sendable (String) -> Bool,
                          emits: @escaping @Sendable (String) -> Bool)
        async -> (ModelProber.ReasoningMechanismFindings, [String]) {
        let rec = Endpoint.Recorder()
        let found = await ModelProber.probeReasoningMechanism(
            apiType: apiType,
            makeProviderForcing: { Endpoint(forced: $0, accepts: accepts, emitsReasoning: emits, seen: rec) })
        return (found, rec.keys)
    }

    /// The regression this exists for. OpenAI has no `thinking` block; it reasons via
    /// `reasoning_effort`, and the old probe recorded that as "cannot enable reasoning".
    @Test("An OpenAI reasoning model is found via reasoning_effort, not written off")
    func openAIReasoningModelIsMeasured() async {
        let (found, keys) = await discover(apiType: .openAICompatible,
                                           accepts: { $0 == "reasoning_effort" },
                                           emits: { $0 == "reasoning_effort" })
        #expect(found.control == .reasoningEffortOnly)
        #expect(found.on.finding.value == true, "the old probe recorded false here")
        #expect(keys.contains("thinking"), "it still tries the block first")
        #expect(keys.contains("reasoning_effort"))
    }

    /// Moonshot accepts BOTH, and the block is the more specific mechanism — `thinking.keep` and
    /// the token budget hang off it, so recording effort-only would lose them.
    @Test("A model that accepts both is recorded as the thinking-block model it is")
    func thinkingBlockWinsWhenBothWork() async {
        let (found, keys) = await discover(apiType: .openAICompatible,
                                           accepts: { _ in true }, emits: { _ in true })
        #expect(found.control == .thinkingBlock)
        #expect(keys.first == "thinking")
        #expect(keys.filter { $0 == "thinking" }.count == 2, "on and off through the same mechanism")
    }

    /// Acceptance is cheap — an endpoint that ignores unknown keys accepts everything — so a
    /// candidate that actually produces thinking beats one that merely did not error.
    @Test("A mechanism that demonstrably thinks beats one that was merely accepted")
    func demonstratedBeatsAccepted() async {
        let (found, _) = await discover(apiType: .openAICompatible,
                                        accepts: { _ in true },
                                        emits: { $0 == "reasoning_effort" })
        #expect(found.control == .reasoningEffortOnly)
        #expect(found.on.reasoningEmitted == true)
    }

    @Test("Anthropic is asked its own way, with the budget its API requires")
    func anthropicUsesItsOwnShape() async {
        let (found, keys) = await discover(apiType: .anthropic,
                                           accepts: { $0 == "thinking" }, emits: { _ in true })
        #expect(found.control == .anthropicThinking)
        #expect(keys.allSatisfy { $0 == "thinking" }, "no other mechanism is tried at Anthropic")
    }

    /// A refusal from every mechanism is a real no — this is what should happen for gpt-4-turbo.
    @Test("Refused everywhere is an established no")
    func refusedEverywhereIsFalse() async {
        let (found, _) = await discover(apiType: .openAICompatible,
                                        accepts: { _ in false }, emits: { _ in false })
        #expect(found.control == nil)
        #expect(found.on.finding.value == false)
        // Every mechanism asked and refused settles BOTH directions: there is no switch here to
        // turn off. Reporting this inconclusive left `reasoningCanBeDisabled` nil, and a caller
        // reading `?.value != false` then treated a model with no reasoning at all as one whose
        // reasoning might be disableable — which is how the tool-choice probe came to force a
        // `thinking` block at endpoints that have none.
        #expect(found.off.finding.value == false)
        #expect(found.mechanismWasDemonstrated == false)
    }
}

/// The budget probes must FORCE their field. Going through production emission meant the value
/// probed was not the value sent — the defect that put fabricated budgets in the shipped catalog.
@Suite("A budget probe sends what it asks for")
struct BudgetForcingOverrideTests {

    @Test("Mechanisms that carry a token budget produce one, and those that don't produce none")
    func onlyBudgetCarryingMechanismsForce() {
        for control in ReasoningControl.allCases {
            let forced = control.budgetForcingOverrides(budget: 4096, pairedMaxTokens: 5120)
            #expect((forced != nil) == control.carriesTokenBudget,
                    "\(control.rawValue): forcing and carriesTokenBudget must agree")
        }
    }

    /// The value has to arrive verbatim: `ThinkingBudget.effective` floors anything below 1024, so a
    /// probe asking 1023 through production sent 1024 and could never measure the real floor.
    @Test("The requested budget reaches the body unfloored")
    func budgetIsNotFloored() {
        let forced = ReasoningControl.anthropicThinking.budgetForcingOverrides(budget: 1, pairedMaxTokens: 1025)
        guard case .dictionary(let thinking)? = forced?["thinking"] else {
            Issue.record("no thinking block"); return
        }
        #expect(thinking["budget_tokens"] == .int(1), "a floored 1024 here is the circularity")
        #expect(forced?["max_tokens"] == .int(1025))
    }

    /// A separate allowance needs no pairing; forcing one would cap the reply rather than the budget.
    @Test("Only output-drawn mechanisms pair a max_tokens")
    func onlyDrawnMechanismsPair() {
        #expect(ReasoningControl.thinkingBlock
            .budgetForcingOverrides(budget: 100, pairedMaxTokens: 200)?["max_tokens"] == .int(200))
        #expect(ReasoningControl.enableThinkingFlag
            .budgetForcingOverrides(budget: 100, pairedMaxTokens: 200)?["max_tokens"] == nil)
        #expect(ReasoningControl.geminiThinkingConfig
            .budgetForcingOverrides(budget: 100, pairedMaxTokens: 200)?["max_tokens"] == nil)
    }
}

/// `describesMalformedRequest` must not swallow the vendor's own capability denials.
@Suite("A capability denial is not a malformed request")
struct MalformedRequestPhraseTests {

    private struct Failing: LLMProvider {
        let body: String
        func send(messages: [LLMMessage], tools: [LLMToolDefinition],
                  overrides: LLMCallOverrides) async throws -> LLMResponse {
            throw LLMProviderError.httpError(statusCode: 400, body: body)
        }
    }

    private func toggle(_ body: String) async -> ProbeFinding<Bool> {
        await ModelProber.probeReasoningToggle(enabled: true, makeProviderForcing: { _ in Failing(body: body) })
    }

    /// OpenAI's canonical denial wording. 34 bodies in the local corpus use it, every one a real
    /// fact — listing it as "malformed" downgraded exactly the answers these probes collect.
    @Test("'Unsupported parameter' is a denial and still establishes false")
    func unsupportedParameterIsADenial() async {
        let body = #"{"error":{"message":"Unsupported parameter: 'thinking' is not supported with this model."}}"#
        #expect(await toggle(body).value == false)
    }

    /// The genuinely-malformed cases must still be caught.
    @Test("A missing required field is still inconclusive")
    func missingFieldStillInconclusive() async {
        let body = #"{"error":{"message":"thinking.enabled.budget_tokens: Field required"}}"#
        #expect(await toggle(body).status == .inconclusive)
    }
}
