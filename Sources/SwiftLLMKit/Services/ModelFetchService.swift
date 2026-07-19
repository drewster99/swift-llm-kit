import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "ModelFetch")

/// Queries provider APIs for available model lists.
public struct ModelFetchService: Sendable {
    /// When true, the full request line and response JSON are logged via ``LLMRequestLogger``,
    /// into whichever directory it is configured to use — the same one the chat traffic uses.
    public nonisolated(unsafe) static var verboseLogging = false

    public init() {}

    /// Fetches available models from a provider endpoint.
    /// - Parameters:
    ///   - provider: The provider to query.
    ///   - apiKey: The API key for authentication (from Keychain).
    /// - Returns: Array of `ModelInfo` with `providerID` populated.
    ///
    /// Convenience over ``fetchModelFacts(from:apiKey:)``: materializes each tri-state record into
    /// the flattened `ModelInfo` shape (unknown capability → `false`, unknown chat → `true`).
    /// Callers that care about the stated/unknown distinction — the merge pipeline, the probe
    /// seeder — should use the facts variant instead.
    public func fetchModels(
        from provider: ModelProvider,
        apiKey: String?
    ) async throws -> [ModelInfo] {
        try await fetchModelFacts(from: provider, apiKey: apiKey)
            .map { $0.facts.materialize(providerID: provider.id, modelID: $0.modelID) }
    }

    /// Fetches the provider's `/models` and decodes it into per-model ``ModelFacts`` — the
    /// tri-state records where `nil` means "the vendor did not say", never "no".
    ///
    /// This is the authoritative layer's source of truth: each decoder emits ONLY what its vendor
    /// actually stated, per the stated-facts audit (Anthropic's capabilities block has no tool key,
    /// so `toolUse` stays nil; Mistral states each capability leaf both directions; OpenRouter's
    /// modality/parameter arrays are bidirectional when non-empty; plain OpenAI states almost
    /// nothing). Getting this right is what makes "authoritative wins for fields it supplies" sound.
    public func fetchModelFacts(
        from provider: ModelProvider,
        apiKey: String?
    ) async throws -> [DecodedModelFacts] {
        let modelsURL: URL
        switch provider.apiType {
        case .ollama:
            modelsURL = provider.endpoint.appendingPathComponent("tags")
        case .anthropic:
            modelsURL = provider.endpoint.strippingAnthropicV1().appendingPathComponent("v1/models")
        case .openAICompatible, .lmStudio, .mistral, .huggingFace, .xAI, .zAI, .metaModel, .alibabaCloud, .openRouter:
            modelsURL = provider.endpoint.appendingPathComponent("models")
        case .gemini:
            // API key goes in the `x-goog-api-key` header below (not the query
            // string) so it doesn't appear in verbose request logs.
            let base = provider.endpoint.appendingPathComponent("models")
            if var components = URLComponents(url: base, resolvingAgainstBaseURL: false) {
                var items = components.queryItems ?? []
                // Default page size is 50; request max to avoid missing models
                items.append(URLQueryItem(name: "pageSize", value: "1000"))
                components.queryItems = items
                modelsURL = components.url ?? base
            } else {
                modelsURL = base
            }
        }

        var request = URLRequest(url: modelsURL)
        request.httpMethod = "GET"
        request.timeoutInterval = 15

        switch provider.apiType {
        case .ollama:
            if let apiKey, !apiKey.isEmpty {
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            }
        case .anthropic:
            if let apiKey {
                request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
            }
            request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        case .openAICompatible, .lmStudio, .mistral, .huggingFace, .xAI, .zAI, .metaModel, .alibabaCloud, .openRouter:
            if let apiKey, !apiKey.isEmpty {
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            }
            if provider.apiType == .zAI {
                request.setValue("en-US,en", forHTTPHeaderField: "Accept-Language")
            }
        case .gemini:
            if let apiKey, !apiKey.isEmpty {
                request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")
            }
        }

        // Logged through LLMRequestLogger so model calls land in the same directory and timeline
        // as chat traffic. They used to write to a directory of their own, which made them look
        // absent rather than merely elsewhere.
        let label = "ModelFetch_\(provider.name.replacingOccurrences(of: " ", with: "_"))"
        logger.debug("Model fetch: GET \(modelsURL.absoluteString, privacy: .public)")
        if Self.verboseLogging {
            LLMRequestLogger.logBodylessRequest(label: label, method: "GET", url: modelsURL)
        }

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            let body = String(data: data, encoding: .utf8) ?? "(non-utf8)"
            logger.error("Model fetch failed: HTTP \(code, privacy: .public) body=\(body, privacy: .private)")
            if Self.verboseLogging {
                LLMRequestLogger.logResponse(label: "\(label)_error", statusCode: code, data: data)
            }
            throw ModelFetchError.httpError(statusCode: code)
        }

        if Self.verboseLogging {
            LLMRequestLogger.logResponse(label: label, statusCode: http.statusCode, data: data)
        }

        let decoded: [DecodedModelFacts]
        switch provider.apiType {
        case .ollama:
            decoded = try decodeOllamaFacts(from: data)
        case .anthropic:
            decoded = try decodeAnthropicFacts(from: data)
        case .xAI:
            decoded = try decodeXAIFacts(from: data)
        case .openRouter:
            decoded = try decodeOpenRouterFacts(from: data)
        case .huggingFace:
            decoded = try decodeHuggingFaceFacts(from: data)
        case .openAICompatible, .lmStudio, .zAI, .metaModel, .alibabaCloud:
            decoded = try decodeOpenAIFacts(from: data)
        case .mistral:
            decoded = try decodeMistralFacts(from: data)
        case .gemini:
            decoded = try decodeGeminiFacts(from: data)
        }

        // Deduplicate by model ID. Some APIs return the same model ID multiple
        // times (e.g. Mistral aliases).
        var seen = Set<String>()
        return decoded.filter { seen.insert($0.modelID).inserted }
    }

    // MARK: - Ollama

    func decodeOllamaModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeOllamaFacts(from: data).map { $0.facts.materialize(providerID: providerID, modelID: $0.modelID) }
    }

    private func decodeOllamaFacts(from data: Data) throws -> [DecodedModelFacts] {
        let decoded = try JSONDecoder().decode(OllamaTagsResponse.self, from: data)
        return decoded.models
            .map { model in
                var facts = ModelFacts()
                // Tool use is POSITIVE-ONLY on purpose: the tags capability list reliably names
                // "tools" when present, but absence is a manifest hint, not a vendor statement of
                // "cannot" — and a wrong stated-false in the authoritative layer would mask a probed
                // true forever under authoritative-wins. Unknown costs one probe; wrong costs a model.
                if let capabilities = model.capabilities, capabilities.contains("tools") {
                    facts.capabilities.toolUse = true
                }
                // Prefer the stated parameter count ("397B") over a byte-size label: `size` is the
                // on-disk weight file in BYTES, and formatBytes renders it with a "B" suffix that
                // reads like a parameter count but isn't. parameter_size is the real thing — but
                // ollama.com states it as "" (present-but-empty, same convention as
                // quantization_level below), which must fall through to the byte label, and an
                // empty byte label (size 0) must record nil, not "".
                let statedSize = model.details?.parameterSize ?? ""
                let byteSizeLabel = formatBytes(model.size)
                facts.sizeLabel = !statedSize.isEmpty ? statedSize
                    : (byteSizeLabel.isEmpty ? nil : byteSizeLabel)
                let quant = model.details?.quantizationLevel ?? ""
                facts.quantizationLabel = quant.isEmpty ? nil : quant
                facts.createdAt = parseISODate(model.modifiedAt)
                // Only local Ollama's payload carries a context window; cloud omits it (nil).
                facts.maxInputTokens = model.details?.contextLength
                return DecodedModelFacts(modelID: model.name, facts: facts)
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - Anthropic

    /// Test seam: exercises the private decoder against captured payload shapes without a fetch.
    func decodeAnthropicModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeAnthropicFacts(from: data).map { $0.facts.materialize(providerID: providerID, modelID: $0.modelID) }
    }

    private func decodeAnthropicFacts(from data: Data) throws -> [DecodedModelFacts] {
        let decoded = try JSONDecoder().decode(AnthropicModelsResponse.self, from: data)
        return decoded.data
            .map { model in
                var facts = ModelFacts()
                facts.displayName = model.displayName
                facts.createdAt = model.createdAt.flatMap { parseISODate($0) }
                facts.maxInputTokens = model.maxInputTokens
                facts.maxOutputTokens = model.maxTokens

                // The vendor describing its own models is the one source with standing to — the
                // capabilities block states each leaf explicitly, BOTH directions, so a present
                // leaf becomes a stated true/false. A missing leaf stays nil ("didn't say"), and
                // tool use is NEVER set: the block has no tool key, and writing false here would
                // mask a probed true forever under authoritative-wins.
                if let capabilities = model.capabilities {
                    facts.capabilities.vision = capabilities.imageInput?.supported
                    facts.capabilities.pdfInput = capabilities.pdfInput?.supported
                    facts.capabilities.reasoning = capabilities.thinking?.supported
                    facts.capabilities.codeExecution = capabilities.codeExecution?.supported
                    facts.capabilities.responseSchema = capabilities.structuredOutputs?.supported

                    // The payload's own per-model level list, ordered by our rank table (the
                    // payload is a JSON object, so it carries no order itself). An effort block
                    // with supported == false is a STATEMENT: "no effort levels" → stated [].
                    if let effort = capabilities.effort, let supported = effort.supported {
                        facts.validEffortLevels = supported
                            ? effort.levels.filter { $0.value }.map(\.key)
                                .sorted { EffortRank.rank(of: $0) < EffortRank.rank(of: $1) }
                            : []
                    }

                    // DERIVED, not hand-listed: a thinking model whose budget_tokens form is
                    // unsupported (`types.enabled == false`) is adaptive-only — and, verified
                    // live 2026-07-17, adaptive-only models also reject the `temperature`
                    // parameter with HTTP 400 ("deprecated for this model"): opus-4-8 rejects,
                    // while sonnet-4-6 / opus-4-5 (enabled == true) accept. The derivation is
                    // sound BOTH directions (Anthropic rejects temperature exactly on
                    // adaptive-only models), so when the leaf is present we state true or false;
                    // when the thinking block or leaf is absent, we say nothing.
                    if capabilities.thinking?.supported == true,
                       let enabledSupported = capabilities.thinking?.types?.enabled?.supported {
                        facts.behaviorFlags.requiresAdaptiveThinking = !enabledSupported
                        facts.behaviorFlags.mustNeverSendTemperatureParam = !enabledSupported
                    }
                }

                return DecodedModelFacts(modelID: model.id, facts: facts)
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - OpenAI Compatible (plain)

    func decodeOpenAIModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeOpenAIFacts(from: data).map { $0.facts.materialize(providerID: providerID, modelID: $0.modelID) }
    }

    private func decodeOpenAIFacts(from data: Data) throws -> [DecodedModelFacts] {
        let decoded = try JSONDecoder().decode(OpenAIModelsResponse.self, from: data)
        return decoded.data
            .map { model in
                var facts = ModelFacts()
                facts.createdAt = model.created.map { Date(timeIntervalSince1970: TimeInterval($0)) }
                // `context_length` and `prompt_image_token_price` are common OpenAI-compatible
                // extensions many providers emit and plain OpenAI omits. A positive image-token
                // price is a vendor-stated vision POSITIVE; its absence proves nothing, so vision
                // is never stated false here. Plain OpenAI states almost nothing — which is the
                // honest record: nearly every fact about these models must come from elsewhere.
                if let imagePrice = model.promptImageTokenPrice, imagePrice > 0 {
                    facts.capabilities.vision = true
                }
                facts.maxInputTokens = model.contextLength
                return DecodedModelFacts(modelID: model.id, facts: facts)
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - xAI

    func decodeXAIModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeXAIFacts(from: data).map { $0.facts.materialize(providerID: providerID, modelID: $0.modelID) }
    }

    /// xAI's `/models` is richer than OpenAI's: a context length, an image-token price (which
    /// implies vision), and per-token prices. Token prices are in xAI's documented unit — "USD
    /// cents per 100 million tokens" — so USD-per-token = value / 1e10. See
    /// https://docs.x.ai/developers/cost-tracking (which also documents `cost_in_usd_ticks`, where
    /// 1 USD = 1e10 ticks). Prices >= 0 only; xAI uses no sentinel, but guard anyway.
    private func decodeXAIFacts(from data: Data) throws -> [DecodedModelFacts] {
        // "USD cents per 100 million tokens" -> USD per single token.
        // Per https://docs.x.ai/developers/cost-tracking.
        func usdPerToken(_ centsPer100M: Double?) -> Double? {
            guard let v = centsPer100M, v >= 0 else { return nil }
            return v / 1e10
        }

        let decoded = try JSONDecoder().decode(XAIModelsResponse.self, from: data)
        return decoded.data
            .map { model in
                var facts = ModelFacts()
                facts.createdAt = model.created.map { Date(timeIntervalSince1970: TimeInterval($0)) }
                facts.maxInputTokens = model.contextLength
                // A positive image-token price means the model prices — and therefore accepts —
                // image input: a vendor-stated vision POSITIVE. Absence proves nothing → stays nil.
                if let imagePrice = model.promptImageTokenPrice, imagePrice > 0 {
                    facts.capabilities.vision = true
                }

                let base = PricingTier(
                    input: usdPerToken(model.promptTextTokenPrice),
                    output: usdPerToken(model.completionTextTokenPrice),
                    cacheRead: usdPerToken(model.cachedPromptTextTokenPrice)
                )
                if base.hasAnyRate {
                    var tiers: [TokenThresholdTier] = []
                    // xAI charges higher rates once the request exceeds long_context_threshold.
                    if let threshold = model.longContextThreshold {
                        let longRates = PricingTier(
                            input: usdPerToken(model.promptTextTokenPriceLongContext),
                            output: usdPerToken(model.completionTextTokenPriceLongContext),
                            cacheRead: usdPerToken(model.cachedPromptTextTokenPriceLongContext)
                        )
                        if longRates.hasAnyRate {
                            tiers.append(TokenThresholdTier(tokenThreshold: threshold, rates: longRates))
                        }
                    }
                    facts.pricing = ModelPricing(base: base, tokenThresholdTiers: tiers)
                }

                // xAI publishes alternate callable IDs per entry (the public name
                // "grok-code-fast-1" is an alias of canonical "grok-build-0.1"; every "-latest"
                // is an alias too — 42 IDs across the 2026-07-18 payload). Mistral ships aliases
                // as their own entries; xAI ships an array, so emit each alias as an entry
                // carrying the same facts or publicly documented IDs resolve to nothing.
                // Cross-entry collisions dedupe upstream (first wins).
                let canonical = DecodedModelFacts(modelID: model.id, facts: facts)
                let aliases = (model.aliases ?? []).map { DecodedModelFacts(modelID: $0, facts: facts) }
                return [canonical] + aliases
            }
            .flatMap { $0 }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - OpenRouter

    func decodeOpenRouterModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeOpenRouterFacts(from: data).map { $0.facts.materialize(providerID: providerID, modelID: $0.modelID) }
    }

    /// OpenRouter's `/models` is the richest of the OpenAI-compatible family: it states input
    /// modalities, the parameters each model supports, per-provider limits, and per-token prices.
    /// The modality and parameter ARRAYS are explicit enumerations, so when non-empty they are
    /// stated BOTH directions (absence from the list = stated no); an absent or empty array says
    /// nothing. Schema: https://openrouter.ai/docs/api-reference/list-available-models
    private func decodeOpenRouterFacts(from data: Data) throws -> [DecodedModelFacts] {
        let decoded = try JSONDecoder().decode(OpenRouterModelsResponse.self, from: data)
        return decoded.data
            .map { model in
                var facts = ModelFacts()
                facts.displayName = model.name
                facts.createdAt = model.created.map { Date(timeIntervalSince1970: TimeInterval($0)) }

                // input_modalities is an explicit list → bidirectional when non-empty.
                let inputs = Set(model.architecture?.inputModalities ?? [])
                if !inputs.isEmpty {
                    facts.capabilities.vision = inputs.contains("image")
                    facts.capabilities.pdfInput = inputs.contains("file")
                    facts.capabilities.audioInput = inputs.contains("audio")
                    facts.capabilities.videoInput = inputs.contains("video")
                }
                let outputs = Set(model.architecture?.outputModalities ?? [])
                if !outputs.isEmpty {
                    facts.capabilities.audioOutput = outputs.contains("audio")
                }

                // supported_parameters names the knobs the model honors → bidirectional when
                // non-empty — EXCEPT for keys OpenRouter provably does not enumerate exhaustively.
                // In the 2026-07-18 payload only 4 of 344 models list parallel_tool_calls (none of
                // the 74 tool-capable OpenAI/Anthropic routes, whose native APIs demonstrably
                // support it) and web_search_options is similarly sparse — for those, absence is
                // not a statement, so they seed positive-only.
                let params = Set(model.supportedParameters ?? [])
                if !params.isEmpty {
                    facts.capabilities.toolUse = params.contains("tools")
                    facts.capabilities.toolChoice = params.contains("tool_choice")
                    if params.contains("parallel_tool_calls") { facts.capabilities.parallelToolCalls = true }
                    facts.capabilities.reasoning = params.contains("reasoning") || params.contains("reasoning_effort") || params.contains("include_reasoning")
                    facts.capabilities.responseSchema = params.contains("structured_outputs") || params.contains("response_format")
                    if params.contains("web_search_options") { facts.capabilities.webSearch = true }
                }

                // pricing: decimal USD-per-token strings ("0.000002"); "-1" marks a variable/auto
                // route (the auto-router), which has no fixed price.
                func rate(_ s: String?) -> Double? {
                    guard let s, let v = Double(s), v >= 0 else { return nil }
                    return v
                }
                if let p = model.pricing {
                    let base = PricingTier(input: rate(p.prompt), output: rate(p.completion),
                                           cacheRead: rate(p.inputCacheRead), cacheWrite: rate(p.inputCacheWrite))
                    if base.hasAnyRate {
                        // pricing.overrides states higher rates above a prompt-length threshold
                        // (gpt-5.6-luna-pro: 2x input past 272k tokens) — the same shape as xAI's
                        // long_context_threshold. Dropping them recorded flat rates that
                        // understate long-context cost by 1.5-2x on 43 models.
                        var tiers: [TokenThresholdTier] = []
                        for override in p.overrides ?? [] {
                            guard let threshold = override.minPromptTokens else { continue }
                            let rates = PricingTier(
                                input: rate(override.prompt), output: rate(override.completion),
                                cacheRead: rate(override.inputCacheRead), cacheWrite: rate(override.inputCacheWrite))
                            if rates.hasAnyRate {
                                tiers.append(TokenThresholdTier(tokenThreshold: threshold, rates: rates))
                            }
                        }
                        facts.pricing = ModelPricing(base: base, tokenThresholdTiers: tiers.sorted())
                    }
                }

                // reasoning.supported_efforts is OpenRouter stating the effort ladder outright — the
                // one non-Anthropic provider that does. An absent reasoning block says nothing (nil),
                // it is NOT a statement of "no efforts".
                if let efforts = model.reasoning?.supportedEfforts {
                    facts.validEffortLevels = efforts.sorted { EffortRank.rank(of: $0) < EffortRank.rank(of: $1) }
                }

                let dp = model.defaultParameters
                let defaults = SamplingDefaults(
                    temperature: dp?.temperature, topP: dp?.topP, topK: dp?.topK,
                    frequencyPenalty: dp?.frequencyPenalty, presencePenalty: dp?.presencePenalty,
                    repetitionPenalty: dp?.repetitionPenalty)
                facts.samplingDefaults = defaults.isEmpty ? nil : defaults

                // top_provider.context_length is the limit of the route OpenRouter actually
                // serves by default and can be smaller than the model's headline context; prefer
                // it, falling back to the top-level figure.
                facts.maxInputTokens = model.topProvider?.contextLength ?? model.contextLength
                facts.maxOutputTokens = model.topProvider?.maxCompletionTokens
                // expiration_date is OpenRouter's scheduled-removal date; treat like a
                // deprecation — but far-future placeholders ("2098-12-31" on active flagship
                // models) mean "no scheduled removal", not a deprecation. Genuine removals in the
                // same payload are dated months out, so anything past a decade is a sentinel.
                facts.deprecatedOn = model.expirationDate.flatMap(parseYearMonthDay)
                    .flatMap { $0.timeIntervalSinceNow > 10 * 365.25 * 86400 ? nil : $0 }
                facts.modelDescription = model.description
                facts.benchmarks = (model.benchmarks?.isEmpty ?? true) ? nil : model.benchmarks
                facts.huggingFaceID = model.huggingFaceID

                return DecodedModelFacts(modelID: model.id, facts: facts)
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - HuggingFace

    func decodeHuggingFaceModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeHuggingFaceFacts(from: data).map { $0.facts.materialize(providerID: providerID, modelID: $0.modelID) }
    }

    /// HuggingFace's router lists several concrete inference providers per model, each with its own
    /// context length, tool/structured-output support, and pricing. Because those genuinely differ
    /// per provider — and the meta-routers (`auto`/`cheapest`/`fastest`/`preferred`) aren't listed
    /// and can change per request — we enumerate one model per concrete provider, keyed
    /// `org/model:provider` (HF's documented routing suffix). Providers with no usable data are
    /// skipped rather than given fabricated defaults, and no meta entry is synthesized.
    /// `supports_tools` / `supports_structured_output` are tri-state: a present leaf is a stated
    /// true or FALSE (both believed), an absent leaf says nothing.
    /// Pricing is USD per million tokens, so USD-per-token = value / 1e6.
    private func decodeHuggingFaceFacts(from data: Data) throws -> [DecodedModelFacts] {
        // USD per million tokens -> USD per single token.
        func usdPerToken(_ perMillion: Double?) -> Double? {
            guard let v = perMillion, v >= 0 else { return nil }
            return v / 1_000_000
        }

        let decoded = try JSONDecoder().decode(HuggingFaceModelsResponse.self, from: data)
        return decoded.data
            .flatMap { model -> [DecodedModelFacts] in
                // Modalities are stated at the model level and shared by every provider;
                // the arrays are explicit enumerations → bidirectional when non-empty.
                var shared = ModelFacts()
                let inputs = Set(model.architecture?.inputModalities ?? [])
                if !inputs.isEmpty {
                    shared.capabilities.vision = inputs.contains("image")
                    shared.capabilities.pdfInput = inputs.contains("file")
                    shared.capabilities.audioInput = inputs.contains("audio")
                    shared.capabilities.videoInput = inputs.contains("video")
                }
                let outputs = Set(model.architecture?.outputModalities ?? [])
                if !outputs.isEmpty {
                    shared.capabilities.audioOutput = outputs.contains("audio")
                }
                shared.createdAt = model.created.map { Date(timeIntervalSince1970: TimeInterval($0)) }

                return (model.providers ?? [])
                    .filter { $0.status == "live" && $0.hasUsableData }
                    .map { entry in
                        // Start from the shared modalities; layer this provider's own statements.
                        var facts = shared
                        facts.displayName = "\(model.id) (\(entry.provider))"
                        facts.capabilities.toolUse = entry.supportsTools
                        facts.capabilities.responseSchema = entry.supportsStructuredOutput
                        facts.maxInputTokens = entry.contextLength
                        facts.isFree = entry.isFree
                        if let p = entry.pricing {
                            let base = PricingTier(input: usdPerToken(p.input), output: usdPerToken(p.output))
                            if base.hasAnyRate { facts.pricing = ModelPricing(base: base) }
                        }
                        return DecodedModelFacts(modelID: "\(model.id):\(entry.provider)", facts: facts)
                    }
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - Mistral

    func decodeMistralModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeMistralFacts(from: data).map { $0.facts.materialize(providerID: providerID, modelID: $0.modelID) }
    }

    private func decodeMistralFacts(from data: Data) throws -> [DecodedModelFacts] {
        let decoded = try JSONDecoder().decode(MistralModelsResponse.self, from: data)
        // Mistral returns both aliases (e.g. "mistral-large-latest") and specific versions
        // (e.g. "mistral-large-2512") which share the same `name` field, so displayName is left
        // unset (materializes to the model ID) to avoid visual duplicates in the picker.
        return decoded.data
            .map { model in
                var facts = ModelFacts()
                facts.createdAt = model.created.map { Date(timeIntervalSince1970: TimeInterval($0)) }
                facts.maxInputTokens = model.maxContextLength
                // Mistral's capabilities block states each leaf explicitly, BOTH directions —
                // including function_calling, which almost no vendor publishes. A present leaf is a
                // stated true/false; an absent leaf stays nil (the old `?? false` fabricated a "no").
                if let abilities = model.capabilities {
                    facts.capabilities.toolUse = abilities.functionCalling
                    facts.capabilities.vision = abilities.vision
                    facts.capabilities.reasoning = abilities.reasoning
                    facts.capabilities.audioInput = abilities.audio
                    facts.capabilities.audioOutput = abilities.audioSpeech
                    facts.supportsChatCompletions = abilities.completionChat
                }
                facts.deprecatedOn = model.deprecation.flatMap(parseISODate)
                facts.deprecationReplacement = model.deprecationReplacementModel
                facts.modelDescription = model.description
                facts.samplingDefaults = model.defaultModelTemperature.map { SamplingDefaults(temperature: $0) }
                return DecodedModelFacts(modelID: model.id, facts: facts)
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - Gemini

    func decodeGeminiModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeGeminiFacts(from: data).map { $0.facts.materialize(providerID: providerID, modelID: $0.modelID) }
    }

    private func decodeGeminiFacts(from data: Data) throws -> [DecodedModelFacts] {
        let decoded = try JSONDecoder().decode(GeminiModelsResponse.self, from: data)
        return decoded.models
            .map { model in
                // Gemini model names are "models/gemini-pro" — strip the prefix for display
                let modelID = model.name.hasPrefix("models/")
                    ? String(model.name.dropFirst("models/".count))
                    : model.name

                var facts = ModelFacts()
                facts.displayName = model.displayName
                // The methods list is an explicit enumeration: when present, "generateContent"
                // distinguishes chat models from embedding/image ones both directions. Absent →
                // stays nil (materializes to the true-by-default convention).
                facts.supportsChatCompletions = model.supportedGenerationMethods
                    .map { $0.contains("generateContent") }
                // `thinking` is stated only when present; absent says nothing.
                facts.capabilities.reasoning = model.thinking
                facts.maxInputTokens = model.inputTokenLimit
                facts.maxOutputTokens = model.outputTokenLimit
                facts.maxTemperature = model.maxTemperature
                facts.modelDescription = model.description
                let defaults = SamplingDefaults(temperature: model.temperature, topP: model.topP, topK: model.topK)
                facts.samplingDefaults = defaults.isEmpty ? nil : defaults
                return DecodedModelFacts(modelID: modelID, facts: facts)
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - Test seams

    /// Tri-state test seam: decodes a captured payload into per-model facts for the given apiType,
    /// so tests can assert the stated/unknown distinction (`nil` vs `false`) the materialized
    /// `ModelInfo` seams flatten away.
    func decodeModelFactsForTesting(from data: Data, apiType: ProviderAPIType) throws -> [DecodedModelFacts] {
        switch apiType {
        case .ollama: return try decodeOllamaFacts(from: data)
        case .anthropic: return try decodeAnthropicFacts(from: data)
        case .xAI: return try decodeXAIFacts(from: data)
        case .openRouter: return try decodeOpenRouterFacts(from: data)
        case .huggingFace: return try decodeHuggingFaceFacts(from: data)
        case .openAICompatible, .lmStudio, .zAI, .metaModel, .alibabaCloud: return try decodeOpenAIFacts(from: data)
        case .mistral: return try decodeMistralFacts(from: data)
        case .gemini: return try decodeGeminiFacts(from: data)
        }
    }

    // MARK: - Helpers

    /// Converts a byte count to a compact parameter-style label: M / B / T.
    func formatBytes(_ bytes: Int64) -> String {
        guard bytes > 0 else { return "" }
        let d = Double(bytes)
        let trillion: Double = 1_000_000_000_000
        let billion: Double  = 1_000_000_000
        let million: Double  = 1_000_000
        let value: Double
        let suffix: String
        if d >= trillion      { value = d / trillion; suffix = "T" }
        else if d >= billion  { value = d / billion;  suffix = "B" }
        else                  { value = d / million;  suffix = "M" }
        return value < 10
            ? String(format: "%.1f\(suffix)", value)
            : String(format: "%.0f\(suffix)", value)
    }

    /// Parses an ISO 8601 date string into a `Date`.
    /// Handles nanosecond-precision timestamps (e.g. from Ollama) by truncating to milliseconds.
    func parseISODate(_ iso: String) -> Date? {
        let truncated = iso.replacingOccurrences(
            of: #"(\.\d{3})\d+"#,
            with: "$1",
            options: .regularExpression
        )
        let parser = ISO8601DateFormatter()
        parser.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = parser.date(from: truncated) {
            return date
        }
        parser.formatOptions = [.withInternetDateTime]
        return parser.date(from: iso)
    }

    /// Parses a bare `yyyy-MM-dd` date (OpenRouter's `expiration_date`) as midnight UTC. Falls back
    /// to the full ISO parser for anything carrying a time component.
    func parseYearMonthDay(_ raw: String) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "GMT")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.date(from: raw) ?? parseISODate(raw)
    }

}

/// Formats a token count as a compact "16K" / "1M" label.
public func formatTokenCount(_ count: Int) -> String {
    if count >= 1_000_000 {
        let value = Double(count) / 1_000_000
        return value.truncatingRemainder(dividingBy: 1) == 0
            ? String(format: "%.0fM", value)
            : String(format: "%.1fM", value)
    } else if count >= 1_000 {
        let value = Double(count) / 1_000
        return value.truncatingRemainder(dividingBy: 1) == 0
            ? String(format: "%.0fK", value)
            : String(format: "%.1fK", value)
    }
    return "\(count)"
}

/// Formats a per-million-token cost as a compact dollar string, e.g. "$3.00", "$0.28".
public func formatCostPerMillion(_ cost: Double) -> String {
    if cost < 0.01 {
        return String(format: "$%.4f", cost)
    } else {
        return String(format: "$%.2f", cost)
    }
}

// MARK: - Errors

/// Errors from model list fetching.
private enum ModelFetchError: Error, LocalizedError {
    case httpError(statusCode: Int)

    var errorDescription: String? {
        switch self {
        case .httpError(let code):
            return "Server returned HTTP \(code). Check the endpoint URL and API key."
        }
    }
}

// MARK: - API Response Types

private struct AnthropicModelsResponse: Decodable {
    /// Anthropic's per-capability leaf: `{"supported": true}`. Everything in the capabilities
    /// block bottoms out in this shape.
    struct Supported: Decodable {
        let supported: Bool?
    }

    /// The `thinking` block: overall support plus which mechanisms exist. `types.enabled` is the
    /// budget_tokens form; `types.adaptive` is `thinking: {type: "adaptive"}`. A model with
    /// `enabled.supported == false` is adaptive-ONLY — which, verified live, also means it
    /// rejects the `temperature` parameter.
    struct Thinking: Decodable {
        struct Types: Decodable {
            let adaptive: Supported?
            let enabled: Supported?
        }
        let supported: Bool?
        let types: Types?
    }

    /// The `effort` block: an overall `supported` plus one `{"supported": Bool}` per level. The
    /// level names arrive as dynamic keys, so it decodes from a keyed container by hand.
    struct Effort: Decodable {
        let supported: Bool?
        let levels: [String: Bool]

        struct DynamicKey: CodingKey {
            var stringValue: String
            var intValue: Int? { nil }
            init?(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { nil }
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: DynamicKey.self)
            var supported: Bool?
            var levels: [String: Bool] = [:]
            for key in container.allKeys {
                if key.stringValue == "supported" {
                    supported = try? container.decode(Bool.self, forKey: key)
                } else if let leaf = try? container.decode(Supported.self, forKey: key) {
                    levels[key.stringValue] = leaf.supported ?? false
                }
            }
            self.supported = supported
            self.levels = levels
        }
    }

    struct Capabilities: Decodable {
        let imageInput: Supported?
        let pdfInput: Supported?
        let thinking: Thinking?
        let effort: Effort?
        let codeExecution: Supported?
        let structuredOutputs: Supported?
        enum CodingKeys: String, CodingKey {
            case imageInput = "image_input"
            case pdfInput = "pdf_input"
            case thinking
            case effort
            case codeExecution = "code_execution"
            case structuredOutputs = "structured_outputs"
        }
    }

    struct ModelEntry: Decodable {
        let id: String
        let displayName: String?
        let createdAt: String?
        let maxTokens: Int?
        let maxInputTokens: Int?
        let capabilities: Capabilities?
        enum CodingKeys: String, CodingKey {
            case id
            case displayName = "display_name"
            case createdAt = "created_at"
            case maxTokens = "max_tokens"
            case maxInputTokens = "max_input_tokens"
            case capabilities
        }
    }
    let data: [ModelEntry]
}

private struct OpenAIModelsResponse: Decodable {
    struct ModelEntry: Decodable {
        let id: String
        let created: Int?
        let ownedBy: String?
        // Common OpenAI-compatible extensions; nil for endpoints (incl. plain OpenAI) that omit them.
        let contextLength: Int?
        let promptImageTokenPrice: Int?
        enum CodingKeys: String, CodingKey {
            case id, created
            case ownedBy = "owned_by"
            case contextLength = "context_length"
            case promptImageTokenPrice = "prompt_image_token_price"
        }
    }
    let data: [ModelEntry]
}

/// xAI's `/models` extends the OpenAI shape with a context length, an image-token price (which
/// implies vision), and per-token prices. Token prices are in xAI's documented unit — "USD cents
/// per 100 million tokens" — so USD-per-token = value / 1e10. See
/// https://docs.x.ai/developers/cost-tracking (which also states 1 USD = 1e10 of the
/// `cost_in_usd_ticks` the API reports per request).
private struct XAIModelsResponse: Decodable {
    struct ModelEntry: Decodable {
        let id: String
        let created: Int?
        let contextLength: Int?
        // Prices typed Double, not Int: xAI currently emits integer "cents per 100M tokens", but a
        // single fractional value would otherwise throw and fail the ENTIRE catalog fetch. Double
        // costs nothing (they feed a Double conversion anyway) and is strictly safer.
        let promptImageTokenPrice: Double?
        let promptTextTokenPrice: Double?
        let completionTextTokenPrice: Double?
        let cachedPromptTextTokenPrice: Double?
        let promptTextTokenPriceLongContext: Double?
        let completionTextTokenPriceLongContext: Double?
        let cachedPromptTextTokenPriceLongContext: Double?
        let longContextThreshold: Int?
        let aliases: [String]?
        enum CodingKeys: String, CodingKey {
            case id, created, aliases
            case contextLength = "context_length"
            case promptImageTokenPrice = "prompt_image_token_price"
            case promptTextTokenPrice = "prompt_text_token_price"
            case completionTextTokenPrice = "completion_text_token_price"
            case cachedPromptTextTokenPrice = "cached_prompt_text_token_price"
            case promptTextTokenPriceLongContext = "prompt_text_token_price_long_context"
            case completionTextTokenPriceLongContext = "completion_text_token_price_long_context"
            case cachedPromptTextTokenPriceLongContext = "cached_prompt_text_token_price_long_context"
            case longContextThreshold = "long_context_threshold"
        }
    }
    let data: [ModelEntry]
}

/// OpenRouter's `/models` states input/output modalities, the parameters each model supports,
/// per-provider limits, and per-token prices (decimal USD-per-token strings; "-1" = variable).
/// Schema: https://openrouter.ai/docs/api-reference/list-available-models
private struct OpenRouterModelsResponse: Decodable {
    struct Architecture: Decodable {
        let inputModalities: [String]?
        let outputModalities: [String]?
        let modality: String?
        enum CodingKeys: String, CodingKey {
            case inputModalities = "input_modalities"
            case outputModalities = "output_modalities"
            case modality
        }
    }
    struct Pricing: Decodable {
        let prompt: String?
        let completion: String?
        let inputCacheRead: String?
        let inputCacheWrite: String?
        let overrides: [PricingOverride]?
        enum CodingKeys: String, CodingKey {
            case prompt, completion, overrides
            case inputCacheRead = "input_cache_read"
            case inputCacheWrite = "input_cache_write"
        }
    }
    /// One long-context pricing tier: rates that replace the base above `min_prompt_tokens`.
    struct PricingOverride: Decodable {
        let minPromptTokens: Int?
        let prompt: String?
        let completion: String?
        let inputCacheRead: String?
        let inputCacheWrite: String?
        enum CodingKeys: String, CodingKey {
            case prompt, completion
            case minPromptTokens = "min_prompt_tokens"
            case inputCacheRead = "input_cache_read"
            case inputCacheWrite = "input_cache_write"
        }
    }
    struct TopProvider: Decodable {
        let contextLength: Int?
        let maxCompletionTokens: Int?
        enum CodingKeys: String, CodingKey {
            case contextLength = "context_length"
            case maxCompletionTokens = "max_completion_tokens"
        }
    }
    struct Reasoning: Decodable {
        let supportedEfforts: [String]?
        enum CodingKeys: String, CodingKey {
            case supportedEfforts = "supported_efforts"
        }
    }
    struct DefaultParameters: Decodable {
        let temperature: Double?
        let topP: Double?
        let topK: Int?
        let frequencyPenalty: Double?
        let presencePenalty: Double?
        let repetitionPenalty: Double?
        enum CodingKeys: String, CodingKey {
            case temperature
            case topP = "top_p"
            case topK = "top_k"
            case frequencyPenalty = "frequency_penalty"
            case presencePenalty = "presence_penalty"
            case repetitionPenalty = "repetition_penalty"
        }
    }
    struct ModelEntry: Decodable {
        let id: String
        let name: String?
        let created: Int?
        let contextLength: Int?
        let architecture: Architecture?
        let pricing: Pricing?
        let topProvider: TopProvider?
        let supportedParameters: [String]?
        let description: String?
        let reasoning: Reasoning?
        let defaultParameters: DefaultParameters?
        let expirationDate: String?
        let huggingFaceID: String?
        let benchmarks: ModelBenchmarks?
        enum CodingKeys: String, CodingKey {
            case id, name, created, architecture, pricing, description, reasoning, benchmarks
            case contextLength = "context_length"
            case topProvider = "top_provider"
            case supportedParameters = "supported_parameters"
            case defaultParameters = "default_parameters"
            case expirationDate = "expiration_date"
            case huggingFaceID = "hugging_face_id"
        }
    }
    let data: [ModelEntry]
}

/// HuggingFace's router `/models` lists, per model, an array of concrete inference providers —
/// each with its own context length, tool/structured-output support, and pricing. There is no
/// single set of capabilities for a bare model ID: the same model varies 4× in context and flips
/// tool/schema support across providers, and the meta-routers (`auto`/`cheapest`/`fastest`/
/// `preferred`) aren't even listed here — you only learn which concrete provider served a request
/// from the response after the fact, and being meta they can change between requests. So we
/// enumerate each concrete provider as its own model (`org/model:provider`, HF's documented routing
/// suffix) and never synthesize a meta entry. Pricing is USD per million tokens.
/// Schema: https://huggingface.co/docs/inference-providers
private struct HuggingFaceModelsResponse: Decodable {
    struct Architecture: Decodable {
        let inputModalities: [String]?
        let outputModalities: [String]?
        enum CodingKeys: String, CodingKey {
            case inputModalities = "input_modalities"
            case outputModalities = "output_modalities"
        }
    }
    struct Pricing: Decodable {
        let input: Double?
        let output: Double?
    }
    struct Provider: Decodable {
        let provider: String
        let status: String?
        let contextLength: Int?
        let pricing: Pricing?
        let supportsTools: Bool?
        let supportsStructuredOutput: Bool?
        let isFree: Bool?
        enum CodingKeys: String, CodingKey {
            case provider, status, pricing
            case contextLength = "context_length"
            case supportsTools = "supports_tools"
            case supportsStructuredOutput = "supports_structured_output"
            case isFree = "is_free"
        }

        /// Whether this entry carries anything worth enumerating. A provider that lists only a
        /// name and `is_free` tells us nothing about the model — skip it rather than fabricate
        /// defaults for it.
        var hasUsableData: Bool {
            contextLength != nil || pricing != nil || supportsTools != nil || supportsStructuredOutput != nil
        }
    }
    struct ModelEntry: Decodable {
        let id: String
        let created: Int?
        let architecture: Architecture?
        let providers: [Provider]?
    }
    let data: [ModelEntry]
}

private struct OllamaTagsResponse: Decodable {
    struct Details: Decodable {
        let quantizationLevel: String?
        // The stated parameter count ("397B"); a truer size label than the byte-size `size` field.
        let parameterSize: String?
        // The model's context window — only local Ollama's payload includes it; cloud omits it.
        let contextLength: Int?
        enum CodingKeys: String, CodingKey {
            case quantizationLevel = "quantization_level"
            case parameterSize = "parameter_size"
            case contextLength = "context_length"
        }
    }
    struct Model: Decodable {
        let name: String
        let size: Int64
        let modifiedAt: String
        let details: Details?
        let capabilities: [String]?
        enum CodingKeys: String, CodingKey {
            case name, size, capabilities
            case modifiedAt = "modified_at"
            case details
        }
    }
    let models: [Model]
}

private struct MistralModelsResponse: Decodable {
    struct Capabilities: Decodable {
        let completionChat: Bool?
        let functionCalling: Bool?
        let vision: Bool?
        let reasoning: Bool?
        let audio: Bool?
        let audioSpeech: Bool?
        enum CodingKeys: String, CodingKey {
            case completionChat = "completion_chat"
            case functionCalling = "function_calling"
            case vision, reasoning, audio
            case audioSpeech = "audio_speech"
        }
    }
    struct ModelEntry: Decodable {
        let id: String
        let name: String?
        let created: Int?
        let maxContextLength: Int?
        let capabilities: Capabilities?
        let description: String?
        // ISO-8601 date the model is (or will be) deprecated; and the model Mistral points to next.
        let deprecation: String?
        let deprecationReplacementModel: String?
        // The model's default temperature (Mistral's only published sampling default).
        let defaultModelTemperature: Double?
        enum CodingKeys: String, CodingKey {
            case id, name, created, capabilities, description, deprecation
            case maxContextLength = "max_context_length"
            case deprecationReplacementModel = "deprecation_replacement_model"
            case defaultModelTemperature = "default_model_temperature"
        }
    }
    let data: [ModelEntry]
}

private struct GeminiModelsResponse: Decodable {
    struct ModelEntry: Decodable {
        let name: String
        let displayName: String?
        let description: String?
        let inputTokenLimit: Int?
        let outputTokenLimit: Int?
        let supportedGenerationMethods: [String]?
        let thinking: Bool?
        // Gemini alone publishes a temperature ceiling (usually 2). temperature/topK/topP are the
        // model's default sampling parameters (not limits) — captured as reference metadata.
        let maxTemperature: Double?
        let temperature: Double?
        let topK: Int?
        let topP: Double?
    }
    let models: [ModelEntry]
}
