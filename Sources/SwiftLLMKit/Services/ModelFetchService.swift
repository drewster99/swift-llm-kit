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
    public func fetchModels(
        from provider: ModelProvider,
        apiKey: String?
    ) async throws -> [ModelInfo] {
        let modelsURL: URL
        switch provider.apiType {
        case .ollama:
            modelsURL = provider.endpoint.appendingPathComponent("tags")
        case .anthropic:
            modelsURL = provider.endpoint.strippingAnthropicV1().appendingPathComponent("v1/models")
        case .openAICompatible, .lmStudio, .mistral, .huggingFace, .xAI, .zAI, .metaLlama, .alibabaCloud, .openRouter:
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
        case .openAICompatible, .lmStudio, .mistral, .huggingFace, .xAI, .zAI, .metaLlama, .alibabaCloud, .openRouter:
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

        let decoded: [ModelInfo]
        switch provider.apiType {
        case .ollama:
            decoded = try decodeOllamaModels(from: data, providerID: provider.id)
        case .anthropic:
            decoded = try decodeAnthropicModels(from: data, providerID: provider.id)
        case .xAI:
            decoded = try decodeXAIModels(from: data, providerID: provider.id)
        case .openRouter:
            decoded = try decodeOpenRouterModels(from: data, providerID: provider.id)
        case .openAICompatible, .lmStudio, .huggingFace, .zAI, .metaLlama, .alibabaCloud:
            decoded = try decodeOpenAIModels(from: data, providerID: provider.id)
        case .mistral:
            decoded = try decodeMistralModels(from: data, providerID: provider.id)
        case .gemini:
            decoded = try decodeGeminiModels(from: data, providerID: provider.id)
        }

        // Deduplicate by composite ID (providerID/modelID). Some APIs return
        // the same model ID multiple times (e.g. Mistral aliases).
        var seen = Set<String>()
        return decoded.filter { seen.insert($0.id).inserted }
    }

    // MARK: - Ollama

    private func decodeOllamaModels(from data: Data, providerID: String) throws -> [ModelInfo] {
        let decoded = try JSONDecoder().decode(OllamaTagsResponse.self, from: data)
        return decoded.models
            .map { model in
                let quant = model.details?.quantizationLevel ?? ""
                var caps = ModelCapabilities()
                if let capabilities = model.capabilities {
                    caps.toolUse = capabilities.contains("tools")
                }
                return ModelInfo(
                    providerID: providerID,
                    modelID: model.name,
                    createdAt: parseISODate(model.modifiedAt),
                    capabilities: caps,
                    sizeLabel: formatBytes(model.size),
                    quantizationLabel: quant.isEmpty ? nil : quant
                )
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - Anthropic

    /// Test seam: exercises the private decoder against captured payload shapes without a fetch.
    func decodeAnthropicModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeAnthropicModels(from: data, providerID: providerID)
    }

    private func decodeAnthropicModels(from data: Data, providerID: String) throws -> [ModelInfo] {
        let decoded = try JSONDecoder().decode(AnthropicModelsResponse.self, from: data)
        return decoded.data
            .map { model in
                // The vendor describing its own models is the one source with standing to — so
                // the capabilities block is decoded and believed, not discarded (as it was until
                // 2026-07: image_input and pdf_input were being thrown away and re-derived from
                // LiteLLM's third-party claims instead).
                var caps = ModelCapabilities()
                var flags = BehaviorFlags()
                var effortLevels: [String] = []

                if let capabilities = model.capabilities {
                    caps.vision = capabilities.imageInput?.supported ?? false
                    caps.pdfInput = capabilities.pdfInput?.supported ?? false
                    caps.reasoning = capabilities.thinking?.supported ?? false
                    caps.codeExecution = capabilities.codeExecution?.supported ?? false
                    caps.responseSchema = capabilities.structuredOutputs?.supported ?? false

                    // The payload's own per-model level list, ordered by our rank table (the
                    // payload is a JSON object, so it carries no order itself).
                    if capabilities.effort?.supported == true {
                        effortLevels = capabilities.effort?.levels
                            .filter { $0.value }
                            .map(\.key)
                            .sorted { EffortRank.rank(of: $0) < EffortRank.rank(of: $1) } ?? []
                    }

                    // DERIVED, not hand-listed: a thinking model whose budget_tokens form is
                    // unsupported (`types.enabled == false`) is adaptive-only — and, verified
                    // live 2026-07-17, adaptive-only models also reject the `temperature`
                    // parameter with HTTP 400 ("deprecated for this model"): opus-4-8 rejects,
                    // while sonnet-4-6 / opus-4-5 (enabled == true) accept. Deriving both flags
                    // here means the next adaptive-only model Anthropic ships is handled the day
                    // it appears, instead of 400-ing until someone edits the bundled JSON. The
                    // bundled entries remain as gap-fill for cold starts with no fetched catalog.
                    if capabilities.thinking?.supported == true,
                       capabilities.thinking?.types?.enabled?.supported == false {
                        flags.requiresAdaptiveThinking = true
                        flags.mustNeverSendTemperatureParam = true
                    }
                }

                return ModelInfo(
                    providerID: providerID,
                    modelID: model.id,
                    displayName: model.displayName ?? model.id,
                    createdAt: model.createdAt.flatMap { parseISODate($0) },
                    maxInputTokens: model.maxInputTokens,
                    maxOutputTokens: model.maxTokens,
                    capabilities: caps,
                    validEffortLevels: effortLevels,
                    behaviorFlags: flags
                )
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - OpenAI Compatible (plain)

    func decodeOpenAIModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeOpenAIModels(from: data, providerID: providerID)
    }

    private func decodeOpenAIModels(from data: Data, providerID: String) throws -> [ModelInfo] {
        let decoded = try JSONDecoder().decode(OpenAIModelsResponse.self, from: data)
        return decoded.data
            .map { model in
                ModelInfo(
                    providerID: providerID,
                    modelID: model.id,
                    createdAt: model.created.map { Date(timeIntervalSince1970: TimeInterval($0)) }
                )
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - xAI

    func decodeXAIModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeXAIModels(from: data, providerID: providerID)
    }

    /// xAI's `/models` is richer than OpenAI's: a context length, an image-token price (which
    /// implies vision), and per-token prices. Token prices are in xAI's documented unit — "USD
    /// cents per 100 million tokens" — so USD-per-token = value / 1e10. See
    /// https://docs.x.ai/developers/cost-tracking (which also documents `cost_in_usd_ticks`, where
    /// 1 USD = 1e10 ticks). Prices >= 0 only; xAI uses no sentinel, but guard anyway.
    private func decodeXAIModels(from data: Data, providerID: String) throws -> [ModelInfo] {
        // "USD cents per 100 million tokens" -> USD per single token.
        // Per https://docs.x.ai/developers/cost-tracking.
        func usdPerToken(_ centsPer100M: Int?) -> Double? {
            guard let v = centsPer100M, v >= 0 else { return nil }
            return Double(v) / 1e10
        }

        let decoded = try JSONDecoder().decode(XAIModelsResponse.self, from: data)
        return decoded.data
            .map { model in
                // A positive image-token price means the model prices — and therefore accepts —
                // image input: a vendor-stated vision signal, not a LiteLLM guess. Only set true;
                // absence isn't proof of no vision.
                var caps = ModelCapabilities()
                if let imagePrice = model.promptImageTokenPrice, imagePrice > 0 {
                    caps.vision = true
                }

                var pricing: ModelPricing?
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
                    pricing = ModelPricing(base: base, tokenThresholdTiers: tiers)
                }

                return ModelInfo(
                    providerID: providerID,
                    modelID: model.id,
                    createdAt: model.created.map { Date(timeIntervalSince1970: TimeInterval($0)) },
                    maxInputTokens: model.contextLength,
                    capabilities: caps,
                    pricing: pricing
                )
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - OpenRouter

    func decodeOpenRouterModelsForTesting(from data: Data, providerID: String) throws -> [ModelInfo] {
        try decodeOpenRouterModels(from: data, providerID: providerID)
    }

    /// OpenRouter's `/models` is the richest of the OpenAI-compatible family: it states input
    /// modalities, the parameters each model supports, per-provider limits, and per-token prices.
    /// All of it is decoded from the vendor rather than guessed. Schema:
    /// https://openrouter.ai/docs/api-reference/list-available-models
    private func decodeOpenRouterModels(from data: Data, providerID: String) throws -> [ModelInfo] {
        let decoded = try JSONDecoder().decode(OpenRouterModelsResponse.self, from: data)
        return decoded.data
            .map { model in
                var caps = ModelCapabilities()

                // input_modalities is an explicit list, so both directions are decodable.
                let inputs = Set(model.architecture?.inputModalities ?? [])
                if !inputs.isEmpty {
                    caps.vision = inputs.contains("image")
                    caps.pdfInput = inputs.contains("file")
                    caps.audioInput = inputs.contains("audio")
                    caps.videoInput = inputs.contains("video")
                }
                let outputs = Set(model.architecture?.outputModalities ?? [])
                if !outputs.isEmpty {
                    caps.audioOutput = outputs.contains("audio")
                }

                // supported_parameters names the knobs the model honors.
                let params = Set(model.supportedParameters ?? [])
                if !params.isEmpty {
                    caps.toolUse = params.contains("tools")
                    caps.toolChoice = params.contains("tool_choice")
                    caps.parallelToolCalls = params.contains("parallel_tool_calls")
                    caps.reasoning = params.contains("reasoning") || params.contains("reasoning_effort") || params.contains("include_reasoning")
                    caps.responseSchema = params.contains("structured_outputs") || params.contains("response_format")
                    caps.webSearch = params.contains("web_search_options")
                }

                // pricing: decimal USD-per-token strings ("0.000002"); "-1" marks a variable/auto
                // route (the auto-router), which has no fixed price.
                func rate(_ s: String?) -> Double? {
                    guard let s, let v = Double(s), v >= 0 else { return nil }
                    return v
                }
                var pricing: ModelPricing?
                if let p = model.pricing {
                    let base = PricingTier(input: rate(p.prompt), output: rate(p.completion),
                                           cacheRead: rate(p.inputCacheRead), cacheWrite: rate(p.inputCacheWrite))
                    if base.hasAnyRate { pricing = ModelPricing(base: base) }
                }

                return ModelInfo(
                    providerID: providerID,
                    modelID: model.id,
                    displayName: model.name ?? model.id,
                    createdAt: model.created.map { Date(timeIntervalSince1970: TimeInterval($0)) },
                    maxInputTokens: model.contextLength,
                    maxOutputTokens: model.topProvider?.maxCompletionTokens,
                    capabilities: caps,
                    pricing: pricing
                )
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - Mistral

    private func decodeMistralModels(from data: Data, providerID: String) throws -> [ModelInfo] {
        let decoded = try JSONDecoder().decode(MistralModelsResponse.self, from: data)
        // Mistral returns both aliases (e.g. "mistral-large-latest") and specific versions
        // (e.g. "mistral-large-2512") which share the same `name` field. Use model ID as
        // display name to avoid visual duplicates in the picker.
        return decoded.data
            .map { model in
                var caps = ModelCapabilities()
                let supportsChat: Bool
                if let abilities = model.capabilities {
                    caps.toolUse = abilities.functionCalling ?? false
                    caps.vision = abilities.vision ?? false
                    caps.reasoning = abilities.reasoning ?? false
                    caps.audioInput = abilities.audio ?? false
                    caps.audioOutput = abilities.audioSpeech ?? false
                    supportsChat = abilities.completionChat ?? true
                } else {
                    supportsChat = true
                }
                return ModelInfo(
                    providerID: providerID,
                    modelID: model.id,
                    displayName: model.id,
                    createdAt: model.created.map { Date(timeIntervalSince1970: TimeInterval($0)) },
                    maxInputTokens: model.maxContextLength,
                    capabilities: caps,
                    supportsChatCompletions: supportsChat
                )
            }
            .sorted { $0.modelID < $1.modelID }
    }

    // MARK: - Gemini

    private func decodeGeminiModels(from data: Data, providerID: String) throws -> [ModelInfo] {
        let decoded = try JSONDecoder().decode(GeminiModelsResponse.self, from: data)
        return decoded.models
            .map { model in
                // Gemini model names are "models/gemini-pro" — strip the prefix for display
                let modelID = model.name.hasPrefix("models/")
                    ? String(model.name.dropFirst("models/".count))
                    : model.name

                let methods = model.supportedGenerationMethods ?? []
                let supportsChat = methods.contains("generateContent")

                var caps = ModelCapabilities()
                caps.reasoning = model.thinking ?? false

                return ModelInfo(
                    providerID: providerID,
                    modelID: modelID,
                    displayName: model.displayName ?? modelID,
                    maxInputTokens: model.inputTokenLimit,
                    maxOutputTokens: model.outputTokenLimit,
                    capabilities: caps,
                    supportsChatCompletions: supportsChat
                )
            }
            .sorted { $0.modelID < $1.modelID }
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
        enum CodingKeys: String, CodingKey {
            case id, created
            case ownedBy = "owned_by"
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
        let promptImageTokenPrice: Int?
        let promptTextTokenPrice: Int?
        let completionTextTokenPrice: Int?
        let cachedPromptTextTokenPrice: Int?
        let promptTextTokenPriceLongContext: Int?
        let completionTextTokenPriceLongContext: Int?
        let cachedPromptTextTokenPriceLongContext: Int?
        let longContextThreshold: Int?
        enum CodingKeys: String, CodingKey {
            case id, created
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
        enum CodingKeys: String, CodingKey {
            case prompt, completion
            case inputCacheRead = "input_cache_read"
            case inputCacheWrite = "input_cache_write"
        }
    }
    struct TopProvider: Decodable {
        let maxCompletionTokens: Int?
        enum CodingKeys: String, CodingKey {
            case maxCompletionTokens = "max_completion_tokens"
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
        enum CodingKeys: String, CodingKey {
            case id, name, created, architecture, pricing
            case contextLength = "context_length"
            case topProvider = "top_provider"
            case supportedParameters = "supported_parameters"
        }
    }
    let data: [ModelEntry]
}

private struct OllamaTagsResponse: Decodable {
    struct Details: Decodable {
        let quantizationLevel: String?
        enum CodingKeys: String, CodingKey {
            case quantizationLevel = "quantization_level"
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
        enum CodingKeys: String, CodingKey {
            case id, name, created, capabilities, description
            case maxContextLength = "max_context_length"
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
    }
    let models: [ModelEntry]
}
