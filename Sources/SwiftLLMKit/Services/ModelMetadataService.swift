import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "ModelMetadata")

/// Manages the LiteLLM model metadata cache.
///
/// Downloads the LiteLLM model cost map from GitHub, caches it locally,
/// and provides lookup by model ID. Uses conditional HTTP requests
/// (ETag/Last-Modified) to avoid redundant downloads.
public actor ModelMetadataService {
    /// When true, the LiteLLM fetch's request line and response are logged via ``LLMRequestLogger``,
    /// into whichever directory it is configured to use — the same one the chat traffic uses.
    public nonisolated(unsafe) static var verboseLogging = false

    private let storageDirectory: URL
    private let userDefaults: UserDefaults
    private let userDefaultsSuiteName: String

    /// In-memory index: `litellm_provider` → (normalized model name → parsed metadata).
    ///
    /// Keyed off the entry's `litellm_provider` FIELD rather than its key prefix, because the
    /// prefix is not a provider: LiteLLM keys carry image sizes (`1024-x-1024/dall-e-2` is
    /// `openai`), quality tiers (`high/1536-x-1024/gpt-image-1.5`), and AWS inference-profile
    /// segments (`global.anthropic.claude-fable-5` is `bedrock_converse`, NOT `anthropic`).
    /// See ``modelName(fromKey:provider:)`` for how the model name is derived.
    private var providerIndex: [String: [String: LiteLLMEntry]] = [:]

    private static let liteLLMURL: URL = {
        guard let url = URL(string: "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json") else {
            preconditionFailure("Invalid LiteLLM metadata URL literal")
        }
        return url
    }()
    private static let metadataFilename = "litellm_metadata.json"
    private static let headersFilename = "litellm_headers.json"
    private static let lastRefreshKey = "LastModelRefreshDate"

    public init(storageDirectory: URL, userDefaultsSuiteName: String) {
        self.storageDirectory = storageDirectory
        self.userDefaultsSuiteName = userDefaultsSuiteName
        guard let defaults = UserDefaults(suiteName: userDefaultsSuiteName) else {
            preconditionFailure("Failed to create UserDefaults suite: \(userDefaultsSuiteName)")
        }
        self.userDefaults = defaults
    }

    /// Whether a refresh is needed (different YYYYMMDD from last refresh).
    public func needsRefresh() -> Bool {
        let today = Self.todayString()
        let last = userDefaults.string(forKey: Self.lastRefreshKey) ?? ""
        return last != today
    }

    /// Refreshes the LiteLLM cache if the YYYYMMDD gate allows it.
    public func refreshIfNeeded() async {
        guard needsRefresh() else {
            // Load from disk if not in memory
            if providerIndex.isEmpty {
                loadFromDisk()
            }
            return
        }
        await forceRefresh()
    }

    /// Always fetches fresh metadata, bypassing the date gate.
    public func forceRefresh() async {
        do {
            let changed = try await fetchIfChanged()
            if changed {
                logger.info("LiteLLM metadata updated")
            } else {
                logger.info("LiteLLM metadata unchanged (304)")
            }
        } catch {
            logger.error("LiteLLM fetch failed: \(error.localizedDescription, privacy: .public)")
            // Fall back to cached data
            if providerIndex.isEmpty {
                loadFromDisk()
            }
        }
        userDefaults.set(Self.todayString(), forKey: Self.lastRefreshKey)
    }

    /// Looks up a model's LiteLLM metadata, or `nil` when the provider is unmapped or the
    /// provider has no entry under this exact model name.
    ///
    /// The match is exact and deliberately has NO fallbacks: `liteLLMProviderName` must equal the
    /// entry's `litellm_provider`, and `modelID` must equal the entry's derived model name
    /// (case-insensitively). A miss means "LiteLLM does not catalogue this model for this
    /// provider" — a real, reportable answer that ``resolution(forModelID:liteLLMProviderName:)``
    /// surfaces, rather than something to paper over with a guess.
    ///
    /// Known limitation: LiteLLM's own resolver treats some provider values as families
    /// (`bedrock` ⊇ `bedrock_converse`, `vertex_ai` ⊇ `vertex_ai-*`, `fireworks_ai` ⊇
    /// `fireworks_ai-*`). We do not, so a provider must be mapped to the exact value its models
    /// carry. Every value present in the data is offered by ``allLiteLLMProviderNames()``.
    public func metadata(for modelID: String, liteLLMProviderName: String?) -> LiteLLMEntry? {
        guard let provider = liteLLMProviderName else { return nil }
        return lookup(provider: provider, modelID: modelID)
    }

    /// Exact match first, then the `-cloud` alias: LiteLLM catalogues Ollama's HOSTED models with
    /// a `-cloud` suffix (`ollama/gpt-oss:120b-cloud`) while Ollama Cloud's own /models lists
    /// them bare (`gpt-oss:120b`). The suffixed key is the same model on the same host, so the
    /// fallback recovers real data; for providers without that convention it simply never matches.
    private func lookup(provider: String, modelID: String) -> LiteLLMEntry? {
        guard let models = providerIndex[provider] else { return nil }
        let normalized = Self.normalize(modelID)
        return models[normalized] ?? models[normalized + "-cloud"]
    }

    /// Populates the index directly from raw LiteLLM JSON, bypassing the network and disk cache.
    /// Exists so the matching rules can be tested against fixed upstream key shapes.
    func ingestForTesting(_ data: Data) {
        parseMetadata(data)
    }

    /// Why a `(provider, model)` pair does or doesn't resolve to LiteLLM metadata. Distinguishes
    /// the failure levels so the UI can say which one to fix.
    public enum Resolution: Sendable, Equatable {
        /// Metadata found.
        case resolved
        /// The provider has no `liteLLMProviderName` — nothing was attempted.
        case providerNotMapped
        /// The mapped name matches no `litellm_provider` value in the data.
        case providerNotFound
        /// The provider exists, but it catalogues no model under this name.
        case modelNotFound
    }

    /// Classifies a `(provider, model)` pair — powers the coverage view's check/X and the
    /// inspector's "missing metadata for this provider/model" distinction.
    public func resolution(forModelID modelID: String, liteLLMProviderName: String?) -> Resolution {
        guard let provider = liteLLMProviderName else { return .providerNotMapped }
        guard providerIndex[provider] != nil else { return .providerNotFound }
        return lookup(provider: provider, modelID: modelID) == nil ? .modelNotFound : .resolved
    }

    /// Classifies many models against one provider in a single actor hop — the coverage view
    /// asks about every model a provider lists, and a hop per model would be hundreds.
    public func resolutions(forModelIDs modelIDs: [String], liteLLMProviderName: String?) -> [String: Resolution] {
        var result: [String: Resolution] = [:]
        for modelID in modelIDs {
            result[modelID] = resolution(forModelID: modelID, liteLLMProviderName: liteLLMProviderName)
        }
        return result
    }

    /// Every distinct `litellm_provider` value present in the data, sorted, with the number of
    /// models each catalogues. This is the authoritative picker list — it comes from the data
    /// itself, not from LiteLLM's `LlmProviders` enum, which the file does not conform to.
    public func allLiteLLMProviderNames() -> [(name: String, modelCount: Int)] {
        providerIndex
            .map { (name: $0.key, modelCount: $0.value.count) }
            .sorted { $0.name < $1.name }
    }

    /// The `litellm_provider` values that catalogue a model under this exact name — lets the
    /// picker offer "providers that actually have this model" when a mapping is being fixed.
    public func liteLLMProviderNames(matchingModelID modelID: String) -> [String] {
        let name = Self.normalize(modelID)
        return providerIndex
            .filter { $0.value[name] != nil }
            .map(\.key)
            .sorted()
    }

    /// Derives the model name to index an entry under, by removing the entry's OWN provider
    /// prefix when the key carries one.
    ///
    /// Only the entry's own prefix is stripped, which is what makes the junk keys self-reject:
    /// `1024-x-1024/dall-e-2` is `openai`, so `1024-x-1024` ≠ `openai`, nothing is stripped, and
    /// it can never match a query for `dall-e-2`. Meanwhile `mistral/codestral-latest` →
    /// `codestral-latest` and `openrouter/anthropic/claude-3-haiku` →
    /// `anthropic/claude-3-haiku` both reduce to exactly what the provider's API reports, and
    /// bare keys (`claude-fable-5`) pass through untouched.
    static func modelName(fromKey key: String, provider: String) -> String {
        let prefix = "\(provider)/"
        let name = key.hasPrefix(prefix) ? String(key.dropFirst(prefix.count)) : key
        return normalize(name)
    }

    /// Case-folds a model name. LiteLLM and provider APIs disagree on casing for the same model
    /// (`deepinfra/meta-llama/Llama-3.2-11B-Vision-Instruct` vs a lowercased API listing), and
    /// that disagreement is never meaningful.
    static func normalize(_ modelName: String) -> String {
        modelName.lowercased()
    }

    // MARK: - Private

    private func fetchIfChanged() async throws -> Bool {
        let cachedHeaders = loadCachedHeaders()

        var request = URLRequest(url: Self.liteLLMURL)
        request.httpMethod = "GET"
        request.timeoutInterval = 30

        if let etag = cachedHeaders["ETag"] {
            request.setValue(etag, forHTTPHeaderField: "If-None-Match")
        }
        if let lastModified = cachedHeaders["Last-Modified"] {
            request.setValue(lastModified, forHTTPHeaderField: "If-Modified-Since")
        }

        let requestLog: LLMRequestLogger.RequestLogToken? = Self.verboseLogging
            ? LLMRequestLogger.logBodylessRequest(label: "LiteLLM", method: "GET", url: Self.liteLLMURL)
            : nil

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ModelMetadataError.invalidResponse
        }

        // Gated on the token, not on `verboseLogging` again: the flag is a mutable static, and
        // re-reading it can log half a pair if it flips during the round trip.
        if requestLog != nil {
            logger.debug("LiteLLM fetch: HTTP \(http.statusCode) bytes=\(data.count)")
            // Through the shared logger so the metadata fetch sits in the same directory and
            // timeline as everything else, and so the status code is recorded — a 304 is the
            // difference between "fetched" and "reused the cache", and it used to be invisible.
            LLMRequestLogger.logResponse(label: "LiteLLM", statusCode: http.statusCode, data: data, for: requestLog)
        }

        if http.statusCode == 304 {
            // Not modified — load from disk if needed
            if providerIndex.isEmpty {
                loadFromDisk()
            }
            return false
        }

        guard (200..<300).contains(http.statusCode) else {
            throw ModelMetadataError.httpError(statusCode: http.statusCode)
        }

        // Save raw data and headers
        try ensureDirectory()
        let metadataURL = storageDirectory.appendingPathComponent(Self.metadataFilename)
        try data.write(to: metadataURL, options: .atomic)

        var headers: [String: String] = [:]
        if let etag = http.value(forHTTPHeaderField: "ETag") {
            headers["ETag"] = etag
        }
        if let lastModified = http.value(forHTTPHeaderField: "Last-Modified") {
            headers["Last-Modified"] = lastModified
        }
        let headersURL = storageDirectory.appendingPathComponent(Self.headersFilename)
        let headersData = try JSONEncoder().encode(headers)
        try headersData.write(to: headersURL, options: .atomic)

        // Parse into memory
        parseMetadata(data)
        return true
    }

    private func loadFromDisk() {
        let metadataURL = storageDirectory.appendingPathComponent(Self.metadataFilename)
        guard FileManager.default.fileExists(atPath: metadataURL.path) else { return }
        do {
            let data = try Data(contentsOf: metadataURL)
            parseMetadata(data)
            logger.debug("Loaded LiteLLM metadata from disk (\(self.providerIndex.count) providers, \(self.providerIndex.values.reduce(0) { $0 + $1.count }) models)")
        } catch {
            logger.error("Failed to load cached LiteLLM metadata: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func parseMetadata(_ data: Data) {
        do {
            guard let rawDict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                logger.error("LiteLLM metadata is not a JSON dictionary")
                return
            }

            var index: [String: [String: LiteLLMEntry]] = [:]

            for (key, value) in rawDict {
                // "sample_spec" is LiteLLM's schema documentation, not a model — its
                // litellm_provider is the literal string "one of https://docs.litellm.ai/…",
                // which sails through the string guard below and minted a phantom provider.
                if key == "sample_spec" { continue }
                guard let modelDict = value as? [String: Any] else { continue }
                // The `litellm_provider` field is the only authoritative provider marker, so an
                // entry without one is unroutable and is dropped. (Upstream, a nil provider means
                // "no constraint"; we have no use for a wildcard entry.)
                guard let provider = modelDict["litellm_provider"] as? String else { continue }

                let entry = LiteLLMEntry(
                    maxInputTokens: modelDict["max_input_tokens"] as? Int,
                    maxOutputTokens: modelDict["max_output_tokens"] as? Int,
                    pricing: Self.parsePricing(from: modelDict),
                    supportsToolUse: modelDict["supports_function_calling"] as? Bool ?? false,
                    supportsVision: modelDict["supports_vision"] as? Bool ?? false,
                    supportsReasoning: modelDict["supports_reasoning"] as? Bool ?? false,
                    supportsPromptCaching: modelDict["supports_prompt_caching"] as? Bool ?? false,
                    supportsComputerUse: modelDict["supports_computer_use"] as? Bool ?? false,
                    supportsAudioInput: modelDict["supports_audio_input"] as? Bool ?? false,
                    supportsAudioOutput: modelDict["supports_audio_output"] as? Bool ?? false,
                    supportsVideoInput: modelDict["supports_video_input"] as? Bool ?? false,
                    supportsResponseSchema: modelDict["supports_response_schema"] as? Bool ?? false,
                    supportsParallelToolCalls: modelDict["supports_parallel_tool_calls"] as? Bool ?? false,
                    supportsPdfInput: modelDict["supports_pdf_input"] as? Bool ?? false,
                    supportsWebSearch: modelDict["supports_web_search"] as? Bool ?? false,
                    supportsSystemMessages: modelDict["supports_system_messages"] as? Bool ?? false,
                    supportsAssistantPrefill: modelDict["supports_assistant_prefill"] as? Bool ?? false,
                    supportsToolChoice: modelDict["supports_tool_choice"] as? Bool ?? false,
                    mode: modelDict["mode"] as? String,
                    supportedEndpoints: modelDict["supported_endpoints"] as? [String]
                )

                index[provider, default: [:]][Self.modelName(fromKey: key, provider: provider)] = entry
            }

            providerIndex = index
        } catch {
            logger.error("Failed to parse LiteLLM JSON: \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Pricing Parser

    /// Parses rich pricing data from a LiteLLM model dictionary.
    ///
    /// Handles base rates, token-count threshold tiers (e.g. `_above_200k_tokens`),
    /// service tier variants (`_priority`, `_flex`, `_batches`), and extended cache
    /// TTL rates (`_above_1hr`).
    private static func parsePricing(from dict: [String: Any]) -> ModelPricing? {
        let baseInput = dict["input_cost_per_token"] as? Double
        let baseOutput = dict["output_cost_per_token"] as? Double
        let baseCacheRead = dict["cache_read_input_token_cost"] as? Double
        let baseCacheWrite = dict["cache_creation_input_token_cost"] as? Double

        guard baseInput != nil || baseOutput != nil else { return nil }

        let base = PricingTier(
            input: baseInput,
            output: baseOutput,
            cacheRead: baseCacheRead,
            cacheWrite: baseCacheWrite
        )

        // Scan for token threshold tiers (e.g. "input_cost_per_token_above_200k_tokens").
        var thresholdMap: [Int: PricingTier] = [:]
        for (key, value) in dict {
            guard let cost = value as? Double else { continue }
            guard let threshold = Self.parseTokenThreshold(from: key) else { continue }
            // Skip service-tier and time-tier combined keys
            if key.hasSuffix("_priority") || key.hasSuffix("_flex") || key.hasSuffix("_batches") { continue }
            if key.contains("_above_1hr") { continue }

            var tier = thresholdMap[threshold] ?? PricingTier()
            if key.contains("input_cost_per_token") { tier.input = cost }
            else if key.contains("output_cost_per_token") { tier.output = cost }
            else if key.contains("cache_read_input_token_cost") { tier.cacheRead = cost }
            else if key.contains("cache_creation_input_token_cost") { tier.cacheWrite = cost }
            thresholdMap[threshold] = tier
        }
        let tokenThresholdTiers = thresholdMap.map { TokenThresholdTier(tokenThreshold: $0.key, rates: $0.value) }.sorted()

        // Scan for service tier variants (_priority, _flex, _batches).
        var serviceTiers: [String: PricingTier] = [:]
        let serviceSuffixes = ["_priority", "_flex", "_batches"]
        for (key, value) in dict {
            guard let cost = value as? Double else { continue }
            for suffix in serviceSuffixes {
                guard key.hasSuffix(suffix) else { continue }
                // Skip combined threshold+service keys
                if Self.parseTokenThreshold(from: key) != nil { continue }
                let tierName = String(suffix.dropFirst()) // remove leading underscore
                var tier = serviceTiers[tierName] ?? PricingTier()
                if key.contains("input_cost_per_token") { tier.input = cost }
                else if key.contains("output_cost_per_token") { tier.output = cost }
                else if key.contains("cache_read_input_token_cost") { tier.cacheRead = cost }
                else if key.contains("cache_creation_input_token_cost") { tier.cacheWrite = cost }
                serviceTiers[tierName] = tier
            }
        }

        // Extended cache TTL (Anthropic's _above_1hr).
        var extendedCacheTier: CacheWriteOverride?
        if let extCacheWrite = dict["cache_creation_input_token_cost_above_1hr"] as? Double {
            var thresholdOverrides: [TokenThresholdCacheWrite] = []
            // Check for combined 1hr + threshold keys (e.g. "cache_creation_input_token_cost_above_1hr_above_200k_tokens")
            for (key, value) in dict {
                guard key.contains("cache_creation_input_token_cost_above_1hr_above_"),
                      let cost = value as? Double,
                      let threshold = Self.parseTokenThreshold(from: key) else { continue }
                thresholdOverrides.append(TokenThresholdCacheWrite(tokenThreshold: threshold, cacheWrite: cost))
            }
            thresholdOverrides.sort { $0.tokenThreshold < $1.tokenThreshold }
            extendedCacheTier = CacheWriteOverride(cacheWrite: extCacheWrite, thresholdOverrides: thresholdOverrides)
        }

        return ModelPricing(
            base: base,
            tokenThresholdTiers: tokenThresholdTiers,
            serviceTiers: serviceTiers,
            extendedCacheTier: extendedCacheTier
        )
    }

    /// Extracts a token threshold from a LiteLLM key like "input_cost_per_token_above_200k_tokens".
    /// Returns the threshold in tokens (e.g. 200_000 for "200k"), or `nil` if the key
    /// doesn't contain a threshold pattern.
    private static func parseTokenThreshold(from key: String) -> Int? {
        // Match patterns like "_above_128k_tokens", "_above_200k_tokens", "_above_272k_tokens"
        guard let range = key.range(of: #"_above_(\d+)k?_tokens"#, options: .regularExpression) else {
            return nil
        }
        let match = key[range]
        // Extract the numeric part
        guard let digitRange = match.range(of: #"\d+"#, options: .regularExpression) else {
            return nil
        }
        guard let number = Int(match[digitRange]) else { return nil }
        // "k" suffix means thousands
        if match.contains("k_tokens") {
            return number * 1000
        }
        return number
    }

    private func loadCachedHeaders() -> [String: String] {
        let headersURL = storageDirectory.appendingPathComponent(Self.headersFilename)
        guard FileManager.default.fileExists(atPath: headersURL.path) else { return [:] }
        do {
            let data = try Data(contentsOf: headersURL)
            return try JSONDecoder().decode([String: String].self, from: data)
        } catch {
            return [:]
        }
    }

    private func ensureDirectory() throws {
        try FileManager.default.createDirectory(
            at: storageDirectory,
            withIntermediateDirectories: true
        )
    }

    private static func todayString() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd"
        formatter.timeZone = .current
        return formatter.string(from: Date())
    }

    /// Strips common provider prefixes like "anthropic/", "openai/", "ollama/" from model IDs.
}

/// A parsed entry from the LiteLLM model cost/metadata map.
public struct LiteLLMEntry: Sendable {
    public let maxInputTokens: Int?
    public let maxOutputTokens: Int?
    /// Rich pricing data parsed from all cost-related fields.
    public let pricing: ModelPricing?
    public let supportsToolUse: Bool
    public let supportsVision: Bool
    public let supportsReasoning: Bool
    public let supportsPromptCaching: Bool
    public let supportsComputerUse: Bool
    public let supportsAudioInput: Bool
    public let supportsAudioOutput: Bool
    public let supportsVideoInput: Bool
    public let supportsResponseSchema: Bool
    public let supportsParallelToolCalls: Bool
    public let supportsPdfInput: Bool
    public let supportsWebSearch: Bool
    public let supportsSystemMessages: Bool
    public let supportsAssistantPrefill: Bool
    public let supportsToolChoice: Bool
    public let mode: String?
    /// API endpoints this model supports (e.g. `["/v1/chat/completions"]`, `["/v1/realtime"]`).
    public let supportedEndpoints: [String]?

    /// Whether this model supports the standard chat completions endpoint.
    ///
    /// Answers only "will `/v1/chat/completions` accept this model" — nothing about whether the
    /// model is a good agent. Gemini generates images *through* the chat endpoint, so its image
    /// models legitimately report `true` here; whether they can call tools is a separate
    /// question that `capabilities.toolUse` answers on evidence.
    ///
    /// Precedence:
    /// 1. `supported_endpoints` when present — an explicit list outranks `mode`, which names a
    ///    model's *primary* use rather than its only one (`o3-deep-research` is
    ///    `mode: "responses"` yet lists `/v1/chat/completions`). Only ~560 of 2968 carry it.
    /// 2. `mode` otherwise — set on 2959 of 2968, so it covers nearly everything else.
    /// 3. Fail open when neither is known.
    public var supportsChatCompletions: Bool {
        if let endpoints = supportedEndpoints { return endpoints.contains("/v1/chat/completions") }
        if let mode { return mode == "chat" }
        return true
    }

    /// Merges this LiteLLM entry's capabilities into an existing ``ModelCapabilities``,
    /// filling in only fields that are currently `false`.
    public func mergeCapabilities(into capabilities: inout ModelCapabilities) {
        if supportsToolUse { capabilities.toolUse = true }
        if supportsVision { capabilities.vision = true }
        if supportsReasoning { capabilities.reasoning = true }
        if supportsPromptCaching { capabilities.promptCaching = true }
        if supportsComputerUse { capabilities.computerUse = true }
        if supportsAudioInput { capabilities.audioInput = true }
        if supportsAudioOutput { capabilities.audioOutput = true }
        if supportsVideoInput { capabilities.videoInput = true }
        if supportsResponseSchema { capabilities.responseSchema = true }
        if supportsParallelToolCalls { capabilities.parallelToolCalls = true }
        if supportsPdfInput { capabilities.pdfInput = true }
        if supportsWebSearch { capabilities.webSearch = true }
        if supportsSystemMessages { capabilities.systemMessages = true }
        if supportsAssistantPrefill { capabilities.assistantPrefill = true }
        if supportsToolChoice { capabilities.toolChoice = true }
    }
}

/// Errors from the metadata service.
private enum ModelMetadataError: Error, LocalizedError {
    case invalidResponse
    case httpError(statusCode: Int)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid HTTP response from LiteLLM metadata server"
        case .httpError(let code):
            return "LiteLLM metadata fetch failed with HTTP \(code)"
        }
    }
}
