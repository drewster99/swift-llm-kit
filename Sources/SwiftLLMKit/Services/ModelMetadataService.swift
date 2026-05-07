import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "ModelMetadata")

/// Manages the LiteLLM model metadata cache.
///
/// Downloads the LiteLLM model cost map from GitHub, caches it locally,
/// and provides lookup by model ID. Uses conditional HTTP requests
/// (ETag/Last-Modified) to avoid redundant downloads.
public actor ModelMetadataService {
    /// When true, full LiteLLM fetch responses are logged to `$TMPDIR/SwiftLLMKit-Logs/`.
    public nonisolated(unsafe) static var verboseLogging = false

    private let storageDirectory: URL
    private let userDefaults: UserDefaults
    private let userDefaultsSuiteName: String

    /// In-memory index: raw model ID → parsed metadata.
    private var metadataIndex: [String: LiteLLMEntry] = [:]
    /// Secondary index with provider prefixes stripped for fuzzy matching.
    private var strippedIndex: [String: LiteLLMEntry] = [:]

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
            if metadataIndex.isEmpty {
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
            if metadataIndex.isEmpty {
                loadFromDisk()
            }
        }
        userDefaults.set(Self.todayString(), forKey: Self.lastRefreshKey)
    }

    /// Looks up metadata for a model by its raw ID.
    /// When a `providerType` is supplied, tries the provider's LiteLLM-prefixed key first
    /// (e.g. `mistral/mistral-large-2512`), then falls back to exact match and stripped index.
    public func metadata(for modelID: String, providerType: ProviderAPIType? = nil) -> LiteLLMEntry? {
        // Try provider-prefixed key first (most precise)
        if let prefix = providerType?.liteLLMPrefix {
            if let entry = metadataIndex["\(prefix)/\(modelID)"] {
                return entry
            }
        }
        // Exact match on raw model ID
        if let entry = metadataIndex[modelID] {
            return entry
        }
        // Stripped index fallback
        let stripped = Self.stripProviderPrefix(modelID)
        if let entry = strippedIndex[stripped] {
            return entry
        }
        // Alias resolution: "-latest" / "-latest-YYYY-MM-DD" suffixes don't appear in
        // LiteLLM, which uses versioned names (e.g. "ministral-3-14b-2512" vs "ministral-14b-latest").
        // Extract the base name from the alias and find the best match in the provider's entries.
        if let prefix = providerType?.liteLLMPrefix {
            if let entry = fuzzyMatchAlias(modelID, prefix: prefix) {
                return entry
            }
        }
        return fuzzyMatchAlias(modelID, prefix: nil)
    }

    /// Attempts to match a model alias (e.g. "ministral-14b-latest") against LiteLLM entries
    /// by stripping the "-latest" suffix and finding a key whose base name contains the same stem.
    private func fuzzyMatchAlias(_ modelID: String, prefix: String?) -> LiteLLMEntry? {
        // Strip common alias suffixes
        let aliasSuffixes = ["-latest"]
        var stem = modelID
        for suffix in aliasSuffixes {
            if stem.hasSuffix(suffix) {
                stem = String(stem.dropLast(suffix.count))
                break
            }
        }
        // Also strip date suffixes like "-20250301"
        if let lastDash = stem.lastIndex(of: "-") {
            let tail = stem[stem.index(after: lastDash)...]
            if tail.count >= 8, tail.allSatisfy(\.isNumber) {
                stem = String(stem[..<lastDash])
            }
        }
        guard stem != modelID else { return nil } // No alias suffix was stripped

        // Search for a matching key in the index
        let searchPrefix = prefix.map { "\($0)/" } ?? ""
        var bestMatch: (key: String, entry: LiteLLMEntry)?
        for (key, entry) in metadataIndex {
            guard key.hasPrefix(searchPrefix) else { continue }
            let keyBase = searchPrefix.isEmpty ? key : String(key.dropFirst(searchPrefix.count))
            // Check if the key's base name contains the stem (handles "ministral-3-14b-2512" matching "ministral-14b")
            if keyBase.contains(stem) || stem.contains(keyBase) {
                // Prefer shorter keys (more specific match)
                if bestMatch == nil || key.count < bestMatch!.key.count {
                    bestMatch = (key, entry)
                }
            }
        }
        return bestMatch?.entry
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

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ModelMetadataError.invalidResponse
        }

        if Self.verboseLogging {
            logger.debug("LiteLLM fetch: HTTP \(http.statusCode) bytes=\(data.count)")
            ModelFetchService.logData(label: "LiteLLM", data: data)
        }

        if http.statusCode == 304 {
            // Not modified — load from disk if needed
            if metadataIndex.isEmpty {
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
            logger.debug("Loaded LiteLLM metadata from disk (\(self.metadataIndex.count) entries)")
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

            var index: [String: LiteLLMEntry] = [:]
            var stripped: [String: LiteLLMEntry] = [:]

            for (key, value) in rawDict {
                // Skip the "sample_spec" key and any non-dict entries
                guard let modelDict = value as? [String: Any] else { continue }

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

                index[key] = entry
                let strippedKey = Self.stripProviderPrefix(key)
                if strippedKey != key {
                    stripped[strippedKey] = entry
                }
            }

            metadataIndex = index
            strippedIndex = stripped
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
    static func stripProviderPrefix(_ modelID: String) -> String {
        let prefixes = [
            "anthropic/", "openai/", "ollama/", "azure/", "bedrock/",
            "vertex_ai/", "cohere/", "mistral/", "groq/", "deepseek/",
            "together_ai/", "fireworks_ai/", "perplexity/", "anyscale/",
            "gemini/", "huggingface/", "xai/"
        ]
        for prefix in prefixes {
            if modelID.hasPrefix(prefix) {
                return String(modelID.dropFirst(prefix.count))
            }
        }
        return modelID
    }
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
    public var supportsChatCompletions: Bool {
        guard let endpoints = supportedEndpoints else { return true }
        return endpoints.contains("/v1/chat/completions")
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
