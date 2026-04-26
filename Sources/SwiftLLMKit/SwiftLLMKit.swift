import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "SwiftLLMKit")

/// Central manager for LLM providers, model discovery, and configuration.
///
/// Provides CRUD for providers (with Keychain-stored API keys), model catalog
/// management (provider APIs + LiteLLM metadata enrichment), and configuration
/// management (provider + model + user settings). Also prepares authenticated
/// URLRequests for the app to complete with messages/tools.
///
/// Usage:
/// ```swift
/// let kit = LLMKitManager(
///     appIdentifier: Bundle.main.bundleIdentifier ?? "com.example.app",
///     keychainServicePrefix: "com.example.SwiftLLMKit"
/// )
/// kit.load()
/// await kit.refreshIfNeeded()
/// ```
@Observable
@MainActor
public final class LLMKitManager {
    // MARK: - Published State

    /// All registered providers.
    public private(set) var providers: [ModelProvider] = []
    /// All known models across all providers.
    public private(set) var models: [ModelInfo] = []
    /// All user-defined model configurations.
    public private(set) var configurations: [ModelConfiguration] = []
    /// Whether a model refresh is in progress.
    public private(set) var isRefreshing: Bool = false
    /// Most recent persistence error, if any. Observable so UI can show an alert/banner.
    public var persistenceError: String?
    /// When true, providers created by `makeProvider(for:)` will log full request/response JSON.
    public var verboseLogging: Bool = false
    /// Errors from the most recent model refresh, keyed by provider name.
    public private(set) var refreshErrors: [String: String] = [:]
    /// Incremented every time any provider's API key is written via `addProvider`,
    /// `updateProvider`, or `setBuiltInProviderAPIKey`. Observable so SwiftUI views
    /// that read from `apiKey(for:)` (which goes through Keychain and is therefore
    /// not reactive by itself) can key off this counter to re-render when a key
    /// changes externally (e.g. undo/redo from another window).
    public private(set) var apiKeyChangeCounter: Int = 0

    // MARK: - Services

    private let storage: StorageManager
    private let keychain: KeychainService
    private let fetchService: ModelFetchService
    private let metadataService: ModelMetadataService

    // MARK: - Override Layers

    /// App-bundled model metadata overrides (gap-fill priority above LiteLLM).
    private let bundledRegistry: BundledModelMetadataRegistry

    /// User-provided model metadata overrides (highest priority, force-replaces).
    /// Keyed by `"providerID/modelID"`.
    private var userOverrides: [String: ModelMetadataOverride] = [:]

    /// Tracks whether each dataset loaded successfully. Prevents saving empty data
    /// over a file that failed to decode (e.g. after a schema change).
    private var providersLoadedOK = false
    private var configurationsLoadedOK = false

    // MARK: - Init

    /// Creates a new SwiftLLMKit instance.
    /// - Parameters:
    ///   - appIdentifier: Typically `Bundle.main.bundleIdentifier`.
    ///   - keychainServicePrefix: Reverse-DNS prefix for Keychain entries.
    public init(
        appIdentifier: String,
        keychainServicePrefix: String
    ) {
        let storage = StorageManager(appIdentifier: appIdentifier)
        self.storage = storage
        self.keychain = KeychainService(
            keychainServicePrefix: keychainServicePrefix,
            appIdentifier: appIdentifier
        )
        self.fetchService = ModelFetchService()
        let suiteName = "SwiftLLMKit.\(appIdentifier)"
        self.metadataService = ModelMetadataService(
            storageDirectory: storage.baseDirectory,
            userDefaultsSuiteName: suiteName
        )
        self.bundledRegistry = BundledModelMetadataRegistry.load()
    }

    /// Updates user-provided model metadata overrides at runtime.
    ///
    /// Overrides are keyed by `"providerID/modelID"` and force-replace all
    /// lower-priority metadata sources. Call this when the user edits overrides
    /// in the UI, then trigger a refresh to re-enrich the model catalog.
    public func setUserOverrides(_ overrides: [String: ModelMetadataOverride]) {
        self.userOverrides = overrides
    }

    // MARK: - Persistence

    /// Loads providers, configurations, and cached models from disk, then seeds/migrates
    /// the bundled built-in providers (see `seedBuiltInProviders()`).
    public func load() {
        persistenceError = nil
        do {
            providers = try storage.loadProviders()
            providersLoadedOK = true
        } catch {
            let msg = "Failed to load providers: \(error.localizedDescription)"
            logger.error("\(msg, privacy: .public)")
            persistenceError = msg
        }
        do {
            configurations = try storage.loadConfigurations()
            configurationsLoadedOK = true
        } catch {
            let msg = "Failed to load configurations: \(error.localizedDescription)"
            logger.error("\(msg, privacy: .public)")
            persistenceError = msg
        }
        do {
            models = try storage.loadModelCatalog()
        } catch {
            let msg = "Failed to load model catalog: \(error.localizedDescription)"
            logger.error("\(msg, privacy: .public)")
            persistenceError = msg
        }

        seedBuiltInProviders()
    }

    /// Persists current state to disk.
    public func save() {
        var errors: [String] = []
        if providersLoadedOK {
            do {
                try storage.saveProviders(providers)
            } catch {
                let msg = "Failed to save providers: \(error.localizedDescription)"
                logger.error("\(msg, privacy: .public)")
                errors.append(msg)
            }
        }
        if configurationsLoadedOK {
            do {
                try storage.saveConfigurations(configurations)
            } catch {
                let msg = "Failed to save configurations: \(error.localizedDescription)"
                logger.error("\(msg, privacy: .public)")
                errors.append(msg)
            }
        }
        do {
            try storage.saveModelCatalog(models)
        } catch {
            let msg = "Failed to save model catalog: \(error.localizedDescription)"
            logger.error("\(msg, privacy: .public)")
            errors.append(msg)
        }
        persistenceError = errors.isEmpty ? nil : errors.joined(separator: "; ")
    }

    // MARK: - Provider CRUD

    /// Adds a new provider and stores its API key in Keychain.
    ///
    /// - Throws: If the Keychain operation fails. The provider is not persisted on failure.
    public func addProvider(_ provider: ModelProvider, apiKey: String) throws {
        if !apiKey.isEmpty {
            try keychain.save(apiKey: apiKey, forProviderID: provider.id)
            apiKeyChangeCounter &+= 1
        }
        providers.append(provider)
        saveProviders()
    }

    /// Updates an existing provider. If `apiKey` is non-nil, updates the Keychain.
    ///
    /// For built-in providers, only the API key may change — `name`, `apiType`, and
    /// `endpoint` are restored from the bundled preset, ignoring any incoming changes.
    ///
    /// - Throws: If the Keychain operation fails, or if a built-in provider is asked
    ///           to mutate an immutable field with a new value. The provider is not
    ///           updated on failure.
    public func updateProvider(_ provider: ModelProvider, apiKey: String?) throws {
        guard let index = providers.firstIndex(where: { $0.id == provider.id }) else { return }
        if let apiKey {
            if apiKey.isEmpty {
                try keychain.delete(forProviderID: provider.id)
            } else {
                try keychain.save(apiKey: apiKey, forProviderID: provider.id)
            }
            apiKeyChangeCounter &+= 1
        }

        // Built-in providers refuse mutation of name/apiType/endpoint — clamp to the preset.
        if let preset = BuiltInProviders.preset(id: provider.id) {
            providers[index] = ModelProvider(builtIn: preset)
        } else {
            providers[index] = provider
        }
        saveProviders()
    }

    /// Sets the API key for a built-in provider.
    ///
    /// Convenience entry point for the per-built-in API-key UI: pass an empty string
    /// to clear the key. Throws `SwiftLLMKitError.notABuiltInProvider` if the ID is
    /// not a known built-in.
    public func setBuiltInProviderAPIKey(id: String, apiKey: String) throws {
        guard let preset = BuiltInProviders.preset(id: id) else {
            throw SwiftLLMKitError.notABuiltInProvider(id: id)
        }
        // Ensure the provider row exists (it should, after seeding, but be defensive).
        if !providers.contains(where: { $0.id == id }) {
            providers.append(ModelProvider(builtIn: preset))
            saveProviders()
        }
        if apiKey.isEmpty {
            try keychain.delete(forProviderID: id)
        } else {
            try keychain.save(apiKey: apiKey, forProviderID: id)
        }
        apiKeyChangeCounter &+= 1
    }

    /// Deletes a provider. Throws if any configuration references it, or if it is a
    /// built-in provider (which cannot be removed).
    public func deleteProvider(id: String) throws {
        if BuiltInProviders.isBuiltIn(id: id) {
            throw SwiftLLMKitError.providerIsBuiltIn(id: id)
        }
        let referencingConfigs = configurations.filter { $0.providerID == id }
        if !referencingConfigs.isEmpty {
            let names = referencingConfigs.map(\.name).joined(separator: ", ")
            throw SwiftLLMKitError.providerInUse(
                providerID: id,
                configNames: names
            )
        }

        providers.removeAll { $0.id == id }
        models.removeAll { $0.providerID == id }

        do {
            try keychain.delete(forProviderID: id)
        } catch {
            logger.error("Failed to delete API key: \(error.localizedDescription, privacy: .public)")
        }

        saveProviders()
        saveModelCatalog()
    }

    /// Retrieves the API key for a provider from Keychain.
    public func apiKey(for providerID: String) -> String? {
        keychain.apiKey(forProviderID: providerID)
    }

    // MARK: - Configuration CRUD

    /// Adds a new model configuration.
    public func addConfiguration(_ config: ModelConfiguration) {
        configurations.append(config)
        validateConfigurations()
    }

    /// Updates an existing model configuration.
    public func updateConfiguration(_ config: ModelConfiguration) {
        guard let index = configurations.firstIndex(where: { $0.id == config.id }) else { return }
        configurations[index] = config
        validateConfigurations()
    }

    /// Deletes a model configuration.
    public func deleteConfiguration(id: UUID) {
        configurations.removeAll { $0.id == id }
        validateConfigurations()
    }

    /// Creates a duplicate of an existing configuration with a new ID and "(Copy)" suffix.
    @discardableResult
    public func duplicateConfiguration(id: UUID) -> ModelConfiguration? {
        guard let original = configurations.first(where: { $0.id == id }) else { return nil }
        let newConfig = ModelConfiguration(
            name: "\(original.name) (Copy)",
            providerID: original.providerID,
            modelID: original.modelID,
            temperature: original.temperature,
            maxOutputTokens: original.maxOutputTokens,
            maxContextTokens: original.maxContextTokens,
            thinkingBudget: original.thinkingBudget,
            streaming: original.streaming
        )
        configurations.append(newConfig)
        validateConfigurations()
        return newConfig
    }

    // MARK: - Model Catalog

    /// Returns models available for a specific provider.
    public func models(for providerID: String) -> [ModelInfo] {
        models.filter { $0.providerID == providerID }
    }

    /// Returns info for a specific model by provider and model ID.
    public func modelInfo(providerID: String, modelID: String) -> ModelInfo? {
        models.first { $0.providerID == providerID && $0.modelID == modelID }
    }

    // MARK: - Refresh

    /// Refreshes model lists if the YYYYMMDD gate allows, and additionally fills in
    /// any "straggler" providers whose cache slice is empty but which have a usable
    /// credential (or are no-auth local providers).
    ///
    /// Without the straggler pass, providers that failed a prior fetch — or that were
    /// added in a later app version after the catalog was first populated — would stay
    /// empty forever, because the YYYYMMDD metadata gate skips the full refresh as soon
    /// as *any* provider has cached models.
    public func refreshIfNeeded() async {
        let needsMetadataRefresh = await metadataService.needsRefresh()

        if needsMetadataRefresh || models.isEmpty {
            await performRefresh()
            return
        }

        // Catalog looks fresh overall; fill in per-provider gaps for providers we
        // could plausibly refresh (saved key or no-auth local type).
        let stragglers = providers.filter { provider in
            models.contains(where: { $0.providerID == provider.id }) == false
                && providerIsRefreshable(provider)
        }
        guard !stragglers.isEmpty else { return }

        isRefreshing = true
        defer { isRefreshing = false }

        // Ensure metadata is at least loaded before enrichment. `forceRefresh` goes
        // through an ETag-gated fetch, so this is cheap when already current.
        await metadataService.forceRefresh()

        for provider in stragglers {
            await refreshProviderSliceLocked(provider)
        }
        saveModelCatalog()
        validateConfigurations()
    }

    /// Always performs a full refresh of every provider's models.
    public func forceRefresh() async {
        await performRefresh()
    }

    /// Refreshes the model catalog for a single provider.
    ///
    /// Used by the per-built-in API key UI: after the user pastes a key, we refetch just
    /// that provider's models so the model dropdown for the agents populates without
    /// re-querying every other provider.
    ///
    /// Replaces only this provider's entries in the cached `models` array. Other
    /// providers' cached models are preserved untouched.
    public func refreshModels(forProviderID providerID: String) async {
        guard let provider = providers.first(where: { $0.id == providerID }) else {
            logger.warning("refreshModels called with unknown providerID: \(providerID, privacy: .public)")
            return
        }

        isRefreshing = true
        defer { isRefreshing = false }

        // Refresh metadata first so enrichment uses the latest LiteLLM data.
        await metadataService.forceRefresh()

        await refreshProviderSliceLocked(provider)
        saveModelCatalog()
        validateConfigurations()
    }

    /// Fetches one provider's models and replaces just that provider's slice in the
    /// cached `models` array, updating `refreshErrors[provider.name]` accordingly.
    ///
    /// Does NOT touch `isRefreshing`, refresh LiteLLM metadata, persist the catalog to
    /// disk, or re-validate configurations — callers are expected to bracket a sequence
    /// of these with those operations so that batched refreshes only pay those costs
    /// once.
    private func refreshProviderSliceLocked(_ provider: ModelProvider) async {
        let result = await fetchAndEnrich(provider: provider)

        var combined = models.filter { $0.providerID != provider.id }
        combined.append(contentsOf: result.models)
        models = combined

        if let error = result.error {
            refreshErrors[provider.name] = error
        } else {
            refreshErrors.removeValue(forKey: provider.name)
        }
    }

    /// Whether it's worth automatically attempting a refresh for this provider:
    /// either a non-empty API key is in the Keychain, or the provider's API type is
    /// one that works without authentication (local servers).
    ///
    /// Used by `refreshIfNeeded` to avoid hammering cloud providers with guaranteed
    /// 401s every launch when the user hasn't configured a key yet.
    private func providerIsRefreshable(_ provider: ModelProvider) -> Bool {
        if let key = keychain.apiKey(forProviderID: provider.id), !key.isEmpty {
            return true
        }
        switch provider.apiType {
        case .ollama, .lmStudio:
            return true
        case .anthropic, .openAICompatible, .mistral, .gemini, .huggingFace,
             .xAI, .zAI, .metaLlama, .alibabaCloud, .openRouter:
            return false
        }
    }

    private func performRefresh() async {
        isRefreshing = true
        defer { isRefreshing = false }

        // 1. Refresh LiteLLM metadata
        await metadataService.forceRefresh()

        // 2. Fetch models from each provider
        var allModels: [ModelInfo] = []
        var errors: [String: String] = [:]
        for provider in providers {
            let result = await fetchAndEnrich(provider: provider)
            allModels.append(contentsOf: result.models)
            if let error = result.error {
                errors[provider.name] = error
            }
        }
        refreshErrors = errors

        models = allModels
        saveModelCatalog()

        // Re-validate configurations against the updated catalog
        validateConfigurations()
    }

    /// Fetches models for one provider and applies the layered metadata enrichment.
    /// On failure, returns the previously cached models for that provider plus an error
    /// string so the caller can preserve state without losing the catalog.
    private func fetchAndEnrich(provider: ModelProvider) async -> (models: [ModelInfo], error: String?) {
        let apiKey = keychain.apiKey(forProviderID: provider.id)
        do {
            var providerModels = try await fetchService.fetchModels(
                from: provider,
                apiKey: apiKey
            )

            // Enrich with layered metadata overrides.
            // Provider API is the base. Gap-fillers run highest-priority first so
            // they claim nil fields before lower-priority sources can.
            // Priority: user (force-replace) > provider API (base) > bundled (gap-fill) > LiteLLM (gap-fill)
            for i in providerModels.indices {
                let modelID = providerModels[i].modelID

                // Layer 0a: App-bundled provider-wide defaults (gap-fill).
                // Applies to every model from this providerID, e.g. all OpenAI
                // models opting into `useMaxCompletionTokens`. Per-model entries
                // below override on a flag-by-flag basis.
                if let providerWide = bundledRegistry.defaults(providerID: provider.id) {
                    providerWide.apply(to: &providerModels[i], forceReplace: false)
                }

                // Layer 0b: App-bundled per-(providerID, modelID) (gap-fill).
                // Pinpoints a single built-in provider's model — used when the
                // apiType is shared between built-in and user-created providers.
                if let providerScoped = bundledRegistry.override(providerID: provider.id, modelID: modelID) {
                    providerScoped.apply(to: &providerModels[i], forceReplace: false)
                }

                // Layer 1: App-bundled per-(apiType, modelID) (gap-fill, higher priority than LiteLLM)
                if let bundled = bundledRegistry.override(providerAPIType: provider.apiType.rawValue, modelID: modelID) {
                    bundled.apply(to: &providerModels[i], forceReplace: false)
                }

                // Layer 2: LiteLLM (gap-fill, fills anything still nil after bundled)
                if let litellm = await metadataService.metadata(for: modelID, providerType: provider.apiType) {
                    let litellmOverride = ModelMetadataOverride(
                        maxInputTokens: litellm.maxInputTokens,
                        maxOutputTokens: litellm.maxOutputTokens,
                        capabilities: ModelCapabilitiesOverride(
                            toolUse: litellm.supportsToolUse ? true : nil,
                            vision: litellm.supportsVision ? true : nil,
                            reasoning: litellm.supportsReasoning ? true : nil,
                            promptCaching: litellm.supportsPromptCaching ? true : nil,
                            computerUse: litellm.supportsComputerUse ? true : nil,
                            audioInput: litellm.supportsAudioInput ? true : nil,
                            audioOutput: litellm.supportsAudioOutput ? true : nil,
                            videoInput: litellm.supportsVideoInput ? true : nil,
                            responseSchema: litellm.supportsResponseSchema ? true : nil,
                            parallelToolCalls: litellm.supportsParallelToolCalls ? true : nil,
                            pdfInput: litellm.supportsPdfInput ? true : nil,
                            webSearch: litellm.supportsWebSearch ? true : nil,
                            systemMessages: litellm.supportsSystemMessages ? true : nil,
                            assistantPrefill: litellm.supportsAssistantPrefill ? true : nil,
                            toolChoice: litellm.supportsToolChoice ? true : nil
                        ),
                        pricing: litellm.pricing,
                        supportsChatCompletions: litellm.supportsChatCompletions ? nil : false
                    )
                    litellmOverride.apply(to: &providerModels[i], forceReplace: false)
                }

                // Layer 3: Provider API data is already the base (providerModels[i]).

                // Layer 4: User overrides (force-replace — user always wins)
                let userKey = "\(provider.id)/\(modelID)"
                if let userOverride = userOverrides[userKey] {
                    userOverride.apply(to: &providerModels[i], forceReplace: true)
                }
            }

            logger.info("Fetched \(providerModels.count, privacy: .public) models from \(provider.name, privacy: .public)")
            return (providerModels, nil)
        } catch {
            let errorMsg = error.localizedDescription
            logger.error("Failed to fetch models from \(provider.name, privacy: .public): \(errorMsg, privacy: .public)")
            // Keep any previously cached models for this provider so we don't blank out the UI.
            let cached = models.filter { $0.providerID == provider.id }
            return (cached, errorMsg)
        }
    }

    // MARK: - Validation

    /// Validates all configurations against current providers and models.
    public func validateConfigurations() {
        for i in configurations.indices {
            validateConfiguration(at: i)
        }
        saveConfigurations()
    }

    private func validateConfiguration(at index: Int) {
        let config = configurations[index]

        // Check provider exists
        guard let provider = providers.first(where: { $0.id == config.providerID }) else {
            configurations[index].isValid = false
            configurations[index].validationError = "Provider '\(config.providerID)' not found"
            return
        }

        // Temperature bounds (per-provider)
        let tempRange = provider.apiType.temperatureRange
        guard tempRange.contains(config.temperature) else {
            configurations[index].isValid = false
            configurations[index].validationError = "Temperature must be between \(tempRange.lowerBound) and \(tempRange.upperBound) for \(provider.apiType.displayName)"
            return
        }

        // Thinking budget is only supported for Anthropic and Alibaba Cloud
        if let budget = config.thinkingBudget, budget > 0,
           provider.apiType != .anthropic && provider.apiType != .alibabaCloud {
            configurations[index].isValid = false
            configurations[index].validationError = "Thinking budget is only supported for Anthropic and Alibaba Cloud providers"
            return
        }

        // Check model exists in that provider's models
        let providerModels = models.filter { $0.providerID == config.providerID }
        if !providerModels.isEmpty {
            guard let modelInfo = providerModels.first(where: { $0.modelID == config.modelID }) else {
                configurations[index].isValid = false
                configurations[index].validationError = "Model '\(config.modelID)' not found for this provider"
                return
            }

            // Check model supports chat completions
            if !modelInfo.supportsChatCompletions {
                configurations[index].isValid = false
                configurations[index].validationError = "Model '\(config.modelID)' does not support the chat completions endpoint"
                return
            }

            // Check maxOutputTokens doesn't exceed model's reported max
            if let modelMax = modelInfo.maxOutputTokens, config.maxOutputTokens > modelMax {
                configurations[index].isValid = false
                configurations[index].validationError = "Max output tokens (\(config.maxOutputTokens)) exceeds model limit (\(modelMax))"
                return
            }
        } else {
            // Models not yet loaded — allow starting but warn the user
            configurations[index].isValid = true
            configurations[index].validationError = "Models not yet loaded — will re-validate after refresh"
            return
        }

        configurations[index].isValid = true
        configurations[index].validationError = nil
    }

    // MARK: - Request Preparation

    /// Prepares an authenticated URLRequest stub for the given configuration.
    ///
    /// The returned `PreparedRequest` contains the URL, auth headers, and base body
    /// parameters (model, temperature, max_tokens, thinking, stream). The app adds
    /// messages/tools to the body and sends the request.
    public func prepareRequest(for configurationID: UUID) throws -> PreparedRequest {
        guard let config = configurations.first(where: { $0.id == configurationID }) else {
            throw SwiftLLMKitError.configurationNotFound(id: configurationID)
        }
        guard let provider = providers.first(where: { $0.id == config.providerID }) else {
            throw SwiftLLMKitError.providerNotFound(id: config.providerID)
        }

        let apiKey = keychain.apiKey(forProviderID: provider.id)

        // Build URL
        let url: URL
        switch provider.apiType {
        case .anthropic:
            url = provider.endpoint.ensureAnthropicV1().appendingPathComponent("messages")
        case .openAICompatible, .lmStudio, .mistral, .huggingFace, .xAI, .zAI, .metaLlama, .alibabaCloud, .openRouter:
            url = provider.endpoint.appendingPathComponent("chat/completions")
        case .ollama:
            url = provider.endpoint.appendingPathComponent("chat")
        case .gemini:
            let base = provider.endpoint.appendingPathComponent("models/\(config.modelID):generateContent")
            if let apiKey, !apiKey.isEmpty,
               var components = URLComponents(url: base, resolvingAgainstBaseURL: false) {
                var items = components.queryItems ?? []
                items.append(URLQueryItem(name: "key", value: apiKey))
                components.queryItems = items
                url = components.url ?? base
            } else {
                url = base
            }
        }

        // Build URLRequest with headers
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        switch provider.apiType {
        case .anthropic:
            if let apiKey {
                request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
            }
            request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        case .openAICompatible, .lmStudio, .mistral, .huggingFace, .xAI, .zAI, .metaLlama, .alibabaCloud, .openRouter:
            if let apiKey, !apiKey.isEmpty {
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            }
            if provider.apiType == .openRouter {
                let appName = (Bundle.main.object(forInfoDictionaryKey: "CFBundleName") as? String)
                    ?? (Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String)
                    ?? "SwiftLLMKit"
                let bundleID = Bundle.main.bundleIdentifier ?? "com.swiftllmkit.app"
                request.setValue("https://\(bundleID)", forHTTPHeaderField: "HTTP-Referer")
                request.setValue(appName, forHTTPHeaderField: "X-Title")
            }
        case .ollama:
            if let apiKey, !apiKey.isEmpty {
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            }
        case .gemini:
            // API key already in URL query parameter
            break
        }

        // Build base body
        var body: [String: Any] = [
            "model": config.modelID,
            "stream": config.streaming
        ]
        if !config.useDefaultTemperature {
            body["temperature"] = config.temperature
        }

        switch provider.apiType {
        case .anthropic:
            body["max_tokens"] = config.maxOutputTokens
            // Anthropic requires temperature = 1 when extended thinking is enabled.
            if let budget = config.thinkingBudget, budget > 0 {
                body["temperature"] = 1.0
                body["thinking"] = ["type": "enabled", "budget_tokens": max(budget, 1024)] as [String: Any]
            }
            // Enable prompt caching for all Anthropic requests.
            body["cache_control"] = config.extendedCacheTTL
                ? ["type": "ephemeral", "ttl": "1h"] as [String: Any]
                : ["type": "ephemeral"] as [String: Any]
        case .openAICompatible, .lmStudio, .mistral, .huggingFace, .xAI, .zAI, .metaLlama, .alibabaCloud, .openRouter:
            // GPT-5.x and o-series reject `max_tokens` and require
            // `max_completion_tokens`. Bundled-defaults JSON flags affected models
            // with `useMaxCompletionTokens: true`; users override per-model. Used to
            // be a hardcoded `provider.id == BuiltInProviders.ID.openai` check.
            let prepFlags = behaviorFlags(forProviderID: provider.id, modelID: config.modelID)
            let tokenLimitKey = prepFlags.useMaxCompletionTokens ? "max_completion_tokens" : "max_tokens"
            body[tokenLimitKey] = config.maxOutputTokens
        case .ollama:
            body["options"] = ["num_predict": config.maxOutputTokens] as [String: Any]
        case .gemini:
            var genConfig: [String: Any] = ["maxOutputTokens": config.maxOutputTokens]
            if !config.useDefaultTemperature {
                genConfig["temperature"] = config.temperature
            }
            body["generationConfig"] = genConfig
        }

        return PreparedRequest(
            urlRequest: request,
            baseBody: body,
            providerType: provider.apiType,
            streaming: config.streaming
        )
    }

    // MARK: - Provider Factory

    /// Creates a configured LLM provider for the given configuration ID.
    ///
    /// Resolves the `ModelConfiguration`, its associated `ModelProvider`, and builds
    /// an API key closure that reads from Keychain at point of use.
    ///
    /// - Parameter configurationID: The UUID of a `ModelConfiguration`.
    /// - Returns: A fully configured `LLMProvider` ready to send messages.
    /// - Throws: `SwiftLLMKitError` if the configuration or provider cannot be found.
    public func makeProvider(for configurationID: UUID) throws -> any LLMProvider {
        guard let config = configurations.first(where: { $0.id == configurationID }) else {
            throw SwiftLLMKitError.configurationNotFound(id: configurationID)
        }
        guard let modelProvider = providers.first(where: { $0.id == config.providerID }) else {
            throw SwiftLLMKitError.providerNotFound(id: config.providerID)
        }

        let providerID = modelProvider.id
        let providerName = modelProvider.name
        let keychainRef = self.keychain
        let readAPIKey: @Sendable () -> String = {
            guard let key = keychainRef.apiKey(forProviderID: providerID), !key.isEmpty else {
                logger.error("API key missing for provider '\(providerName, privacy: .public)' (id: \(providerID, privacy: .public)) — Keychain returned nil. Requests will fail with authentication errors.")
                return ""
            }
            return key
        }
        let verbose = self.verboseLogging

        // Resolve the merged behavior flags for this (provider, model) so providers
        // can read knobs without reaching back into the manager. `behaviorFlags`
        // returns the catalog-resolved flags (bundled + LiteLLM + user overrides
        // already merged by `fetchProviderModels`).
        let flags = behaviorFlags(forProviderID: modelProvider.id, modelID: config.modelID)

        switch modelProvider.apiType {
        case .anthropic:
            return AnthropicProvider(
                configuration: config, provider: modelProvider,
                readAPIKey: readAPIKey, verboseLogging: verbose
            )
        case .openAICompatible, .lmStudio, .mistral, .huggingFace, .xAI, .zAI, .metaLlama, .alibabaCloud, .openRouter:
            // `parallel_tool_calls` is enabled when the model's catalog capability says so
            // OR when behavior flags force it on (used to be a hardcoded `apiType == .mistral`
            // branch — now the bundled defaults JSON flags Mistral models with
            // `forceParallelToolCalls: true` and the hardcode is gone).
            let supportsParallel = modelInfo(providerID: modelProvider.id, modelID: config.modelID)?
                .capabilities.parallelToolCalls ?? false
            let enableParallel = supportsParallel || flags.forceParallelToolCalls
            return OpenAICompatibleProvider(
                configuration: config, provider: modelProvider,
                readAPIKey: readAPIKey, verboseLogging: verbose,
                parallelToolCalls: enableParallel,
                behaviorFlags: flags
            )
        case .ollama:
            return OllamaProvider(
                configuration: config, provider: modelProvider,
                readAPIKey: readAPIKey, verboseLogging: verbose,
                behaviorFlags: flags
            )
        case .gemini:
            return GeminiProvider(
                configuration: config, provider: modelProvider,
                readAPIKey: readAPIKey, verboseLogging: verbose
            )
        }
    }

    /// Returns the merged `BehaviorFlags` for a given `(providerID, modelID)`. If the
    /// catalog already has a `ModelInfo` for this pair, returns its flags (which already
    /// reflect the layered enrichment pass). Otherwise computes a fresh resolution from
    /// bundled + user overrides — useful for callsites that need flags before the catalog
    /// has been fetched (or for models the catalog doesn't list).
    public func behaviorFlags(forProviderID providerID: String, modelID: String) -> BehaviorFlags {
        if let info = modelInfo(providerID: providerID, modelID: modelID) {
            return info.behaviorFlags
        }
        // No catalog entry yet — synthesize from overrides directly, mirroring the
        // layering rules in `fetchProviderModels`.
        var flags = BehaviorFlags()
        // Layer 0a: provider-wide defaults
        if let providerWide = bundledRegistry.defaults(providerID: providerID),
           let wideFlags = providerWide.behaviorFlags {
            wideFlags.apply(to: &flags, forceReplace: false)
        }
        // Layer 0b: per-(providerID, modelID) bundled
        if let providerScoped = bundledRegistry.override(providerID: providerID, modelID: modelID),
           let scopedFlags = providerScoped.behaviorFlags {
            scopedFlags.apply(to: &flags, forceReplace: false)
        }
        // Layer 1: per-(apiType, modelID) bundled
        let providerAPIType = providers.first(where: { $0.id == providerID })?.apiType.rawValue
        if let apiType = providerAPIType,
           let bundled = bundledRegistry.override(providerAPIType: apiType, modelID: modelID),
           let bundledFlags = bundled.behaviorFlags {
            bundledFlags.apply(to: &flags, forceReplace: false)
        }
        // Layer 4: user overrides force-replace
        let userKey = "\(providerID)/\(modelID)"
        if let user = userOverrides[userKey], let userFlags = user.behaviorFlags {
            userFlags.apply(to: &flags, forceReplace: true)
        }
        return flags
    }

    // MARK: - Built-in Provider Seeding

    /// Ensures every preset in `BuiltInProviders.all` is represented in `providers`.
    ///
    /// For each preset, in order:
    /// 1. If a provider with the preset's ID already exists, refresh its `name`,
    ///    `apiType`, and `endpoint` to the preset values (so renamed presets propagate).
    ///    The API key is not touched.
    /// 2. Otherwise, look for an existing user-created provider whose `apiType` AND
    ///    `endpoint` both match the preset. If exactly one such candidate exists, adopt
    ///    it: rewrite its ID to the built-in ID, copy its Keychain entry from old → new,
    ///    rewrite all `ModelConfiguration.providerID` references, and remove the old row.
    ///    This is the upgrade path for users who created (e.g.) a "My Anthropic" provider
    ///    before built-ins existed.
    /// 3. If zero or multiple candidates match, just add a new built-in row with no key.
    ///
    /// Idempotent: re-running on subsequent launches is a no-op once seeding has stabilised.
    public func seedBuiltInProviders() {
        guard providersLoadedOK else {
            logger.warning("Skipping built-in provider seeding — providers failed to load")
            return
        }

        var didMutate = false

        for preset in BuiltInProviders.all {
            if let existingIndex = providers.firstIndex(where: { $0.id == preset.id }) {
                // Already present — refresh fixed fields from the preset.
                let canonical = ModelProvider(builtIn: preset)
                if providers[existingIndex] != canonical {
                    providers[existingIndex] = canonical
                    didMutate = true
                }
                continue
            }

            // Look for a unique adoption candidate.
            let candidates = providers.filter {
                !BuiltInProviders.isBuiltIn(id: $0.id)
                    && $0.apiType == preset.apiType
                    && $0.endpoint == preset.endpoint
            }

            if candidates.count == 1 {
                let oldProvider = candidates[0]
                logger.info("Adopting existing provider '\(oldProvider.name, privacy: .public)' (id: \(oldProvider.id, privacy: .public)) into built-in slot \(preset.id, privacy: .public)")

                // Migrate the Keychain entry from old → new.
                if let oldKey = keychain.apiKey(forProviderID: oldProvider.id), !oldKey.isEmpty {
                    do {
                        try keychain.save(apiKey: oldKey, forProviderID: preset.id)
                    } catch {
                        logger.error("Failed to migrate API key for built-in adoption (\(preset.id, privacy: .public)): \(error.localizedDescription, privacy: .public)")
                    }
                }
                // Best-effort: remove the old keychain entry.
                do {
                    try keychain.delete(forProviderID: oldProvider.id)
                } catch {
                    logger.warning("Failed to delete old API key for adopted provider \(oldProvider.id, privacy: .public): \(error.localizedDescription, privacy: .public)")
                }

                // Rewrite all configuration references that pointed at the old provider.
                for i in configurations.indices where configurations[i].providerID == oldProvider.id {
                    configurations[i].providerID = preset.id
                }

                // Replace the old row with the canonical built-in row.
                if let oldIndex = providers.firstIndex(where: { $0.id == oldProvider.id }) {
                    providers[oldIndex] = ModelProvider(builtIn: preset)
                }
                didMutate = true
            } else {
                // Zero matches or multiple ambiguous matches — just add an empty built-in.
                providers.append(ModelProvider(builtIn: preset))
                didMutate = true
            }
        }

        if didMutate {
            saveProviders()
            saveConfigurations()
        }
    }

    // MARK: - Private persistence helpers

    private func saveProviders() {
        guard providersLoadedOK else {
            logger.warning("Skipping provider save — initial load failed, refusing to overwrite on-disk data")
            return
        }
        do {
            try storage.saveProviders(providers)
            persistenceError = nil
        } catch {
            let msg = "Failed to save providers: \(error.localizedDescription)"
            logger.error("\(msg, privacy: .public)")
            persistenceError = msg
        }
    }

    private func saveConfigurations() {
        guard configurationsLoadedOK else {
            logger.warning("Skipping configuration save — initial load failed, refusing to overwrite on-disk data")
            return
        }
        do {
            try storage.saveConfigurations(configurations)
            persistenceError = nil
        } catch {
            let msg = "Failed to save configurations: \(error.localizedDescription)"
            logger.error("\(msg, privacy: .public)")
            persistenceError = msg
        }
    }

    private func saveModelCatalog() {
        do {
            try storage.saveModelCatalog(models)
            persistenceError = nil
        } catch {
            let msg = "Failed to save model catalog: \(error.localizedDescription)"
            logger.error("\(msg, privacy: .public)")
            persistenceError = msg
        }
    }
}

/// Errors thrown by SwiftLLMKit operations.
public enum SwiftLLMKitError: Error, LocalizedError {
    case providerInUse(providerID: String, configNames: String)
    case configurationNotFound(id: UUID)
    case providerNotFound(id: String)
    case providerIsBuiltIn(id: String)
    case notABuiltInProvider(id: String)

    public var errorDescription: String? {
        switch self {
        case .providerInUse(_, let names):
            return "Cannot delete provider — it is referenced by configurations: \(names)"
        case .configurationNotFound(let id):
            return "Configuration not found: \(id)"
        case .providerNotFound(let id):
            return "Provider not found: \(id)"
        case .providerIsBuiltIn(let id):
            return "Cannot delete built-in provider '\(id)' — built-in providers are permanent"
        case .notABuiltInProvider(let id):
            return "Provider '\(id)' is not a known built-in provider"
        }
    }
}
