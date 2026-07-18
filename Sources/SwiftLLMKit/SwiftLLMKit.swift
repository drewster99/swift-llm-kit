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

    /// Per-model composition detail from the most recent merge, keyed `providerID/modelID`:
    /// each layer's own record, which layer won each field, and where layers disagreed. In-memory
    /// only — recomputed on every refresh — and the data the provenance inspector reads.
    public private(set) var metadataCompositions: [String: MergedModelComposition] = [:]

    /// The discovery ledger: every `providerID/modelID` ever observed in a `/models` listing.
    /// Persisted; seeded silently from the existing catalog on first load (see ``SeenModelsLedger``).
    private var seenModels = SeenModelsLedger()

    /// Model keys first observed during THIS session's refreshes — genuinely new to this install,
    /// never populated by the seeding pass. Nothing consumes it automatically yet (probing is
    /// manual by design); the coming probe queue and inspector read it.
    public private(set) var newlyDiscoveredModelKeys: Set<String> = []
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
    private let probeStore: ProbeRecordStore

    // MARK: - Empirical (probe) layer

    /// Locally-discovered probe records, keyed by their model-scoped identity. Loaded from the
    /// per-record store at startup; updated through ``storeProbeResult(profile:provider:modelID:)``.
    private var localProbeRecords: [ProbeRecordKey: ProbeRecord] = [:]

    /// Downloaded/shipped probe records — the slot the design reserves for server- or
    /// app-distributed evidence. Loaded from `downloaded_probes.json` when present; an absent
    /// file is simply an empty layer. Combined with local records per-field, newest wins.
    private var downloadedProbeRecords: [ProbeRecordKey: ProbeRecord] = [:]

    // MARK: - Override Layers

    /// App-bundled model metadata overrides — the shipped half of the downloaded-overrides layer.
    private let bundledRegistry: BundledModelMetadataRegistry

    /// The EFFECTIVE downloaded-overrides registry: the bundled registry with
    /// `downloaded_overrides.json` (same schema, App Support) laid over it when present —
    /// fresher curation supersedes shipped curation inside the one layer. Recomputed at load
    /// and at every refresh so a newly downloaded file takes effect without a restart.
    private var overridesRegistry: BundledModelMetadataRegistry

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
        self.overridesRegistry = self.bundledRegistry
        self.probeStore = ProbeRecordStore(baseDirectory: storage.baseDirectory)
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
        do {
            seenModels = try storage.loadSeenModels()
        } catch {
            // Non-fatal: a fresh ledger re-seeds silently from the current catalog below.
            logger.error("Failed to load seen-models ledger: \(error.localizedDescription, privacy: .public)")
        }
        // Seed the ledger from the persisted catalog when it has never been seeded — the whole
        // existing install base must start with its catalog marked seen (per provider), not
        // "all new".
        if seenModels.seededAt == nil, !models.isEmpty {
            let byProvider = Dictionary(grouping: models, by: \.providerID)
            let now = Date()
            for (providerID, providerModels) in byProvider {
                seenModels.observe(providerID: providerID, keys: providerModels.map(\.id), at: now)
            }
            saveSeenModels()
            logger.info("Seeded seen-models ledger with \(self.models.count, privacy: .public) existing catalog entries across \(byProvider.count, privacy: .public) providers")
        }

        // The empirical layer: locally-probed records plus the downloaded slot (absent = empty).
        reloadProbeRecords()
        reloadOverridesRegistry()

        seedBuiltInProviders()
    }

    /// Recomputes the effective downloaded-overrides registry from the bundled data plus
    /// `downloaded_overrides.json` when present. Same cadence as `reloadProbeRecords()`.
    private func reloadOverridesRegistry() {
        let url = storage.baseDirectory.appendingPathComponent("downloaded_overrides.json")
        if let downloaded = BundledModelMetadataRegistry.load(from: url) {
            overridesRegistry = bundledRegistry.overlaying(downloaded)
        } else {
            overridesRegistry = bundledRegistry
        }
    }

    /// Re-reads the probe evidence from disk. Called at load and at the start of every refresh —
    /// the headless eval runner is a SEPARATE PROCESS writing the same per-record store, so a
    /// running GUI must pick up its records on the next refresh, not the next restart.
    private func reloadProbeRecords() {
        localProbeRecords = Dictionary(
            probeStore.loadAll().map { ($0.key, $0) },
            uniquingKeysWith: { a, b in a.recordedAt >= b.recordedAt ? a : b }
        )
        downloadedProbeRecords = Dictionary(
            storage.loadDownloadedProbeRecords().map { ($0.key, $0) },
            uniquingKeysWith: { a, b in a.recordedAt >= b.recordedAt ? a : b }
        )
    }

    // MARK: - Probe evidence

    /// Persists a completed probe run and folds it into the live empirical layer. Returns false
    /// (and stores nothing) when the profile established no probed findings — an aborted or
    /// rate-gutted run must not overwrite a real record. The catalog is NOT recomposed here;
    /// the next refresh reads the updated records. Callers wanting immediacy refresh the provider.
    @discardableResult
    public func storeProbeResult(profile: ModelProfile, provider: ModelProvider, modelID: String) throws -> Bool {
        let key = ProbeRecordKey(apiType: provider.apiType, endpoint: provider.endpoint, modelID: modelID)
        let stored = try probeStore.upsert(
            profile: profile, key: key, providerID: provider.id,
            proberVersion: ModelProber.proberVersion
        )
        if stored, let record = probeStore.record(forKey: key) {
            localProbeRecords[key] = record
        }
        return stored
    }

    /// The probe evidence for a model as the merge sees it: local and downloaded records for its
    /// model-scoped key. For the inspector; nil entries mean that layer holds nothing.
    public func probeRecords(provider: ModelProvider, modelID: String) -> (local: ProbeRecord?, downloaded: ProbeRecord?) {
        let key = ProbeRecordKey(apiType: provider.apiType, endpoint: provider.endpoint, modelID: modelID)
        return (localProbeRecords[key], downloadedProbeRecords[key])
    }

    /// Every locally-discovered probe record, EXPORT-STRIPPED (account-scoped findings removed) —
    /// the exact array shape the downloaded-probe slot consumes, so the CLI's export file IS the
    /// shipped format. Re-reads the store first so a headless run exports records written by any
    /// process, ordered stably by key.
    public func exportableProbeRecords() -> [ProbeRecord] {
        reloadProbeRecords()
        return localProbeRecords.values
            .map(\.strippedForExport)
            .sorted { $0.key.storageKey < $1.key.storageKey }
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
        // The ledger's own save path only logs on failure; retrying here means a transiently
        // failed ledger write heals on the next general save instead of resurrecting
        // already-reported discoveries after a restart.
        do {
            try storage.saveSeenModels(seenModels)
        } catch {
            let msg = "Failed to save seen-models ledger: \(error.localizedDescription)"
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
        // `liteLLMProviderName` is exempt: it is user-editable on built-ins too, so it is taken
        // from the incoming value verbatim (nil included — that's an explicit "not mapped", not
        // an absent field). Clamping it here would silently discard the mapping editor's save.
        if let preset = BuiltInProviders.preset(id: provider.id) {
            var clamped = ModelProvider(builtIn: preset)
            clamped.liteLLMProviderName = provider.liteLLMProviderName
            providers[index] = clamped
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
        // Compositions are keyed providerID/modelID and only refreshed by a live provider —
        // evict the deleted provider's entries here or they linger in memory forever.
        let deletedPrefix = "\(id)/"
        metadataCompositions = metadataCompositions.filter { !$0.key.hasPrefix(deletedPrefix) }

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
    ///
    /// Uses the "mutate-from-existing" pattern (`var newConfig = original`)
    /// instead of rebuilding via named init parameters. This guarantees every
    /// field present on `ModelConfiguration` is preserved into the duplicate
    /// — including any fields added in future releases. The named-init path
    /// the old implementation used would silently drop new fields (e.g.
    /// `thinkingEffort`, `extendedCacheTTL`, `extraJSONOverrides` were all
    /// dropped before this fix).
    @discardableResult
    public func duplicateConfiguration(id: UUID) -> ModelConfiguration? {
        guard let original = configurations.first(where: { $0.id == id }) else { return nil }
        var newConfig = original
        newConfig.id = UUID()
        newConfig.name = "\(original.name) (Copy)"
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

    // MARK: - LiteLLM Coverage

    /// Whether a `(provider, model)` pair resolves to LiteLLM metadata, and if not, which level
    /// failed — the provider's mapping or the model itself.
    public func liteLLMResolution(providerID: String, modelID: String) async -> ModelMetadataService.Resolution {
        let name = providers.first(where: { $0.id == providerID })?.liteLLMProviderName
        return await metadataService.resolution(forModelID: modelID, liteLLMProviderName: name)
    }

    /// Classifies every model this provider lists, in one actor hop.
    public func liteLLMResolutions(forProviderID providerID: String) async -> [String: ModelMetadataService.Resolution] {
        let name = providers.first(where: { $0.id == providerID })?.liteLLMProviderName
        let modelIDs = models(for: providerID).map(\.modelID)
        return await metadataService.resolutions(forModelIDs: modelIDs, liteLLMProviderName: name)
    }

    /// Whether a provider's mapping names a `litellm_provider` that LiteLLM actually has data
    /// for. `nil` (unmapped) is reported as `false`.
    public func liteLLMProviderIsKnown(_ liteLLMProviderName: String?) async -> Bool {
        guard let liteLLMProviderName else { return false }
        return await metadataService.allLiteLLMProviderNames().contains { $0.name == liteLLMProviderName }
    }

    /// Every distinct `litellm_provider` value LiteLLM's data actually contains, with model
    /// counts — the authoritative choice list for mapping a provider.
    public func allLiteLLMProviderNames() async -> [(name: String, modelCount: Int)] {
        await metadataService.allLiteLLMProviderNames()
    }

    /// The `litellm_provider` values that catalogue this exact model name — lets a mapping fix
    /// offer "providers that actually have this model" rather than the full list.
    public func liteLLMProviderNames(matchingModelID modelID: String) async -> [String] {
        await metadataService.liteLLMProviderNames(matchingModelID: modelID)
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
        await refreshModels(provider: provider)
    }

    /// Refreshes the model catalog for a `ModelProvider` the caller already holds.
    /// **This is the implementation**; ``refreshModels(forProviderID:)`` resolves the ID and
    /// funnels here.
    ///
    /// Looks nothing up, so unlike the ID overload it has no unknown-provider case to warn about
    /// and silently do nothing for. A caller iterating ``providers`` — refreshing every configured
    /// provider in turn, say — has the struct already and shouldn't round-trip through an ID only
    /// to re-find what it passed.
    public func refreshModels(provider: ModelProvider) async {
        isRefreshing = true
        defer { isRefreshing = false }

        // Refresh metadata first so enrichment uses the latest LiteLLM data, and pick up probe
        // records another process (the headless eval runner) may have written since our load,
        // plus any newly downloaded overrides file.
        await metadataService.forceRefresh()
        reloadProbeRecords()
        reloadOverridesRegistry()

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

    /// Records a successful `/models` listing in the discovery ledger. Lives on the ONE path every
    /// refresh flavor shares (`fetchAndEnrich`) so a full refresh and a single-provider refresh
    /// cannot diverge — an earlier draft recorded only in the per-slice path, and the full-refresh
    /// path silently bypassed the ledger. A provider's first-ever listing seeds silently and
    /// reports nothing as new (see ``SeenModelsLedger``).
    private func recordModelObservation(providerID: String, keys: [String], providerName: String) {
        let wasSeeded = seenModels.seededProviders.contains(providerID)
        let fresh = seenModels.observe(providerID: providerID, keys: keys, at: Date())
        if !fresh.isEmpty {
            newlyDiscoveredModelKeys.formUnion(fresh)
            logger.info("Discovered \(fresh.count, privacy: .public) new models from \(providerName, privacy: .public)")
        }
        if !wasSeeded || !fresh.isEmpty {
            saveSeenModels()
        }
    }

    private func saveSeenModels() {
        do {
            try storage.saveSeenModels(seenModels)
        } catch {
            logger.error("Failed to save seen-models ledger: \(error.localizedDescription, privacy: .public)")
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

    /// Refreshes LiteLLM metadata and then every configured provider's models, unconditionally.
    ///
    /// ``refreshIfNeeded()`` is the launch path and is gated: once metadata is same-day and the
    /// catalog is non-empty it does only a straggler pass, so calling it twice in a day refreshes
    /// nothing. That is right for launch and wrong for anything that must actually re-fetch —
    /// a capability probe wants the provider's live model list, not yesterday's cache, however
    /// many times it runs. This is the ungated door to the same work.
    ///
    /// Per-provider failures do not throw; they land in ``refreshErrors`` keyed by provider name,
    /// so one dead endpoint can't abort the rest.
    public func refreshAllModels() async {
        await performRefresh()
    }

    private func performRefresh() async {
        isRefreshing = true
        defer { isRefreshing = false }

        // 1. Refresh LiteLLM metadata, and pick up probe records another process may have
        // written, plus any newly downloaded overrides file.
        await metadataService.forceRefresh()
        reloadProbeRecords()
        reloadOverridesRegistry()

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

    /// Fetches models for one provider and composes the layered metadata merge.
    /// On failure, returns the previously cached models for that provider plus an error
    /// string so the caller can preserve state without losing the catalog.
    ///
    /// The five-layer composition (see ``ModelFactsMerger``), applied per model, per field:
    ///   1. authoritative — the provider's own `/models` decode (tri-state facts; the base)
    ///   2. empirical — probe results (EMPTY until the probe store is wired in)
    ///   3. downloaded overrides — app-curated fixes, FORCE (three key axes folded
    ///      least-specific-first so the most specific wins)
    ///   4. enrichment — LiteLLM, gap-fill only
    ///   5. user overrides — FORCE, the user always wins
    /// The full composition (per-layer records, per-field provenance, disagreements) is retained
    /// in ``metadataCompositions`` for the inspector; the materialized `ModelInfo` is what the
    /// rest of the app consumes, unchanged in shape.
    private func fetchAndEnrich(provider: ModelProvider) async -> (models: [ModelInfo], error: String?) {
        let apiKey = keychain.apiKey(forProviderID: provider.id)
        do {
            let decodedFacts = try await fetchService.fetchModelFacts(
                from: provider,
                apiKey: apiKey
            )

            // Drop compositions from this provider's previous refresh so delisted models don't
            // leave stale inspector entries behind.
            let providerPrefix = "\(provider.id)/"
            metadataCompositions = metadataCompositions.filter { !$0.key.hasPrefix(providerPrefix) }

            var providerModels: [ModelInfo] = []
            providerModels.reserveCapacity(decodedFacts.count)
            for decoded in decodedFacts {
                providerModels.append(await composeModel(
                    modelID: decoded.modelID, authoritative: decoded.facts, provider: provider
                ))
            }

            // Union-of-layers existence: a model an override pinpoints but /models no longer
            // lists still materializes (Anthropic delists old snapshots that remain callable
            // through their deprecation window; a user override can add an unlisted model).
            // Only the enumerable providerID axes contribute — see providerScopedModelIDs.
            let decodedIDs = Set(decodedFacts.map(\.modelID))
            var overrideOnlyIDs = Set(overridesRegistry.providerScopedModelIDs(providerID: provider.id))
            overrideOnlyIDs.formUnion(
                userOverrides.keys
                    .filter { $0.hasPrefix(providerPrefix) }
                    .map { String($0.dropFirst(providerPrefix.count)) }
            )
            for modelID in overrideOnlyIDs.subtracting(decodedIDs).sorted() {
                providerModels.append(await composeModel(
                    modelID: modelID, authoritative: ModelFacts(), provider: provider
                ))
            }

            // The ledger records what the LISTING contained — decoded models only. Synthesized
            // override-only rows were never in the listing, and recording them would make the
            // ledger lie about its own definition.
            recordModelObservation(
                providerID: provider.id,
                keys: decodedFacts.map { "\(provider.id)/\($0.modelID)" },
                providerName: provider.name
            )

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


    /// Composes ONE model through the five-layer merge and records its composition for the
    /// inspector. `authoritative` is empty for override-only models (union-of-layers existence:
    /// the merge doesn't care which layer created the key, and the inspector shows the absence
    /// of provider-stated fields for what it is).
    private func composeModel(modelID: String, authoritative: ModelFacts, provider: ModelProvider) async -> ModelInfo {
        // Downloaded-overrides layer: three key axes, least specific applied first so the
        // most specific wins — provider-wide defaults (e.g. all-OpenAI useMaxCompletionTokens),
        // then (apiType, modelID), then (providerID, modelID).
        var downloadedOverrides = ModelFacts()
        if let providerWide = overridesRegistry.defaults(providerID: provider.id) {
            downloadedOverrides.overlay(providerWide.asFacts)
        }
        if let apiTypeScoped = overridesRegistry.override(providerAPIType: provider.apiType.rawValue, modelID: modelID) {
            downloadedOverrides.overlay(apiTypeScoped.asFacts)
        }
        if let providerScoped = overridesRegistry.override(providerID: provider.id, modelID: modelID) {
            downloadedOverrides.overlay(providerScoped.asFacts)
        }

        var enrichment = ModelFacts()
        if let litellm = await metadataService.metadata(for: modelID, liteLLMProviderName: provider.liteLLMProviderName) {
            enrichment = litellm.asFacts
        }

        var userFacts = ModelFacts()
        if let userOverride = userOverrides["\(provider.id)/\(modelID)"] {
            userFacts = userOverride.asFacts
        }

        // Empirical layer: local + downloaded probe evidence for this model's model-scoped key,
        // combined per-field with the newer record winning.
        let probeKey = ProbeRecordKey(apiType: provider.apiType, endpoint: provider.endpoint, modelID: modelID)
        let empirical = ProbeEvidenceCombiner.combinedFacts(
            local: localProbeRecords[probeKey],
            downloaded: downloadedProbeRecords[probeKey],
            forProviderID: provider.id
        )

        let composition = ModelFactsMerger.merge(
            authoritative: authoritative,
            empirical: empirical,
            downloadedOverrides: downloadedOverrides,
            enrichment: enrichment,
            userOverrides: userFacts
        )
        metadataCompositions["\(provider.id)/\(modelID)"] = composition
        return composition.merged.materialize(providerID: provider.id, modelID: modelID)
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

        // Temperature bounds (per-provider). Skip when temperature is nil
        // ("use model's default" — nothing for us to validate).
        if let temperature = config.temperature {
            let tempRange = provider.apiType.temperatureRange
            guard tempRange.contains(temperature) else {
                configurations[index].isValid = false
                configurations[index].validationError = "Temperature must be between \(tempRange.lowerBound) and \(tempRange.upperBound) for \(provider.apiType.displayName)"
                return
            }
        }

        // Thinking budget is only supported for Anthropic and Alibaba Cloud
        if let budget = config.thinkingBudget, budget > 0,
           provider.apiType != .anthropic && provider.apiType != .alibabaCloud {
            configurations[index].isValid = false
            configurations[index].validationError = "Thinking budget is only supported for Anthropic and Alibaba Cloud providers"
            return
        }

        // thinkingEffort validation: must be a recognized effort enum value
        // AND only emitted by providers that actually consume it. Anthropic
        // accepts effort on Opus 4.5+ / Sonnet 4.6+; OpenAI on reasoning
        // models flagged with `supportsReasoningEffort`. On every other
        // provider (Ollama, Gemini, alibabaCloud, etc.) the field is silently
        // dropped — flag that as invalid so users don't set it expecting it
        // to fire. Anthropic provider always emits it (no flag gating), so
        // a model that doesn't accept the field will 400 — preferable to
        // silent drop but worth surfacing pre-flight.
        if let effort = config.thinkingEffort {
            let validEfforts: Set<String> = ["minimal", "low", "medium", "high", "xhigh", "max"]
            guard validEfforts.contains(effort) else {
                configurations[index].isValid = false
                configurations[index].validationError = "thinkingEffort '\(effort)' is not a recognized value (valid: \(validEfforts.sorted().joined(separator: ", ")))"
                return
            }
            let configFlags = behaviorFlags(forProviderID: provider.id, modelID: config.modelID)
            let providerSupports: Bool = {
                switch provider.apiType {
                case .anthropic, .alibabaCloud:
                    return true   // Anthropic emits effort unconditionally; alibaba uses thinking_budget but effort field harmless
                case .openAICompatible, .lmStudio, .mistral, .huggingFace, .xAI, .zAI, .metaLlama, .openRouter:
                    return configFlags.supportsReasoningEffort
                case .gemini, .ollama:
                    return false
                }
            }()
            if !providerSupports {
                configurations[index].isValid = false
                configurations[index].validationError = "thinkingEffort is not supported by \(provider.apiType.displayName) for this model (drop it or pick a model with reasoning support)"
                return
            }
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

            // When the model's OWN effort list is known (today: Anthropic publishes it per model),
            // validate the chosen effort against it — not against the vendor-blind union above,
            // which happily accepts `minimal` on a Claude model (a guaranteed 400) and can't know
            // that sonnet-4-6 takes `max` but not `xhigh`. An empty list means nobody told us the
            // levels, not "no levels", so it stays permissive.
            if let effort = config.thinkingEffort,
               !modelInfo.validEffortLevels.isEmpty,
               !modelInfo.validEffortLevels.contains(effort) {
                configurations[index].isValid = false
                configurations[index].validationError = "Model '\(config.modelID)' does not accept effort '\(effort)' (it accepts: \(modelInfo.validEffortLevels.joined(separator: ", ")))"
                return
            }

            // Check maxOutputTokens doesn't exceed model's reported max
            if let modelMax = modelInfo.maxOutputTokens, config.maxOutputTokens > modelMax {
                configurations[index].isValid = false
                configurations[index].validationError = "Max output tokens (\(config.maxOutputTokens)) exceeds model limit (\(modelMax))"
                return
            }
            // Context-bound models (gpt-4: "maximum context length is 8192") have no independent
            // output cap, so the check above is skipped — but the output request still cannot
            // exceed the whole context. Validate against context instead, closing the hole where
            // an unbounded output value would silently pass and then 400 at runtime.
            if modelInfo.outputBoundedByContext, let context = modelInfo.maxInputTokens,
               config.maxOutputTokens > context {
                configurations[index].isValid = false
                configurations[index].validationError = "Max output tokens (\(config.maxOutputTokens)) exceeds this model's context length (\(context)); its output is bounded by context, with no separate cap"
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
    /// parameters (model, temperature, max_tokens, thinking, output_config.effort,
    /// reasoning_effort, stream, cache_control). The app adds messages/tools to
    /// the body and sends the request.
    ///
    /// **`tool_choice` is NOT pre-populated** — it's a per-call field that the
    /// caller must add to the body alongside `tools`. Use `LLMToolChoice`'s
    /// per-provider wire shapes (see the type's documentation). For most uses,
    /// prefer the high-level `LLMProvider.send(messages:tools:toolChoice:)`
    /// path which handles all per-provider translation automatically.
    public func prepareRequest(for configurationID: UUID) throws -> PreparedRequest {
        guard let config = configurations.first(where: { $0.id == configurationID }) else {
            throw SwiftLLMKitError.configurationNotFound(id: configurationID)
        }
        return try prepareRequest(configuration: config)
    }

    /// Resolves the `ModelProvider` for a caller-supplied configuration, then prepares the
    /// request. See ``prepareRequest(configuration:provider:)`` for the implementation.
    public func prepareRequest(configuration config: ModelConfiguration) throws -> PreparedRequest {
        guard let provider = providers.first(where: { $0.id == config.providerID }) else {
            throw SwiftLLMKitError.providerNotFound(id: config.providerID)
        }
        return prepareRequest(configuration: config, provider: provider)
    }

    /// Prepares a request from a configuration and a `ModelProvider` the caller already holds.
    /// **This is the implementation**; the other overloads resolve something and funnel here.
    /// It looks nothing up, so it cannot fail and does not throw.
    public func prepareRequest(
        configuration config: ModelConfiguration,
        provider: ModelProvider
    ) -> PreparedRequest {
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

        // Resolve behavior flags once — used by multiple per-apiType branches below.
        let prepFlags = behaviorFlags(forProviderID: provider.id, modelID: config.modelID)

        // Reasoning models (o-series, GPT-5 family) reject `temperature` outright — omit it
        // when the model is flagged.
        if !prepFlags.mustNeverSendTemperatureParam, let temperature = config.temperature {
            body["temperature"] = temperature
        }

        switch provider.apiType {
        case .anthropic:
            body["max_tokens"] = config.maxOutputTokens
            // Anthropic thinking: adaptive (Opus 4.7/4.8) vs manual (older
            // models). Mirrors AnthropicProvider.buildRequestBody. The
            // `thinkingBudget` field acts as a boolean signal on adaptive
            // models (>0 = on, 0/nil = off); manual models use it as the
            // token budget. temperature pinned to 1.0 when thinking is on
            // (both modes — Anthropic-thinking convention).
            let thinkingEnabled = (config.thinkingBudget ?? 0) > 0
            if thinkingEnabled {
                body["temperature"] = 1.0
                if prepFlags.requiresAdaptiveThinking {
                    body["thinking"] = ["type": "adaptive"] as [String: Any]
                } else if let budget = config.thinkingBudget {
                    body["thinking"] = [
                        "type": "enabled",
                        "budget_tokens": max(budget, 1024)
                    ] as [String: Any]
                }
            }
            // Top-level `output_config.effort` — independent of thinking mode.
            if let effort = config.thinkingEffort {
                body["output_config"] = ["effort": effort] as [String: Any]
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
            let tokenLimitKey = prepFlags.useMaxCompletionTokens ? "max_completion_tokens" : "max_tokens"
            // For context-bound models, clamp the outgoing cap to the context length so a config
            // that predates validation (or was set elsewhere) can never send an impossible
            // max_tokens. Clamping only ever REDUCES the value — the safe direction — and the
            // catalog lookup here is the manager's own, no extra fetch.
            var outgoingMaxTokens = config.maxOutputTokens
            if let info = modelInfo(providerID: provider.id, modelID: config.modelID),
               info.outputBoundedByContext, let context = info.maxInputTokens {
                outgoingMaxTokens = min(outgoingMaxTokens, context)
            }
            body[tokenLimitKey] = outgoingMaxTokens
            // OpenAI reasoning_effort — depth control for reasoning models
            // (o-series, GPT-5 family). Gated on `supportsReasoningEffort`
            // because non-reasoning models reject the field with HTTP 400.
            if prepFlags.supportsReasoningEffort,
               let effort = config.thinkingEffort {
                body["reasoning_effort"] = effort
            }
            // OpenRouter passes top-level cache_control through to Anthropic upstreams.
            // Mirrors AnthropicProvider's automatic-caching shape; gated to OpenRouter +
            // Anthropic-prefixed model IDs so other upstreams don't see unfamiliar fields.
            if provider.apiType == .openRouter,
               config.modelID.lowercased().hasPrefix("anthropic/") {
                body["cache_control"] = config.extendedCacheTTL
                    ? ["type": "ephemeral", "ttl": "1h"] as [String: Any]
                    : ["type": "ephemeral"] as [String: Any]
            }
        case .ollama:
            body["options"] = ["num_predict": config.maxOutputTokens] as [String: Any]
        case .gemini:
            var genConfig: [String: Any] = ["maxOutputTokens": config.maxOutputTokens]
            if !prepFlags.mustNeverSendTemperatureParam, let temperature = config.temperature {
                genConfig["temperature"] = temperature
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
        guard var config = configurations.first(where: { $0.id == configurationID }) else {
            throw SwiftLLMKitError.configurationNotFound(id: configurationID)
        }

        // Clamp the requested output-token cap to the model's known maximum so we never
        // build a provider that will send a request the backend rejects with
        // "max_tokens (X) exceeds model's maximum output tokens (Y)". The known max comes
        // from the catalog (LiteLLM) OR a learned user override, so a limit discovered at
        // runtime and saved as an override automatically clamps every future provider.
        // min-semantics: clamping (never raising) means a later increase to the known limit
        // — an upstream metadata refresh or a raised override — takes effect immediately,
        // while a user-configured value below the limit is left untouched.
        //
        // This reconciles a *saved* setting against the catalog, so it belongs here rather than
        // in the overload below: a caller supplying its own configuration owns its own caps.
        if let modelMax = modelInfo(providerID: config.providerID, modelID: config.modelID)?.maxOutputTokens,
           config.maxOutputTokens > modelMax {
            config.maxOutputTokens = modelMax
        }

        // The provider-existence check lives in the overload, which throws providerNotFound.
        return try makeProvider(configuration: config)
    }

    /// Resolves the `ModelProvider` for a caller-supplied configuration, then builds the
    /// `LLMProvider`. See ``makeProvider(configuration:provider:)`` for the implementation.
    public func makeProvider(configuration config: ModelConfiguration) throws -> any LLMProvider {
        guard let modelProvider = providers.first(where: { $0.id == config.providerID }) else {
            throw SwiftLLMKitError.providerNotFound(id: config.providerID)
        }
        return makeProvider(configuration: config, provider: modelProvider)
    }

    /// Builds an `LLMProvider` from a configuration and a `ModelProvider` the caller already
    /// holds. **This is the implementation**; every other `makeProvider` overload resolves
    /// something and funnels here, so they can never drift apart on how an `apiType` is wired.
    ///
    /// It looks nothing up, so there is nothing to fail on and it does not throw. Callers
    /// enumerating ``providers`` already have the struct in hand and shouldn't pay for a lookup
    /// that can only re-find what they passed.
    ///
    /// Taking both structs directly is what makes an un-saved model callable: a configuration is
    /// something a user creates deliberately, so the overwhelming majority of catalog models have
    /// none, and capability probing has to reach exactly those. It also wants to pin
    /// `temperature` / `streaming` / `maxOutputTokens` itself. Note what is deliberately NOT done
    /// here: no clamping against ``modelInfo(providerID:modelID:)``, because that value is
    /// LiteLLM-derived and a probe that let it shape the request would assume its own conclusion.
    /// The caller owns the knobs; ``makeProvider(for:)`` is where a *saved* config gets reconciled
    /// with the catalog.
    ///
    /// `behaviorFlags` ARE still applied, which is not a contradiction: they are our own
    /// hand-authored knobs describing *how to form a valid request* (o1 rejects `temperature`;
    /// OpenAI needs `max_completion_tokens`), not claims about what a model can do. Dropping them
    /// would yield 400s that look like capability failures but are really our malformed requests.
    ///
    /// - Note: the configuration is used as given and never stored — nothing is added to
    ///   ``configurations``, no validation runs, nothing is written to disk.
    /// Builds a provider for CAPABILITY PROBING — the one place that decides which behavior
    /// flags a probe runs under, so the rule lives here and not in each runner:
    ///
    /// - **Necessity flags stay** (`useMaxCompletionTokens`, `glmTemplateSalvage`,
    ///   `requiresAdaptiveThinking`, …): without them, requests fail for reasons unrelated to the
    ///   capability under test, and the probe would fabricate capability negatives out of our own
    ///   malformed requests.
    /// - **Restriction flags the probe MEASURES are stripped**: `mustNeverSendTemperatureParam`
    ///   makes the provider omit temperature, so a probe run under it "passes" trivially and can
    ///   never observe the very rejection the flag encodes. Stripping it lets the probe measure
    ///   the raw endpoint — the rejection re-derives the flag empirically, and a stale flag on a
    ///   model that now accepts temperature becomes detectable instead of self-sealing.
    /// - **Optional request knobs the probe does not measure are omitted**:
    ///   `disableParallelToolCalls` is forced ON so probe requests never carry
    ///   `parallel_tool_calls: true`. No probe measures that knob, and endpoints that reject it
    ///   (OpenAI o-series) 400 with the word "tool" in the body — which the tool probe's
    ///   classifier read as "cannot call tools", writing false verdicts for eight o-series
    ///   models in the 2026-07-18 audit. An un-sent parameter cannot poison a verdict.
    /// - **A probe-scoped 60s timeout** (`probeURLSession`): probe calls are tiny, so a stall is
    ///   an endpoint fault, not a long generation — and a timeout grades inconclusive, never a
    ///   verdict. Production providers keep the standard long-generation session untouched.
    public func makeProbeProvider(
        configuration config: ModelConfiguration,
        provider modelProvider: ModelProvider
    ) -> any LLMProvider {
        var probeFlags = behaviorFlags(forProviderID: modelProvider.id, modelID: config.modelID)
        probeFlags.mustNeverSendTemperatureParam = false
        probeFlags.disableParallelToolCalls = true
        return makeProvider(configuration: config, provider: modelProvider,
                            behaviorFlags: probeFlags, session: probeURLSession)
    }

    public func makeProvider(
        configuration config: ModelConfiguration,
        provider modelProvider: ModelProvider,
        behaviorFlags flagsOverride: BehaviorFlags? = nil,
        session: URLSession = llmURLSession
    ) -> any LLMProvider {
        // Resolve the merged behavior flags for this (provider, model) so providers can read
        // knobs without reaching back into the manager. Keyed on the ModelProvider's ID, so it
        // works for any model that provider serves, saved configuration or not. An explicit
        // override skips the resolution — the probe factory below uses it to strip the flags
        // whose restrictions the probe is trying to measure.
        let flags = flagsOverride ?? behaviorFlags(forProviderID: modelProvider.id, modelID: config.modelID)

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

        switch modelProvider.apiType {
        case .anthropic:
            return AnthropicProvider(
                configuration: config, provider: modelProvider,
                readAPIKey: readAPIKey, verboseLogging: verbose,
                session: session,
                behaviorFlags: flags
            )
        case .openAICompatible, .lmStudio, .mistral, .huggingFace, .xAI, .zAI, .metaLlama, .alibabaCloud, .openRouter:
            // `parallel_tool_calls: true` is sent by default for every OpenAI-compatible
            // model — the agent loop pairs multi-call turns safely and most endpoints
            // default the param to true anyway, so the (unreliable) per-model catalog
            // capability isn't consulted. A strict endpoint that rejects the field can
            // opt out with the `disableParallelToolCalls` behavior flag.
            let enableParallel = !flags.disableParallelToolCalls
            return OpenAICompatibleProvider(
                configuration: config, provider: modelProvider,
                readAPIKey: readAPIKey, verboseLogging: verbose,
                parallelToolCalls: enableParallel,
                behaviorFlags: flags,
                session: session
            )
        case .ollama:
            // Ollama Cloud (ollama.com) proxies to upstream model backends that enforce
            // strict OpenAI-style tool_call/tool_result pairing and natively support the
            // `tool` role + tool_call_id. OllamaProvider's message normalization (textualizing
            // tool results into user text while leaving the assistant's tool_calls structured)
            // produces an unmatched call/response count that ollama.com rejects with HTTP 400
            // "Not the same number of function calls and responses". It also drops tool_call IDs,
            // which the cloud backends require for pairing. Route cloud through the OpenAI-
            // compatible provider against Ollama's documented `/v1` surface so the whole
            // exchange — request encoding AND response parsing — stays structured and paired.
            // Local Ollama keeps OllamaProvider: its weaker chat templates (e.g. gemma3's strict
            // user/assistant alternation with no tool role) need the textual rewrite.
            if let openAIEndpoint = Self.ollamaCloudOpenAIEndpoint(for: modelProvider.endpoint) {
                let enableParallel = !flags.disableParallelToolCalls
                var cloudProvider = modelProvider
                cloudProvider.endpoint = openAIEndpoint
                return OpenAICompatibleProvider(
                    configuration: config, provider: cloudProvider,
                    readAPIKey: readAPIKey, verboseLogging: verbose,
                    parallelToolCalls: enableParallel,
                    behaviorFlags: flags,
                    session: session
                )
            }
            return OllamaProvider(
                configuration: config, provider: modelProvider,
                readAPIKey: readAPIKey, verboseLogging: verbose,
                behaviorFlags: flags,
                session: session
            )
        case .gemini:
            return GeminiProvider(
                configuration: config, provider: modelProvider,
                readAPIKey: readAPIKey, verboseLogging: verbose,
                session: session
            )
        }
    }

    /// Maps an Ollama Cloud base endpoint (e.g. `https://ollama.com/api`) to its
    /// OpenAI-compatible surface (`https://ollama.com/v1`, which `OpenAICompatibleProvider`
    /// turns into `/v1/chat/completions`). Returns `nil` for local or self-hosted Ollama
    /// endpoints — those keep the native `OllamaProvider` path. Host match is exact on
    /// `ollama.com` (and subdomains) so a local box named to contain "ollama" isn't misrouted.
    private static func ollamaCloudOpenAIEndpoint(for endpoint: URL) -> URL? {
        guard let host = endpoint.host?.lowercased(),
              host == "ollama.com" || host.hasSuffix(".ollama.com") else {
            return nil
        }
        var components = URLComponents()
        components.scheme = endpoint.scheme ?? "https"
        components.host = endpoint.host
        components.port = endpoint.port
        components.path = "/v1"
        return components.url
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
        // No catalog entry yet — synthesize through the SAME fold `composeModel` uses (the
        // effective registry incl. downloaded_overrides.json, least-specific-first with force
        // semantics, then user overrides forcing on top). An earlier version read only the
        // bundled registry with gap-fill ordering and could disagree with the composed catalog —
        // a downloaded no-temperature fix would vanish exactly for the not-yet-fetched models
        // that need it most.
        var facts = ModelFacts()
        if let providerWide = overridesRegistry.defaults(providerID: providerID) {
            facts.overlay(providerWide.asFacts)
        }
        if let apiType = providers.first(where: { $0.id == providerID })?.apiType.rawValue,
           let apiTypeScoped = overridesRegistry.override(providerAPIType: apiType, modelID: modelID) {
            facts.overlay(apiTypeScoped.asFacts)
        }
        if let providerScoped = overridesRegistry.override(providerID: providerID, modelID: modelID) {
            facts.overlay(providerScoped.asFacts)
        }
        if let user = userOverrides["\(providerID)/\(modelID)"] {
            facts.overlay(user.asFacts)
        }
        var flags = BehaviorFlags()
        facts.behaviorFlags.apply(to: &flags, forceReplace: true)
        return flags
    }

    // MARK: - Built-in Provider Seeding

    /// The form a stored built-in row should take at seed time: the preset's fixed fields, but
    /// preserving a stored `liteLLMProviderName`.
    ///
    /// Seeding exists to force renamed/moved presets back onto their canonical values, which is
    /// why it overwrites. `liteLLMProviderName` is the one field the user may legitimately edit
    /// on a built-in, so overwriting it would revert their mapping on the very next launch — and
    /// the mapping editor tells them to restart. A stored `nil` means the row predates the field,
    /// so the preset seeds it and existing installs self-heal.
    ///
    /// Consequence worth knowing: once a row carries a non-nil value, a corrected preset mapping
    /// no longer reaches it automatically; that user re-picks in Settings → Metadata.
    nonisolated static func canonicalBuiltIn(preset: BuiltInProviderPreset, existing: ModelProvider?) -> ModelProvider {
        var canonical = ModelProvider(builtIn: preset)
        if let stored = existing?.liteLLMProviderName {
            canonical.liteLLMProviderName = stored
        }
        return canonical
    }

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
                let canonical = Self.canonicalBuiltIn(preset: preset, existing: providers[existingIndex])
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
private enum SwiftLLMKitError: Error, LocalizedError {
    case providerInUse(providerID: String, configNames: String)
    case configurationNotFound(id: UUID)
    case providerNotFound(id: String)
    case providerIsBuiltIn(id: String)
    case notABuiltInProvider(id: String)

    var errorDescription: String? {
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
