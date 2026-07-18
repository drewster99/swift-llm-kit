import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "Storage")

/// Manages file-based persistence for SwiftLLMKit data.
///
/// Storage layout:
/// ```
/// <ApplicationSupport>/SwiftLLMKit/<appBundleID>/
///     providers.json
///     model_configurations.json
///     model_catalog.json           (cached, rebuilt on refresh)
///     seen_models.json             (discovery ledger: model keys ever observed)
///     litellm_metadata.json        (cached from GitHub)
///     litellm_headers.json         (ETag/Last-Modified for conditional fetch)
/// ```
struct StorageManager: Sendable {
    let baseDirectory: URL

    /// Creates a storage manager for the given app identifier.
    /// - Parameter appIdentifier: Typically `Bundle.main.bundleIdentifier`.
    init(appIdentifier: String) {
        guard let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            preconditionFailure("Application Support directory unavailable")
        }
        self.baseDirectory = appSupport
            .appendingPathComponent("SwiftLLMKit", isDirectory: true)
            .appendingPathComponent(appIdentifier, isDirectory: true)
    }

    /// Ensures the storage directory exists.
    func ensureDirectory() throws {
        try FileManager.default.createDirectory(
            at: baseDirectory,
            withIntermediateDirectories: true
        )
    }

    // MARK: - Providers

    func saveProviders(_ providers: [ModelProvider]) throws {
        try ensureDirectory()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(providers)
        try data.write(to: providersURL, options: .atomic)
    }

    func loadProviders() throws -> [ModelProvider] {
        guard FileManager.default.fileExists(atPath: providersURL.path) else { return [] }
        let data = try Data(contentsOf: providersURL)
        return try JSONDecoder().decode([ModelProvider].self, from: data)
    }

    // MARK: - Model Configurations

    func saveConfigurations(_ configurations: [ModelConfiguration]) throws {
        try ensureDirectory()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(configurations)
        try data.write(to: configurationsURL, options: .atomic)
    }

    func loadConfigurations() throws -> [ModelConfiguration] {
        guard FileManager.default.fileExists(atPath: configurationsURL.path) else { return [] }
        let data = try Data(contentsOf: configurationsURL)
        return try JSONDecoder().decode([ModelConfiguration].self, from: data)
    }

    // MARK: - Model Catalog (cache)

    func saveModelCatalog(_ models: [ModelInfo]) throws {
        try ensureDirectory()
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(models)
        try data.write(to: modelCatalogURL, options: .atomic)
    }

    func loadModelCatalog() throws -> [ModelInfo] {
        guard FileManager.default.fileExists(atPath: modelCatalogURL.path) else { return [] }
        let data = try Data(contentsOf: modelCatalogURL)
        return try JSONDecoder().decode([ModelInfo].self, from: data)
    }

    // MARK: - Seen-models ledger

    func saveSeenModels(_ ledger: SeenModelsLedger) throws {
        try ensureDirectory()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(ledger)
        try data.write(to: seenModelsURL, options: .atomic)
    }

    func loadSeenModels() throws -> SeenModelsLedger {
        guard FileManager.default.fileExists(atPath: seenModelsURL.path) else { return SeenModelsLedger() }
        do {
            let data = try Data(contentsOf: seenModelsURL)
            return try JSONDecoder().decode(SeenModelsLedger.self, from: data)
        } catch {
            // Ledger keys are never supposed to be forgotten (a delisted model must stay "seen").
            // Re-seeding over an unreadable file would silently erase that history and let a
            // relisted model masquerade as a discovery — so preserve the corrupt file for
            // forensics/recovery instead of letting the reseed clobber it.
            let corruptURL = seenModelsURL.deletingPathExtension().appendingPathExtension("corrupt.json")
            try? FileManager.default.removeItem(at: corruptURL)
            try? FileManager.default.moveItem(at: seenModelsURL, to: corruptURL)
            logger.error("Seen-models ledger unreadable; moved aside to \(corruptURL.lastPathComponent, privacy: .public)")
            throw error
        }
    }

    // MARK: - URLs

    private var providersURL: URL {
        baseDirectory.appendingPathComponent("providers.json")
    }

    private var seenModelsURL: URL {
        baseDirectory.appendingPathComponent("seen_models.json")
    }

    private var configurationsURL: URL {
        baseDirectory.appendingPathComponent("model_configurations.json")
    }

    private var modelCatalogURL: URL {
        baseDirectory.appendingPathComponent("model_catalog.json")
    }
}
