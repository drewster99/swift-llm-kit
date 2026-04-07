import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "BundledMetadata")

/// Registry of app-bundled model metadata overrides.
///
/// These ship with the library binary and fill gaps between LiteLLM data
/// (which may be stale) and provider APIs (which often lack pricing/capability
/// data). Entries are keyed by `"providerAPIType/modelID"`.
public struct BundledModelMetadataRegistry: Sendable {
    /// Overrides keyed by `"providerAPIType/modelID"`.
    public let entries: [String: ModelMetadataOverride]

    /// Loads the bundled metadata from the package's resource bundle.
    /// Returns an empty registry if the resource is missing or malformed.
    public static func load() -> BundledModelMetadataRegistry {
        guard let url = Bundle.module.url(forResource: "bundled_model_metadata", withExtension: "json") else {
            logger.warning("Bundled model metadata resource not found")
            return BundledModelMetadataRegistry(entries: [:])
        }
        do {
            let data = try Data(contentsOf: url)
            let wrapper = try JSONDecoder().decode(BundledWrapper.self, from: data)
            logger.info("Loaded \(wrapper.entries.count) bundled model metadata entries (v\(wrapper.version))")
            return BundledModelMetadataRegistry(entries: wrapper.entries)
        } catch {
            logger.error("Failed to load bundled model metadata: \(error.localizedDescription, privacy: .public)")
            return BundledModelMetadataRegistry(entries: [:])
        }
    }

    /// Looks up an override by provider API type and model ID.
    public func override(providerAPIType: String, modelID: String) -> ModelMetadataOverride? {
        entries["\(providerAPIType)/\(modelID)"]
    }
}

/// Internal JSON wrapper matching the bundled_model_metadata.json schema.
private struct BundledWrapper: Codable {
    let version: Int
    let entries: [String: ModelMetadataOverride]
}
