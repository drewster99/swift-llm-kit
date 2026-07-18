import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "BundledMetadata")

/// Registry of app-bundled model metadata overrides.
///
/// These ship with the library binary and fill gaps between LiteLLM data
/// (which may be stale) and provider APIs (which often lack pricing/capability
/// data).
///
/// Three lookup axes, listed by priority (more-specific wins on conflict):
///
///   1. **Per-(apiType, modelID)** — `entries`. Applies to every provider that
///      uses this apiType. Best for cross-provider model traits (e.g. Mistral
///      models supporting parallel tool calls regardless of which Mistral
///      gateway is hosting them).
///
///   2. **Per-(providerID, modelID)** — `providerEntries`. Pinpoints a single
///      built-in provider's model. Used when an apiType is shared between the
///      built-in provider and user-created providers (e.g. `openAICompatible`
///      covers both `builtin.openai` and a user's DeepSeek), but the trait
///      only applies to the built-in.
///
///   3. **Per-providerID defaults** — `providerDefaults`. Applies to *every*
///      model from a given built-in provider. Used when a provider-wide trait
///      holds (e.g. all `builtin.openai` models accept `max_completion_tokens`).
///      Per-model entries can override.
struct BundledModelMetadataRegistry: Sendable {
    /// Per-(apiType, modelID) overrides.
    let entries: [String: ModelMetadataOverride]
    /// Per-(providerID, modelID) overrides — pinpoints a single built-in provider.
    let providerEntries: [String: ModelMetadataOverride]
    /// Per-providerID defaults — applies to every model from that provider.
    let providerDefaults: [String: ModelMetadataOverride]

    /// Loads the bundled metadata from the package's resource bundle.
    /// Returns an empty registry if the resource is missing or malformed.
    public static func load() -> BundledModelMetadataRegistry {
        guard let url = Bundle.module.url(forResource: "bundled_model_metadata", withExtension: "json") else {
            logger.warning("Bundled model metadata resource not found")
            return BundledModelMetadataRegistry(entries: [:], providerEntries: [:], providerDefaults: [:])
        }
        do {
            let data = try Data(contentsOf: url)
            let wrapper = try JSONDecoder().decode(BundledWrapper.self, from: data)
            logger.info("Loaded bundled model metadata v\(wrapper.version): \(wrapper.entries.count) by-apiType, \(wrapper.providerEntries?.count ?? 0) by-providerID, \(wrapper.providerDefaults?.count ?? 0) provider defaults")
            return BundledModelMetadataRegistry(
                entries: wrapper.entries,
                providerEntries: wrapper.providerEntries ?? [:],
                providerDefaults: wrapper.providerDefaults ?? [:]
            )
        } catch {
            logger.error("Failed to load bundled model metadata: \(error.localizedDescription, privacy: .public)")
            return BundledModelMetadataRegistry(entries: [:], providerEntries: [:], providerDefaults: [:])
        }
    }

    /// Loads a registry from an external file in the SAME schema as the bundled resource — the
    /// downloaded-overrides slot. Returns nil when the file is absent; logs and returns nil when
    /// it is unreadable (a bad downloaded file must never take out the bundled layer).
    public static func load(from url: URL) -> BundledModelMetadataRegistry? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        do {
            let wrapper = try JSONDecoder().decode(BundledWrapper.self, from: data)
            logger.info("Loaded downloaded overrides v\(wrapper.version): \(wrapper.entries.count) by-apiType, \(wrapper.providerEntries?.count ?? 0) by-providerID, \(wrapper.providerDefaults?.count ?? 0) provider defaults")
            return BundledModelMetadataRegistry(
                entries: wrapper.entries,
                providerEntries: wrapper.providerEntries ?? [:],
                providerDefaults: wrapper.providerDefaults ?? [:]
            )
        } catch {
            logger.error("Downloaded overrides at \(url.lastPathComponent, privacy: .public) unreadable, ignoring: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    /// This registry with `newer`'s entries laid over it — same-key entries in `newer` replace
    /// this registry's. Used to let a downloaded overrides file supersede the bundled one inside
    /// the single downloaded-overrides layer (fresher curation wins; both remain one layer).
    public func overlaying(_ newer: BundledModelMetadataRegistry) -> BundledModelMetadataRegistry {
        BundledModelMetadataRegistry(
            entries: entries.merging(newer.entries) { _, new in new },
            providerEntries: providerEntries.merging(newer.providerEntries) { _, new in new },
            providerDefaults: providerDefaults.merging(newer.providerDefaults) { _, new in new }
        )
    }

    /// Model IDs this registry pinpoints for one provider — the enumerable axis behind
    /// union-of-layers existence (an override-only entry materializes a model the provider's
    /// `/models` no longer lists, e.g. a delisted-but-callable snapshot). Only the
    /// providerID axis enumerates: an apiType-scoped entry applies to every provider of that
    /// type, and materializing a model on all of them from one entry would fabricate listings.
    public func providerScopedModelIDs(providerID: String) -> [String] {
        let prefix = "\(providerID)/"
        return providerEntries.keys
            .filter { $0.hasPrefix(prefix) }
            .map { String($0.dropFirst(prefix.count)) }
    }

    /// Looks up an override by provider API type and model ID.
    public func override(providerAPIType: String, modelID: String) -> ModelMetadataOverride? {
        entries["\(providerAPIType)/\(modelID)"]
    }

    /// Looks up a per-(providerID, modelID) override. Returns nil if no entry.
    public func override(providerID: String, modelID: String) -> ModelMetadataOverride? {
        providerEntries["\(providerID)/\(modelID)"]
    }

    /// Looks up the provider-wide defaults (applies to every model from this provider).
    public func defaults(providerID: String) -> ModelMetadataOverride? {
        providerDefaults[providerID]
    }
}

/// Internal JSON wrapper matching the bundled_model_metadata.json schema.
///
/// Schema v1: `entries` only (apiType-keyed).
/// Schema v2 (current): adds `providerEntries` and `providerDefaults` for cases
/// where apiType-keyed lookup is too coarse (e.g., distinguishing
/// `builtin.openai` from a user's DeepSeek both running `openAICompatible`).
private struct BundledWrapper: Codable {
    let version: Int
    let entries: [String: ModelMetadataOverride]
    let providerEntries: [String: ModelMetadataOverride]?
    let providerDefaults: [String: ModelMetadataOverride]?
}
