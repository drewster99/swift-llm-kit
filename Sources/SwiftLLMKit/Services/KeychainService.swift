import Foundation
import os
import Security

private let logger = Logger(subsystem: "SwiftLLMKit", category: "Keychain")

/// Manages API key storage in the macOS Keychain with an in-memory cache.
///
/// Each provider's API key is stored as a generic password with:
/// - Service: `<keychainServicePrefix>.<appBundleID>`
/// - Account: provider ID
///
/// Keys are cached in RAM after the first successful Keychain read to avoid
/// transient Keychain access failures (contention, lock-screen delays) that
/// would otherwise produce empty API keys and downstream 401 errors.
struct KeychainService: Sendable {
    private let service: String

    /// In-memory cache of API keys, keyed by provider ID. Protected by a lock
    /// since `KeychainService` is a value-type `Sendable` struct shared across
    /// multiple provider closures.
    private let cache: APIKeyCache

    /// Creates a keychain service scoped to the given identifiers.
    /// - Parameters:
    ///   - keychainServicePrefix: A reverse-DNS prefix, e.g. "com.yourname.SwiftLLMKit".
    ///   - appIdentifier: Typically `Bundle.main.bundleIdentifier`.
    public init(keychainServicePrefix: String, appIdentifier: String) {
        self.service = "\(keychainServicePrefix).\(appIdentifier)"
        self.cache = APIKeyCache()
    }

    /// Stores or updates an API key for the given provider.
    ///
    /// Invalidates the in-memory cache for this provider so subsequent reads
    /// pick up the new key.
    ///
    /// **Keychain fallback:** the Data Protection Keychain (DPK) requires the
    /// process to be code-signed with a `keychain-access-groups` entitlement.
    /// GUI apps built via Xcode get this for free; unsigned/ad-hoc-signed CLI
    /// binaries built via Swift Package Manager don't. When DPK returns
    /// `errSecMissingEntitlement` we transparently fall back to the legacy
    /// (login) keychain so CLI tools work out of the box. Read path
    /// (`apiKey(forProviderID:)`) already prefers DPK and falls back to legacy.
    /// - Throws: `KeychainError` if the operation fails.
    public func save(apiKey: String, forProviderID providerID: String) throws {
        guard let data = apiKey.data(using: .utf8) else {
            throw KeychainError.encodingFailed
        }

        do {
            try saveImpl(data: data, providerID: providerID, useDataProtection: true)
        } catch KeychainError.saveFailed(let status) where status == errSecMissingEntitlement {
            logger.info("DPK save rejected with missing-entitlement; falling back to legacy keychain for provider \(providerID, privacy: .public)")
            try saveImpl(data: data, providerID: providerID, useDataProtection: false)
        }

        // Update the cache with the new key so subsequent reads are immediate.
        cache.set(apiKey, forProviderID: providerID)
    }

    /// Inner save that targets either DPK or legacy based on `useDataProtection`.
    /// Splits the update-first-then-add logic so the outer `save` can retry on
    /// fallback without duplicating it.
    private func saveImpl(data: Data, providerID: String, useDataProtection: Bool) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: providerID,
            kSecUseDataProtectionKeychain as String: useDataProtection
        ]

        let updateAttributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock
        ]

        let updateStatus = SecItemUpdate(query as CFDictionary, updateAttributes as CFDictionary)

        if updateStatus == errSecItemNotFound {
            var addQuery = query
            addQuery[kSecValueData as String] = data
            addQuery[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
            let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
            guard addStatus == errSecSuccess else {
                throw KeychainError.saveFailed(status: addStatus)
            }
        } else if updateStatus != errSecSuccess {
            throw KeychainError.saveFailed(status: updateStatus)
        }
    }

    /// Retrieves the API key for the given provider, or `nil` if not stored.
    ///
    /// Returns the cached value if available. Otherwise reads from the data protection
    /// keychain (falling back to legacy keychain with automatic migration) and caches
    /// the result on success. If the Keychain read fails transiently but a cached value
    /// exists, the cached value is returned.
    public func apiKey(forProviderID providerID: String) -> String? {
        // Fast path: return cached key if available.
        if let cached = cache.get(forProviderID: providerID) {
            return cached
        }

        // Try data protection keychain first
        let dpQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: providerID,
            kSecUseDataProtectionKeychain as String: true,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let dpStatus = SecItemCopyMatching(dpQuery as CFDictionary, &result)

        if dpStatus == errSecSuccess, let data = result as? Data,
           let key = String(data: data, encoding: .utf8) {
            migrateAccessibilityIfNeeded(query: dpQuery)
            cache.set(key, forProviderID: providerID)
            return key
        }

        if dpStatus != errSecItemNotFound {
            logger.warning("Keychain read failed for provider \(providerID, privacy: .public): OSStatus \(dpStatus)")
        }

        // Fall back to legacy (login) keychain for pre-migration entries
        let legacyQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: providerID,
            kSecUseDataProtectionKeychain as String: false,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var legacyResult: AnyObject?
        let legacyStatus = SecItemCopyMatching(legacyQuery as CFDictionary, &legacyResult)

        guard legacyStatus == errSecSuccess, let legacyData = legacyResult as? Data,
              let key = String(data: legacyData, encoding: .utf8) else {
            return nil
        }

        // Try to migrate to the data protection keychain ONLY if DPK is actually
        // usable for this process (entitled). We use saveImpl directly with
        // useDataProtection: true so it bypasses the legacy fallback path —
        // if DPK rejects with errSecMissingEntitlement (the typical case for
        // unsigned/ad-hoc-signed CLI binaries), the throw is caught here and
        // we leave the legacy entry in place. WITHOUT this guard, the
        // fallback-save in our public save(...) would write back to legacy and
        // we'd then delete the legacy entry below, losing the only copy.
        // `key` came from `legacyData` decoded as UTF-8 just above, so re-encoding
        // to UTF-8 cannot fail — Data(key.utf8) avoids the dead optional path.
        let data = Data(key.utf8)
        do {
            try saveImpl(data: data, providerID: providerID, useDataProtection: true)
            logger.info("Migrated API key for \(providerID, privacy: .public) to data protection keychain")
            // Only safe to delete legacy after the DPK-only save actually succeeded.
            SecItemDelete(legacyQuery as CFDictionary)
        } catch {
            // Expected when this process lacks the keychain-access-groups
            // entitlement; the key stays in legacy and we keep reading from there.
            logger.debug("Leaving API key for \(providerID, privacy: .public) in legacy keychain (DPK unavailable: \(error.localizedDescription, privacy: .public))")
        }

        cache.set(key, forProviderID: providerID)
        return key
    }

    /// Ensures a keychain item uses `kSecAttrAccessibleAfterFirstUnlock`.
    ///
    /// Called after a successful read to migrate items that were saved with the
    /// default accessibility (`WhenUnlocked`). Only writes if the current
    /// accessibility differs from the target.
    private func migrateAccessibilityIfNeeded(query: [String: Any]) {
        // Build an attributes-only query to check the current accessibility.
        var attrQuery = query
        attrQuery.removeValue(forKey: kSecReturnData as String)
        attrQuery[kSecReturnAttributes as String] = true
        attrQuery[kSecMatchLimit as String] = kSecMatchLimitOne

        var attrResult: AnyObject?
        let attrStatus = SecItemCopyMatching(attrQuery as CFDictionary, &attrResult)

        if attrStatus == errSecSuccess,
           let attrs = attrResult as? [String: Any],
           let current = attrs[kSecAttrAccessible as String] as? String,
           current == kSecAttrAccessibleAfterFirstUnlock as String {
            return  // Already correct — no write needed.
        }

        // Strip read-specific keys to build a match-only query for the update.
        var matchQuery = query
        matchQuery.removeValue(forKey: kSecReturnData as String)
        matchQuery.removeValue(forKey: kSecMatchLimit as String)

        let updateAttrs: [String: Any] = [
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock
        ]

        let providerID = query[kSecAttrAccount as String] as? String ?? "unknown"
        let status = SecItemUpdate(matchQuery as CFDictionary, updateAttrs as CFDictionary)
        if status == errSecSuccess {
            logger.info("Migrated keychain accessibility to AfterFirstUnlock for provider \(providerID, privacy: .public)")
        } else {
            logger.warning("Accessibility migration failed for provider \(providerID, privacy: .public): OSStatus \(status)")
        }
    }

    /// Deletes the API key for the given provider.
    ///
    /// Deletes from BOTH the Data Protection Keychain and the legacy keychain
    /// so a key that was saved via the fallback path doesn't linger after a
    /// later deletion attempt. Missing items are not an error in either store.
    /// - Throws: `KeychainError` if a non-fallback delete fails.
    public func delete(forProviderID providerID: String) throws {
        try deleteImpl(providerID: providerID, useDataProtection: true)
        try deleteImpl(providerID: providerID, useDataProtection: false)
        cache.remove(forProviderID: providerID)
    }

    private func deleteImpl(providerID: String, useDataProtection: Bool) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: providerID,
            kSecUseDataProtectionKeychain as String: useDataProtection
        ]
        let status = SecItemDelete(query as CFDictionary)
        // errSecItemNotFound: nothing to delete in this store — fine.
        // errSecMissingEntitlement: this binary can't access this store — fine,
        //   the entry can't be there anyway and the matching `save` would have
        //   fallen back to the other store.
        guard status == errSecSuccess
                || status == errSecItemNotFound
                || status == errSecMissingEntitlement else {
            throw KeychainError.deleteFailed(status: status)
        }
    }

    /// Drops all cached API keys, forcing the next read to go to Keychain.
    public func invalidateAllCachedKeys() {
        cache.removeAll()
    }
}

/// Thread-safe in-memory cache for API keys.
///
/// Uses `NSLock` for synchronization since accesses are brief and non-blocking.
/// Shared by reference across `KeychainService` value copies.
private final class APIKeyCache: Sendable {
    private let lock = NSLock()
    private nonisolated(unsafe) var store: [String: String] = [:]

    func get(forProviderID providerID: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return store[providerID]
    }

    func set(_ key: String, forProviderID providerID: String) {
        lock.lock()
        defer { lock.unlock() }
        store[providerID] = key
    }

    func remove(forProviderID providerID: String) {
        lock.lock()
        defer { lock.unlock() }
        store[providerID] = nil
    }

    func removeAll() {
        lock.lock()
        defer { lock.unlock() }
        store.removeAll()
    }
}

/// Errors from Keychain operations.
private enum KeychainError: Error, LocalizedError {
    case encodingFailed
    case saveFailed(status: OSStatus)
    case deleteFailed(status: OSStatus)

    var errorDescription: String? {
        switch self {
        case .encodingFailed:
            return "Failed to encode API key as UTF-8"
        case .saveFailed(let status):
            return "Keychain save failed (OSStatus \(status))"
        case .deleteFailed(let status):
            return "Keychain delete failed (OSStatus \(status))"
        }
    }
}
