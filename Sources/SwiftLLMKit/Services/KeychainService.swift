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
    /// - Throws: `KeychainError` if the operation fails.
    public func save(apiKey: String, forProviderID providerID: String) throws {
        guard let data = apiKey.data(using: .utf8) else {
            throw KeychainError.encodingFailed
        }

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: providerID,
            kSecUseDataProtectionKeychain as String: true
        ]

        // Try to update first — also set accessibility so existing items are migrated.
        let updateAttributes: [String: Any] = [
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock
        ]

        let updateStatus = SecItemUpdate(query as CFDictionary, updateAttributes as CFDictionary)

        if updateStatus == errSecItemNotFound {
            // Item doesn't exist yet — add it
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

        // Update the cache with the new key so subsequent reads are immediate.
        cache.set(apiKey, forProviderID: providerID)
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

        // Migrate to data protection keychain
        logger.info("Migrating API key for \(providerID, privacy: .public) to data protection keychain")
        do {
            try save(apiKey: key, forProviderID: providerID)
            // Remove legacy entry after successful migration
            SecItemDelete(legacyQuery as CFDictionary)
        } catch {
            logger.warning("Migration to data protection keychain failed for \(providerID, privacy: .public): \(error.localizedDescription, privacy: .public)")
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
    /// - Throws: `KeychainError` if deletion fails (not found is not an error).
    public func delete(forProviderID providerID: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: providerID,
            kSecUseDataProtectionKeychain as String: true
        ]

        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deleteFailed(status: status)
        }

        cache.remove(forProviderID: providerID)
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
