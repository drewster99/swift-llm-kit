import Foundation

/// Which `providerID/modelID` keys this install has ever observed in a `/models` listing.
///
/// Purely a discovery ledger: keys are added and never removed — a delisted model stays "seen",
/// so a temporary delisting followed by re-listing is not a discovery event. Nothing consumes
/// discoveries automatically yet (probing is manual by design); the ledger exists so that when
/// discovery-driven features land, "new" already has a trustworthy meaning.
///
/// Seeding is PER PROVIDER, and the rule is load-bearing: a provider's first-ever listing seeds
/// silently and reports NOTHING as new. Global seeding wasn't enough — on a fresh install the
/// first provider's fetch would seed, and every later provider's first fetch would then report
/// its entire catalog as "discovered"; likewise pasting a new OpenRouter key would mark ~343
/// models new in one shot (and, once probe-on-discovery exists, fire a paid probe storm).
/// "New" means: appeared in a provider's listing AFTER we first learned that provider's catalog.
public struct SeenModelsLedger: Sendable, Equatable {
    /// When the ledger first recorded anything. Informational.
    public var seededAt: Date?
    /// Provider IDs whose first listing has been recorded. A provider not in this set gets its
    /// next observation seeded silently.
    public var seededProviders: Set<String>
    /// Every `providerID/modelID` key ever observed.
    public var seenKeys: Set<String>

    public init(seededAt: Date? = nil, seededProviders: Set<String> = [], seenKeys: Set<String> = []) {
        self.seededAt = seededAt
        self.seededProviders = seededProviders
        self.seenKeys = seenKeys
    }

    /// Records one provider's listing and returns the keys that are genuinely NEW — always `[]`
    /// on that provider's first-ever listing, per the seeding rule above.
    @discardableResult
    public mutating func observe(providerID: String, keys: some Sequence<String>, at date: Date) -> Set<String> {
        if seededAt == nil { seededAt = date }
        let isFirstListing = seededProviders.insert(providerID).inserted
        if isFirstListing {
            seenKeys.formUnion(keys)
            return []
        }
        let fresh = Set(keys).subtracting(seenKeys)
        seenKeys.formUnion(fresh)
        return fresh
    }
}

extension SeenModelsLedger: Codable {
    private enum CodingKeys: String, CodingKey {
        case seededAt, seededProviders, seenKeys
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        seededAt = try container.decodeIfPresent(Date.self, forKey: .seededAt)
        // decodeIfPresent throughout so a ledger written by an older schema (or a future one
        // missing a key) degrades to defaults instead of throwing the whole file away.
        seededProviders = try container.decodeIfPresent(Set<String>.self, forKey: .seededProviders) ?? []
        seenKeys = try container.decodeIfPresent(Set<String>.self, forKey: .seenKeys) ?? []
    }
}
