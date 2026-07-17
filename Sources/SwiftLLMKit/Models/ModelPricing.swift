import Foundation

/// A complete set of per-token rates for a single pricing context.
/// All values are in USD per single token (matching LiteLLM's native unit).
/// Conversion to per-million display happens only at the UI layer.
public struct PricingTier: Codable, Sendable, Equatable {
    /// Cost per uncached input token.
    public var input: Double?
    /// Cost per output token.
    public var output: Double?
    /// Cost per cache-read input token.
    public var cacheRead: Double?
    /// Cost per cache-write (creation) input token.
    public var cacheWrite: Double?

    public init(
        input: Double? = nil,
        output: Double? = nil,
        cacheRead: Double? = nil,
        cacheWrite: Double? = nil
    ) {
        self.input = input
        self.output = output
        self.cacheRead = cacheRead
        self.cacheWrite = cacheWrite
    }

    /// Whether any rate is populated.
    public var hasAnyRate: Bool {
        input != nil || output != nil || cacheRead != nil || cacheWrite != nil
    }
}

/// A pricing tier that activates when total input context exceeds a token threshold.
public struct TokenThresholdTier: Codable, Sendable, Equatable, Comparable {
    /// The token count above which this tier's rates apply (e.g. 200_000, 128_000).
    public var tokenThreshold: Int
    /// Rates that apply above this threshold. Only non-nil fields override the base tier.
    public var rates: PricingTier

    public init(tokenThreshold: Int, rates: PricingTier) {
        self.tokenThreshold = tokenThreshold
        self.rates = rates
    }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.tokenThreshold < rhs.tokenThreshold
    }
}

/// Override for cache write cost under extended TTL (e.g. Anthropic 1hr cache).
public struct CacheWriteOverride: Codable, Sendable, Equatable {
    /// Cache write cost per token when using extended TTL.
    public var cacheWrite: Double
    /// Additional threshold-specific overrides that apply above a token count.
    public var thresholdOverrides: [TokenThresholdCacheWrite]

    public init(cacheWrite: Double, thresholdOverrides: [TokenThresholdCacheWrite] = []) {
        self.cacheWrite = cacheWrite
        self.thresholdOverrides = thresholdOverrides
    }
}

/// Cache write cost above a token threshold under extended TTL.
public struct TokenThresholdCacheWrite: Codable, Sendable, Equatable {
    public var tokenThreshold: Int
    public var cacheWrite: Double

    public init(tokenThreshold: Int, cacheWrite: Double) {
        self.tokenThreshold = tokenThreshold
        self.cacheWrite = cacheWrite
    }
}

/// Rich pricing data for an LLM model, supporting tiered pricing,
/// cache pricing, and service tier variants.
///
/// Most models only need the ``base`` tier. More complex pricing structures
/// (Anthropic's 200K threshold, OpenAI's priority/flex tiers, Gemini's 128K
/// threshold) are captured in the optional fields.
public struct ModelPricing: Codable, Sendable, Equatable {
    /// The base (default) pricing tier.
    public var base: PricingTier

    /// Tiers that activate above a token-count threshold.
    /// Sorted by threshold ascending. Typically 0-2 entries.
    public var tokenThresholdTiers: [TokenThresholdTier]

    /// Service-tier variants (e.g. "priority", "flex", "batches").
    /// Empty for most models. Keyed by tier name.
    public var serviceTiers: [String: PricingTier]

    /// Extended cache TTL rates (e.g. Anthropic's 1hr+ cache write cost).
    public var extendedCacheTier: CacheWriteOverride?

    public init(
        base: PricingTier = PricingTier(),
        tokenThresholdTiers: [TokenThresholdTier] = [],
        serviceTiers: [String: PricingTier] = [:],
        extendedCacheTier: CacheWriteOverride? = nil
    ) {
        self.base = base
        self.tokenThresholdTiers = tokenThresholdTiers.sorted()
        self.serviceTiers = serviceTiers
        self.extendedCacheTier = extendedCacheTier
    }

    /// Returns the effective rates for a given context by resolving threshold
    /// tiers and service tier into a single ``PricingTier``.
    ///
    /// - Parameters:
    ///   - totalInputTokens: Total input tokens in the request. Used to determine
    ///     whether threshold tiers apply. Pass `nil` to use base rates only.
    ///   - serviceTier: Service tier name (e.g. "priority", "flex"). Pass `nil`
    ///     for standard pricing.
    ///   - extendedCache: Whether extended cache TTL is active.
    /// - Returns: A resolved ``PricingTier`` with rates for this context.
    public func effectiveRates(
        totalInputTokens: Int? = nil,
        serviceTier: String? = nil,
        extendedCache: Bool = false
    ) -> PricingTier {
        var resolved = base

        // Apply the highest matching threshold tier. Sort here rather than trust stored order: the
        // memberwise init sorts, but the *synthesized* Codable init does not, so a decoded pricing
        // blob can carry tiers in any order — and this loop is last-wins, so an unsorted list would
        // apply a lower threshold's rates last and misprice. Tiers are 0–2 entries; the sort is free.
        if let inputTokens = totalInputTokens {
            for tier in tokenThresholdTiers.sorted() where inputTokens > tier.tokenThreshold {
                if let v = tier.rates.input { resolved.input = v }
                if let v = tier.rates.output { resolved.output = v }
                if let v = tier.rates.cacheRead { resolved.cacheRead = v }
                if let v = tier.rates.cacheWrite { resolved.cacheWrite = v }
            }
        }

        // Apply service tier (completely overrides where populated).
        if let tierName = serviceTier, let tier = serviceTiers[tierName] {
            if let v = tier.input { resolved.input = v }
            if let v = tier.output { resolved.output = v }
            if let v = tier.cacheRead { resolved.cacheRead = v }
            if let v = tier.cacheWrite { resolved.cacheWrite = v }
        }

        // Apply extended cache write rate.
        if extendedCache, let ext = extendedCacheTier {
            resolved.cacheWrite = ext.cacheWrite
            if let inputTokens = totalInputTokens {
                for override in ext.thresholdOverrides where inputTokens > override.tokenThreshold {
                    resolved.cacheWrite = override.cacheWrite
                }
            }
        }

        return resolved
    }

    /// Computes the estimated cost in USD for a single API call.
    ///
    /// - Parameters:
    ///   - usage: Token counts from the API response.
    ///   - totalInputTokens: Total input tokens in the request (for threshold
    ///     tier resolution). Defaults to `usage.inputTokens` if not provided.
    ///   - serviceTier: Service tier name if applicable.
    ///   - extendedCache: Whether extended cache TTL was used.
    /// - Returns: Estimated cost in USD, or `nil` if base pricing lacks both
    ///   input and output rates.
    public func estimatedCost(
        for usage: TokenUsage,
        totalInputTokens: Int? = nil,
        serviceTier: String? = nil,
        extendedCache: Bool = false
    ) -> Double? {
        guard base.input != nil || base.output != nil else { return nil }

        let rates = effectiveRates(
            totalInputTokens: totalInputTokens ?? usage.inputTokens,
            serviceTier: serviceTier,
            extendedCache: extendedCache
        )

        var cost = 0.0

        // Uncached input: total input minus cache-read and cache-write tokens.
        // Anthropic's input_tokens includes all three categories; each is billed
        // at its own rate, so we must not double-count.
        let uncachedInput = max(0, usage.inputTokens - usage.cacheReadTokens - usage.cacheWriteTokens)
        cost += Double(uncachedInput) * (rates.input ?? 0)

        // Output tokens.
        cost += Double(usage.outputTokens) * (rates.output ?? 0)

        // Cache read tokens.
        cost += Double(usage.cacheReadTokens) * (rates.cacheRead ?? 0)

        // Cache write tokens.
        cost += Double(usage.cacheWriteTokens) * (rates.cacheWrite ?? 0)

        return cost
    }
}
