import Foundation

/// A special availability condition a model can be in. Absence of any of these is the ordinary,
/// fully-usable case — a "normal" model carries an EMPTY set of these.
///
/// The concrete conditions are derived from ``ModelInfo``'s tri-state fields (`isAvailable`,
/// `isAccessDenied`, `deprecatedOn`); see ``ModelInfo/availabilityStates``. `.all` is not a
/// condition a model reports — it's the aggregate marker (present only when every real condition is)
/// and, in a query's include-set, the shorthand for "any special state is allowed."
public enum ModelAvailabilityState: String, CaseIterable, Sendable, Codable, Hashable {
    /// A probe established the model is gone (`ModelInfo.isAvailable == false`).
    case isUnavailable
    /// A probe by this provider's own key was refused for access reasons (`ModelInfo.isAccessDenied == true`).
    case isAccessDenied
    /// The provider scheduled a deprecation that is still in the future (`deprecatedOn` in the future).
    case isFutureDeprecated
    /// The provider's deprecation date has passed (`deprecatedOn` in the past).
    case isDeprecated
    /// Aggregate marker: present in a model's set only when every real state is (see
    /// ``realStates``). Because `isFutureDeprecated` and `isDeprecated` are mutually exclusive, a
    /// model can never actually carry `.all`; its real use is in a query's include-set, where it
    /// means "permit models in ANY special state."
    case all

    /// The concrete conditions — every case except the `.all` aggregate. Used to derive `.all` and to
    /// expand `.all` when it appears in a query's include-set.
    public static let realStates: Set<ModelAvailabilityState> = [
        .isUnavailable, .isAccessDenied, .isFutureDeprecated, .isDeprecated
    ]
}

// MARK: - ModelInfo availability + capability filtering

extension ModelInfo {
    /// The special availability conditions this model is currently in, derived live from its
    /// tri-state fields. Empty for an ordinary, fully-usable model. `.all` is included only when
    /// every real state is present (see ``ModelAvailabilityState/all`` — in practice never, since a
    /// deprecation date is either future or past, not both).
    public var availabilityStates: Set<ModelAvailabilityState> {
        var states: Set<ModelAvailabilityState> = []
        if isAvailable == false { states.insert(.isUnavailable) }
        if isAccessDenied == true { states.insert(.isAccessDenied) }
        if let deprecatedOn {
            if deprecatedOn > Date() {
                states.insert(.isFutureDeprecated)
            } else {
                states.insert(.isDeprecated)
            }
        }
        if ModelAvailabilityState.realStates.isSubset(of: states) {
            states.insert(.all)
        }
        return states
    }

    /// Whether this model passes a capability/availability filter.
    ///
    /// Capabilities are TRI-STATE, so filtering acts only on KNOWN facts — an unmeasured capability
    /// never disqualifies a model:
    /// - `requiredCapabilities`: rejected only when a required capability is KNOWN-FALSE. Known-true
    ///   and unknown both pass — an unprobed model stays visible.
    /// - `mustNotBePresent`: rejected only when a forbidden capability is KNOWN-TRUE. Known-false and
    ///   unknown both pass.
    /// - `includedAvailabilityStates`: the model's special states must ALL be permitted here — so a
    ///   model with no special states always passes, and a model with a special state passes only if
    ///   that state was explicitly included. `.all` here permits any special state.
    public func satisfies(
        requiredCapabilities: Set<ModelCapability>,
        mustNotBePresent: Set<ModelCapability>,
        includedAvailabilityStates: Set<ModelAvailabilityState>
    ) -> Bool {
        for capability in requiredCapabilities where capabilities.state(of: capability) == false {
            return false
        }
        for capability in mustNotBePresent where capabilities.state(of: capability) == true {
            return false
        }
        let allowed = includedAvailabilityStates.contains(.all)
            ? ModelAvailabilityState.realStates
            : includedAvailabilityStates
        let modelSpecialStates = availabilityStates.intersection(ModelAvailabilityState.realStates)
        return modelSpecialStates.isSubset(of: allowed)
    }
}

/// A reusable bundle of the three arguments to ``LLMKitManager/availableModels(requiredCapabilities:mustNotBePresent:alsoIncludingAvailabilityStates:)``.
///
/// Lets a caller (e.g. an app's agent role) declare, in one readable value, exactly what its models
/// must and must not do and which special states it tolerates.
public struct ModelRequirements: Sendable, Equatable {
    /// Every capability a matching model MUST have.
    public var requiredCapabilities: Set<ModelCapability>
    /// Every capability a matching model must NOT have.
    public var mustNotBePresent: Set<ModelCapability>
    /// Special availability states to permit. Empty = only fully-normal models; `.all` = any state.
    public var includedAvailabilityStates: Set<ModelAvailabilityState>

    public init(
        requiredCapabilities: Set<ModelCapability> = [],
        mustNotBePresent: Set<ModelCapability> = [],
        includedAvailabilityStates: Set<ModelAvailabilityState> = []
    ) {
        self.requiredCapabilities = requiredCapabilities
        self.mustNotBePresent = mustNotBePresent
        self.includedAvailabilityStates = includedAvailabilityStates
    }
}
