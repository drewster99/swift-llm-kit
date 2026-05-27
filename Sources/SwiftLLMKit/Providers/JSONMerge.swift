import Foundation

/// Deep-merges `overrides` into `body` in place.
///
/// **Merge rule:** for each key in `overrides`:
/// - If BOTH `body[key]` and the override value are dictionaries, recurse so
///   sibling sub-keys from `body` survive (e.g. setting
///   `generationConfig.thinkingConfig` doesn't wipe `generationConfig.temperature`).
/// - Otherwise (scalars, arrays, strings, type mismatches), the override
///   replaces the body value outright.
///
/// Arrays do NOT merge element-wise — there's no sensible "combine two
/// arrays of message content blocks" semantic. Callers that want to add
/// to an array must build the full replacement array themselves.
///
/// Used by every provider to apply `ModelConfiguration.extraJSONOverrides`
/// to the outbound request body without clobbering structured sub-keys the
/// provider built (cache_control breakpoints, generationConfig defaults, etc.).
func mergeJSONOverrides(_ body: inout [String: Any], with overrides: [String: AnyCodable]) {
    for (key, value) in overrides {
        if case .dictionary(let nestedOverride) = value,
           var nestedExisting = body[key] as? [String: Any] {
            mergeJSONOverrides(&nestedExisting, with: nestedOverride)
            body[key] = nestedExisting
        } else {
            body[key] = value.rawValue
        }
    }
}
