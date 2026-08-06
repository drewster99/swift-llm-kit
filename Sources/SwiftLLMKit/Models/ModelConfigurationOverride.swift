import Foundation

/// A sparse per-(caller, model) override of the RUNTIME inference settings — the sibling of
/// ``ModelMetadataOverride`` (which corrects model FACTS). Every field is optional: `nil` means
/// "inherit the model's resolved default", so the override holds *only* what was deliberately set.
///
/// It is never a source of resolved values. The effective ``ModelConfiguration`` is COMPUTED fresh
/// from the latest ``ModelInfo`` plus this delta (see ``resolved(against:name:)``), so a later probe
/// or refresh that changes the model's limits flows through automatically for every un-set field.
///
/// Overrides are PERMISSIVE: a value may exceed what the catalog believes about the model (the
/// catalog is merged evidence, not ground truth — the user may know better). Out-of-range values are
/// surfaced as non-blocking ``OverrideWarning``s (see ``warnings(against:)``), never clamped.
public struct ModelConfigurationOverride: Codable, Sendable, Equatable {
    /// Sampling temperature. `nil` inherits the model's `samplingDefaults` (or the provider default).
    public var temperature: Double?
    /// Max tokens to generate. `nil` inherits the model's reported output ceiling.
    public var maxOutputTokens: Int?
    /// Context-window budget for pruning. `nil` inherits the model's reported input window.
    public var maxContextTokens: Int?
    /// Extended-thinking token budget (Anthropic / Alibaba). `nil` inherits (off / model default).
    public var thinkingBudget: Int?
    /// The thinking SWITCH: force reasoning on or off per the model's discovered mechanism.
    /// `nil` inherits the model/provider default. See ``ModelConfiguration/reasoningEnabled``.
    public var reasoningEnabled: Bool?
    /// Adaptive-thinking effort hint. `nil` inherits the model/provider default.
    /// GENERAL effort override (Anthropic `output_config.effort`).
    public var effort: String?
    /// REASONING effort override (`reasoning_effort`).
    public var reasoningEffort: String?
    /// 1-hour prompt-cache TTL (Anthropic). `nil` inherits the default (false).
    public var extendedCacheTTL: Bool?
    /// Whether to stream. `nil` inherits the default (true).
    public var streaming: Bool?
    /// Free-form top-level request-body overrides. `nil` inherits none.
    public var extraJSONOverrides: [String: AnyCodable]?

    public init(
        temperature: Double? = nil,
        maxOutputTokens: Int? = nil,
        maxContextTokens: Int? = nil,
        thinkingBudget: Int? = nil,
        reasoningEnabled: Bool? = nil,
        effort: String? = nil,
        reasoningEffort: String? = nil,
        extendedCacheTTL: Bool? = nil,
        streaming: Bool? = nil,
        extraJSONOverrides: [String: AnyCodable]? = nil
    ) {
        self.temperature = temperature
        self.maxOutputTokens = maxOutputTokens
        self.maxContextTokens = maxContextTokens
        self.thinkingBudget = thinkingBudget
        self.reasoningEnabled = reasoningEnabled
        self.effort = effort
        self.reasoningEffort = reasoningEffort
        self.extendedCacheTTL = extendedCacheTTL
        self.streaming = streaming
        self.extraJSONOverrides = extraJSONOverrides
    }

    /// Whether nothing is overridden — the caller inherits every value from the model. An empty
    /// override should not be persisted (it is indistinguishable from having no entry at all).
    public var isEmpty: Bool {
        temperature == nil && maxOutputTokens == nil && maxContextTokens == nil
            && thinkingBudget == nil && reasoningEnabled == nil && effort == nil && reasoningEffort == nil
            && extendedCacheTTL == nil && streaming == nil
            && (extraJSONOverrides?.isEmpty ?? true)
    }

    /// The effective ``ModelConfiguration`` = the model's resolved defaults with this delta overlaid.
    /// Pure and cheap — call it fresh wherever the config is needed rather than storing the result.
    /// Fallbacks (4096 / 128000) apply only when the model reports no limit AND the field is un-set.
    public func resolved(against modelInfo: ModelInfo, name: String? = nil) -> ModelConfiguration {
        ModelConfiguration(
            // Deterministic so this projection is genuinely PURE (see the doc above): recomputing it
            // yields an identical value rather than a fresh random `id` that would defeat every
            // equality/`.task(id:)` check downstream. NOT the random `init(id:)` default.
            id: ModelConfiguration.deterministicID(providerID: modelInfo.providerID, modelID: modelInfo.modelID),
            name: name ?? modelInfo.displayName,
            providerID: modelInfo.providerID,
            modelID: modelInfo.modelID,
            temperature: temperature ?? modelInfo.samplingDefaults?.temperature,
            maxOutputTokens: maxOutputTokens ?? modelInfo.maxOutputTokens ?? 4096,
            maxContextTokens: maxContextTokens ?? modelInfo.maxInputTokens ?? 128_000,
            thinkingBudget: thinkingBudget,
            reasoningEnabled: reasoningEnabled,
            effort: effort,
            reasoningEffort: reasoningEffort,
            extendedCacheTTL: extendedCacheTTL ?? false,
            streaming: streaming ?? true,
            extraJSONOverrides: extraJSONOverrides
        )
    }

    /// Non-blocking warnings for overrides that exceed what the catalog believes about the model.
    /// Fires ONLY against a KNOWN bound — an unknown limit (`nil` / empty) can't be exceeded, so it
    /// yields no warning. Derived, never stored: a refresh that reveals the real ceiling clears or
    /// raises these automatically.
    public func warnings(against modelInfo: ModelInfo) -> [OverrideWarning] {
        var out: [OverrideWarning] = []
        if let v = maxOutputTokens, let known = modelInfo.maxOutputTokens, v > known {
            out.append(OverrideWarning(field: .maxOutputTokens,
                message: "Above the model's reported max output (\(known.formatted()) tokens)."))
        }
        if let v = maxContextTokens, let known = modelInfo.maxInputTokens, v > known {
            out.append(OverrideWarning(field: .maxContextTokens,
                message: "Above the model's reported context window (\(known.formatted()) tokens)."))
        }
        if let v = temperature, let maxT = modelInfo.maxTemperature, v > maxT {
            out.append(OverrideWarning(field: .temperature,
                message: "Above the model's reported maximum temperature (\(maxT.formatted()))."))
        }
        // Per construct, against that construct's own record. `rejects` fails safe, so an
        // unknown ladder never produces a warning.
        let effortChecks: [(OverrideWarning.Field, String?, EffortSupport?)] = [
            (.effort, effort, modelInfo.generalEffort),
            (.reasoningEffort, reasoningEffort, modelInfo.reasoningEffort)
        ]
        for (field, value, support) in effortChecks {
            guard let value, let support, support.rejects(value) else { continue }
            let detail = support.knownLevels.map { "reported levels (\($0.joined(separator: ", ")))" }
                ?? "this parameter, which the model does not support"
            out.append(OverrideWarning(field: field, message: "Not among the model's \(detail)."))
        }
        // The thinking switch: warn when an explicit request provably won't reach the wire,
        // via the SAME resolver the settings UI shows — the warning cannot disagree with
        // emission. `.unknown` on a RECORDED mechanism means "nothing sent, model default";
        // `.unsupported` means there is no switch at all. An unrecorded mechanism warns
        // nothing: the legacy fallbacks honor the switch, and warnings fire only on evidence.
        if reasoningEnabled != nil, let control = modelInfo.reasoningControl {
            switch ReasoningControl.plannedThinkingState(PlannedThinkingState.Inputs(
                control: control, apiType: nil, capabilities: modelInfo.capabilities,
                reasoningEnabled: reasoningEnabled, thinkingBudget: thinkingBudget,
                reasoningEffort: reasoningEffort, reasoningEffortSupport: modelInfo.reasoningEffort)) {
            case .unknown(let detail):
                out.append(OverrideWarning(field: .reasoningEnabled, message: detail + "."))
            case .unsupported:
                out.append(OverrideWarning(field: .reasoningEnabled,
                    message: "The model exposes no reasoning control; nothing will be sent."))
            case .on, .off:
                break
            }
        }
        // Budget against the MEASURED range; emission clamps, so out-of-range means "not what
        // you asked for" rather than an error — still worth saying.
        // `else if`: an inverted measured range (max 0 recorded when even the minimum was
        // rejected, min from the separate floor probe) would otherwise fire both, and
        // OverrideWarning's id is its field — two warnings, one Identifiable id. The ceiling
        // message wins: a 0 ceiling means no budget is usable at all, subsuming the floor.
        if let v = thinkingBudget, v > 0 {
            if let known = modelInfo.maxThinkingBudgetTokens, v > known {
                out.append(OverrideWarning(field: .thinkingBudget,
                    message: "Above the measured maximum thinking budget (\(known.formatted()) tokens); the send is clamped."))
            } else if let floor = modelInfo.minThinkingBudgetTokens, v < floor {
                out.append(OverrideWarning(field: .thinkingBudget,
                    message: "Below the measured minimum thinking budget (\(floor.formatted()) tokens); the send is floored."))
            }
        }
        return out
    }
}

/// A non-blocking advisory that one overridden field is outside the model's known/expected range.
public struct OverrideWarning: Sendable, Equatable, Identifiable {
    public enum Field: String, Sendable, CaseIterable {
        case temperature, maxOutputTokens, maxContextTokens, effort, reasoningEffort, reasoningEnabled,
             thinkingBudget
    }
    public let field: Field
    public let message: String
    public var id: String { field.rawValue }

    public init(field: Field, message: String) {
        self.field = field
        self.message = message
    }
}
