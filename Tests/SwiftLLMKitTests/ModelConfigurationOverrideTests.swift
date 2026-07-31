import Testing
import Foundation
@testable import SwiftLLMKit

/// The 2026-07-31 per-(role, model) runtime override: a sparse delta resolved fresh against the
/// latest ``ModelInfo``, permissive (no clamp) with non-blocking warnings. These pin resolve
/// (delta ?? model default), the warning rules (known-bound-only), isEmpty, and Codable sparsity.
@Suite("ModelConfigurationOverride: resolve + warnings")
struct ModelConfigurationOverrideTests {

    private func model(
        maxOutput: Int? = 8000, maxInput: Int? = 200_000,
        temp: Double? = 0.7, maxTemp: Double? = nil, efforts: [String] = []
    ) -> ModelInfo {
        ModelInfo(
            providerID: "p", modelID: "m", displayName: "M",
            maxInputTokens: maxInput, maxOutputTokens: maxOutput,
            maxTemperature: maxTemp,
            samplingDefaults: temp == nil ? nil : SamplingDefaults(temperature: temp)
        ).with { $0.validEffortLevels = efforts }
    }

    // MARK: resolve

    @Test("An empty override inherits every value from the model")
    func emptyInheritsModelDefaults() {
        let resolved = ModelConfigurationOverride().resolved(against: model())
        #expect(resolved.temperature == 0.7)          // samplingDefaults
        #expect(resolved.maxOutputTokens == 8000)      // model ceiling
        #expect(resolved.maxContextTokens == 200_000)  // model window
        #expect(resolved.providerID == "p")
        #expect(resolved.modelID == "m")
    }

    @Test("A set field wins; unset fields still inherit")
    func overlayWinsPerField() {
        let override = ModelConfigurationOverride(temperature: 0.2, maxOutputTokens: 16000)
        let resolved = override.resolved(against: model())
        #expect(resolved.temperature == 0.2)           // overridden
        #expect(resolved.maxOutputTokens == 16000)     // overridden (permissive, even above 8000)
        #expect(resolved.maxContextTokens == 200_000)  // still inherited
    }

    @Test("Fallbacks apply only when the model reports no limit and the field is unset")
    func fallbacksOnlyWhenUnknownAndUnset() {
        let resolved = ModelConfigurationOverride().resolved(against: model(maxOutput: nil, maxInput: nil, temp: nil))
        #expect(resolved.maxOutputTokens == 4096)
        #expect(resolved.maxContextTokens == 128_000)
        #expect(resolved.temperature == nil)           // no default → omit
    }

    // MARK: warnings

    @Test("An override above a KNOWN bound warns; at/below does not")
    func warnsAboveKnownBound() {
        let m = model(maxOutput: 8000)
        #expect(ModelConfigurationOverride(maxOutputTokens: 16000).warnings(against: m).map(\.field) == [.maxOutputTokens])
        #expect(ModelConfigurationOverride(maxOutputTokens: 8000).warnings(against: m).isEmpty)
        #expect(ModelConfigurationOverride(maxOutputTokens: 4000).warnings(against: m).isEmpty)
    }

    @Test("An UNKNOWN bound is never exceeded — no warning")
    func unknownBoundNeverWarns() {
        let m = model(maxOutput: nil, maxTemp: nil)
        #expect(ModelConfigurationOverride(temperature: 5.0, maxOutputTokens: 999_999).warnings(against: m).isEmpty)
    }

    @Test("Temperature above the model's max, and effort not in the list, warn")
    func temperatureAndEffortWarnings() {
        let m = model(maxTemp: 1.0, efforts: ["low", "medium", "high"])
        #expect(ModelConfigurationOverride(temperature: 1.5).warnings(against: m).map(\.field) == [.temperature])
        #expect(ModelConfigurationOverride(thinkingEffort: "ultra").warnings(against: m).map(\.field) == [.thinkingEffort])
        #expect(ModelConfigurationOverride(thinkingEffort: "high").warnings(against: m).isEmpty)
    }

    @Test("Effort warning is suppressed when the model reports no effort levels")
    func noEffortLevelsNoWarning() {
        #expect(ModelConfigurationOverride(thinkingEffort: "high").warnings(against: model(efforts: [])).isEmpty)
    }

    // MARK: isEmpty + Codable

    @Test("isEmpty is true only when nothing is set")
    func isEmptySemantics() {
        #expect(ModelConfigurationOverride().isEmpty)
        #expect(!ModelConfigurationOverride(temperature: 0.5).isEmpty)
        #expect(!ModelConfigurationOverride(streaming: false).isEmpty)
    }

    @Test("Codable is sparse: an empty override encodes to an empty object; round-trips")
    func codableSparse() throws {
        let emptyData = try JSONEncoder().encode(ModelConfigurationOverride())
        let emptyObj = try #require(try JSONSerialization.jsonObject(with: emptyData) as? [String: Any])
        #expect(emptyObj.isEmpty)

        let original = ModelConfigurationOverride(temperature: 0.3, maxOutputTokens: 16000, thinkingEffort: "max")
        let restored = try JSONDecoder().decode(ModelConfigurationOverride.self, from: JSONEncoder().encode(original))
        #expect(restored == original)
    }
}

private extension ModelInfo {
    func with(_ mutate: (inout ModelInfo) -> Void) -> ModelInfo {
        var copy = self
        mutate(&copy)
        return copy
    }
}
