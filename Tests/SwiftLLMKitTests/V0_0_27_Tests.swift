import Foundation
import Testing
@testable import SwiftLLMKit

/// Tests for the 0.0.27 release: adaptive thinking + output_config.effort.
///
/// Anthropic moved newer models (Opus 4.7, Opus 4.8) to a new thinking API
/// that REJECTS the legacy `thinking: {type: "enabled", budget_tokens: N}`
/// format with HTTP 400. The new format is `thinking: {type: "adaptive"}`,
/// with depth controlled via a top-level `output_config: {effort: "..."}`
/// field. swift-llm-kit detects required-adaptive models via the new
/// `BehaviorFlags.requiresAdaptiveThinking` flag (auto-set in the bundled
/// metadata registry for known model IDs) and emits the correct wire shape.
///
/// New `ModelConfiguration.thinkingEffort: String?` field gives users
/// effort control. Emitted as top-level `output_config.effort` whenever
/// non-nil — independent of the thinking mode.
@Suite("0.0.27: adaptive thinking + output_config.effort")
struct V0_0_27_Tests {

    private static let dummyKey: @Sendable () -> String = { "" }

    private func anthropic(
        thinkingBudget: Int? = nil,
        thinkingEffort: String? = nil,
        flags: BehaviorFlags = BehaviorFlags()
    ) throws -> AnthropicProvider {
        try AnthropicProvider(
            configuration: ModelConfiguration(
                name: "t",
                providerID: "p",
                modelID: "m",
                thinkingBudget: thinkingBudget,
                thinkingEffort: thinkingEffort
            ),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .anthropic,
                endpoint: #require(URL(string: "https://api.anthropic.com/v1"))
            ),
            readAPIKey: Self.dummyKey,
            behaviorFlags: flags
        )
    }

    // MARK: - Legacy manual thinking (unchanged behavior)

    @Test("legacy: thinkingBudget > 0 with no flag emits thinking.type = enabled + budget_tokens")
    func legacyManualThinkingUnchanged() throws {
        let provider = try anthropic(thinkingBudget: 4096)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["type"] as? String == "enabled")
        #expect(thinking["budget_tokens"] as? Int == 4096)
        #expect(body["output_config"] == nil, "no effort set, so no output_config")
    }

    @Test("legacy: thinkingBudget == 0 with no flag → no thinking field, no temperature override")
    func legacyManualThinkingOffEmitsNoThinking() throws {
        let provider = try anthropic(thinkingBudget: 0)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["thinking"] == nil)
    }

    // MARK: - Adaptive thinking (requires flag)

    @Test("adaptive: flag on + thinkingBudget > 0 emits thinking.type = adaptive, no budget_tokens")
    func adaptiveThinkingEmittedWhenFlagOn() throws {
        let flags = BehaviorFlags(requiresAdaptiveThinking: true)
        let provider = try anthropic(thinkingBudget: 4096, flags: flags)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["type"] as? String == "adaptive")
        #expect(thinking["budget_tokens"] == nil,
                "adaptive thinking must NOT include budget_tokens — Anthropic rejects this combo")
        // temperature gets pegged to 1.0 when thinking is on (legacy and adaptive both).
        #expect(body["temperature"] as? Double == 1.0)
    }

    @Test("adaptive: flag on + thinkingBudget == 0 → no thinking field (user opted out)")
    func adaptiveFlagOnButBudgetZeroOmitsThinking() throws {
        let flags = BehaviorFlags(requiresAdaptiveThinking: true)
        let provider = try anthropic(thinkingBudget: 0, flags: flags)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["thinking"] == nil,
                "thinkingBudget = 0 means thinking OFF even with adaptive flag set")
    }

    @Test("adaptive: flag on + thinkingBudget nil → no thinking field")
    func adaptiveFlagOnBudgetNilOmitsThinking() throws {
        let flags = BehaviorFlags(requiresAdaptiveThinking: true)
        let provider = try anthropic(thinkingBudget: nil, flags: flags)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["thinking"] == nil)
    }

    // MARK: - output_config.effort

    @Test("output_config.effort emitted when thinkingEffort is set")
    func effortEmittedWhenSet() throws {
        let provider = try anthropic(thinkingEffort: "xhigh")
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        let oc = try #require(body["output_config"] as? [String: Any])
        #expect(oc["effort"] as? String == "xhigh")
    }

    @Test("output_config NOT emitted when thinkingEffort is nil")
    func noEffortNoOutputConfig() throws {
        let provider = try anthropic(thinkingEffort: nil)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["output_config"] == nil)
    }

    @Test("output_config.effort + adaptive thinking work together")
    func effortPlusAdaptive() throws {
        let flags = BehaviorFlags(requiresAdaptiveThinking: true)
        let provider = try anthropic(thinkingBudget: 4096, thinkingEffort: "high", flags: flags)
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["type"] as? String == "adaptive")
        let oc = try #require(body["output_config"] as? [String: Any])
        #expect(oc["effort"] as? String == "high")
    }

    @Test("output_config.effort + legacy manual thinking work together (Opus 4.5 case)")
    func effortPlusManualThinking() throws {
        // No adaptive flag → manual thinking. Opus 4.5 supports effort alongside.
        let provider = try anthropic(thinkingBudget: 4096, thinkingEffort: "medium")
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [])
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["type"] as? String == "enabled",
                "no adaptive flag means we keep legacy thinking shape")
        #expect(thinking["budget_tokens"] as? Int == 4096)
        let oc = try #require(body["output_config"] as? [String: Any])
        #expect(oc["effort"] as? String == "medium")
    }

    // MARK: - max_tokens behavior

    @Test("adaptive thinking does NOT enforce max_tokens > budget_tokens constraint")
    func adaptiveSkipsBudgetMaxTokensFloor() throws {
        // Manual thinking forces max_tokens >= budget+1. Adaptive has no
        // budget concept so the override is passed through as-is.
        let flags = BehaviorFlags(requiresAdaptiveThinking: true)
        let provider = try anthropic(thinkingBudget: 4096, flags: flags)
        let body = try provider.buildRequestBody(
            messages: [.user("hi")],
            tools: [],
            maxOutputTokensOverride: 2048
        )
        #expect(body["max_tokens"] as? Int == 2048,
                "adaptive thinking does not have the manual max_tokens > budget_tokens floor")
    }

    @Test("manual thinking still enforces max_tokens > budget_tokens floor")
    func manualEnforcesBudgetMaxTokensFloor() throws {
        let provider = try anthropic(thinkingBudget: 4096)
        let body = try provider.buildRequestBody(
            messages: [.user("hi")],
            tools: [],
            maxOutputTokensOverride: 2048
        )
        // override 2048 < budget+1 (4097) → clamped to 4097
        #expect(body["max_tokens"] as? Int == 4097,
                "manual thinking clamps override to max(override, budget+1)")
    }

    // MARK: - Codable + Override

    @Test("BehaviorFlags Codable round-trips requiresAdaptiveThinking")
    func flagsCodableRoundTrip() throws {
        let flags = BehaviorFlags(requiresAdaptiveThinking: true)
        let data = try JSONEncoder().encode(flags)
        let decoded = try JSONDecoder().decode(BehaviorFlags.self, from: data)
        #expect(decoded.requiresAdaptiveThinking == true)
        // Default (false) must NOT be serialized — compact wire format.
        let defaultFlags = BehaviorFlags()
        let defaultData = try JSONEncoder().encode(defaultFlags)
        let json = String(data: defaultData, encoding: .utf8) ?? ""
        #expect(!json.contains("requiresAdaptiveThinking"),
                "default value must be omitted from JSON")
    }

    @Test("ModelConfiguration Codable round-trips thinkingEffort")
    func configCodableRoundTripsEffort() throws {
        let config = ModelConfiguration(
            name: "t", providerID: "p", modelID: "m",
            thinkingEffort: "xhigh"
        )
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(ModelConfiguration.self, from: data)
        #expect(decoded.thinkingEffort == "xhigh")
    }

    @Test("Legacy ModelConfiguration JSON without thinkingEffort decodes cleanly")
    func legacyConfigDecodes() throws {
        // Pre-0.0.27 saved configs have no thinkingEffort key — must round-trip
        // through `decodeIfPresent` and leave the field nil.
        let legacy = """
        {
          "id": "11111111-1111-1111-1111-111111111111",
          "name": "legacy",
          "providerID": "p",
          "modelID": "m",
          "temperature": 0.7,
          "maxOutputTokens": 4096,
          "maxContextTokens": 128000,
          "thinkingBudget": 4096,
          "extendedCacheTTL": false,
          "streaming": true,
          "isValid": true
        }
        """
        let decoded = try JSONDecoder().decode(ModelConfiguration.self, from: Data(legacy.utf8))
        #expect(decoded.thinkingEffort == nil)
        #expect(decoded.thinkingBudget == 4096)
    }

    @Test("BehaviorFlagsOverride patches requiresAdaptiveThinking")
    func overrideAppliesFlag() throws {
        var flags = BehaviorFlags()
        let patch = BehaviorFlagsOverride(requiresAdaptiveThinking: true)
        patch.apply(to: &flags, forceReplace: false)
        #expect(flags.requiresAdaptiveThinking == true)
    }
}
