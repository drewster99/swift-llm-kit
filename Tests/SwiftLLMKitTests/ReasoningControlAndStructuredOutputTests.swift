import Testing
import Foundation
@testable import SwiftLLMKit

/// The knobs added alongside the effort split: reasoning on/off by MECHANISM, structured output,
/// and `strict` tool definitions. Every one of them is an HTTP 400 when sent to a model that
/// doesn't take it, so all of them fail CLOSED — emitted only on a KNOWN-true capability.
@Suite("Reasoning control, structured output, strict tools")
struct ReasoningControlEmissionTests {

    private func provider(
        control: ReasoningControl? = nil,
        capabilities: ModelCapabilities = ModelCapabilities(),
        apiType: ProviderAPIType = .openAICompatible,
        thinkingBudget: Int? = nil
    ) throws -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m",
                                              thinkingBudget: thinkingBudget),
            provider: ModelProvider(id: "p", name: "p", apiType: apiType,
                                    endpoint: try #require(URL(string: "https://x.test/v1"))),
            readAPIKey: { "" },
            reasoningControl: control,
            modelCapabilities: capabilities
        )
    }

    // MARK: Reasoning on/off

    @Test("thinkingBlock: enabling is emitted only when the model can be switched ON")
    func thinkingBlockEnable() throws {
        let allowed = try provider(control: .thinkingBlock,
                                   capabilities: ModelCapabilities([.reasoningEnableable]))
        let body = allowed.buildRequestBody(messages: [.user("hi")], tools: [],
                                            overrides: LLMCallOverrides(reasoningEnabled: true))
        #expect((body["thinking"] as? [String: Any])?["type"] as? String == "enabled")

        // Kimi documents models that can be switched on but NOT off — so the two directions are
        // gated independently, and an unknown one stays silent rather than guessing.
        let unknown = try provider(control: .thinkingBlock)
        let silent = unknown.buildRequestBody(messages: [.user("hi")], tools: [],
                                              overrides: LLMCallOverrides(reasoningEnabled: true))
        #expect(silent["thinking"] == nil)
    }

    @Test("thinkingBlock: disabling requires the DISABLE capability, not the enable one")
    func thinkingBlockDisableIsSeparate() throws {
        let onlyEnableable = try provider(control: .thinkingBlock,
                                          capabilities: ModelCapabilities([.reasoningEnableable]))
        let body = onlyEnableable.buildRequestBody(messages: [.user("hi")], tools: [],
                                                   overrides: LLMCallOverrides(reasoningEnabled: false))
        #expect(body["thinking"] == nil, "enable-able must not imply disable-able")

        let disableable = try provider(control: .thinkingBlock,
                                       capabilities: ModelCapabilities([.reasoningDisableable]))
        let ok = disableable.buildRequestBody(messages: [.user("hi")], tools: [],
                                              overrides: LLMCallOverrides(reasoningEnabled: false))
        #expect((ok["thinking"] as? [String: Any])?["type"] as? String == "disabled")
    }

    @Test("thinkingBlock: keep and budget ride the same object, each on its own capability")
    func thinkingBlockKeepAndBudget() throws {
        let caps = ModelCapabilities([.reasoningEnableable, .thinkingKeepAll, .thinkingBudgetTokens])
        let p = try provider(control: .thinkingBlock, capabilities: caps)
        let body = p.buildRequestBody(messages: [.user("hi")], tools: [],
                                      overrides: LLMCallOverrides(reasoningEnabled: true,
                                                                  thinkingBudgetTokens: 2048,
                                                                  keepThinking: true))
        let thinking = try #require(body["thinking"] as? [String: Any])
        #expect(thinking["keep"] as? String == "all")
        #expect(thinking["budget_tokens"] as? Int == 2048)

        // Without the keep capability the rest still goes; only the unsupported key is dropped.
        let noKeep = try provider(control: .thinkingBlock,
                                  capabilities: ModelCapabilities([.reasoningEnableable]))
        let partial = noKeep.buildRequestBody(messages: [.user("hi")], tools: [],
                                              overrides: LLMCallOverrides(reasoningEnabled: true, keepThinking: true))
        #expect((partial["thinking"] as? [String: Any])?["keep"] == nil)
        #expect((partial["thinking"] as? [String: Any])?["type"] as? String == "enabled")
    }

    /// An unknown mechanism must not silently disable reasoning on every model whose control has
    /// not been recorded yet — Alibaba's legacy apiType behaviour is preserved.
    @Test("An unknown control keeps the legacy apiType behaviour for Alibaba")
    func unknownControlKeepsLegacyAlibaba() throws {
        let p = try provider(control: nil, apiType: .alibabaCloud, thinkingBudget: 4096)
        let body = p.buildRequestBody(messages: [.user("hi")], tools: [])
        #expect(body["enable_thinking"] as? Bool == true)
        #expect(body["thinking_budget"] as? Int == 4096)
    }

    @Test("reasoningEffortOnly and unsupported emit no on/off object at all")
    func noOnOffMechanisms() throws {
        for control in [ReasoningControl.reasoningEffortOnly, .unsupported] {
            let p = try provider(control: control,
                                 capabilities: ModelCapabilities([.reasoningEnableable]))
            let body = p.buildRequestBody(messages: [.user("hi")], tools: [],
                                          overrides: LLMCallOverrides(reasoningEnabled: true))
            #expect(body["thinking"] == nil, "\(control) has no on/off knob")
            #expect(body["enable_thinking"] == nil)
        }
    }

    // MARK: Structured output

    @Test("response_format is emitted only for the mode the model actually supports")
    func structuredOutputGating() throws {
        let objectOnly = try provider(capabilities: ModelCapabilities([.structuredOutputJSONObject]))
        let ok = objectOnly.buildRequestBody(messages: [.user("hi")], tools: [],
                                             overrides: LLMCallOverrides(responseFormat: .jsonObject))
        #expect((ok["response_format"] as? [String: Any])?["type"] as? String == "json_object")

        // The two modes are separate capabilities because a model may take one and reject the
        // other — DeepSeek documents json_object with json_schema unconfirmed.
        let schema = objectOnly.buildRequestBody(
            messages: [.user("hi")], tools: [],
            overrides: LLMCallOverrides(responseFormat: .jsonSchema(name: "s", schema: [:])))
        #expect(schema["response_format"] == nil, "json_object support must not imply json_schema")

        let unknown = try provider()
        let none = unknown.buildRequestBody(messages: [.user("hi")], tools: [],
                                            overrides: LLMCallOverrides(responseFormat: .jsonObject))
        #expect(none["response_format"] == nil, "fails closed on unknown")
    }

    @Test("json_schema carries name, strict and the schema at the documented path")
    func schemaWireShape() throws {
        let p = try provider(capabilities: ModelCapabilities([.responseSchema]))
        let body = p.buildRequestBody(
            messages: [.user("hi")], tools: [],
            overrides: LLMCallOverrides(responseFormat: .jsonSchema(
                name: "answer", schema: ["type": AnyCodable.string("object")], strict: true)))
        let format = try #require(body["response_format"] as? [String: Any])
        #expect(format["type"] as? String == "json_schema")
        let schema = try #require(format["json_schema"] as? [String: Any])
        #expect(schema["name"] as? String == "answer")
        #expect(schema["strict"] as? Bool == true)
        #expect((schema["schema"] as? [String: Any])?["type"] as? String == "object")
    }

    // MARK: strict tool definitions

    @Test("strict rides the function definition only when the model is known to take it")
    func strictToolGating() throws {
        let tool = LLMToolDefinition(name: "t", description: "d", parameters: [:], strict: true)

        let supported = try provider(capabilities: ModelCapabilities([.strictToolDefinitions]))
        let body = supported.buildRequestBody(messages: [.user("hi")], tools: [tool])
        let fn = try #require((body["tools"] as? [[String: Any]])?.first?["function"] as? [String: Any])
        #expect(fn["strict"] as? Bool == true)

        let unknown = try provider()
        let plain = unknown.buildRequestBody(messages: [.user("hi")], tools: [tool])
        let plainFn = try #require((plain["tools"] as? [[String: Any]])?.first?["function"] as? [String: Any])
        #expect(plainFn["strict"] == nil, "omitted on unknown — endpoints vary between ignore and reject")
        #expect(plainFn["name"] as? String == "t", "the tool itself still goes")
    }
}
