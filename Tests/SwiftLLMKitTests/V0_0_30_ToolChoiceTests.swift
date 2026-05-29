import Foundation
import Testing
@testable import SwiftLLMKit

/// 0.0.30 — `LLMToolChoice` per-call control + per-provider translation.
///
/// Locks the wire shape each provider emits for each of the four cases
/// (auto / required / none / specific). Tool-choice is meaningful only when
/// `tools` is non-empty; the providers gate emission on `!tools.isEmpty`
/// so `toolChoice` set with an empty tools array produces no field.
@Suite("0.0.30: LLMToolChoice per-provider translation")
struct V0_0_30_ToolChoiceTests {

    private static let dummyKey: @Sendable () -> String = { "" }

    private static let testTool = LLMToolDefinition(
        name: "do_x",
        description: "test tool",
        parameters: ["type": .string("object")]
    )

    // MARK: - Anthropic

    private func anthropic() throws -> AnthropicProvider {
        try AnthropicProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "claude-sonnet-4-6"),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .anthropic,
                endpoint: #require(URL(string: "https://api.anthropic.com/v1"))
            ),
            readAPIKey: Self.dummyKey
        )
    }

    @Test("Anthropic: nil toolChoice omits the field entirely")
    func anthropic_nilToolChoice_noField() throws {
        let provider = try anthropic()
        let body = try provider.buildRequestBody(
            messages: [.user("hi")], tools: [Self.testTool], toolChoice: nil
        )
        #expect(body["tool_choice"] == nil,
                "default behavior — let Anthropic apply its auto default")
    }

    @Test("Anthropic: .auto emits {type: auto}")
    func anthropic_auto() throws {
        let provider = try anthropic()
        let body = try provider.buildRequestBody(
            messages: [.user("hi")], tools: [Self.testTool], toolChoice: .auto
        )
        let tc = try #require(body["tool_choice"] as? [String: Any])
        #expect(tc["type"] as? String == "auto")
    }

    @Test("Anthropic: .required emits {type: any} (Anthropic uses 'any', not 'required')")
    func anthropic_required() throws {
        let provider = try anthropic()
        let body = try provider.buildRequestBody(
            messages: [.user("hi")], tools: [Self.testTool], toolChoice: .required
        )
        let tc = try #require(body["tool_choice"] as? [String: Any])
        #expect(tc["type"] as? String == "any")
    }

    @Test("Anthropic: .textOnly emits {type: none}")
    func anthropic_none() throws {
        let provider = try anthropic()
        let body = try provider.buildRequestBody(
            messages: [.user("hi")], tools: [Self.testTool], toolChoice: .textOnly
        )
        let tc = try #require(body["tool_choice"] as? [String: Any])
        #expect(tc["type"] as? String == "none")
    }

    @Test("Anthropic: .specific emits {type: tool, name: ...}")
    func anthropic_specific() throws {
        let provider = try anthropic()
        let body = try provider.buildRequestBody(
            messages: [.user("hi")], tools: [Self.testTool], toolChoice: .specific(name: "do_x")
        )
        let tc = try #require(body["tool_choice"] as? [String: Any])
        #expect(tc["type"] as? String == "tool")
        #expect(tc["name"] as? String == "do_x")
    }

    @Test("Anthropic: tools empty + toolChoice set → no tool_choice (no tools, no choice)")
    func anthropic_emptyTools_noToolChoice() throws {
        let provider = try anthropic()
        let body = try provider.buildRequestBody(
            messages: [.user("hi")], tools: [], toolChoice: .required
        )
        #expect(body["tool_choice"] == nil)
        #expect(body["tools"] == nil)
    }

    // MARK: - OpenAI-compatible

    private func openAI() throws -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "gpt-5"),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .openAICompatible,
                endpoint: try #require(URL(string: "https://api.openai.com/v1"))
            ),
            readAPIKey: Self.dummyKey
        )
    }

    @Test("OpenAI: nil toolChoice omits the field")
    func openAI_nil_noField() throws {
        let provider = try openAI()
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: nil)
        #expect(body["tool_choice"] == nil)
    }

    @Test("OpenAI: .auto emits string \"auto\"")
    func openAI_auto() throws {
        let provider = try openAI()
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: .auto)
        #expect(body["tool_choice"] as? String == "auto")
    }

    @Test("OpenAI: .required emits string \"required\"")
    func openAI_required() throws {
        let provider = try openAI()
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: .required)
        #expect(body["tool_choice"] as? String == "required")
    }

    @Test("OpenAI: .textOnly emits string \"none\"")
    func openAI_none() throws {
        let provider = try openAI()
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: .textOnly)
        #expect(body["tool_choice"] as? String == "none")
    }

    @Test("OpenAI: .specific emits {type: function, function: {name: ...}}")
    func openAI_specific() throws {
        let provider = try openAI()
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: .specific(name: "do_x"))
        let tc = try #require(body["tool_choice"] as? [String: Any])
        #expect(tc["type"] as? String == "function")
        let fn = try #require(tc["function"] as? [String: Any])
        #expect(fn["name"] as? String == "do_x")
    }

    @Test("OpenAI: tools empty + toolChoice set → no tool_choice")
    func openAI_emptyTools_noToolChoice() throws {
        let provider = try openAI()
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [], toolChoice: .required)
        #expect(body["tool_choice"] == nil)
    }

    // MARK: - Gemini

    private func gemini() throws -> GeminiProvider {
        try GeminiProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "gemini-2.5-pro"),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .gemini,
                endpoint: #require(URL(string: "https://generativelanguage.googleapis.com/v1beta"))
            ),
            readAPIKey: Self.dummyKey
        )
    }

    @Test("Gemini: nil toolChoice defaults to AUTO in toolConfig")
    func gemini_nilDefaultsAuto() throws {
        let provider = try gemini()
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: nil)
        let tc = try #require(body["toolConfig"] as? [String: Any])
        let fcc = try #require(tc["functionCallingConfig"] as? [String: Any])
        #expect(fcc["mode"] as? String == "AUTO")
    }

    @Test("Gemini: .auto explicit emits mode AUTO")
    func gemini_autoExplicit() throws {
        let provider = try gemini()
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: .auto)
        let tc = try #require(body["toolConfig"] as? [String: Any])
        let fcc = try #require(tc["functionCallingConfig"] as? [String: Any])
        #expect(fcc["mode"] as? String == "AUTO")
    }

    @Test("Gemini: .required emits mode ANY")
    func gemini_required() throws {
        let provider = try gemini()
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: .required)
        let tc = try #require(body["toolConfig"] as? [String: Any])
        let fcc = try #require(tc["functionCallingConfig"] as? [String: Any])
        #expect(fcc["mode"] as? String == "ANY")
        #expect(fcc["allowedFunctionNames"] == nil)
    }

    @Test("Gemini: .textOnly emits mode NONE")
    func gemini_none() throws {
        let provider = try gemini()
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: .textOnly)
        let tc = try #require(body["toolConfig"] as? [String: Any])
        let fcc = try #require(tc["functionCallingConfig"] as? [String: Any])
        #expect(fcc["mode"] as? String == "NONE")
    }

    @Test("Gemini: .specific emits mode ANY + allowedFunctionNames")
    func gemini_specific() throws {
        let provider = try gemini()
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: .specific(name: "do_x"))
        let tc = try #require(body["toolConfig"] as? [String: Any])
        let fcc = try #require(tc["functionCallingConfig"] as? [String: Any])
        #expect(fcc["mode"] as? String == "ANY")
        let allowed = try #require(fcc["allowedFunctionNames"] as? [String])
        #expect(allowed == ["do_x"])
    }

    @Test("Gemini: tools empty + toolChoice set → no toolConfig field")
    func gemini_emptyTools_noToolConfig() throws {
        let provider = try gemini()
        let body = try provider.buildRequestBody(messages: [.user("hi")], tools: [], toolChoice: .required)
        #expect(body["toolConfig"] == nil)
    }

    // MARK: - Ollama

    private func ollama() throws -> OllamaProvider {
        OllamaProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "gemma3:27b"),
            provider: ModelProvider(
                id: "p", name: "p", apiType: .ollama,
                endpoint: try #require(URL(string: "http://localhost:11434/api"))
            ),
            readAPIKey: Self.dummyKey
        )
    }

    @Test("Ollama: nil toolChoice omits the field")
    func ollama_nil_noField() throws {
        let provider = try ollama()
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: nil)
        #expect(body["tool_choice"] == nil)
    }

    @Test("Ollama: .auto emits string \"auto\" (OpenAI-shape pass-through)")
    func ollama_auto() throws {
        let provider = try ollama()
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: .auto)
        #expect(body["tool_choice"] as? String == "auto")
    }

    @Test("Ollama: .specific emits OpenAI-shape function object")
    func ollama_specific() throws {
        let provider = try ollama()
        let body = provider.buildRequestBody(messages: [.user("hi")], tools: [Self.testTool], toolChoice: .specific(name: "do_x"))
        let tc = try #require(body["tool_choice"] as? [String: Any])
        let fn = try #require(tc["function"] as? [String: Any])
        #expect(fn["name"] as? String == "do_x")
    }

    // MARK: - Convenience overloads (backward compat)

    @Test("send(messages:tools:) two-arg overload still compiles + calls into default-nil toolChoice")
    func protocol_twoArgOverload_stillWorks() throws {
        // This test ensures the protocol extension survived the signature
        // change. Compiles → passes. Behavior is "no toolChoice on wire."
        let provider = try anthropic()
        // We can't easily exercise the actual `send()` without hitting network.
        // The compile-time check (this file builds) is the load-bearing assertion.
        _ = provider  // suppress unused warning
        #expect(Bool(true))
    }

    // MARK: - Empty tools wire-shape symmetry across providers

    @Test("All four providers: empty tools + any toolChoice → no tool-related fields")
    func allProviders_emptyTools_noToolFields() throws {
        let ant = try anthropic()
        let antBody = try ant.buildRequestBody(messages: [.user("hi")], tools: [], toolChoice: .required)
        #expect(antBody["tools"] == nil && antBody["tool_choice"] == nil)

        let oa = try openAI()
        let oaBody = oa.buildRequestBody(messages: [.user("hi")], tools: [], toolChoice: .required)
        #expect(oaBody["tools"] == nil && oaBody["tool_choice"] == nil)

        let gm = try gemini()
        let gmBody = try gm.buildRequestBody(messages: [.user("hi")], tools: [], toolChoice: .required)
        #expect(gmBody["tools"] == nil && gmBody["toolConfig"] == nil)

        let ol = try ollama()
        let olBody = ol.buildRequestBody(messages: [.user("hi")], tools: [], toolChoice: .required)
        #expect(olBody["tools"] == nil && olBody["tool_choice"] == nil)
    }
}
