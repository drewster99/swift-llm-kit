import Foundation
import Testing
@testable import SwiftLLMKit

/// Covers the Responses translation both ways, off recorded fixtures — no network, no credential.
///
/// The event names and payload shapes below were captured from live calls to
/// `chatgpt.com/backend-api/codex/responses` on 2026-09-16, not invented.
@Suite("Codex Responses translation")
struct CodexResponsesTests {

    // MARK: Outbound — messages → input

    @Test("System and developer messages fold into instructions and leave the input")
    func systemFoldsIntoInstructions() {
        let messages: [LLMMessage] = [
            LLMMessage(role: .system, content: .text("You are Agent Brown.")),
            LLMMessage(role: .developer, content: .text("Prefer terse answers.")),
            LLMMessage(role: .user, content: .text("hi"))
        ]
        #expect(CodexResponsesProvider.buildInstructions(messages)
                == "You are Agent Brown.\n\nPrefer terse answers.")
        let input = CodexResponsesProvider.buildInput(messages)
        #expect(input.count == 1, "system/developer must not also appear as input items")
        #expect(input[0]["role"] as? String == "user")
    }

    @Test("No identity prefix is injected — the OAuth gate does not require one")
    func noIdentityPrefix() {
        let instructions = CodexResponsesProvider.buildInstructions(
            [LLMMessage(role: .system, content: .text("You are Agent Smith."))])
        // Verified 2026-09-16: a request without the "You are Codex…" string returns 200. Injecting
        // it would put a second, conflicting identity ahead of every role prompt.
        #expect(instructions == "You are Agent Smith.")
        #expect(instructions?.contains("You are Codex") == false)
    }

    @Test("User and assistant text use the right content-part type")
    func contentPartTypes() throws {
        let input = CodexResponsesProvider.buildInput([
            LLMMessage(role: .user, content: .text("ask")),
            LLMMessage(role: .assistant, content: .text("answer"))
        ])
        func partType(_ item: [String: Any]) throws -> String {
            let content = try #require(item["content"] as? [[String: Any]])
            return try #require(content.first?["type"] as? String)
        }
        // Sending `input_text` for an assistant turn (or vice versa) is a 400.
        #expect(try partType(input[0]) == "input_text")
        #expect(try partType(input[1]) == "output_text")
    }

    @Test("Tool calls and their results become top-level items correlated by call_id")
    func toolTrafficRoundTrips() throws {
        let call = LLMToolCall(id: "call_abc", name: "get_weather", arguments: #"{"city":"Paris"}"#)
        let input = CodexResponsesProvider.buildInput([
            LLMMessage(role: .user, content: .text("weather?")),
            LLMMessage(role: .assistant, content: .toolCalls([call])),
            LLMMessage(role: .tool, content: .toolResult(toolCallID: "call_abc", content: "17C"))
        ])
        #expect(input.count == 3)
        #expect(input[1]["type"] as? String == "function_call")
        #expect(input[1]["call_id"] as? String == "call_abc")
        #expect(input[1]["name"] as? String == "get_weather")
        // Arguments travel as a JSON STRING, not a nested object.
        #expect(input[1]["arguments"] as? String == #"{"city":"Paris"}"#)
        #expect(input[2]["type"] as? String == "function_call_output")
        #expect(input[2]["call_id"] as? String == "call_abc", "the result must quote the call it answers")
        #expect(input[2]["output"] as? String == "17C")
    }

    @Test("A mixed turn emits the prose before the calls it accompanied")
    func mixedTurnOrder() {
        let input = CodexResponsesProvider.buildInput([
            LLMMessage(role: .assistant, content: .mixed(
                text: "Looking that up.",
                toolCalls: [LLMToolCall(id: "c1", name: "search", arguments: "{}")]))
        ])
        #expect(input.count == 2)
        #expect(input[0]["type"] as? String == "message")
        #expect(input[1]["type"] as? String == "function_call")
    }

    @Test("Tools are flat, not nested under a function key")
    func toolsAreFlat() throws {
        let encoded = CodexResponsesProvider.encodeTools([
            LLMToolDefinition(name: "bash", description: "run a command",
                              parameters: ["type": .string("object")])
        ])
        let tool = try #require(encoded.first)
        // chat/completions nests these under "function"; Responses does not.
        #expect(tool["type"] as? String == "function")
        #expect(tool["name"] as? String == "bash")
        #expect(tool["function"] == nil)
    }

    @Test("Tool choice maps to the Responses vocabulary")
    func toolChoiceMapping() {
        #expect(CodexResponsesProvider.encodeToolChoice(.auto) as? String == "auto")
        #expect(CodexResponsesProvider.encodeToolChoice(.required) as? String == "required")
        #expect(CodexResponsesProvider.encodeToolChoice(.textOnly) as? String == "none")
        let specific = CodexResponsesProvider.encodeToolChoice(.specific(name: "bash")) as? [String: String]
        #expect(specific?["name"] == "bash")
    }

    @Test("The body is stateless and omits temperature")
    func bodyShape() {
        let body = CodexResponsesProvider.buildRequestBody(
            model: "gpt-5.5",
            messages: [LLMMessage(role: .user, content: .text("hi"))],
            tools: [],
            overrides: LLMCallOverrides())
        #expect(body["store"] as? Bool == false, "the endpoint keeps nothing between turns")
        #expect(body["stream"] as? Bool == true)
        // Deliberately absent: the endpoint 400s on `max_output_tokens`. See
        // `neverSendsAnOutputCap` for the evidence.
        #expect(body["max_output_tokens"] == nil)
        // These models reject temperature outright.
        #expect(body["temperature"] == nil)
        // No tools sent → no tool_choice, which some endpoints reject on its own.
        #expect(body["tool_choice"] == nil)
    }

    @Test("Reasoning effort is sent with a summary so deltas actually stream")
    func reasoningEffort() throws {
        var overrides = LLMCallOverrides()
        overrides.reasoningEffort = "xhigh"
        let body = CodexResponsesProvider.buildRequestBody(
            model: "gpt-5.5", messages: [], tools: [], overrides: overrides)
        let reasoning = try #require(body["reasoning"] as? [String: String])
        #expect(reasoning["effort"] == "xhigh")
        // Without `summary`, a reasoning turn streams nothing until the answer lands.
        #expect(reasoning["summary"] == "auto")
    }

    @Test("No output-token cap is ever sent — the endpoint rejects the field")
    func neverSendsAnOutputCap() {
        func body(_ overrides: LLMCallOverrides) -> [String: Any] {
            CodexResponsesProvider.buildRequestBody(
                model: "gpt-5.5", messages: [], tools: [], overrides: overrides)
        }
        // 400 {"detail":"Unsupported parameter: max_output_tokens"} — verified live 2026-09-17.
        // The builder takes no cap argument at all, so the only way one could reach the wire is
        // someone re-adding the field inside it. Each expectation below would catch that.
        var overridden = LLMCallOverrides()
        overridden.maxOutputTokens = 512
        #expect(body(LLMCallOverrides())["max_output_tokens"] == nil)
        #expect(body(overridden)["max_output_tokens"] == nil, "a per-call override may not leak a cap")
        // Keyed on the PREFIX, so a rename (`max_output_token_count`, `maxOutputTokens`) is caught
        // too — an exact-key check would go quietly green on the misspelling that 400s.
        #expect(body(overridden).keys.filter { $0.contains("max_output") }.isEmpty)
        #expect(body(overridden).keys.filter { $0.lowercased().contains("maxoutput") }.isEmpty)
        // The rest of the body is unaffected — this is one omitted key, not a disabled builder.
        #expect(body(overridden)["model"] as? String == "gpt-5.5")
        #expect(body(overridden)["input"] != nil)
    }

    // MARK: Inbound — SSE → LLMResponse

    /// A text turn, as captured. Deltas arrive split; the parser must rejoin them in order.
    static let textStream = """
        data: {"type":"response.created"}
        data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}
        data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"He"}
        data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"llo"}
        data: {"type":"response.output_text.done","item_id":"msg_1"}
        data: {"type":"response.completed","response":{"usage":{"input_tokens":46,"output_tokens":5,\
        "input_tokens_details":{"cached_tokens":12,"cache_write_tokens":3},\
        "output_tokens_details":{"reasoning_tokens":7}}}}
        """

    @Test("A text stream reassembles its deltas, usage and finish reason")
    func parsesTextStream() throws {
        let response = try CodexResponsesProvider.parseStream(Self.textStream)
        #expect(response.text == "Hello")
        #expect(response.toolCalls.isEmpty)
        #expect(response.finishReason == "completed")
        let usage = try #require(response.usage)
        #expect(usage.inputTokens == 46)
        #expect(usage.outputTokens == 5)
        // The cache and reasoning fields are what `CostBoard.costOf` reads; dropping them would
        // silently misprice every cached turn.
        #expect(usage.cacheReadTokens == 12)
        #expect(usage.cacheWriteTokens == 3)
        #expect(usage.reasoningTokens == 7)
    }

    @Test("A tool call is assembled from its added event and argument deltas")
    func parsesToolCall() throws {
        // Carries BOTH `item_id` and `output_index` on the deltas, exactly as a live stream does.
        // An earlier fixture supplied only `output_index`, which hid a real defect: this side keyed
        // by `item.id` and that side by `item_id`, so the two never joined and the call arrived
        // with empty arguments. Fixtures that are sparser than the wire prove less than they look.
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,"sequence_number":1,\
            "item":{"type":"function_call","id":"fc_1","call_id":"call_xyz","name":"get_weather",\
            "arguments":"","status":"in_progress"}}
            data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,\
            "delta":"{\\"city\\":"}
            data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,\
            "delta":"\\"Paris\\"}"}
            data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":2}}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        let call = try #require(response.toolCalls.first)
        // `call_id` — NOT the item's `id` — is what a later function_call_output must quote.
        #expect(call.id == "call_xyz")
        #expect(call.name == "get_weather")
        #expect(call.arguments == #"{"city":"Paris"}"#)
    }

    @Test("The done event wins over accumulated deltas")
    func doneEventWinsOverDeltas() throws {
        // A dropped delta would otherwise leave invalid JSON in `arguments`.
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,\
            "item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"f"}}
            data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\\"a\\":"}
            data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{\\"a\\":1}"}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        #expect(response.toolCalls.first?.arguments == #"{"a":1}"#)
    }

    @Test("A call with no arguments yields {} rather than an empty string")
    func emptyArgumentsBecomeEmptyObject() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,\
            "item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"now"}}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        // Every consumer parses `arguments` as JSON; "" is not valid JSON.
        #expect(response.toolCalls.first?.arguments == "{}")
    }

    @Test("Reasoning deltas land in reasoning, not in the answer")
    func reasoningIsSeparate() throws {
        let sse = """
            data: {"type":"response.reasoning_text.delta","delta":"thinking…"}
            data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}
            data: {"type":"response.output_text.delta","item_id":"m","delta":"answer"}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        #expect(response.text == "answer")
        #expect(response.reasoning == "thinking…")
    }

    @Test("Truncation reports verbatim and is recognised as an output-token limit")
    func truncationIsRecognised() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}
            data: {"type":"response.output_text.delta","item_id":"m","delta":"partial"}
            data: {"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        // Stored verbatim per the finishReason convention; the vocabulary table lives on LLMResponse.
        #expect(response.finishReason == "max_output_tokens")
        #expect(response.hitOutputTokenLimit, """
            the capability prober needs this to tell a truncated generation from a declined tool call
            """)
    }

    @Test("A failed stream throws rather than returning a half answer")
    func failedStreamThrows() {
        let sse = """
            data: {"type":"response.output_text.delta","item_id":"m","delta":"partial"}
            data: {"type":"response.failed","response":{"error":{"message":"server on fire"}}}
            """
        #expect(throws: LLMProviderError.self) {
            _ = try CodexResponsesProvider.parseStream(sse)
        }
    }

    @Test("A failed stream carries the server's error code, and a cyber_policy code is a typed refusal")
    func failedStreamCarriesCode() {
        let sse = """
            data: {"type":"response.failed","response":{"status":"failed","error":{"code":"cyber_policy","message":"This content was flagged for possible cybersecurity risk."}}}
            """
        do {
            _ = try CodexResponsesProvider.parseStream(sse)
            Issue.record("expected a throw")
        } catch let error as LLMProviderError {
            guard case .responseFailed(let code, let message) = error else {
                Issue.record("expected responseFailed, got \(error)")
                return
            }
            #expect(code == "cyber_policy")
            #expect(message.hasPrefix("This content was flagged"))
            #expect(error.contentPolicyRefusal == .cyberPolicy)
        } catch {
            Issue.record("unexpected error type \(error)")
        }
    }

    @Test("A failed stream with an unrecognised code is not a content-policy refusal")
    func failedStreamUnknownCodeIsNotRefusal() {
        let sse = """
            data: {"type":"response.failed","response":{"error":{"code":"server_error","message":"on fire"}}}
            """
        do {
            _ = try CodexResponsesProvider.parseStream(sse)
            Issue.record("expected a throw")
        } catch let error as LLMProviderError {
            #expect(error.contentPolicyRefusal == nil)
        } catch {
            Issue.record("unexpected error type \(error)")
        }
    }

    @Test("Unknown events and [DONE] are ignored, not fatal")
    func unknownEventsIgnored() throws {
        let sse = """
            data: {"type":"response.some_future_event","payload":{"x":1}}
            data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}
            data: {"type":"response.output_text.delta","item_id":"m","delta":"ok"}
            data: {"type":"response.completed","response":{}}
            data: [DONE]
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        #expect(response.text == "ok")
    }

    @Test("An empty answer is nil rather than an empty string")
    func emptyTextIsNil() throws {
        let response = try CodexResponsesProvider.parseStream(
            #"data: {"type":"response.completed","response":{}}"#)
        #expect(response.text == nil)
        #expect(response.reasoning == nil)
    }

    // MARK: Models listing

    @Test("The models listing decodes slugs, context, modalities, effort presence and zero pricing")
    func decodesModelsListing() throws {
        let json = """
            {"models":[
              {"slug":"gpt-5.5","display_name":"GPT-5.5","description":"d","context_window":272000,
               "max_context_window":272000,"input_modalities":["text","image"],"visibility":"list",
               "supported_reasoning_levels":[{"effort":"low"},{"effort":"high"},{"effort":"xhigh"}]},
              {"slug":"gpt-reserve","display_name":"GPT-Reserve","context_window":272000,
               "input_modalities":["text"],"visibility":"hide","supported_reasoning_levels":[]}
            ]}
            """
        let decoded = try ModelFetchService().decodeModelFactsForTesting(
            from: Data(json.utf8), apiType: .codexChatGPT)
        #expect(decoded.count == 2)

        let first = try #require(decoded.first { $0.modelID == "gpt-5.5" })
        #expect(first.facts.displayName == "GPT-5.5")
        #expect(first.facts.maxInputTokens == 272000)
        #expect(first.facts.capabilities.vision == true, "stated by input_modalities, not inferred")
        // The listing's levels are the CLI's menu, not the API's accepted set (it omits `none`
        // where accepted and declares `ultra` where refused). The field is left UNSET — any value
        // here, even "levels unknown", blocks the probe's ladder from gap-filling the merge.
        #expect(first.facts.reasoningEffort == nil)
        #expect(first.facts.hidden == false)
        #expect(first.facts.isFree == true)
        // The endpoint rejects `temperature` for every model; stated as a vendor fact so the
        // provider never sends it in production (a probe strips the flag and measures anyway).
        #expect(first.facts.behaviorFlags.mustNeverSendTemperatureParam == true)
        // Present-but-zero, never nil: `CostBoard.costOf` returns 0 for a FAILED lookup too, so
        // absence would make "free" indistinguishable from "we don't know".
        let pricing = try #require(first.facts.pricing)
        #expect(pricing.base.input == 0)
        #expect(pricing.base.output == 0)

        let hidden = try #require(decoded.first { $0.modelID == "gpt-reserve" })
        #expect(hidden.facts.hidden == true)
        #expect(hidden.facts.capabilities.vision == false)
        #expect(hidden.facts.reasoningEffort == nil, "the listing never states a ladder")
    }
}

/// The link between "a model is assigned to this provider" and "the Codex client is what runs it".
///
/// Everything else in this file tests translation in isolation; if the factory hands back an
/// `OpenAICompatibleProvider` for this apiType, all of it is dead code and every call 401s on an
/// endpoint the token cannot reach.
@Suite("Codex provider factory")
struct CodexProviderFactoryTests {

    @Test("The codexChatGPT apiType constructs the Codex client, not the OpenAI-compatible one")
    @MainActor
    func factoryReturnsCodexProvider() throws {
        // A unique app identifier so this gets its own storage directory rather than the
        // developer's real one.
        let kit = LLMKitManager(
            appIdentifier: "test.codex.factory.\(UUID().uuidString)",
            keychainServicePrefix: "test.codex.factory")
        let provider = ModelProvider(
            id: "builtin.codex-chatgpt",
            name: "ChatGPT Subscription (Codex)",
            apiType: .codexChatGPT,
            endpoint: try #require(URL(string: "https://chatgpt.com/backend-api/codex")))
        let config = ModelConfiguration(
            name: "t", providerID: provider.id, modelID: "gpt-5.5")

        let built = kit.makeProvider(configuration: config, provider: provider)
        #expect(built is CodexResponsesProvider, "got \(type(of: built))")
    }

    @Test("Every other apiType still gets the provider it had")
    @MainActor
    func otherAPITypesUnchanged() throws {
        let kit = LLMKitManager(
            appIdentifier: "test.codex.factory.\(UUID().uuidString)",
            keychainServicePrefix: "test.codex.factory")
        func built(_ apiType: ProviderAPIType, _ endpoint: String) throws -> any LLMProvider {
            let provider = ModelProvider(id: "p", name: "p", apiType: apiType,
                                        endpoint: try #require(URL(string: endpoint)))
            return kit.makeProvider(
                configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
                provider: provider)
        }
        // Adding a case to the factory's switch must not have captured anyone else's traffic.
        #expect(try built(.anthropic, "https://api.anthropic.com") is AnthropicProvider)
        #expect(try built(.openAICompatible, "https://api.openai.com/v1") is OpenAICompatibleProvider)
        #expect(try built(.ollama, "http://localhost:11434") is OllamaProvider)
    }
}

