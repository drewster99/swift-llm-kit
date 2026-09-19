import Foundation
import Testing
@testable import SwiftLLMKit

/// The Codex serializer measured against the chat/completions one, feature by feature.
///
/// Written after a capability probe recorded `vision = false` for gpt-6-astra with the evidence
/// "I don't see an image attached": the request had no image in it. `buildInput` built every user
/// turn as a single `input_text` part, so images and documents never reached the wire — and the
/// same builder ignored `extraJSONOverrides`, so every FORCED probe (effort ladder, structured
/// output, tool_choice options) sent a bare request and graded the ordinary success as support.
///
/// Every wire fact asserted here was checked against the live endpoint on 2026-09-19:
/// `input_image` and `input_file` are read; `text.format` is enforced; a `developer` item is
/// honored; `parallel_tool_calls: false` is accepted; `temperature`, `top_p`, `max_output_tokens`
/// and a `{role: system}` input item are each rejected with `Unsupported parameter` /
/// `System messages are not allowed`.
@Suite("Codex serializer parity")
struct CodexSerializerParityTests {

    private static let png = Data([0x89, 0x50, 0x4E, 0x47])
    private static let pdf = Data("%PDF-1.4".utf8)

    private static func config(
        temperature: Double? = nil, reasoningEffort: String? = nil, reasoningEnabled: Bool? = nil,
        extra: [String: AnyCodable]? = nil
    ) -> ModelConfiguration {
        ModelConfiguration(
            name: "t", providerID: BuiltInProviders.ID.codexChatGPT, modelID: "gpt-5.5",
            temperature: temperature, reasoningEnabled: reasoningEnabled,
            reasoningEffort: reasoningEffort, extraJSONOverrides: extra)
    }

    private static func body(
        _ messages: [LLMMessage] = [.user("hi")], tools: [LLMToolDefinition] = [],
        overrides: LLMCallOverrides = LLMCallOverrides(),
        configuration: ModelConfiguration = config(),
        flags: BehaviorFlags = BehaviorFlags(),
        effort: EffortSupport? = nil,
        capabilities: ModelCapabilities = ModelCapabilities()
    ) -> [String: Any] {
        CodexResponsesProvider.buildRequestBody(
            configuration: configuration, messages: messages, tools: tools, overrides: overrides,
            behaviorFlags: flags, reasoningEffortSupport: effort, modelCapabilities: capabilities)
    }

    private static func parts(_ item: [String: Any]) throws -> [[String: Any]] {
        try #require(item["content"] as? [[String: Any]])
    }

    // MARK: Attachments

    @Test("A user turn's images travel as input_image parts with a data URL and the detail hint")
    func imagesBecomeInputImageParts() throws {
        let image = LLMImageContent(data: Self.png, mimeType: "image/png", detail: .low)
        let input = CodexResponsesProvider.buildInput([.user("what is this?", images: [image])])
        let content = try Self.parts(try #require(input.first))
        let imagePart = try #require(content.first { $0["type"] as? String == "input_image" })
        #expect(imagePart["image_url"] as? String
                == "data:image/png;base64,\(Self.png.base64EncodedString())")
        #expect(imagePart["detail"] as? String == "low")
        // The text is still there, as the same part type it always was.
        let textPart = try #require(content.first { $0["type"] as? String == "input_text" })
        #expect(textPart["text"] as? String == "what is this?")
        #expect(content.count == 2)
    }

    @Test("A detail-less image sends no detail key rather than a made-up one")
    func imageWithoutDetail() throws {
        let image = LLMImageContent(data: Self.png, mimeType: "image/png")
        let input = CodexResponsesProvider.buildInput([.user("?", images: [image])])
        let content = try Self.parts(try #require(input.first))
        let imagePart = try #require(content.first { $0["type"] as? String == "input_image" })
        #expect(imagePart["detail"] == nil)
    }

    @Test("Documents travel as input_file parts, with a filename synthesized when none was given")
    func documentsBecomeInputFileParts() throws {
        let named = LLMDocumentContent(data: Self.pdf, mimeType: "application/pdf", filename: "spec.pdf")
        let unnamed = LLMDocumentContent(data: Self.pdf, mimeType: "application/pdf")
        let input = CodexResponsesProvider.buildInput([
            .user("read these", images: [], documents: [named, unnamed])
        ])
        let files = try Self.parts(try #require(input.first)).filter { $0["type"] as? String == "input_file" }
        #expect(files.count == 2)
        #expect(files[0]["filename"] as? String == "spec.pdf")
        #expect(files[0]["file_data"] as? String
                == "data:application/pdf;base64,\(Self.pdf.base64EncodedString())")
        // Required beside `file_data`; omitting it is a 400, so it is never left absent.
        #expect(files[1]["filename"] as? String == "document.pdf")
    }

    @Test("Attachments and text are one message item, and the text part comes last")
    func attachmentsShareTheMessage() throws {
        let image = LLMImageContent(data: Self.png, mimeType: "image/png")
        let doc = LLMDocumentContent(data: Self.pdf, mimeType: "application/pdf", filename: "a.pdf")
        let input = CodexResponsesProvider.buildInput([.user("both", images: [image], documents: [doc])])
        #expect(input.count == 1, "one turn is one item, not one item per attachment")
        let types = try Self.parts(try #require(input.first)).compactMap { $0["type"] as? String }
        #expect(types == ["input_image", "input_file", "input_text"])
    }

    @Test("A plain text turn is byte-for-byte what it was before attachments existed")
    func textOnlyTurnUnchanged() throws {
        let input = CodexResponsesProvider.buildInput([.user("hi")])
        let content = try Self.parts(try #require(input.first))
        #expect(content.count == 1)
        #expect(content[0]["type"] as? String == "input_text")
        #expect(content[0]["text"] as? String == "hi")
    }

    // MARK: Roles

    @Test("A developer turn folds into instructions by default and becomes a developer item when the model reads that role")
    func developerRoleFollowsTheFlag() throws {
        let messages: [LLMMessage] = [.system("base"), .developer("terse"), .user("hi")]

        let folded = CodexResponsesProvider.buildInput(messages)
        #expect(folded.count == 1)
        #expect(CodexResponsesProvider.buildInstructions(messages) == "base\n\nterse")

        var flags = BehaviorFlags()
        flags.supportsDeveloperRole = true
        let emitted = CodexResponsesProvider.buildInput(messages, behaviorFlags: flags)
        #expect(emitted.count == 2)
        #expect(emitted[0]["role"] as? String == "developer")
        #expect(try Self.parts(emitted[0]).first?["type"] as? String == "input_text")
        // Emitted once: it must not ALSO appear in instructions.
        #expect(CodexResponsesProvider.buildInstructions(messages, behaviorFlags: flags) == "base")
    }

    @Test("A trailing system turn is folded even where chat/completions would keep it at the tail")
    func trailingSystemTurnIsFolded() {
        // The endpoint rejects a `{role: system}` input item outright (400 "System messages are
        // not allowed", live 2026-09-19), so unlike the OpenAI-compatible provider the flag cannot
        // buy a trailing item here — instructions is the only place a system turn can go.
        var flags = BehaviorFlags()
        flags.supportsTrailingSystemMessage = true
        let messages: [LLMMessage] = [.system("base"), .user("hi"), .system("steer")]
        let input = CodexResponsesProvider.buildInput(messages, behaviorFlags: flags)
        #expect(input.count == 1)
        #expect(input.allSatisfy { ($0["role"] as? String) != "system" })
        #expect(CodexResponsesProvider.buildInstructions(messages, behaviorFlags: flags) == "base\n\nsteer")
    }

    // MARK: Sampling

    @Test("Temperature is sent when asked for and withheld under mustNeverSendTemperatureParam")
    func temperatureFollowsTheFlag() {
        // Sent — so a probe measures the endpoint's real (400) answer and derives the flag.
        #expect(Self.body(configuration: Self.config(temperature: 0.2))["temperature"] as? Double == 0.2)
        #expect(Self.body(overrides: LLMCallOverrides(temperature: 0))["temperature"] as? Double == 0)
        // The per-call override outranks the configuration.
        #expect(Self.body(overrides: LLMCallOverrides(temperature: 1),
                          configuration: Self.config(temperature: 0.2))["temperature"] as? Double == 1)
        // Nothing asked for → nothing sent.
        #expect(Self.body()["temperature"] == nil)
        // Flagged → withheld regardless of who asked.
        var flags = BehaviorFlags()
        flags.mustNeverSendTemperatureParam = true
        #expect(Self.body(overrides: LLMCallOverrides(temperature: 0),
                          configuration: Self.config(temperature: 0.2), flags: flags)["temperature"] == nil)
    }

    @Test("top_p is a per-call knob; stop sequences and penalties have no Responses field")
    func otherSamplingKnobs() {
        var overrides = LLMCallOverrides()
        overrides.topP = 0.5
        overrides.stopSequences = ["END"]
        overrides.frequencyPenalty = 0.1
        overrides.presencePenalty = 0.2
        let body = Self.body(overrides: overrides)
        #expect(body["top_p"] as? Double == 0.5)
        #expect(body["stop"] == nil)
        #expect(body["frequency_penalty"] == nil)
        #expect(body["presence_penalty"] == nil)
    }

    // MARK: Reasoning

    @Test("Reasoning effort layers override > explicit off > configuration, and only KNOWN unsupported withholds it")
    func reasoningEffortLayering() throws {
        func effort(_ body: [String: Any]) -> String? { (body["reasoning"] as? [String: Any])?["effort"] as? String }

        #expect(effort(Self.body(configuration: Self.config(reasoningEffort: "high"))) == "high")
        #expect(effort(Self.body(overrides: LLMCallOverrides(reasoningEffort: "xhigh"),
                                 configuration: Self.config(reasoningEffort: "high"))) == "xhigh")
        // An explicit off has one wire form on an effort-only model.
        #expect(effort(Self.body(overrides: LLMCallOverrides(reasoningEnabled: false),
                                 configuration: Self.config(reasoningEffort: "high"))) == "none")
        #expect(effort(Self.body(configuration: Self.config(reasoningEffort: "high", reasoningEnabled: false))) == "none")
        // …unless the ladder is known to reject `none`; then the configured depth stands.
        #expect(effort(Self.body(configuration: Self.config(reasoningEffort: "high", reasoningEnabled: false),
                                 effort: .levels(["low", "high"]))) == "high")
        // The override still outranks an explicit off.
        #expect(effort(Self.body(overrides: LLMCallOverrides(reasoningEffort: "low", reasoningEnabled: false))) == "low")
        // Fail-OPEN: an unrecorded ladder does not withhold the field…
        #expect(effort(Self.body(configuration: Self.config(reasoningEffort: "high"), effort: nil)) == "high")
        #expect(effort(Self.body(configuration: Self.config(reasoningEffort: "high"), effort: .supportedLevelsUnknown)) == "high")
        // …only a KNOWN unsupported does.
        #expect(Self.body(configuration: Self.config(reasoningEffort: "high"), effort: .unsupported)["reasoning"] == nil)
        // Nothing asked for → nothing sent.
        #expect(Self.body()["reasoning"] == nil)
        // The summary rides along whenever effort is sent, or reasoning deltas never stream.
        let reasoning = try #require(Self.body(configuration: Self.config(reasoningEffort: "medium"))["reasoning"] as? [String: Any])
        #expect(reasoning["summary"] as? String == "auto")
    }

    @Test("Every request asks for encrypted reasoning so a stateless conversation can carry it forward")
    func includesEncryptedReasoning() {
        #expect(Self.body()["include"] as? [String] == ["reasoning.encrypted_content"])
    }

    @Test("Reasoning items are captured from the stream and replayed ahead of the assistant turn")
    func reasoningItemsRoundTrip() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}
            data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1",\
            "summary":[{"type":"summary_text","text":"thought"}],"encrypted_content":"SEALED"}}
            data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1",\
            "call_id":"call_1","name":"get_code","arguments":""}}
            data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_1",\
            "call_id":"call_1","name":"get_code","arguments":"{}"}}
            data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        let items = try #require(response.continuation?.codexReasoningItems)
        #expect(items == [CodexReasoningItem(id: "rs_1", encryptedContent: "SEALED", summary: ["thought"])])
        // The reasoning item is not a call and must not become one.
        #expect(response.toolCalls.map(\.id) == ["call_1"])

        // Recording the turn the misuse-resistant way carries the items into the next request…
        let input = CodexResponsesProvider.buildInput([
            .user("go"),
            .assistant(from: response),
            .toolResult("K9", callID: "call_1")
        ])
        #expect(input.map { $0["type"] as? String } == ["message", "reasoning", "function_call", "function_call_output"])
        let replayed = input[1]
        #expect(replayed["id"] as? String == "rs_1")
        #expect(replayed["encrypted_content"] as? String == "SEALED")
        let summary = try #require(replayed["summary"] as? [[String: Any]])
        #expect(summary.first?["type"] as? String == "summary_text")
        #expect(summary.first?["text"] as? String == "thought")
    }

    @Test("A reasoning item with no encrypted payload is not carried, and an assistant turn without continuation replays nothing")
    func emptyReasoningIsNotCarried() throws {
        let sse = """
            data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}
            data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"ok"}
            data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        #expect(response.continuation == nil)
        #expect(response.text == "ok")
        let input = CodexResponsesProvider.buildInput([.user("go"), .assistant(from: response)])
        #expect(input.map { $0["type"] as? String } == ["message", "message"])
    }

    // MARK: Structured output

    @Test("Structured output is text.format, flat for a schema, and fails closed on the capability")
    func structuredOutputGating() throws {
        let schema: [String: AnyCodable] = ["type": .string("object")]
        let mode = LLMResponseFormat.jsonSchema(name: "probe", schema: schema, strict: true)
        // Unknown capability → nothing sent (a rejected format is a 400).
        #expect(Self.body(overrides: LLMCallOverrides(responseFormat: mode))["text"] == nil)
        // Known false → nothing sent.
        #expect(Self.body(overrides: LLMCallOverrides(responseFormat: mode),
                          capabilities: ModelCapabilities(states: [.structuredOutputSupportsJSONSchema: false]))["text"] == nil)
        // Known true → the Responses spelling.
        let body = Self.body(overrides: LLMCallOverrides(responseFormat: mode),
                             capabilities: ModelCapabilities(states: [.structuredOutputSupportsJSONSchema: true]))
        let format = try #require((body["text"] as? [String: Any])?["format"] as? [String: Any])
        #expect(format["type"] as? String == "json_schema")
        #expect(format["name"] as? String == "probe")
        #expect(format["strict"] as? Bool == true)
        #expect((format["schema"] as? [String: Any])?["type"] as? String == "object")
        #expect(format["json_schema"] == nil, "chat/completions nests these; Responses does not")
        // The two modes are gated separately.
        let object = Self.body(overrides: LLMCallOverrides(responseFormat: .jsonObject),
                               capabilities: ModelCapabilities(states: [.structuredOutputSupportsJSONObject: true]))
        #expect(((object["text"] as? [String: Any])?["format"] as? [String: Any])?["type"] as? String == "json_object")
    }

    @Test("A probe forcing structured output at Codex forces text.format, not response_format")
    func forcedOverridesPerDialect() throws {
        let codex = LLMResponseFormat.jsonObject.forcedOverrides(for: .codexChatGPT)
        #expect(codex["response_format"] == nil)
        guard case .dictionary(let text) = try #require(codex["text"]),
              case .dictionary(let format) = try #require(text["format"]) else {
            Issue.record("text.format not nested as dictionaries"); return
        }
        #expect(format["type"] == .string("json_object"))
        let chat = LLMResponseFormat.jsonObject.forcedOverrides(for: .openAICompatible)
        #expect(chat["text"] == nil)
        #expect(chat["response_format"] == .dictionary(["type": .string("json_object")]))
    }

    // MARK: Tools

    private static let tool = LLMToolDefinition(name: "bash", description: "run", parameters: ["type": .string("object")])

    @Test("tool_choice is sent only when the caller set one, gated per option, in the flat Responses shape")
    func toolChoiceGating() throws {
        // Not set → not sent; the endpoint's own default applies.
        #expect(Self.body(tools: [Self.tool])["tool_choice"] == nil)
        // Set, nothing known → sent (fail-open, like chat/completions).
        #expect(Self.body(tools: [Self.tool], overrides: LLMCallOverrides(toolChoice: .required))["tool_choice"] as? String == "required")
        // Set, option KNOWN rejected → withheld.
        #expect(Self.body(tools: [Self.tool], overrides: LLMCallOverrides(toolChoice: .required),
                          capabilities: ModelCapabilities(states: [.toolChoiceSupportsValueRequired: false]))["tool_choice"] == nil)
        // A named function is flat — no `function` nesting.
        let named = try #require(Self.body(tools: [Self.tool],
                                            overrides: LLMCallOverrides(toolChoice: .specific(name: "bash")))["tool_choice"] as? [String: Any])
        #expect(named["type"] as? String == "function")
        #expect(named["name"] as? String == "bash")
        #expect(named["function"] == nil)
        // The probe forces the SAME shape the provider ships.
        #expect(LLMToolChoice.specific(name: "bash").wireValue(for: .codexChatGPT)
                == .dictionary(["type": .string("function"), "name": .string("bash")]))
        #expect(LLMToolChoice.specific(name: "bash").wireValue(for: .openAICompatible)
                != LLMToolChoice.specific(name: "bash").wireValue(for: .codexChatGPT))
    }

    @Test("parallel_tool_calls is stated only to disable it, and only when tools are present")
    func parallelToolCallsFlag() {
        var flags = BehaviorFlags()
        flags.disableParallelToolCalls = true
        #expect(Self.body(tools: [Self.tool], flags: flags)["parallel_tool_calls"] as? Bool == false)
        // The endpoint's default is already true; saying so buys nothing.
        #expect(Self.body(tools: [Self.tool])["parallel_tool_calls"] == nil)
        // Without tools the field is meaningless.
        #expect(Self.body(flags: flags)["parallel_tool_calls"] == nil)
    }

    // MARK: Overrides

    @Test("extraJSONOverrides merge last, so a forced probe parameter reaches the wire past every gate")
    func extraOverridesMergeLast() throws {
        let extra: [String: AnyCodable] = [
            "reasoning_effort": .string("xhigh"),
            "reasoning": .dictionary(["effort": .string("low")]),
            "text": .dictionary(["format": .dictionary(["type": .string("json_object")])])
        ]
        let body = Self.body(configuration: Self.config(reasoningEffort: "high", extra: extra))
        // A key the builder never emits arrives verbatim (this is how the ladder probe works).
        #expect(body["reasoning_effort"] as? String == "xhigh")
        // A key the builder DID emit is deep-merged: the forced effort wins, the summary survives.
        let reasoning = try #require(body["reasoning"] as? [String: Any])
        #expect(reasoning["effort"] as? String == "low")
        #expect(reasoning["summary"] as? String == "auto")
        // A gated key is forced straight past its (unknown) capability gate.
        #expect(((body["text"] as? [String: Any])?["format"] as? [String: Any])?["type"] as? String == "json_object")
    }

    // MARK: Usage

    @Test("The raw usage object is preserved, as every other adapter preserves it")
    func rawUsagePreserved() throws {
        let sse = """
            data: {"type":"response.completed","response":{"usage":{"input_tokens":3,"output_tokens":4,\
            "attribution":{"items":{}}}}}
            """
        let usage = try #require(try CodexResponsesProvider.parseStream(sse).usage)
        let raw = try #require(usage.rawUsage)
        #expect(raw.contains("\"attribution\""))
        #expect(raw.contains("\"input_tokens\":3"))
    }

    // MARK: Prepared requests

    @Test("A prepared Codex request carries no output cap")
    @MainActor
    func preparedRequestHasNoCap() throws {
        let kit = LLMKitManager(
            appIdentifier: "test.codex.prepared.\(UUID().uuidString)",
            keychainServicePrefix: "test.codex.prepared")
        let provider = ModelProvider(
            id: BuiltInProviders.ID.codexChatGPT, name: "ChatGPT Subscription (Codex)",
            apiType: .codexChatGPT, endpoint: try #require(URL(string: "https://chatgpt.com/backend-api/codex")))
        let config = ModelConfiguration(name: "p", providerID: provider.id, modelID: "gpt-5.5", maxOutputTokens: 512)
        let prepared = kit.prepareRequest(configuration: config, provider: provider)
        #expect(prepared.baseBody["max_output_tokens"] == nil, "the endpoint 400s on the field")
        #expect(prepared.baseBody["temperature"] == nil)
        #expect(prepared.baseBody["model"] as? String == "gpt-5.5")
    }

    // MARK: Forced probes speak the dialect

    @Test("A forced reasoning-effort level is spelled per dialect, and the disable payload follows it")
    func reasoningEffortDialect() {
        #expect(ReasoningControl.reasoningEffortOverrides(level: "high", for: .openAICompatible)
                == ["reasoning_effort": .string("high")])
        #expect(ReasoningControl.reasoningEffortOverrides(level: "high", for: .codexChatGPT)
                == ["reasoning": .dictionary(["effort": .string("high")])])
        #expect(ReasoningControl.reasoningEffortOnly.reasoningDisableOverrides(for: .codexChatGPT)
                == ["reasoning": .dictionary(["effort": .string("none")])])
        #expect(ReasoningControl.reasoningEffortOnly.reasoningDisableOverrides(for: .openAICompatible)
                == ["reasoning_effort": .string("none")])
        // The other mechanisms are not dialect-sensitive and are unchanged.
        #expect(ReasoningControl.thinkingBlock.reasoningDisableOverrides(for: .codexChatGPT)
                == ["thinking": .dictionary(["type": .string("disabled")])])
        #expect(ReasoningControl.unsupported.reasoningDisableOverrides(for: .codexChatGPT) == nil)
    }

    /// Answers a forced body the way the Codex endpoint does (verified live 2026-09-19): `thinking`
    /// and top-level `reasoning_effort` are "Unsupported parameter"; `reasoning.effort` is taken and
    /// the reply bills reasoning tokens.
    private struct CodexLikeEndpoint: LLMProvider, @unchecked Sendable {
        let forced: [String: AnyCodable]
        let seen: Recorder
        final class Recorder: @unchecked Sendable { var bodies: [[String: AnyCodable]] = [] }
        func send(messages: [LLMMessage], tools: [LLMToolDefinition],
                  overrides: LLMCallOverrides) async throws -> LLMResponse {
            seen.bodies.append(forced)
            for key in forced.keys where key != "reasoning" && key != "tools" {
                throw LLMProviderError.httpError(statusCode: 400, body: #"{"detail":"Unsupported parameter: \#(key)"}"#)
            }
            if case .array(let tools)? = forced["tools"], case .dictionary(let tool)? = tools.first, tool["name"] == nil {
                throw LLMProviderError.httpError(statusCode: 400, body: "Missing required parameter: 'tools[0].name'.")
            }
            let reasons = forced["reasoning"] != nil
            return LLMResponse(text: "ok", usage: TokenUsage(inputTokens: 5, outputTokens: 2,
                                                             reasoningTokens: reasons ? 64 : 0))
        }
    }

    @Test("Mechanism discovery at Codex finds reasoning.effort instead of writing the model off")
    func mechanismDiscoveryAtCodex() async {
        let rec = CodexLikeEndpoint.Recorder()
        let found = await ModelProber.probeReasoningMechanism(
            apiType: .codexChatGPT, makeProviderForcing: { CodexLikeEndpoint(forced: $0, seen: rec) })
        // Before the dialect fix the effort candidate forced `reasoning_effort`, was refused like
        // the others, and the sweep recorded "refused every reasoning mechanism tried".
        #expect(found.control == .reasoningEffortOnly)
        #expect(found.mechanismWasEstablished)
        #expect(found.on.finding.value == true)
        #expect(found.off.finding.value == true)
        #expect(rec.bodies.contains { $0["reasoning"] == .dictionary(["effort": .string("low")]) })
        #expect(rec.bodies.contains { $0["reasoning"] == .dictionary(["effort": .string("none")]) })
        #expect(!rec.bodies.contains { $0["reasoning_effort"] != nil }, "the chat/completions key never reaches this endpoint")
    }

    @Test("The strict-tools probe forces the flat Responses tool shape at Codex")
    func strictProbeShapeAtCodex() async throws {
        let rec = CodexLikeEndpoint.Recorder()
        let finding = await ModelProber.probeStrictToolDefinitions(
            apiType: .codexChatGPT, makeProviderForcing: { CodexLikeEndpoint(forced: $0, seen: rec) })
        #expect(finding?.value == true)
        guard case .array(let tools)? = try #require(rec.bodies.first)["tools"],
              case .dictionary(let tool)? = tools.first else { Issue.record("no tools forced"); return }
        #expect(tool["name"] == .string(CapabilityProbe.probeToolName))
        #expect(tool["strict"] == .bool(true))
        #expect(tool["function"] == nil, "chat/completions nests the definition; Responses does not")
        // And the chat/completions shape is untouched for everyone else.
        let chatRec = CodexLikeEndpoint.Recorder()
        _ = await ModelProber.probeStrictToolDefinitions(
            apiType: .openAICompatible, makeProviderForcing: { CodexLikeEndpoint(forced: $0, seen: chatRec) })
        guard case .array(let chatTools)? = try #require(chatRec.bodies.first)["tools"],
              case .dictionary(let chatTool)? = chatTools.first else { Issue.record("no tools forced"); return }
        #expect(chatTool["function"] != nil)
    }

    @Test("Continuation with only Codex items is not empty, and round-trips through Codable")
    func continuationCodable() throws {
        let item = CodexReasoningItem(id: "rs_9", encryptedContent: "X", summary: ["s"])
        let continuation = ProviderContinuation(codexReasoningItems: [item])
        #expect(continuation.isEmpty == false)
        let data = try JSONEncoder().encode(continuation)
        let decoded = try JSONDecoder().decode(ProviderContinuation.self, from: data)
        #expect(decoded == continuation)
        // Files written before this field existed still load.
        let legacy = try JSONDecoder().decode(ProviderContinuation.self, from: Data("{}".utf8))
        #expect(legacy.codexReasoningItems == nil)
    }
}
