import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "ModelProber")

/// Runs the full battery of capability probes against one model and assembles a ``ModelProfile``.
///
/// The governing rule, decided with the product: **decode what a vendor publishes and believe it;
/// probe only what no vendor tells us.** So the driver takes a `skip` set — the facts the caller
/// already read out of the `/models` payload — and never spends a call re-establishing them. What
/// remains is the genuinely unknowable: tool calling (no vendor publishes it, bar Ollama's
/// unverified hint), and everything OpenAI omits, which is nearly everything.
///
/// Every individual probe defends against the one failure mode that matters here — a false
/// positive. Accepting an image is not reading it; a 200 with tools present is not a tool call; an
/// endpoint that echoes a prompt is not answering it. Each probe embeds a random token the model
/// can only produce by actually doing the thing, mirroring the tool probe's identifier.
///
/// Probes run **serially**, never in parallel: a capability run is not a load test, and
/// interleaving calls would turn one model's rate-limit into another's `inconclusive`.
///
/// Each probe is also a standalone `public` function — the common case is a full sweep via
/// ``probe(llm:providerID:modelID:skip:effortLevelsToProbe:)``, but any single capability can be
/// checked on its own without one.
public enum ModelProber {

    /// Facts the caller already established by decoding the provider payload, so the driver skips
    /// probing them. Everything absent from the set is probed.
    public struct Skip: OptionSet, Sendable {
        public let rawValue: Int
        public init(rawValue: Int) { self.rawValue = rawValue }
        public static let chat            = Skip(rawValue: 1 << 0)
        public static let vision          = Skip(rawValue: 1 << 1)
        public static let pdfInput        = Skip(rawValue: 1 << 2)
        public static let maxOutputTokens = Skip(rawValue: 1 << 3)
        public static let effort          = Skip(rawValue: 1 << 4)
    }

    /// Probes everything not in `skip` and returns a fully populated profile.
    ///
    /// - Parameters:
    ///   - llm: a provider already bound to the model under test. It exposes no capability data,
    ///     which is what keeps catalog claims out of a measurement of the truth.
    ///   - effortLevelsToProbe: which named efforts to attempt. The caller passes the model's own
    ///     list (decoded, for Anthropic) or a hand-authored one (OpenAI); the driver never guesses
    ///     the universe of levels itself.
    public static func probe(
        llm: any LLMProvider,
        providerID: String,
        modelID: String,
        skip: Skip = [],
        effortLevelsToProbe: [String] = []
    ) async -> ModelProfile {
        let started = Date()
        let calls = ProbeCallCounter()
        var profile = ModelProfile(providerID: providerID, modelID: modelID)

        // 1. Chat is the floor, and temperature rides along in the same request — the chat probe
        //    already sends a nonce echo, so setting temperature: 0 on it establishes both in one
        //    call. Standalone `probeChat`/`probeTemperature` exist for individual use, but a full
        //    sweep should not pay for two calls when one carries both facts. Only a temperature
        //    rejection forces a second, temperature-free call to still get the chat answer.
        //    If chat can't be reached at all, every later probe fails the same way — stop once.
        if skip.contains(.chat) {
            profile.chat = .established(true, "decoded from provider payload")
        } else {
            let combined = await probeChatAndTemperature(llm: llm, modelID: modelID, calls: calls)
            profile.chat = combined.chat
            profile.acceptsTemperature = combined.temperature
            if profile.chat.status == .inconclusive {
                logger.error("Probe \(modelID, privacy: .public): chat inconclusive — halting")
                profile.callCount = calls.value
                profile.duration = Date().timeIntervalSince(started)
                return profile
            }
            if profile.chat.value == false {
                // Not a chat model. Tool calling and the rest are meaningless.
                profile.callCount = calls.value
                profile.duration = Date().timeIntervalSince(started)
                return profile
            }
        }


        // 3. Tool calling, then the result round-trip. The reason the probe exists.
        let toolResult = await CapabilityProbe.probeToolCalling(
            llm: llm, providerID: providerID, modelID: modelID, calls: calls
        )
        profile.toolCalling = toolResult.toolUse
            .map { ProbeFinding<Bool>.established($0, toolResult.errorDescription ?? toolResult.verdict.rawValue) }
            ?? .inconclusive(toolResult.errorDescription ?? "no answer")
        profile.toolResultRoundTrip = {
            switch toolResult.verdict {
            case .roundTripCompleted:    return .established(true, "returned the identifier")
            case .toolCallOnly:          return .established(false, "called the tool but did not return the identifier")
            case .noToolCall, .rejected: return .established(false, "no tool call")
            case .inconclusive:          return .inconclusive(toolResult.errorDescription ?? "no answer")
            }
        }()

        // 4. Vision + PDF — only where the payload didn't already tell us.
        if skip.contains(.vision) {
            profile.vision = .established(true, "decoded from provider payload")
        } else {
            profile.vision = await probeVision(llm: llm, modelID: modelID, calls: calls)
        }
        if skip.contains(.pdfInput) {
            profile.pdfInput = .established(true, "decoded from provider payload")
        } else {
            profile.pdfInput = await probePDFInput(llm: llm, modelID: modelID, calls: calls)
        }

        // 5. Max output — one call, learned from the endpoint's own rejection.
        if !skip.contains(.maxOutputTokens) {
            profile.maxOutputTokens = await probeMaxOutputTokens(llm: llm, modelID: modelID, calls: calls)
        }

        // 6. Effort levels — attempt each caller-supplied level. Only meaningful where the
        //    provider emits the field unconditionally (Anthropic); where emission is gated on a
        //    behavior flag (OpenAI-compatible), a level we haven't flagged is silently dropped, so
        //    a "no error" result there is confirmation, not discovery.
        if !skip.contains(.effort) {
            for level in effortLevelsToProbe {
                profile.effortLevels[level] = await probeEffortLevel(level, llm: llm, modelID: modelID, calls: calls)
            }
        }

        profile.callCount = calls.value
        profile.duration = Date().timeIntervalSince(started)
        return profile
    }

    // MARK: - Combined probe (efficiency)

    /// Establishes chat AND whether temperature is accepted in a single call, because both facts
    /// come from the same nonce-echo request — the chat probe with `temperature: 0` on it. The
    /// full sweep uses this rather than the two standalone probes to avoid paying for a second
    /// call it doesn't need.
    ///
    /// The only branch that costs a second call is a temperature rejection: a 400 naming
    /// temperature settles `temperature = false` but leaves chat unknown, so chat is re-probed
    /// without the parameter. Every other outcome is one call.
    public static func probeChatAndTemperature(
        llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil
    ) async -> (chat: ProbeFinding<Bool>, temperature: ProbeFinding<Bool>) {
        let nonce = CapabilityProbe.makeIdentifier()
        let messages: [LLMMessage] = [
            .system("You are a test harness. Reply with exactly what is asked and nothing else."),
            .user("Reply with exactly this identifier and nothing else: \(nonce)")
        ]
        let started = Date()
        calls?.increment()
        do {
            let response = try await llm.send(messages: messages, tools: [],
                                              overrides: LLMCallOverrides(temperature: 0))
            // Success with temperature present: both facts settled at once.
            return (chatFinding(response.text, nonce: nonce, started: started),
                    .established(true, "accepted temperature alongside a successful chat call",
                                 duration: Date().timeIntervalSince(started)))
        } catch {
            let detail = CapabilityProbe.rejectionDetail(error)
            guard detail.lowercased().contains("temperature") else {
                // Not a temperature problem — chat failed for some other reason, and we learned
                // nothing about temperature.
                return (finding(fromError: error, capabilityKeyword: nil, started: started),
                        .inconclusive(detail, duration: Date().timeIntervalSince(started)))
            }
            // Temperature is the culprit: settle it false, then re-probe chat without it so the
            // model still gets a fair chat reading.
            let temperature = ProbeFinding<Bool>.established(false, detail, duration: Date().timeIntervalSince(started))
            let chat = await probeChat(llm: llm, modelID: modelID, calls: calls)
            return (chat, temperature)
        }
    }

    // MARK: - Individual probes (each standalone-callable)

    /// Establishes chat by demanding a random identifier back verbatim. A fixed prompt could be
    /// answered from a canned response; the nonce forces a real, current generation.
    public static func probeChat(llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil) async -> ProbeFinding<Bool> {
        let nonce = CapabilityProbe.makeIdentifier()
        let started = Date()
        calls?.increment()
        do {
            let response = try await llm.send(messages: [
                .system("You are a test harness. Reply with exactly what is asked and nothing else."),
                .user("Reply with exactly this identifier and nothing else: \(nonce)")
            ], tools: [])
            return chatFinding(response.text, nonce: nonce, started: started)
        } catch {
            return finding(fromError: error, capabilityKeyword: nil, started: started)
        }
    }

    /// Grades a chat response body against the nonce it should have echoed.
    private static func chatFinding(_ text: String?, nonce: String, started: Date) -> ProbeFinding<Bool> {
        let body = text ?? ""
        let dur = Date().timeIntervalSince(started)
        if body.contains(nonce) {
            return .established(true, "echoed the identifier", duration: dur)
        }
        // 200 but no echo: the endpoint answered, so it IS a chat endpoint, but the model didn't
        // follow a trivial instruction. Recorded true (chat works) with a note — small or
        // quantized models ramble, and that isn't "not a chat model".
        return .established(true, "responded (\(body.prefix(60))) but did not echo the identifier", duration: dur)
    }

    /// Whether the endpoint tolerates a `temperature` parameter. `established(false)` means it
    /// answered 400 naming temperature — the signal that the model needs
    /// `mustNeverSendTemperatureParam`. Note: if that flag is already set the provider omits the
    /// parameter, so a request succeeds and this reports `true` — correct from the app's view
    /// (requests work) even though the model itself wouldn't accept the field.
    public static func probeTemperature(llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil) async -> ProbeFinding<Bool> {
        let nonce = CapabilityProbe.makeIdentifier()
        let started = Date()
        calls?.increment()
        do {
            _ = try await llm.send(messages: [
                .user("Reply with exactly this identifier: \(nonce)")
            ], tools: [], overrides: LLMCallOverrides(temperature: 0))
            return .established(true, "accepted temperature", duration: Date().timeIntervalSince(started))
        } catch {
            let detail = CapabilityProbe.rejectionDetail(error)
            if detail.lowercased().contains("temperature") {
                return .established(false, detail, duration: Date().timeIntervalSince(started))
            }
            return .inconclusive(detail, duration: Date().timeIntervalSince(started))
        }
    }

    /// Whether the model reads an image. Sends a coloured shape and asks for both the shape and
    /// the colour. Only `true` when it names BOTH — a guesser lands both about 1-in-18, and a
    /// model that merely accepts the attachment without reading it can't produce either. Colour
    /// alone was too guessable and too easy to fake by describing the payload's existence.
    public static func probeVision(llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil) async -> ProbeFinding<Bool> {
        guard let color = ProbeFixtures.namedColors.randomElement(),
              let shape = ProbeFixtures.namedShapes.randomElement() else { return .notAttempted }
        let png = ProbeFixtures.makeShapePNG(shape: shape, red: color.red, green: color.green, blue: color.blue)
        let image = LLMImageContent(data: png, mimeType: "image/png")
        let started = Date()
        calls?.increment()
        do {
            let response = try await llm.send(messages: [
                .system("You are a vision test harness. Describe the image in a few words."),
                .user("What shape is in this image, and what colour is it? Answer briefly.", images: [image])
            ], tools: [])
            let text = (response.text ?? "").lowercased()
            let dur = Date().timeIntervalSince(started)
            let sawColor = text.contains(color.name)
            let sawShape = text.contains(shape.rawValue)
            if sawColor && sawShape {
                return .established(true, "named '\(color.name) \(shape.rawValue)'", duration: dur)
            }
            // Answered without both facts: it accepted the image shape but did not read it, or it
            // can't see. Either way it did not demonstrate vision. Recorded false with what it
            // actually said, so an auditor can tell a hallucination from a partial read.
            let got = [sawColor ? "colour✓" : "colour✗", sawShape ? "shape✓" : "shape✗"].joined(separator: " ")
            return .established(false, "expected '\(color.name) \(shape.rawValue)' — \(got); said '\(text.prefix(50))'", duration: dur)
        } catch {
            return finding(fromError: error, capabilityKeyword: "image", started: started)
        }
    }

    /// Whether the model reads a PDF. Sends a one-page document showing a random code and asks for
    /// it back — text transcription, the hardest signal to fake. `true` only when the code returns.
    public static func probePDFInput(llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil) async -> ProbeFinding<Bool> {
        let code = CapabilityProbe.makeIdentifier()
        let pdf = ProbeFixtures.makePDF(code: code)
        let document = LLMDocumentContent(data: pdf, mimeType: "application/pdf", filename: "probe.pdf")
        let started = Date()
        calls?.increment()
        do {
            let response = try await llm.send(messages: [
                .system("You are a document test harness. Reply with only the code you are asked for."),
                .user("This PDF displays a code. Reply with only that code.", images: [], documents: [document])
            ], tools: [])
            let text = response.text ?? ""
            let dur = Date().timeIntervalSince(started)
            if text.contains(code) {
                return .established(true, "returned the embedded code", duration: dur)
            }
            return .established(false, "expected '\(code)', said '\(text.prefix(40))'", duration: dur)
        } catch {
            return finding(fromError: error, capabilityKeyword: "pdf", started: started)
        }
    }

    /// The model's real output ceiling, read out of its own rejection. Requests an absurd
    /// `max_tokens`; the endpoint answers with the true limit, which we already parse for the
    /// runtime auto-clamp (``LLMProviderError/reportedMaxOutputTokenLimit``). One call, no binary
    /// search — that's the fallback for endpoints whose error doesn't state the number, and is
    /// deferred until it's actually needed. When the rejection is in a format we don't parse, the
    /// raw body is kept as `inconclusive` evidence so an unrecognised shape is visible, not lost.
    public static func probeMaxOutputTokens(llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil) async -> ProbeFinding<Int> {
        let absurd = 100_000_000
        let started = Date()
        calls?.increment()
        do {
            _ = try await llm.send(messages: [
                .user("Reply with the single word: ok")
            ], tools: [], overrides: LLMCallOverrides(maxOutputTokens: absurd))
            // It accepted a preposterous cap. One call can't bound it from above; all we'd know is
            // "at least this", which is never useful. Report inconclusive rather than a fake number.
            return .inconclusive("accepted max_tokens=\(absurd); true ceiling not revealed", duration: Date().timeIntervalSince(started))
        } catch {
            let dur = Date().timeIntervalSince(started)
            if let limit = (error as? LLMProviderError)?.reportedMaxOutputTokenLimit {
                return .established(limit, "endpoint reported its maximum", duration: dur)
            }
            return .inconclusive(CapabilityProbe.rejectionDetail(error), duration: dur)
        }
    }

    /// Whether a named effort level is accepted. See the driver's note on flag-gated emission:
    /// where the provider only sends the field when a behavior flag is set (OpenAI-compatible),
    /// a "no error" here confirms an already-known reasoning model but cannot discover a new one.
    public static func probeEffortLevel(_ level: String, llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil) async -> ProbeFinding<Bool> {
        let started = Date()
        calls?.increment()
        do {
            _ = try await llm.send(messages: [
                .user("Reply with the single word: ok")
            ], tools: [], overrides: LLMCallOverrides(thinkingEffort: level))
            return .established(true, "accepted effort '\(level)'", duration: Date().timeIntervalSince(started))
        } catch {
            let detail = CapabilityProbe.rejectionDetail(error)
            if detail.lowercased().contains("effort") || detail.lowercased().contains("reasoning") {
                return .established(false, detail, duration: Date().timeIntervalSince(started))
            }
            return .inconclusive(detail, duration: Date().timeIntervalSince(started))
        }
    }

    // MARK: - Error → finding

    /// Maps a thrown error to a Bool finding for capabilities where a refusal that NAMES the
    /// capability is a "no", and any other failure is `inconclusive`. The keyword is what lets us
    /// tell "this model can't take images" (a fact) from "our request was malformed" or "the
    /// network dropped" (not facts).
    private static func finding(fromError error: any Error, capabilityKeyword: String?, started: Date) -> ProbeFinding<Bool> {
        let dur = Date().timeIntervalSince(started)
        let detail = CapabilityProbe.rejectionDetail(error)
        switch CapabilityProbe.classifyFailure(error) {
        case .noAnswer:
            return .inconclusive(detail, duration: dur)
        case .refusedTools, .refusedOurRequest:
            guard let keyword = capabilityKeyword else {
                // Chat: any 4xx that got this far is the endpoint declining a plain request, which
                // for chat means "not a usable chat endpoint".
                return CapabilityProbe.classifyFailure(error) == .noAnswer
                    ? .inconclusive(detail, duration: dur)
                    : .established(false, detail, duration: dur)
            }
            // A 400 that names the capability ("image", "pdf") is the endpoint saying it can't. A
            // 400 about something else says nothing about the capability under test.
            return detail.lowercased().contains(keyword)
                ? .established(false, detail, duration: dur)
                : .inconclusive(detail, duration: dur)
        }
    }
}
