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

    /// The prober's request-forming generation, stamped into every persisted ``ProbeRecord``.
    /// Bump whenever a probe's request shape changes in a way that could alter its verdicts
    /// (e.g. the PDF encoding switch that made every earlier pdfInput=false suspect) — records
    /// from an older prober are then identifiable as re-probe candidates.
    ///
    /// v2 (2026-07-18): probe providers strip `mustNeverSendTemperatureParam`
    /// (`makeProbeProvider`), so temperature is genuinely sent — v1 records' `acceptsTemperature
    /// = true` on flagged models are artifacts of the suppressed parameter, not measurements.
    ///
    /// v3 (2026-07-18): tool-probe request/grading overhaul after the log audit. Probe providers
    /// no longer send `parallel_tool_calls` (its rejection poisoned eight o-series verdicts);
    /// the round-trip follow-up answers EVERY tool_call_id (parallel-calling models were graded
    /// on a harness-malformed request); truncated/empty 200s no longer grade `noToolCall`; and
    /// rejections naming our own request knobs no longer read as tool refusals. v2 records'
    /// `toolCalling`/`toolResultRoundTrip` = false are suspect; re-probe.
    public static let proberVersion = 3

    /// Builds a probe seed from a TRI-STATE facts record — the preferred seeding path.
    ///
    /// No per-apiType switch: the decoders already encode statedness (`nil` = the vendor didn't
    /// say), so a field seeds as `decoded` exactly when the vendor stated it, both directions.
    /// This is what the ModelInfo-based overload cannot do — materialization flattens `nil` to
    /// `false`, and seeding from it fabricates `decoded(false)` for fields the vendor never
    /// stated, which both skips the probe AND displays a vendor "no" that doesn't exist.
    public static func seedProfile(fromDecodedFacts decoded: DecodedModelFacts, providerID: String) -> ModelProfile {
        var profile = ModelProfile(providerID: providerID, modelID: decoded.modelID)
        let facts = decoded.facts
        let evidence = "provider /models payload"
        profile.displayName = facts.displayName ?? decoded.modelID
        profile.createdAt = facts.createdAt
        profile.pricing = facts.pricing
        profile.deprecatedOn = facts.deprecatedOn
        profile.maxTemperature = facts.maxTemperature
        profile.samplingDefaults = facts.samplingDefaults
        profile.isFree = facts.isFree
        profile.benchmarks = facts.benchmarks
        if let value = facts.maxInputTokens { profile.maxContextTokens = .decoded(value, evidence) }
        if let value = facts.maxOutputTokens { profile.maxOutputTokens = .decoded(value, evidence) }
        if let value = facts.supportsChatCompletions { profile.chat = .decoded(value, evidence) }
        if let value = facts.capabilities.toolUse { profile.toolCalling = .decoded(value, evidence) }
        if let value = facts.capabilities.vision { profile.vision = .decoded(value, evidence) }
        if let value = facts.capabilities.pdfInput { profile.pdfInput = .decoded(value, evidence) }
        if let levels = facts.validEffortLevels {
            for level in levels { profile.effortLevels[level] = .decoded(true, evidence) }
            // A non-nil list is the vendor enumerating THE valid set (Anthropic's per-level
            // supported flags; a stated [] means "no effort levels at all"), so every known level
            // not listed is a stated "no" — without this, vendor denials (xhigh on sonnet-4-6,
            // everything on haiku-4-5) were indistinguishable from "nobody asked" and the ladder
            // re-probed levels the vendor already answered.
            for level in EffortRank.table.keys where !levels.contains(level) {
                profile.effortLevels[level] = .decoded(false, evidence + " (not in the model's stated effort set)")
            }
        }
        // Listed in /models ⇒ presumed reachable, but only presumed — a live probe can overturn it.
        profile.isAvailable = .decoded(true, "present in provider /models listing")
        return profile
    }

    /// Builds a profile pre-filled with everything the provider's `/models` payload already told
    /// us, so the driver only spends calls on the gaps.
    ///
    /// LEGACY: prefer ``seedProfile(fromDecodedFacts:providerID:)``. This overload consumes the
    /// flattened `ModelInfo`, where `false` may mean "the vendor didn't say" — its per-apiType
    /// switch limits the damage but cannot fully recover the lost tri-state (an Anthropic payload
    /// missing a capability leaf seeds a fabricated `decoded(false)` here).
    ///
    /// **Feed this the freshly-decoded `ModelInfo` from `ModelFetchService.fetchModels`, never the
    /// merged catalog** — the catalog has LiteLLM's third-party claims layered in, and seeding
    /// from it would let those claims wear a `decoded` badge. The rule this whole system runs on
    /// is: the vendor's payload is gospel, everything else gets probed.
    ///
    /// Seeding is deliberately per-apiType and conservative, because most decoders leave
    /// `capabilities` at its all-false default and false-meaning-unknown must never seed a
    /// `decoded(false)`:
    /// - every apiType: `displayName`, `createdAt`, limits when present, `validEffortLevels`
    ///   (each level as `decoded(true)`)
    /// - `.anthropic`: `vision`/`pdfInput` (its capabilities block states them explicitly, both
    ///   directions), and `chat` from `supportsChatCompletions`
    /// - `.ollama`: `toolCalling` when the tags payload lists `tools`
    /// - `.gemini`: `chat` from `supportedGenerationMethods` (decoded into
    ///   `supportsChatCompletions`)
    public static func seedProfile(fromDecoded info: ModelInfo, apiType: ProviderAPIType) -> ModelProfile {
        var profile = ModelProfile(providerID: info.providerID, modelID: info.modelID)
        profile.displayName = info.displayName
        profile.createdAt = info.createdAt
        profile.pricing = info.pricing              // decoded-only, believed as published
        profile.deprecatedOn = info.deprecatedOn    // decoded-only (Mistral publishes it)
        profile.maxTemperature = info.maxTemperature // decoded-only (Gemini publishes it)
        profile.samplingDefaults = info.samplingDefaults // decoded-only (Gemini/Mistral/OpenRouter)
        profile.isFree = info.isFree                  // decoded-only (HuggingFace per-provider)
        profile.benchmarks = info.benchmarks          // decoded-only (OpenRouter)
        // Listed in /models ⇒ presumed reachable, but only presumed — a live probe can overturn it.
        profile.isAvailable = .decoded(true, "present in provider /models listing")
        if let context = info.maxInputTokens {
            profile.maxContextTokens = .decoded(context, "provider /models payload")
        }
        if let output = info.maxOutputTokens {
            profile.maxOutputTokens = .decoded(output, "provider /models payload")
        }
        for level in info.validEffortLevels {
            profile.effortLevels[level] = .decoded(true, "provider /models payload")
        }

        switch apiType {
        case .anthropic:
            // Anthropic's capabilities block states supported: true/false explicitly, so both
            // directions are decodable — unlike decoders where false just means "didn't say".
            profile.vision = .decoded(info.capabilities.vision, "capabilities.image_input")
            profile.pdfInput = .decoded(info.capabilities.pdfInput, "capabilities.pdf_input")
            profile.chat = .decoded(info.supportsChatCompletions, "provider /models payload")
        case .ollama:
            if info.capabilities.toolUse {
                profile.toolCalling = .decoded(true, "tags payload lists 'tools'")
            }
        case .gemini:
            profile.chat = .decoded(info.supportsChatCompletions, "supportedGenerationMethods")
        case .mistral:
            // Mistral's capabilities block states each flag explicitly (true/false), like
            // Anthropic's — so both directions are decodable, including tool calling
            // (`function_calling`), which almost no other vendor publishes.
            profile.chat = .decoded(info.supportsChatCompletions, "capabilities.completion_chat")
            profile.toolCalling = .decoded(info.capabilities.toolUse, "capabilities.function_calling")
            profile.vision = .decoded(info.capabilities.vision, "capabilities.vision")
        default:
            break
        }
        return profile
    }

    /// Probes every field of `seed` still `notAttempted` and returns the completed profile.
    ///
    /// The seed IS the skip logic: a field the payload already established (see
    /// ``seedProfile(fromDecoded:apiType:)``) arrives `established` and is left alone, so "probe
    /// only what we don't get for free" is structural rather than a flag the caller can forget.
    /// Pass a bare `ModelProfile(providerID:modelID:)` to probe everything.
    ///
    /// - Parameters:
    ///   - llm: a provider already bound to the model under test. It exposes no capability data,
    ///     which is what keeps catalog claims out of a measurement of the truth.
    ///   - effortLevelsToProbe: named efforts to attempt beyond what the seed already settled.
    ///     Only meaningful where the provider emits the field unconditionally (Anthropic); where
    ///     emission is flag-gated (OpenAI-compatible), build per-level providers whose
    ///     configuration forces the field via `extraJSONOverrides` and use
    ///     ``probeParameterAcceptance`` instead.
    public static func probe(
        llm: any LLMProvider,
        seed: ModelProfile,
        effortLevelsToProbe: [String] = [],
        preferLowImageDetail: Bool = false
    ) async -> ModelProfile {
        let started = Date()
        let calls = ProbeCallCounter()
        var profile = seed
        let modelID = profile.modelID

        // 1. Chat is the floor, and temperature rides along in the same request — one nonce-echo
        //    call with temperature: 0 settles both. Only a temperature rejection costs a second,
        //    temperature-free call. If chat can't be reached at all, every later probe fails the
        //    same way — stop once.
        var chatDeferred = false
        if profile.chat.status == .notAttempted {
            let combined = await probeChatAndTemperature(llm: llm, modelID: modelID, calls: calls)
            profile.chat = combined.chat
            profile.acceptsTemperature = combined.temperature
            chatDeferred = combined.chatDeferred
        } else if profile.acceptsTemperature.status == .notAttempted {
            // Chat came decoded; temperature still needs its own (cheap) probe.
            profile.acceptsTemperature = await probeTemperature(llm: llm, modelID: modelID, calls: calls)
        }

        // Reachability gate. A decoded chat=true is a claim, not proof: a model can be retired while
        // still listed (Gemini keeps returning gemini-2.0-flash-lite in /models long after every
        // call 404s "no longer available"), or listed but not enabled for this account (Alibaba
        // Cloud's Model.AccessDenied). Scan ONLY the evidence of FAILED (inconclusive) step-1
        // findings: a successful chat reply is echoed back into its evidence string, and a rambling
        // model that says "does not exist" / "model access" in that reply must not be read as gone/
        // denied — the call succeeded, so the model is reachable. Error bodies are the only place
        // these phrases mean what they say.
        let failedEvidence = [profile.chat, profile.acceptsTemperature]
            .filter { $0.status == .inconclusive }
            .compactMap(\.evidence)
        if let gone = failedEvidence.first(where: CapabilityProbe.textIndicatesModelGone) {
            profile.isAvailable = .established(false, gone)
            logger.error("Probe \(modelID, privacy: .public): model unavailable — halting")
            return finish(&profile, calls: calls, started: started)
        }
        if let denied = failedEvidence.first(where: CapabilityProbe.textIndicatesAccessDenied) {
            profile.isAccessDenied = .established(true, denied)
            logger.error("Probe \(modelID, privacy: .public): access denied — halting")
            return finish(&profile, calls: calls, started: started)
        }
        if profile.chat.status == .inconclusive, !chatDeferred {
            logger.error("Probe \(modelID, privacy: .public): chat inconclusive — halting")
            return finish(&profile, calls: calls, started: started)
        }
        if profile.chat.value == false {
            // Not a chat model. Tool calling and the rest are meaningless.
            return finish(&profile, calls: calls, started: started)
        }
        // Reachability must be CONFIRMED by a live call that produced a DEFINITE answer before the
        // structural probes (tool/vision/PDF/max-output) run — each of them grades a 4xx as the
        // capability failing, which is only sound if a plain call to this model currently works. A
        // step-1 finding that is `established` AND `probed` is exactly that proof: the endpoint gave
        // a coherent answer (a 200, or a real 400 like "temperature not supported" — either way it's
        // reachable). If step 1 produced only inconclusive results (rate-limit, auth, network), the
        // assumption is unproven, so halt rather than fabricate falses, and don't claim availability
        // we never confirmed. (A decoded chat=true is a catalog claim, not a live confirmation.)
        let reachableConfirmed = [profile.chat, profile.acceptsTemperature]
            .contains { $0.status == .established && $0.source == .probed }
        guard reachableConfirmed else {
            logger.info("Probe \(modelID, privacy: .public): no confirmed live response — halting before structural probes")
            return finish(&profile, calls: calls, started: started)
        }
        profile.isAvailable = .established(true, "responded to a live call")
        profile.isAccessDenied = .established(false, "responded to a live call")

        // 2. Tool calling, then the result round-trip. The reason the probe exists.
        var toolBattery: CapabilityProbe.ToolCallResult?
        if profile.toolCalling.status == .notAttempted || profile.toolResultRoundTrip.status == .notAttempted {
            let toolResult = await CapabilityProbe.probeToolCalling(
                llm: llm, providerID: profile.providerID, modelID: modelID, calls: calls
            )
            toolBattery = toolResult
            if profile.toolCalling.status == .notAttempted {
                profile.toolCalling = toolResult.toolUse
                    .map { ProbeFinding<Bool>.established($0, toolResult.errorDescription ?? toolResult.verdict.rawValue) }
                    ?? .inconclusive(toolResult.errorDescription ?? "no answer")
            }
            profile.toolResultRoundTrip = {
                switch toolResult.verdict {
                case .roundTripCompleted:      return .established(true, "returned the identifier")
                case .toolCallOnly:            return .established(false, "called the tool but did not return the identifier")
                case .noToolCall, .rejected:   return .established(false, "no tool call")
                case .roundTripInconclusive:   return .inconclusive(toolResult.errorDescription ?? "tool-result request unresolved")
                case .inconclusive:            return .inconclusive(toolResult.errorDescription ?? "no answer")
                }
            }()
        }

        // Deferred-chat resolution (replay elision). A temperature rejection consumed call 1
        // before chat could be observed; any 200 from the tool battery is the same proof a replay
        // would buy — a chat/completions request was served for this model — and the round-trip's
        // identifier echo is stronger (it followed an instruction through a tool result). Only
        // when the battery produced no 200 do we pay for the dedicated chat call, with the same
        // gone/denied evidence scan step 1 applies.
        if chatDeferred {
            if toolBattery?.sawSuccessfulResponse == true {
                profile.chat = .established(true, toolBattery?.verdict == .roundTripCompleted
                    ? "echoed the identifier via the tool round-trip (chat replay elided)"
                    : "tool battery answered a chat/completions request (chat replay elided)")
            } else {
                profile.chat = await probeChat(llm: llm, modelID: modelID, calls: calls)
                if profile.chat.status == .inconclusive, let evidence = profile.chat.evidence {
                    if CapabilityProbe.textIndicatesModelGone(evidence) {
                        profile.isAvailable = .established(false, evidence)
                    } else if CapabilityProbe.textIndicatesAccessDenied(evidence) {
                        profile.isAccessDenied = .established(true, evidence)
                    }
                }
            }
            if profile.chat.value != true {
                logger.error("Probe \(modelID, privacy: .public): chat unresolved after deferral — halting")
                return finish(&profile, calls: calls, started: started)
            }
        }

        // Tool calling is the capability this whole system exists to establish, and a model that
        // demonstrably can't call tools is one we will ultimately discard — so there's no value in
        // spending the remaining calls (vision, PDF, max-output binary search, effort) on it. Stop
        // as soon as tool calling is *established* false. Inconclusive keeps going: not knowing is
        // not the same as a "no", and the later probes may still be worth their calls.
        if profile.toolCalling.value == false {
            logger.info("Probe \(modelID, privacy: .public): tool calling unsupported — skipping remaining probes")
            return finish(&profile, calls: calls, started: started)
        }

        // 3. Vision + PDF — only where the payload didn't already say. When BOTH are open, one
        //    combined call settles the double-positive majority (83% of models reaching this
        //    stage); on any other outcome it returns nil, nothing is graded, and the separate
        //    probes below run exactly as before.
        if profile.vision.status == .notAttempted, profile.pdfInput.status == .notAttempted,
           let combined = await probeVisionAndPDF(llm: llm, modelID: modelID, calls: calls,
                                                  preferLowImageDetail: preferLowImageDetail) {
            profile.vision = combined.vision
            profile.pdfInput = combined.pdf
        }
        if profile.vision.status == .notAttempted {
            profile.vision = await probeVision(llm: llm, modelID: modelID, calls: calls,
                                               preferLowImageDetail: preferLowImageDetail)
        }
        if profile.pdfInput.status == .notAttempted {
            profile.pdfInput = await probePDFInput(llm: llm, modelID: modelID, calls: calls)
        }

        // 4. Max output — one call, learned from the endpoint's own rejection. Skips once
        //    either an output cap OR a context-bound outcome has been settled.
        if profile.maxOutputTokens.status == .notAttempted,
           (profile.maxOutputBoundedByContext?.status ?? .notAttempted) == .notAttempted {
            let result = await probeMaxOutputTokens(llm: llm, modelID: modelID, calls: calls, maxContextTokens: profile.maxContextTokens.value)
            profile.maxOutputTokens = result.cap
            profile.maxOutputBoundedByContext = result.contextBound
        }

        // 5. Effort levels the seed didn't settle.
        for level in effortLevelsToProbe where profile.effortLevels[level] == nil {
            profile.effortLevels[level] = await probeEffortLevel(level, llm: llm, modelID: modelID, calls: calls)
        }

        return finish(&profile, calls: calls, started: started)
    }

    private static func finish(_ profile: inout ModelProfile, calls: ProbeCallCounter, started: Date) -> ModelProfile {
        profile.callCount += calls.value
        profile.duration += Date().timeIntervalSince(started)
        return profile
    }

    /// Whether the endpoint accepts a request at all in this shape — for parameters the normal
    /// provider path won't emit unless a behavior flag says to (OpenAI's `reasoning_effort`). The
    /// caller builds an `LLMProvider` whose configuration forces the parameter into the body via
    /// `extraJSONOverrides` — which merges last and unconditionally, bypassing the flag gate —
    /// and this just sends and grades.
    ///
    /// `established(false)` only when the refusal names one of `rejectionKeywords`; an unrelated
    /// 4xx stays inconclusive, per the rule that a refusal we provoked is not evidence.
    public static func probeParameterAcceptance(
        llm: any LLMProvider,
        parameterDescription: String,
        rejectionKeywords: [String],
        calls: ProbeCallCounter? = nil
    ) async -> ProbeFinding<Bool> {
        let started = Date()
        calls?.increment()
        do {
            _ = try await llm.send(messages: [.user("Reply with the single word: ok")], tools: [])
            return .established(true, "accepted \(parameterDescription)", duration: Date().timeIntervalSince(started))
        } catch {
            let detail = CapabilityProbe.rejectionDetail(error)
            let lowered = detail.lowercased()
            if !CapabilityProbe.classifyFailure(error).meansNoAnswer,
               rejectionKeywords.contains(where: { lowered.contains($0.lowercased()) }) {
                return .established(false, detail, duration: Date().timeIntervalSince(started))
            }
            return .inconclusive(detail, duration: Date().timeIntervalSince(started))
        }
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
    /// `chatDeferred` is true only on the temperature-rejection path: the 400 settles
    /// temperature=false but consumed the call before chat could be observed. The old behavior
    /// re-probed chat immediately; the 2026-07-18 efficiency audit measured that replay at 20
    /// billed calls and ~45% of the sweep's reasoning-token burn per OpenAI run, all spent
    /// re-learning what the tool battery's own 200 proves anyway. The sweep now resolves a
    /// deferred chat from the tool battery and only replays when no 200 ever arrives.
    public static func probeChatAndTemperature(
        llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil
    ) async -> (chat: ProbeFinding<Bool>, temperature: ProbeFinding<Bool>, chatDeferred: Bool) {
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
                                 duration: Date().timeIntervalSince(started)),
                    false)
        } catch {
            let detail = CapabilityProbe.rejectionDetail(error)
            // "Temperature is the culprit" only when a GENUINE refusal (a coherent 4xx) names it. A
            // transport failure / rate limit / auth / missing-model error that merely contains the
            // word "temperature" establishes nothing — and re-probing chat without temperature would
            // just fail the same way. Treat those like any other non-temperature failure.
            let namesTemperature = !CapabilityProbe.classifyFailure(error).meansNoAnswer
                && detail.lowercased().contains("temperature")
            guard namesTemperature else {
                // Not a temperature problem — chat failed for some other reason, and we learned
                // nothing about temperature.
                return (finding(fromError: error, capabilityKeywords: [], started: started),
                        .inconclusive(detail, duration: Date().timeIntervalSince(started)),
                        false)
            }
            // Temperature is the culprit: settle it false and DEFER chat — the tool battery's own
            // 200 (or its round-trip identifier echo) is at least as much proof as a replay, and
            // the sweep falls back to a dedicated chat call only if no 200 ever arrives.
            let temperature = ProbeFinding<Bool>.established(false, detail, duration: Date().timeIntervalSince(started))
            let deferredChat = ProbeFinding<Bool>.inconclusive(
                "temperature rejected; chat deferred to the tool battery",
                duration: Date().timeIntervalSince(started))
            return (deferredChat, temperature, true)
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
            return finding(fromError: error, capabilityKeywords: [], started: started)
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
    /// `mustNeverSendTemperatureParam`. Callers must hand this probe a provider that actually
    /// SENDS temperature — `LLMKitManager.makeProbeProvider` strips the no-temperature flag for
    /// exactly this reason; a provider built with the flag active omits the parameter and this
    /// probe would "pass" without measuring anything.
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
            // Only a genuine refusal (a coherent 4xx) that NAMES temperature is a "no". A transport
            // failure, rate limit, missing model, or auth error that merely happens to contain the
            // word "temperature" establishes nothing.
            if !CapabilityProbe.classifyFailure(error).meansNoAnswer, detail.lowercased().contains("temperature") {
                return .established(false, detail, duration: Date().timeIntervalSince(started))
            }
            return .inconclusive(detail, duration: Date().timeIntervalSince(started))
        }
    }

    /// Whether the model reads an image. Sends a coloured shape and asks for both the shape and
    /// the colour. Only `true` when it names BOTH — a guesser lands both about 1-in-12, and a
    /// model that merely accepts the attachment without reading it can't produce either. Colour
    /// alone was too guessable and too easy to fake by describing the payload's existence.
    /// `preferLowImageDetail` requests OpenAI's `image_url.detail: "low"` (flat ~85-token image
    /// bill instead of tile math — one gpt-4-turbo probe billed 8,542 tokens for a 128px shape).
    /// Callers set it ONLY for endpoints documented to accept the field (api.openai.com); and the
    /// grading invariant holds regardless: a failure is never graded from a request that carried
    /// the hint — on any error the probe retries once hint-free and grades that, so a strict
    /// deserializer rejecting OUR field can't fabricate vision=false.
    public static func probeVision(
        llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil,
        preferLowImageDetail: Bool = false
    ) async -> ProbeFinding<Bool> {
        guard let color = ProbeFixtures.namedColors.randomElement(),
              let shape = ProbeFixtures.namedShapes.randomElement() else { return .notAttempted }
        let png = ProbeFixtures.makeShapePNG(shape: shape, red: color.red, green: color.green, blue: color.blue)
        let started = Date()

        func attempt(detail: LLMImageContent.Detail?) async throws -> ProbeFinding<Bool> {
            let image = LLMImageContent(data: png, mimeType: "image/png", detail: detail)
            let attemptStarted = Date()
            calls?.increment()
            let response = try await llm.send(messages: [
                .system("You are a vision test harness. Describe the image in a few words."),
                .user("What shape is in this image, and what colour is it? Answer briefly.", images: [image])
            ], tools: [])
            let text = (response.text ?? "").lowercased()
            // Per-attempt duration: a graded verdict times only its own call, not a discarded
            // detail-hint attempt that failed before it.
            let dur = Date().timeIntervalSince(attemptStarted)
            let (sawColor, sawShape) = gradeVisionAnswer(text, colorName: color.name, shape: shape)
            if sawColor && sawShape {
                return .established(true, "named '\(color.name) \(shape.rawValue)'", duration: dur)
            }
            // Answered without both facts: it accepted the image shape but did not read it, or it
            // can't see. Either way it did not demonstrate vision. Recorded false with what it
            // actually said, so an auditor can tell a hallucination from a partial read.
            let got = [sawColor ? "colour✓" : "colour✗", sawShape ? "shape✓" : "shape✗"].joined(separator: " ")
            return .established(false, "expected '\(color.name) \(shape.rawValue)' — \(got); said '\(text.prefix(50))'", duration: dur)
        }

        if preferLowImageDetail {
            do {
                return try await attempt(detail: .low)
            } catch {
                // Fall through to the hint-free attempt below; this failure is never graded.
            }
        }
        do {
            return try await attempt(detail: nil)
        } catch {
            return attachmentRejection(error, attachment: "image", started: started)
        }
    }

    /// Grades a vision answer by WHOLE WORDS, not substrings. "red" must not match "colo(red)", and
    /// a shape named with a legitimate synonym ("box"/"rectangle" for a square, "round"/"disc" for a
    /// circle) still counts as seen — otherwise a sighted model that phrases it differently would be
    /// recorded `vision=false`, a fabricated negative. Exposed for testing.
    static func gradeVisionAnswer(_ text: String, colorName: String, shape: ProbeFixtures.Shape) -> (sawColor: Bool, sawShape: Bool) {
        let words = Set(text.lowercased().split { !$0.isLetter }.map(String.init))
        return (words.contains(colorName), !words.isDisjoint(with: ProbeFixtures.shapeSynonyms(for: shape)))
    }

    /// Whether the model reads a PDF. Sends a one-page document showing a random code and asks for
    /// it back — text transcription, the hardest signal to fake. `true` only when the code returns.
    /// One call answering BOTH the vision and PDF questions — but graded ONLY when both answers
    /// are correct (the double-positive), which is 83% of models that reach this stage. Any other
    /// outcome — an error, one answer wrong, an ambiguous reply — returns nil and the caller runs
    /// the two separate probes exactly as before. The signature makes graded failure
    /// unrepresentable: `attachmentRejection`'s "the attachment is the only new variable" premise
    /// doesn't hold with two attachments, so failures from this call must never grade.
    ///
    /// The reply is requested as JSON (in the system prompt only — never `response_format`, an
    /// optional knob some endpoints reject) and graded leniently: parse the first JSON object if
    /// present, else fall back to whole-text grading of the same response. JSON non-compliance
    /// can only cost the fallback calls, never a verdict.
    public static func probeVisionAndPDF(
        llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil,
        preferLowImageDetail: Bool = false
    ) async -> (vision: ProbeFinding<Bool>, pdf: ProbeFinding<Bool>)? {
        guard let color = ProbeFixtures.namedColors.randomElement(),
              let shape = ProbeFixtures.namedShapes.randomElement() else { return nil }
        let png = ProbeFixtures.makeShapePNG(shape: shape, red: color.red, green: color.green, blue: color.blue)
        let code = CapabilityProbe.makeIdentifier()
        let pdf = ProbeFixtures.makePDF(code: code)
        let image = LLMImageContent(data: png, mimeType: "image/png",
                                    detail: preferLowImageDetail ? .low : nil)
        let document = LLMDocumentContent(data: pdf, mimeType: "application/pdf", filename: "probe.pdf")
        let started = Date()
        calls?.increment()
        do {
            let response = try await llm.send(messages: [
                .system("""
                You are a multimodal test harness. Reply with ONLY a JSON object in exactly this \
                format, no other text: {"shape": "<shape in the image>", "color": "<its colour>", \
                "pdf_code": "<the code displayed in the PDF>"}
                """),
                .user("Identify the shape and its colour in the attached image, and the code displayed in the attached PDF.",
                      images: [image], documents: [document])
            ], tools: [])
            let raw = response.text ?? ""
            let dur = Date().timeIntervalSince(started)

            // Layer 1: lenient JSON — first {...} block, if any.
            var visionText = raw.lowercased()
            var pdfText = raw
            if let open = raw.firstIndex(of: "{"), let close = raw.lastIndex(of: "}"), open < close,
               let object = try? JSONSerialization.jsonObject(with: Data(raw[open...close].utf8)) as? [String: Any] {
                let shapeField = (object["shape"] as? String ?? "")
                let colorField = (object["color"] as? String ?? "")
                visionText = "\(colorField) \(shapeField)".lowercased()
                pdfText = object["pdf_code"] as? String ?? raw
            }

            let (sawColor, sawShape) = gradeVisionAnswer(visionText, colorName: color.name, shape: shape)
            let sawCode = pdfText.contains(code)
            guard sawColor, sawShape, sawCode else { return nil }
            return (vision: .established(true, "named '\(color.name) \(shape.rawValue)' (combined vision+PDF call)", duration: dur),
                    pdf: .established(true, "returned the embedded code (combined vision+PDF call)", duration: dur))
        } catch {
            // Never graded: with two attachments a rejection cannot be attributed to either.
            return nil
        }
    }

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
            return attachmentRejection(error, attachment: "PDF", started: started)
        }
    }

    /// Grades a failure from an attachment probe (image or PDF) WITHOUT parsing the error text.
    ///
    /// This is the answer to "how do you classify the rejection?": for an attachment, you don't
    /// have to. The vision/PDF probes run only after a plain chat call to the same model has
    /// already succeeded (the driver establishes chat first; a bare chat is what the temperature
    /// probe and tool probe also send). The one thing added to the failing request is the
    /// attachment — so any 4xx is caused by the attachment, and no keyword matching against the
    /// provider's ever-changing error dialect is needed. (glm-5.2 alone rejects images three
    /// different ways across z.ai / Ollama-Cloud / vLLM; matching all of them is a losing game.)
    ///
    /// "No vision" and "wrong image format for this endpoint" both land here as `false`, and that
    /// is correct: the practical fact is that this (provider, model) will not take the attachment
    /// the way we send it. Only a non-4xx failure — timeout, 429, 5xx — is `inconclusive`, because
    /// that says nothing about the attachment.
    private static func attachmentRejection(_ error: any Error, attachment: String, started: Date) -> ProbeFinding<Bool> {
        let dur = Date().timeIntervalSince(started)
        let detail = CapabilityProbe.rejectionDetail(error)
        // The structural grading below assumes plain chat to this model works, so the attachment is
        // the only new variable. If the body says the MODEL is gone or the account can't access it,
        // that assumption is void — the failure is the model/permission, not the attachment.
        // Inconclusive, never a fabricated "no".
        if CapabilityProbe.textIndicatesModelGone(detail) || CapabilityProbe.textIndicatesAccessDenied(detail) {
            return .inconclusive("model unavailable/denied, not a \(attachment) rejection: \(detail)", duration: dur)
        }
        // OpenAI encodes "this model takes no image input at all" as a 429 whose per-minute image
        // budget is literally zero ("on input-images per min: Limit 0, Requested 1"). The blanket
        // 429-is-transient rule left vision permanently unsettleable on such models — the identical
        // request can never succeed against a structural zero quota, so it IS the answer.
        if detail.lowercased().contains("input-images per min: limit 0") {
            return .established(false, "zero image quota — the endpoint accepts no image input: \(detail)", duration: dur)
        }
        switch CapabilityProbe.classifyFailure(error) {
        case .noAnswer, .paymentRequired:
            return .inconclusive(detail, duration: dur)
        case .refusedTools, .refusedOurRequest:
            // A plain chat call to this model works; the \(attachment) is the only added variable.
            return .established(false, "rejected the request with a \(attachment) attached (plain chat works): \(detail)", duration: dur)
        }
    }

    /// The model's real output ceiling, read out of its own rejection. Requests an absurd
    /// `max_tokens`; the endpoint answers with the true limit, which we already parse for the
    /// runtime auto-clamp (``LLMProviderError/reportedMaxOutputTokenLimit``). One call, no binary
    /// search — that's the fallback for endpoints whose error doesn't state the number, and is
    /// deferred until it's actually needed. When the rejection is in a format we don't parse, the
    /// raw body is kept as `inconclusive` evidence so an unrecognised shape is visible, not lost.
    /// The result of a max-output probe: an output CAP, or the discovery that the endpoint has
    /// no independent cap and bounds output only by context length.
    public struct MaxOutputProbeResult: Sendable {
        /// The output ceiling finding. `.inconclusive` when `contextBound` is established.
        public var cap: ProbeFinding<Int>
        /// Established only when output is bounded solely by context length (value = that length).
        public var contextBound: ProbeFinding<Int>?
    }

    /// A searched max-output value within this many tokens of the context window is treated as
    /// context-bound (the search converged to context-minus-input), not a per-model output cap. A
    /// genuine distinct cap sits a round fraction of context BELOW the window (thousands+ of tokens
    /// of gap); the masquerade gap is only the probe's own input size (hundreds of tokens).
    static let contextBoundMargin = 2048

    public static func probeMaxOutputTokens(
        llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil,
        maxContextTokens: Int? = nil, allowBinarySearch: Bool = true
    ) async -> MaxOutputProbeResult {
        let absurd = 100_000_000
        let started = Date()
        calls?.increment()
        do {
            _ = try await llm.send(messages: [
                .user("Reply with the single word: ok")
            ], tools: [], overrides: LLMCallOverrides(maxOutputTokens: absurd))
            // It accepted a preposterous cap. One call can't bound it from above; all we'd know is
            // "at least this", which is never useful. Report inconclusive rather than a fake number.
            return MaxOutputProbeResult(cap: .inconclusive("accepted max_tokens=\(absurd); true ceiling not revealed", duration: Date().timeIntervalSince(started)))
        } catch {
            let dur = Date().timeIntervalSince(started)
            // Preferred: the endpoint stated its ceiling and we parsed it — one call, exact.
            if let limit = (error as? LLMProviderError)?.reportedMaxOutputTokenLimit {
                return MaxOutputProbeResult(cap: .established(limit, "endpoint reported its maximum", duration: dur))
            }
            // A non-4xx failure (or payment-required) says nothing about the cap.
            guard !CapabilityProbe.classifyFailure(error).meansNoAnswer else {
                return MaxOutputProbeResult(cap: .inconclusive(CapabilityProbe.rejectionDetail(error), duration: dur))
            }
            // The rejection may state the ROUTER's absolute max_tokens PARAMETER ceiling (deepinfra:
            // "Input should be less than or equal to 10000000") rather than the model's output cap.
            // Searching against it converges to ~10M on a model that emits a few dozen tokens (the
            // 2026-07-19 audit's 42 :deepinfra records), so don't search — record the real physical
            // bound (context) when known, else inconclusive.
            if let providerError = error as? LLMProviderError,
               case .httpError(_, let body, _, _) = providerError,
               let paramCeiling = LLMProviderError.reportedParameterCeiling(inBody: body) {
                // "Input should be less than or equal to N" is the endpoint's max_tokens PARAMETER
                // limit. If N is below the context window it's a genuine per-model output cap — record
                // it. Only a ceiling at/above context (the router's absolute limit, e.g. deepinfra's
                // 10,000,000) is not an output cap: treat that as context-bound. With no known context
                // a bare large number can't be told from a real cap, so stay inconclusive rather than
                // record a possibly-bogus value.
                if let ctx = maxContextTokens {
                    if paramCeiling < ctx {
                        return MaxOutputProbeResult(cap: .established(paramCeiling, "endpoint enforces max_tokens ≤ \(paramCeiling), below the context window — the model's output cap", duration: dur))
                    }
                    return MaxOutputProbeResult(
                        cap: .inconclusive("max_tokens parameter limit (\(paramCeiling)) is at/above the context window (\(ctx)) — a router ceiling, not an output cap; output is bounded by context", duration: dur),
                        contextBound: .established(ctx, "endpoint enforces only a router max_tokens ceiling ≥ context; output is bounded by its context length", duration: dur))
                }
                return MaxOutputProbeResult(cap: .inconclusive("endpoint caps the max_tokens parameter at \(paramCeiling); with no known context window it can't be told from a genuine output cap, so it is not recorded", duration: dur))
            }
            // It rejected the absurd cap but didn't state the exact output limit in a form we
            // parse. Find the ceiling empirically — the same "max_tokens is the only variable"
            // logic the attachment probes use, applied as a search.
            guard allowBinarySearch else {
                return MaxOutputProbeResult(cap: .inconclusive("rejected max_tokens=\(absurd) without a parseable limit: \(CapabilityProbe.rejectionDetail(error))", duration: dur))
            }
            // Shrink the search when the same error reveals the top of a max_tokens RANGE —
            // [512, hint] instead of [512, 100M] is a handful of calls, not ~24.
            var knownBad = absurd
            if let providerError = error as? LLMProviderError,
               case .httpError(_, let body, _, _) = providerError {
                if let hint = LLMProviderError.reportedLimitHint(inBody: body), hint > 512, hint < absurd {
                    knownBad = hint + 1   // the bound itself may be rejectable; +1 keeps it a known-bad
                } else if let contextBound = LLMProviderError.reportedContextLengthBound(inBody: body) {
                    // The endpoint bounds max_tokens only by CONTEXT length (gpt-4, OpenRouter,
                    // the HuggingFace router). A search would "converge" to context-minus-input —
                    // the audit caught HF accepting max_tokens=4276225 against a 4.29M context —
                    // an artifact. Record it as an ESTABLISHED context-bound outcome: there is no
                    // output cap to measure, but "output is bounded by this context length" is a
                    // real, useful fact (drives context-based validation and clamping downstream),
                    // and recording it established stops every future sweep re-buying this call.
                    return MaxOutputProbeResult(
                        cap: .inconclusive(
                            "output bounded only by context length (\(contextBound)); no independent output cap",
                            duration: Date().timeIntervalSince(started)),
                        contextBound: .established(contextBound,
                            "endpoint bounds max_tokens only by its context length",
                            duration: Date().timeIntervalSince(started)))
                }
            }
            let searchResult = await binarySearchMaxOutput(llm: llm, knownGood: 512, knownBad: knownBad, calls: calls, started: started)
            // Sanity net for routers that reject WITHOUT a parseable message (nscale's "Failed to
            // process request", together's "Input validation error"): a searched value at or within an
            // input-sized margin of the context window is the endpoint bounding output by CONTEXT, not
            // a per-model cap — the search converged to context-minus-input. Record it context-bound
            // rather than a fabricated (often impossible, output>context) output ceiling. Only the
            // SEARCHED path is guarded; an endpoint-STATED exact limit is trusted over a possibly-stale
            // decoded context window.
            if let ctx = maxContextTokens, let capValue = searchResult.value {
                // Proportional margin so small context windows aren't over-guarded: a fixed 2048 would
                // discard a genuine half-of-context cap on a 4096-token model. The masquerade gap is
                // the probe's small input; a real cap sits a large round fraction below context.
                let margin = min(Self.contextBoundMargin, ctx / 8)
                if capValue >= ctx - margin {
                    return MaxOutputProbeResult(
                        cap: .inconclusive("search reached the context window (\(capValue) vs context \(ctx)); no independent output cap", duration: Date().timeIntervalSince(started)),
                        contextBound: .established(ctx, "output bounded by context length (search converged to the context ceiling)", duration: Date().timeIntervalSince(started)))
                }
            }
            return MaxOutputProbeResult(cap: searchResult)
        }
    }

    /// Finds the largest `max_tokens` the endpoint accepts when it won't state its limit in a
    /// parseable form, by binary search over `[knownGood, knownBad]`.
    ///
    /// No error text is parsed. The invariant is structural: `knownGood` accepts and `knownBad`
    /// rejects, and `max_tokens` is the ONLY thing changing between calls — so at each step a 200
    /// means "ceiling ≥ mid" and any 4xx means "ceiling < mid". A non-4xx failure aborts (it says
    /// nothing about the cap).
    ///
    /// A real binary search: each step probes the MIDPOINT of the live interval and discards the
    /// half that can't contain the ceiling. From `[512, 100M]` the first probe is ~50M, then ~25M,
    /// ~12.5M, and so on down — the interval halves every call. (Tightening `knownBad` with a
    /// parsed hint first, as the caller does, cuts the wasted steps through the empty upper range.)
    ///
    /// Stops when the interval is within `tolerance` of the ceiling — the value feeds a clamp, so
    /// a handful of tokens' slack isn't worth extra calls — or at the step cap. Returns the largest
    /// accepted value, never above the true ceiling (the safe direction for a clamp).
    ///
    /// Only reached on the rare provider that rejects without stating the number, so its cost
    /// lands there, not on Anthropic/OpenAI (one call each). Disable with `allowBinarySearch:
    /// false` on ``probeMaxOutputTokens(llm:modelID:calls:allowBinarySearch:)``.
    static func binarySearchMaxOutput(
        llm: any LLMProvider, knownGood: Int, knownBad: Int, calls: ProbeCallCounter?, started: Date,
        stepCap: Int = 24
    ) async -> ProbeFinding<Int> {
        var low = knownGood, high = knownBad, steps = 0

        func accepts(_ value: Int) async -> Bool? {   // nil = transport failure, abort
            steps += 1
            calls?.increment()
            do {
                _ = try await llm.send(messages: [.user("Reply with the single word: ok")],
                                       tools: [], overrides: LLMCallOverrides(maxOutputTokens: value))
                return true
            } catch {
                return CapabilityProbe.classifyFailure(error).meansNoAnswer ? nil : false
            }
        }

        // Verify the assumed floor before trusting it. Every value the search later reports is one
        // it saw accepted — EXCEPT this initial `low`, which no midpoint ever re-tests (mid is always
        // > low). If the model's real ceiling is below `knownGood`, returning `low` would be a value
        // ABOVE the true ceiling — the one thing a clamp must never do (it 400s in production). So if
        // the floor itself is rejected, we have no verified-accepted value and report inconclusive.
        switch await accepts(low) {
        case .some(true): break
        case .some(false):
            return .inconclusive("rejected even the floor max_tokens=\(low); ceiling is below it, not established",
                                 duration: Date().timeIntervalSince(started))
        case .none:
            return .inconclusive("transport failure verifying the floor max_tokens=\(low)",
                                 duration: Date().timeIntervalSince(started))
        }

        // "Close enough" once the remaining gap is under ~0.5% of the live floor (min 2), since the
        // result only clamps an output cap. Converges to the exact integer for small ceilings and
        // stops a few calls early for large ones. This proportional rule also IS the "don't refine
        // below ~500 tokens" behavior for large ceilings — low/200 ≥ 500 exactly once low ≥ 100k —
        // so big models stop with ~500-token slack while small models still converge exactly.
        while high - low > max(2, low / 200) && steps < stepCap {
            let mid = low + (high - low) / 2
            switch await accepts(mid) {
            case .some(true):  low = mid          // ceiling ≥ mid → search the upper half
            case .some(false): high = mid         // ceiling < mid → search the lower half
            case .none:
                // A transport failure (429/timeout) interrupted the next probe. If the search had
                // already narrowed to within ~1% (or 500 tokens) of the ceiling, the latest
                // accepted value is a usable answer — don't discard it as inconclusive just because
                // the endpoint rate-limited us one step from done. A search still far from
                // converging stays inconclusive.
                if high - low <= max(500, low / 100) {
                    return .established(low, "converged to \(low) before a transport interruption (\(steps) calls, gap \(high - low))",
                                        duration: Date().timeIntervalSince(started))
                }
                return .inconclusive("binary search interrupted after \(steps) steps (narrowed to \(low)–\(high))",
                                     duration: Date().timeIntervalSince(started))
            }
        }
        return .established(low, "largest accepted max_tokens by binary search (\(steps) calls, within \(high - low))",
                            duration: Date().timeIntervalSince(started))
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
            // A genuine refusal (coherent 4xx) naming effort/reasoning is a "no"; a transport/rate-
            // limit/auth/missing-model error that happens to mention "reasoning" establishes nothing.
            if !CapabilityProbe.classifyFailure(error).meansNoAnswer,
               detail.lowercased().contains("effort") || detail.lowercased().contains("reasoning") {
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
    private static func finding(fromError error: any Error, capabilityKeywords: [String], started: Date) -> ProbeFinding<Bool> {
        let dur = Date().timeIntervalSince(started)
        let detail = CapabilityProbe.rejectionDetail(error)
        // A model that is gone (404 "no longer available") or that this account can't access hasn't
        // refused our request on its merits — it isn't reachable. For chat that means inconclusive,
        // not "not a chat model."
        if CapabilityProbe.textIndicatesModelGone(detail) || CapabilityProbe.textIndicatesAccessDenied(detail) {
            return .inconclusive(detail, duration: dur)
        }
        // "This is not a chat model ... not supported in the v1/chat/completions endpoint" is the
        // definitive answer to the chat question, but OpenAI ships it as a 404, which the status
        // gate below reads as unreachable — the audit found babbage/davinci/instruct/tts/transcribe
        // models establishing nothing and re-probing forever. The phrase is a statement about the
        // model (the endpoint recognized it), so it settles chat=false on any status.
        if capabilityKeywords.isEmpty, CapabilityProbe.textIndicatesNotAChatModel(detail) {
            return .established(false, detail, duration: dur)
        }
        switch CapabilityProbe.classifyFailure(error) {
        case .noAnswer, .paymentRequired:
            return .inconclusive(detail, duration: dur)
        case .refusedTools, .refusedOurRequest:
            // Chat (no keywords): any 4xx that got here is the endpoint declining a plain request,
            // which for chat means "not a usable chat endpoint" — EXCEPT a generic request-VALIDATION
            // rejection, which is about the request SHAPE, not the model's chat capability. A safety
            // classifier (Llama-Guard) is reachable via chat/completions but 400s a plain nonce-echo
            // with "Input validation error"; stay inconclusive rather than fabricate "not a chat model".
            guard !capabilityKeywords.isEmpty else {
                return CapabilityProbe.textIndicatesRequestValidationOnly(detail)
                    ? .inconclusive(detail, duration: dur)
                    : .established(false, detail, duration: dur)
            }
            // A 400 whose text names the capability ("image", "pdf", "file content is not
            // supported") is the endpoint saying it can't. A 400 about something else says
            // nothing about the capability under test, so it stays inconclusive.
            let lowered = detail.lowercased()
            return capabilityKeywords.contains(where: { lowered.contains($0) })
                ? .established(false, detail, duration: dur)
                : .inconclusive(detail, duration: dur)
        }
    }
}
