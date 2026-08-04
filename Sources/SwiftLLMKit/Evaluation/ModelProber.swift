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
    ///
    /// v4: the trailing-system-turn probe changed shape — it now sends the REAL production request
    /// (a base system prompt PLUS a trailing `{role: system}` turn) through the provider's own
    /// consumer with `supportsTrailingSystemMessage` forced on, instead of forcing a raw body with no
    /// base system. v3 `trailingSystemMessage` findings were measured in that isolated shape and are
    /// suspect; the caller re-probes JUST that finding when it reuses a v3 record (no full re-sweep).
    public static let proberVersion = 4

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
        if let value = facts.capabilities.chat { profile.chat = .decoded(value, evidence) }
        if let value = facts.capabilities.toolUse { profile.toolCalling = .decoded(value, evidence) }
        if let value = facts.capabilities.vision { profile.vision = .decoded(value, evidence) }
        if let value = facts.capabilities.pdfInput { profile.pdfInput = .decoded(value, evidence) }
        seedEffort(facts.generalEffort, into: &profile.generalEffortLevels, evidence: evidence)
        seedEffort(facts.reasoningEffort, into: &profile.reasoningEffortLevels, evidence: evidence)
        // Listed in /models ⇒ presumed reachable, but only presumed — a live probe can overturn it.
        profile.isAvailable = .decoded(true, "present in provider /models listing")
        return profile
    }

    /// Seeds one effort ladder from a decoded ``EffortSupport``.
    ///
    /// A KNOWN ladder is the vendor enumerating THE valid set, so every level it omits is a stated
    /// "no" — without that, vendor denials (xhigh on sonnet-4-6, everything on haiku-4-5) were
    /// indistinguishable from "nobody asked" and the prober re-probed levels already answered.
    /// ``EffortSupport/unsupported`` states "no" for every level; `supportedLevelsUnknown` states
    /// nothing, leaving the whole ladder to the probe.
    private static func seedEffort(_ support: EffortSupport?,
                                   into ladder: inout [String: ProbeFinding<Bool>],
                                   evidence: String) {
        guard let support else { return }
        switch support {
        case .supportedLevelsUnknown:
            return
        case .unsupported:
            for level in EffortRank.table.keys {
                ladder[level] = .decoded(false, evidence + " (the model states no effort levels)")
            }
        case .levels(let levels):
            for level in levels { ladder[level] = .decoded(true, evidence) }
            for level in EffortRank.table.keys where !levels.contains(level) {
                ladder[level] = .decoded(false, evidence + " (not in the model's stated effort set)")
            }
        }
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
        seedEffort(info.generalEffort, into: &profile.generalEffortLevels, evidence: "provider /models payload")
        seedEffort(info.reasoningEffort, into: &profile.reasoningEffortLevels, evidence: "provider /models payload")

        switch apiType {
        case .anthropic:
            // Anthropic's capabilities block states supported: true/false explicitly, so both
            // directions are decodable — unlike decoders where false just means "didn't say".
            profile.vision = .decoded(info.capabilities.vision, "capabilities.image_input")
            profile.pdfInput = .decoded(info.capabilities.pdfInput, "capabilities.pdf_input")
            if let chat = info.capabilities.state(of: .chat) {
                profile.chat = .decoded(chat, "provider /models payload")
            }
        case .ollama:
            if info.capabilities.toolUse {
                profile.toolCalling = .decoded(true, "tags payload lists 'tools'")
            }
        case .gemini:
            if let chat = info.capabilities.state(of: .chat) {
                profile.chat = .decoded(chat, "supportedGenerationMethods")
            }
        case .mistral:
            // Mistral's capabilities block states each flag explicitly (true/false), like
            // Anthropic's — so both directions are decodable, including tool calling
            // (`function_calling`), which almost no other vendor publishes.
            if let chat = info.capabilities.state(of: .chat) {
                profile.chat = .decoded(chat, "capabilities.completion_chat")
            }
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
    ///     These measure GENERAL effort and are attempted only when the parameter below says the
    ///     endpoint actually emits it.
    ///   - modelCapabilities: the model's catalog record — the same one the provider gates on.
    ///     Without it the tool-calling probe cannot tell a forced call from a free one, because a
    ///     suppressed `tool_choice` looks exactly like a successful forced request.
    ///   - supportsUnconditionalGeneralEffortEmission: whether this endpoint puts the general
    ///     effort field on the wire whether or not the model is known to accept it (Anthropic
    ///     does; flag-gated endpoints do not). Defaults to `false` — the SAFE answer, because on a
    ///     gated endpoint the field is silently dropped and a "no error" would be recorded as a
    ///     false positive. This guard used to live in the caller, where it could be forgotten.
    ///     Only meaningful where the provider emits the field unconditionally (Anthropic); where
    ///     emission is flag-gated (OpenAI-compatible), build per-level providers whose
    ///     configuration forces the field via `extraJSONOverrides` and use
    ///     ``probeParameterAcceptance`` instead.
    public static func probe(
        llm: any LLMProvider,
        seed: ModelProfile,
        effortLevelsToProbe: [String] = [],
        supportsUnconditionalGeneralEffortEmission: Bool = false,
        preferLowImageDetail: Bool = false,
        modelCapabilities: ModelCapabilities = ModelCapabilities()
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
            // The capabilities the PROVIDER will gate on, so `toolChoiceForced` records whether
            // `tool_choice` actually went out rather than assuming it did.
            let toolResult = await CapabilityProbe.probeToolCalling(
                llm: llm, providerID: profile.providerID, modelID: modelID,
                capabilities: modelCapabilities, calls: calls
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
        // GENERAL effort only. `probeEffortLevel` sends the configuration's general effort, which
        // Anthropic emits unconditionally — on a flag-gated endpoint an unsupported model silently
        // drops the field and a "no error" would record a false positive. Callers used to have to
        // know that and pass [] for non-Anthropic; the guard now lives here, where it cannot be
        // forgotten. Reasoning ladders are measured by forcing the raw parameter instead.
        if supportsUnconditionalGeneralEffortEmission {
            for level in effortLevelsToProbe where profile.generalEffortLevels[level] == nil {
                profile.generalEffortLevels[level] = await probeEffortLevel(level, llm: llm, modelID: modelID, calls: calls)
            }
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

    // MARK: - Trailing system turn

    /// One trailing-system-turn test: the messages to SEND and the nonce that grades the answer.
    ///
    /// The two travel together because they are meaningless apart — the nonce is embedded in the
    /// trailing turn, so a caller that built one without the other would be grading against a code the
    /// model was never given.
    public struct TrailingSystemTurnTest: Sendable {
        /// A base system prompt, a user turn, then a trailing system turn carrying the nonce — the
        /// exact production shape. Sent THROUGH the provider (not as a forced raw body): with
        /// ``BehaviorFlags/supportsTrailingSystemMessage`` on, the provider's own consumer keeps the
        /// trailing turn in place.
        public let messages: [LLMMessage]
        /// The code the trailing system turn carries; an echo is the proof it was read.
        public let nonce: String
    }

    /// Builds the messages for a trailing `{role: system}` steering turn, in the REAL shape we send.
    ///
    /// **Probe what we actually send.** Earlier versions forced a raw `messages` body to bypass the
    /// provider's system-hoisting, which measured the endpoint in isolation — a system turn alone,
    /// with NO base system prompt. Production never looks like that: there is always a base system
    /// prompt (top-level `system` on Anthropic, `messages[0]` on OpenAI-compatible / Ollama) AND the
    /// trailing turn. Now that ``BehaviorFlags/supportsTrailingSystemMessage`` exists, the probe
    /// sends these messages THROUGH the provider's own path with the flag forced on, so the wire
    /// shape is exactly production — base system included — and a pass means our real request works,
    /// not that an artificial one does. A provider whose consumer does NOT emit a trailing turn
    /// (Gemini folds it into `systemInstruction`) must not be probed with this — it would echo the
    /// nonce from the top and fabricate a pass; the caller gates that out by `apiType`.
    ///
    /// **Say plainly that this is a capability test**, exactly as ``CapabilityProbe/makeProbeTool``
    /// and the chat probe do: a nonce to echo, and a sentence saying why. The first version dressed
    /// it up as an access-code extraction and models refused it (measured 2026-07-26: `claude-fable-5`
    /// returned `stop_reason: refusal`, `claude-opus-4-8` declined, `claude-sonnet-5` flagged it
    /// suspicious) — three of four supporting models scored as failures. Don't dress a probe up.
    ///
    /// The trailing turn follows a `user` turn and is the last entry — the one shape our consumer
    /// keeps in place and the wire accepts — so a rejection is about model support, not a malformed
    /// request.
    public static func makeTrailingSystemTurnTest() -> TrailingSystemTurnTest {
        let nonce = CapabilityProbe.makeIdentifier()
        let messages: [LLMMessage] = [
            .system("You are a helpful assistant taking part in an automated capability test."),
            .user(
                "We are testing whether you can read a system instruction placed after this "
                + "message. Please follow the instruction that follows."
            ),
            .system(
                "You are in an automated capability test confirming this model can read a system "
                + "instruction placed after the user's message, alongside the base system prompt. "
                + "Reply with exactly this identifier and nothing else: \(nonce)"
            )
        ]
        return TrailingSystemTurnTest(messages: messages, nonce: nonce)
    }

    /// Whether the endpoint accepts a trailing `{"role": "system"}` steering turn AND acts on it.
    ///
    /// Hand this a provider built with ``BehaviorFlags/supportsTrailingSystemMessage`` forced on and
    /// ``makeTrailingSystemTurnTest``'s messages: the provider's own consumer keeps the trailing turn
    /// in place, so this measures the real production shape (base system prompt + trailing turn).
    ///
    /// Grading is deliberately three-way. An echoed nonce is `established(true)`: the model could
    /// not have known the code unless it read a turn that only exists in trailing position, so this
    /// establishes *honored*, not merely *tolerated*. A rejection naming the unsupported role is
    /// `established(false)`. Everything else — a 200 with no echo, a 429, an unrelated 4xx — is
    /// inconclusive, because "accepted but ignored" cannot be told from "the model rambled" in one
    /// call, and a refusal we provoked is not evidence about the model.
    public static func probeTrailingSystemTurn(
        llm: any LLMProvider,
        test: TrailingSystemTurnTest,
        modelID: String,
        calls: ProbeCallCounter? = nil
    ) async -> ProbeFinding<Bool> {
        let started = Date()
        calls?.increment()
        do {
            let response = try await llm.send(messages: test.messages, tools: [])
            let body = response.text ?? ""
            let duration = Date().timeIntervalSince(started)
            if body.contains(test.nonce) {
                return .established(true, "echoed the code carried by the trailing system turn", duration: duration)
            }
            return .inconclusive(
                "accepted the trailing system turn but did not echo its code (\(body.prefix(60)))",
                duration: duration
            )
        } catch {
            let detail = CapabilityProbe.rejectionDetail(error)
            let lowered = detail.lowercased()
            if !CapabilityProbe.classifyFailure(error).meansNoAnswer,
               trailingSystemRejectionKeywords.contains(where: { lowered.contains($0) }) {
                return .established(false, detail, duration: Date().timeIntervalSince(started))
            }
            return .inconclusive(detail, duration: Date().timeIntervalSince(started))
        }
    }

    /// Phrases that mark a refusal as "this model does not support the role", as opposed to one
    /// about placement or anything else we might have got wrong. Deliberately narrow: matching bare
    /// `system` or `role` would swallow a malformed-request 400 and record it as a measured no.
    private static let trailingSystemRejectionKeywords = [
        "not supported on this model",
        "role 'system'",
        "role \"system\""
    ]

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
            .system("You are in an automated capability test confirming this model can hold a chat exchange. Reply with exactly what is asked and nothing else."),
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
                return (finding(fromError: error, started: started),
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
                .system("You are in an automated capability test confirming this model can hold a chat exchange. Reply with exactly what is asked and nothing else."),
                .user("Reply with exactly this identifier and nothing else: \(nonce)")
            ], tools: [])
            return chatFinding(response.text, nonce: nonce, started: started)
        } catch {
            return finding(fromError: error, started: started)
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
                .system("You are in an automated capability test confirming this model can read images. Describe the image in a few words."),
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
        // Lowercase the color operand too so both sides share one normalization (fixtures use
        // single-word lowercase names; a mixed-case fixture would otherwise never match).
        return (words.contains(colorName.lowercased()), !words.isDisjoint(with: ProbeFixtures.shapeSynonyms(for: shape)))
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
                You are in an automated capability test confirming this model can read an image and a PDF in \
                one request. Reply with ONLY a JSON object in exactly this \
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
                .system("You are in an automated capability test confirming this model can read PDF documents. Reply with only the code you are asked for."),
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

    /// Whether a named GENERAL effort level is accepted.
    ///
    /// Sends the configuration's general effort, so it is only meaningful on an endpoint that emits
    /// that field unconditionally (Anthropic). On a flag-gated endpoint the field is silently
    /// dropped and a "no error" here would be a FALSE POSITIVE — which is why `probe(...)` will not
    /// call this unless the caller states the endpoint emits unconditionally. Reasoning ladders are
    /// measured by forcing the raw `reasoning_effort` parameter instead, which cannot be dropped.
    public static func probeEffortLevel(_ level: String, llm: any LLMProvider, modelID: String, calls: ProbeCallCounter? = nil) async -> ProbeFinding<Bool> {
        let started = Date()
        calls?.increment()
        do {
            _ = try await llm.send(messages: [
                .user("Reply with the single word: ok")
            ], tools: [], overrides: LLMCallOverrides(effort: level))
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


    /// Asks in PROSE for a different shape than the forced format demands.
    ///
    /// This disagreement is the whole test. A prompt that asks for the shape it then grades cannot
    /// distinguish "the endpoint held the model to `response_format`" from "the model followed
    /// instructions", and an endpoint that IGNORES the field passes it. Here, obeying the words
    /// produces prose; obeying the format produces JSON. Only the latter is evidence.

    // MARK: - Structured output

    /// Whether the model actually HONORS a structured-output request.
    ///
    /// Graded on the RESPONSE, not on acceptance — and that distinction is the whole point.
    /// `probeParameterAcceptance` discards the body and calls any 200 a pass, which for
    /// `response_format` records every endpoint that silently IGNORES the field as supporting it.
    /// An ignored `response_format` is indistinguishable from an honored one at the status-code
    /// level and completely distinguishable one line deeper, so this probe reads the text and
    /// parses it.
    ///
    /// The prompt asks for something whose correct answer is unambiguous, so a model that answers
    /// in prose fails on the parse rather than on the content.
    /// Returns `nil` where the provider family has no `response_format` field — forcing one there
    /// measures nothing and spends a paid call on a rejection that says nothing about structured
    /// output.
    public static func probeStructuredOutput(
        _ mode: LLMResponseFormat,
        apiType: ProviderAPIType,
        makeProviderForcing: @Sendable ([String: AnyCodable]) async -> any LLMProvider,
        calls: ProbeCallCounter? = nil
    ) async -> ProbeFinding<Bool>? {
        guard LLMResponseFormat.isSupportedWireField(for: apiType) else { return nil }
        let started = Date()
        calls?.increment()
        do {
            // FORCED, not sent through `LLMCallOverrides`: production emission is gated on the very
            // capability this is establishing, so on an unknown it would send no `response_format`
            // at all — and then grade a model that merely followed the prompt's wording as
            // supporting it. Every probe in this file forces for the same reason.
            let response = try await makeProviderForcing(["response_format": mode.forcedWireValue])
                .send(messages: [.user("Answer in one short English sentence: what colour is the sky?")],
                      tools: [])
            let text = response.text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard let data = text.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                // A 200 whose body is not a JSON object means the field was accepted and ignored.
                return .established(false, "followed the prose and returned non-JSON, so response_format was ignored: \(text.prefix(120))",
                                    duration: Date().timeIntervalSince(started))
            }
            // A JSON OBJECT when the prose asked for an English sentence: the only thing that
            // could have produced it is the endpoint enforcing the format.
            _ = object
            return .established(true, "returned JSON though the prompt asked for prose — the format was enforced",
                                duration: Date().timeIntervalSince(started))
        } catch {
            let detail = CapabilityProbe.rejectionDetail(error)
            let lowered = detail.lowercased()
            if !CapabilityProbe.classifyFailure(error).meansNoAnswer,
               lowered.contains("response_format") || lowered.contains("json_schema")
                || lowered.contains("json_object") || lowered.contains("schema") {
                return .established(false, detail, duration: Date().timeIntervalSince(started))
            }
            return .inconclusive(detail, duration: Date().timeIntervalSince(started))
        }
    }

    // MARK: - tool_choice options

    /// Whether one `tool_choice` option is accepted.
    ///
    /// Acceptance-graded on purpose, unlike structured output: an endpoint that ignored
    /// `tool_choice: required` would still have to return SOMETHING, and grading "did it really
    /// force a call" against a model free to answer directly produces false negatives on exactly
    /// the models that behave best. A refusal naming the parameter is the reliable signal.
    /// Returns `nil` when this provider family has no `tool_choice` field, so there is nothing to
    /// measure and no call is spent.
    ///
    /// The wire shape is derived from `apiType` rather than accepted from the caller. A probe that
    /// can be handed a shape is a probe that can be handed the WRONG one: forcing OpenAI's bare
    /// `"required"` at Anthropic — which requires `{"type": "any"}` — is rejected for the shape,
    /// the rejection names `tool_choice`, and it would be recorded as "this model cannot force a
    /// tool call". Flatly wrong for Claude, and it exports as shipped data.
    public static func probeToolChoice(
        _ choice: LLMToolChoice,
        apiType: ProviderAPIType,
        makeProviderForcing: @Sendable ([String: AnyCodable]) async -> any LLMProvider,
        calls: ProbeCallCounter? = nil
    ) async -> ProbeFinding<Bool>? {
        guard let forcedWireValue = choice.wireValue(for: apiType) else { return nil }
        let started = Date()
        calls?.increment()
        do {
            // FORCED: production emission is gated per option, so re-probing an option already
            // recorded false would send nothing and grade the ordinary success as support.
            _ = try await makeProviderForcing(["tool_choice": forcedWireValue])
                .send(messages: [.user("Reply with the single word: ok")],
                      tools: [CapabilityProbe.makeProbeTool()])
            return .established(true, "accepted tool_choice", duration: Date().timeIntervalSince(started))
        } catch {
            let detail = CapabilityProbe.rejectionDetail(error)
            if !CapabilityProbe.classifyFailure(error).meansNoAnswer,
               detail.lowercased().contains("tool_choice") || detail.lowercased().contains("tool choice") {
                return .established(false, detail, duration: Date().timeIntervalSince(started))
            }
            return .inconclusive(detail, duration: Date().timeIntervalSince(started))
        }
    }


    // MARK: - Thinking budget range

    /// The largest reasoning token budget the endpoint accepts.
    ///
    /// Replaces the hardcoded 1024 floor (see ``ThinkingBudget``) with a measured per-model fact.
    ///
    /// **The ceiling is derived, never a constant.** When the budget is drawn from the output
    /// allowance (Anthropic: `max_tokens` must EXCEED `budget_tokens`) a value above that allowance
    /// is unreachable by construction, so searching past it buys only refusals that say nothing
    /// about the budget. When the budget is a separate allowance the context window is the physical
    /// bound. ``ThinkingBudgetAccounting/searchCeiling(maxOutputTokens:maxContextTokens:)`` picks
    /// between them, and when neither limit is known this probe declines rather than inventing one.
    ///
    /// **An accepted absurd value reports `inconclusive`, not a number** — the same rule
    /// ``probeMaxOutputTokens(llm:modelID:calls:maxContextTokens:allowBinarySearch:)`` follows. One
    /// accepting call bounds nothing from above, and "at least this" was never useful.
    ///
    /// **The caller's factory must pair `max_tokens` with each budget.** Anthropic requires
    /// `max_tokens > budget_tokens`, and the provider's clamp only fires when a per-call max-output
    /// override is present — which this probe does not supply. Without pairing, the first attempt
    /// at the full allowance violates the constraint and the search converges on the PAIRING
    /// boundary, recording it as the model's intrinsic budget ceiling. `pairedMaxOutputTokens`
    /// below is passed to the factory so the caller can honour this.
    public static func probeThinkingBudgetRange(
        llm: any LLMProvider,
        modelID: String,
        accounting: ThinkingBudgetAccounting?,
        maxOutputTokens: Int?,
        maxContextTokens: Int?,
        makeProviderWithBudget: @Sendable (_ budget: Int, _ pairedMaxOutputTokens: Int) async -> any LLMProvider,
        calls: ProbeCallCounter? = nil
    ) async -> ProbeFinding<Int> {
        let started = Date()

        /// One attempt: does the endpoint accept this budget? `nil` = it said nothing usable, which
        /// must abort the search rather than be read as a rejection and halve the range.
        func accepts(_ budget: Int) async -> Bool? {
            calls?.increment()
            do {
                // Always above the budget, so a refusal is about the BUDGET, not the pairing.
                _ = try await makeProviderWithBudget(budget, budget + ThinkingBudget.minimumTokens)
                    .send(messages: [.user("Reply with the single word: ok")], tools: [])
                return true
            } catch {
                if CapabilityProbe.classifyFailure(error).meansNoAnswer { return nil }
                return false
            }
        }

        // Reserve room for the pairing when the budget is drawn from the output allowance: probing
        // AT `maxOutputTokens` needs a `max_tokens` ABOVE it, so the refusal would be about the
        // output cap and the search would converge on `maxOutputTokens - floor` every time.
        let effectiveAccounting = accounting ?? .drawnFromMaxOutputTokens
        let rawCeiling = effectiveAccounting.searchCeiling(maxOutputTokens: maxOutputTokens,
                                                           maxContextTokens: maxContextTokens)
        guard let ceiling = rawCeiling.map({
                  effectiveAccounting == .drawnFromMaxOutputTokens ? $0 - ThinkingBudget.minimumTokens : $0
              }),
              ceiling > ThinkingBudget.minimumTokens else {
            return .inconclusive(
                "no known output or context limit to bound the search — a ceiling here would be invented",
                duration: Date().timeIntervalSince(started))
        }

        // If the whole allowance is accepted there is no ceiling to find inside it.
        switch await accepts(ceiling) {
        case true:
            return .established(ceiling, "accepted a budget at the model's full \(ceiling)-token allowance",
                                duration: Date().timeIntervalSince(started))
        case nil:
            return .inconclusive("the endpoint gave no usable answer at \(ceiling)",
                                 duration: Date().timeIntervalSince(started))
        case false:
            break
        }

        // Binary search the accepted/rejected boundary. `low` is always known-good.
        var low = ThinkingBudget.minimumTokens
        var high = ceiling
        switch await accepts(low) {
        case false:
            return .established(0, "rejected even the minimum budget of \(low) — no usable budget range",
                                duration: Date().timeIntervalSince(started))
        case nil:
            return .inconclusive("the endpoint gave no usable answer at the \(low)-token minimum",
                                 duration: Date().timeIntervalSince(started))
        case true:
            break
        }

        // Converge to within 1024 tokens: finer than the floor itself is precision the caller
        // cannot use, and each extra step is a live API call.
        while high - low > ThinkingBudget.minimumTokens {
            let midpoint = low + (high - low) / 2
            switch await accepts(midpoint) {
            case true: low = midpoint
            case false: high = midpoint
            case nil:
                return .established(low, "largest accepted budget before the endpoint stopped answering (≥ \(low))",
                                    duration: Date().timeIntervalSince(started))
            }
        }
        return .established(low, "largest accepted budget, converged within \(ThinkingBudget.minimumTokens) tokens",
                            duration: Date().timeIntervalSince(started))
    }


    // MARK: - Reasoning toggling, keep, strict tools

    /// Whether reasoning can be switched to a given state.
    ///
    /// The two directions are probed SEPARATELY and neither implies the other: Kimi documents
    /// models that can be switched on but never off, and thinking-only models that reject being
    /// switched off while accepting being switched on.
    ///
    /// Forces the raw `thinking` object rather than going through `LLMCallOverrides`, because the
    /// production path is gated on the very capability this is trying to establish — asking through
    /// it would send nothing and grade the silence as success.
    public static func probeReasoningToggle(
        enabled: Bool,
        makeProviderForcing: @Sendable ([String: AnyCodable]) async -> any LLMProvider,
        calls: ProbeCallCounter? = nil
    ) async -> ProbeFinding<Bool> {
        await probeForcedField(
            ["thinking": .dictionary(["type": .string(enabled ? "enabled" : "disabled")])],
            description: "thinking.type=\(enabled ? "enabled" : "disabled")",
            rejectionKeywords: ["thinking", "reasoning"],
            makeProviderForcing: makeProviderForcing, calls: calls)
    }

    /// Whether `thinking.keep: "all"` is accepted — retaining reasoning content across turns.
    /// Returns `nil` unless the model's mechanism actually HAS a `keep` key.
    ///
    /// Acceptance-graded, so an endpoint that merely ignores unknown body keys would record
    /// `true` — a false vendor fact written into the catalog and the shipped seed corpus. `keep`
    /// is a key of the `thinking` object, so only ``ReasoningControl/thinkingBlock`` can answer.
    public static func probeThinkingKeep(
        reasoningControl: ReasoningControl?,
        makeProviderForcing: @Sendable ([String: AnyCodable]) async -> any LLMProvider,
        calls: ProbeCallCounter? = nil
    ) async -> ProbeFinding<Bool>? {
        guard reasoningControl == .thinkingBlock else { return nil }
        return await probeForcedField(
            ["thinking": .dictionary(["type": .string("enabled"), "keep": .string("all")])],
            description: "thinking.keep=all",
            rejectionKeywords: ["keep", "thinking"],
            makeProviderForcing: makeProviderForcing, calls: calls)
    }

    /// Whether `strict: true` is accepted on a function definition.
    ///
    /// Replaces the WHOLE `tools` array: `mergeJSONOverrides` deep-merges dictionaries but replaces
    /// arrays outright, so reaching a key inside one means supplying the array entire. Strict mode
    /// also constrains the schema itself (no open-ended objects), so the probe tool is declared
    /// with an explicit closed schema — otherwise a rejection would be about the SCHEMA rather than
    /// about `strict`, and would be recorded as the wrong fact.
    public static func probeStrictToolDefinitions(
        makeProviderForcing: @Sendable ([String: AnyCodable]) async -> any LLMProvider,
        calls: ProbeCallCounter? = nil
    ) async -> ProbeFinding<Bool> {
        let strictTool: AnyCodable = .dictionary([
            "type": .string("function"),
            "function": .dictionary([
                "name": .string(CapabilityProbe.probeToolName),
                "description": .string("Returns the test identifier string."),
                "strict": .bool(true),
                "parameters": .dictionary([
                    "type": .string("object"),
                    "properties": .dictionary([:]),
                    "required": .array([]),
                    "additionalProperties": .bool(false)
                ])
            ])
        ])
        return await probeForcedField(
            ["tools": .array([strictTool])],
            description: "strict tool definition",
            rejectionKeywords: ["strict", "schema", "additionalproperties"],
            makeProviderForcing: makeProviderForcing, calls: calls)
    }

    /// Sends one request with `overrides` forced into the raw body and grades the endpoint's answer.
    ///
    /// Acceptance-graded: these fields change what the model DOES, not what it returns in a
    /// checkable shape, so there is nothing to parse. A refusal naming the field is the signal; any
    /// other failure establishes nothing and stays inconclusive rather than becoming a false "no".
    private static func probeForcedField(
        _ forced: [String: AnyCodable],
        description: String,
        rejectionKeywords: [String],
        makeProviderForcing: @Sendable ([String: AnyCodable]) async -> any LLMProvider,
        calls: ProbeCallCounter?
    ) async -> ProbeFinding<Bool> {
        let started = Date()
        calls?.increment()
        do {
            _ = try await makeProviderForcing(forced)
                .send(messages: [.user("Reply with the single word: ok")], tools: [])
            return .established(true, "accepted \(description)", duration: Date().timeIntervalSince(started))
        } catch {
            let detail = CapabilityProbe.rejectionDetail(error)
            let lowered = detail.lowercased()
            if !CapabilityProbe.classifyFailure(error).meansNoAnswer,
               rejectionKeywords.contains(where: { lowered.contains($0) }) {
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
    private static func finding(fromError error: any Error, started: Date) -> ProbeFinding<Bool> {
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
        if CapabilityProbe.textIndicatesNotAChatModel(detail) {
            return .established(false, detail, duration: dur)
        }
        switch CapabilityProbe.classifyFailure(error) {
        case .noAnswer, .paymentRequired:
            return .inconclusive(detail, duration: dur)
        case .refusedTools, .refusedOurRequest:
            // A coherent 4xx that ISN'T an affirmative non-chat/modality statement (already settled
            // above by textIndicatesNotAChatModel) is too ambiguous to conclude "not a chat model".
            // Aggregators like OpenRouter return routing errors ("does not support endpoint"),
            // availability errors ("model_not_available / non-serverless"), not-found ("is not a valid
            // model ID"), rate-limits, and bare request-shape 400s ("Input validation error", "Request
            // contains an invalid argument") — none of which mean the model can't chat. The 2026-07-19
            // audit caught flagship chat models (qwen-2.5-72b, arcee virtuoso) stamped non-chat this
            // way. A false "non-chat" on a good model costs far more than a re-probe.
            return .inconclusive(detail, duration: dur)
        }
    }
}
