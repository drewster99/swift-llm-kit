import Testing
import Foundation
@testable import SwiftLLMKit

/// The 2026-07-30 chat fold: `ModelInfo.supportsChatCompletions` stopped being a stored field and
/// became a 2-state view over `capabilities[.chat]` — one source of truth for chat support, so it can
/// appear in a role's `requiredCapabilities`. These pin the shim's default-true behavior, the legacy
/// migration, that chat round-trips INSIDE capabilities (no separate key), and the filter behavior.
@Suite("ModelInfo chat capability fold")
struct ModelInfoChatCapabilityTests {

    private func model(supportsChatCompletions: Bool? = nil) -> ModelInfo {
        ModelInfo(providerID: "p", modelID: "m", supportsChatCompletions: supportsChatCompletions)
    }

    @Test("Unstated chat is UNKNOWN in capabilities but reads true through the shim")
    func unknownChatReadsTrue() {
        let m = model()
        #expect(m.capabilities.state(of: .chat) == nil)     // genuinely unknown
        #expect(m.supportsChatCompletions)                  // shim defaults true
    }

    @Test("An explicit chat value is recorded as a KNOWN capability")
    func explicitChatRecorded() {
        #expect(model(supportsChatCompletions: false).capabilities.state(of: .chat) == false)
        #expect(!model(supportsChatCompletions: false).supportsChatCompletions)
        #expect(model(supportsChatCompletions: true).capabilities.state(of: .chat) == true)
    }

    @Test("Setting supportsChatCompletions writes through to the capability")
    func setterWritesCapability() {
        var m = model()
        m.supportsChatCompletions = false
        #expect(m.capabilities.state(of: .chat) == false)
        m.capabilities[.chat] = true
        #expect(m.supportsChatCompletions)
    }

    @Test("A legacy standalone supportsChatCompletions key migrates into capabilities[.chat]")
    func legacyKeyMigrates() throws {
        let json = #"{"providerID":"p","modelID":"m","displayName":"m","supportsChatCompletions":false}"#
        let decoded = try JSONDecoder().decode(ModelInfo.self, from: Data(json.utf8))
        #expect(decoded.capabilities.state(of: .chat) == false)
        #expect(!decoded.supportsChatCompletions)
    }

    @Test("Chat round-trips INSIDE capabilities, with no separate top-level key")
    func chatRoundTripsInsideCapabilities() throws {
        let original = model(supportsChatCompletions: false)
        let data = try JSONEncoder().encode(original)
        let obj = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(obj["supportsChatCompletions"] == nil)      // no standalone key any more
        let caps = try #require(obj["capabilities"] as? [String: Bool])
        #expect(caps["chat"] == false)                      // it lives here now
        let restored = try JSONDecoder().decode(ModelInfo.self, from: data)
        #expect(!restored.supportsChatCompletions)
    }

    @Test("required: [.chat] hides only a KNOWN non-chat model; unknown and chat pass")
    func requiredChatHidesOnlyKnownNonChat() {
        let unknown = model()                               // chat unmeasured
        let chats = model(supportsChatCompletions: true)
        let noChat = model(supportsChatCompletions: false)  // KNOWN non-chat (e.g. embedding)
        #expect(unknown.satisfies(requiredCapabilities: [.chat], mustNotBePresent: [], includedAvailabilityStates: []))
        #expect(chats.satisfies(requiredCapabilities: [.chat], mustNotBePresent: [], includedAvailabilityStates: []))
        #expect(!noChat.satisfies(requiredCapabilities: [.chat], mustNotBePresent: [], includedAvailabilityStates: []))
    }
}
