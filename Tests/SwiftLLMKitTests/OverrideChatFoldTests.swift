import Testing
import Foundation
@testable import SwiftLLMKit

/// The 2026-07-30 full chat fold at the OVERRIDE layer: `ModelCapabilitiesOverride` gained `chat`
/// plus an enum subscript, and `ModelMetadataOverride.supportsChatCompletions` became a shim over
/// `capabilities.chat` with a Codable migration off the legacy standalone key. These pin the
/// subscript, the shim, the migration, and that every capability carries editor copy.
@Suite("Chat fold: override layer + editor metadata")
struct OverrideChatFoldTests {

    @Test("ModelCapabilitiesOverride subscript reaches chat and the other flags")
    func overrideSubscript() {
        var override = ModelCapabilitiesOverride()
        #expect(override[.chat] == nil)
        override[.chat] = false
        override[.toolUse] = true
        #expect(override[.chat] == false)
        #expect(override.chat == false)
        #expect(override[.toolUse] == true)
        #expect(override[.vision] == nil)
        override[.chat] = nil
        #expect(override.chat == nil)
    }

    @Test("supportsChatCompletions is a shim over capabilities.chat")
    func metadataOverrideShim() {
        var override = ModelMetadataOverride()
        #expect(override.supportsChatCompletions == nil)
        override.supportsChatCompletions = false
        #expect(override.capabilities?[.chat] == false)      // routed into capabilities
        #expect(override.supportsChatCompletions == false)
        // The init convenience param routes the same way.
        let viaInit = ModelMetadataOverride(supportsChatCompletions: true)
        #expect(viaInit.capabilities?[.chat] == true)
    }

    @Test("A nil supportsChatCompletions init arg never clobbers a chat set via capabilities")
    func nilInitArgKeepsCapabilitiesChat() {
        let override = ModelMetadataOverride(capabilities: ModelCapabilitiesOverride(chat: false))
        #expect(override.capabilities?[.chat] == false)
        #expect(override.supportsChatCompletions == false)
    }

    @Test("Codable migrates the legacy standalone key into capabilities.chat")
    func codableMigratesLegacyKey() throws {
        let json = #"{"displayName":"X","supportsChatCompletions":false}"#
        let decoded = try JSONDecoder().decode(ModelMetadataOverride.self, from: Data(json.utf8))
        #expect(decoded.capabilities?[.chat] == false)
        #expect(decoded.supportsChatCompletions == false)
        #expect(decoded.displayName == "X")
    }

    @Test("New-format capabilities.chat wins over a stray legacy key; encode emits no standalone key")
    func newFormatWinsAndEncodesClean() throws {
        let json = #"{"capabilities":{"chat":true},"supportsChatCompletions":false}"#
        let decoded = try JSONDecoder().decode(ModelMetadataOverride.self, from: Data(json.utf8))
        #expect(decoded.capabilities?[.chat] == true)         // capability wins over the legacy key

        let reencoded = try JSONEncoder().encode(ModelMetadataOverride(supportsChatCompletions: true))
        let obj = try #require(try JSONSerialization.jsonObject(with: reencoded) as? [String: Any])
        #expect(obj["supportsChatCompletions"] == nil)        // no standalone key on write
        let caps = try #require(obj["capabilities"] as? [String: Bool])
        #expect(caps["chat"] == true)                          // it lives in capabilities
    }

    @Test("Every capability carries non-empty editor title and description")
    func everyCapabilityHasEditorCopy() {
        for capability in ModelCapability.allCases {
            #expect(!capability.editorTitle.isEmpty)
            #expect(!capability.editorDescription.isEmpty)
        }
    }
}
