import Foundation
import Testing
@testable import SwiftLLMKit

/// Coverage for native document (e.g. PDF) content in outbound request bodies — Anthropic
/// `document` blocks, OpenAI `file` parts, Gemini `inlineData`. Verifies the wire SHAPE; live
/// provider acceptance isn't exercised here.
@Suite("Document content serialization")
struct DocumentContentTests {
    private static let dummyKey: @Sendable () -> String = { "test-key" }

    private static func config(_ providerID: String) -> ModelConfiguration {
        ModelConfiguration(name: "t", providerID: providerID, modelID: "m")
    }

    private static func provider(_ id: String, type: ProviderAPIType, endpoint: String) throws -> ModelProvider {
        ModelProvider(id: id, name: id, apiType: type, endpoint: try #require(URL(string: endpoint)))
    }

    private static let pdf = LLMDocumentContent(
        data: Data([0x25, 0x50, 0x44, 0x46]), // "%PDF"
        mimeType: "application/pdf",
        filename: "doc.pdf"
    )

    @Test func anthropic_emitsDocumentBlock() throws {
        let p = AnthropicProvider(
            configuration: Self.config("builtin.anthropic"),
            provider: try Self.provider("builtin.anthropic", type: .anthropic, endpoint: "https://api.anthropic.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let body = try p.buildRequestBody(messages: [.user("read this", images: [], documents: [Self.pdf])], tools: [])
        let messages = try #require(body["messages"] as? [[String: Any]])
        let content = try #require(messages.first?["content"] as? [[String: Any]])
        let doc = try #require(content.first { $0["type"] as? String == "document" })
        let source = try #require(doc["source"] as? [String: Any])
        #expect(source["type"] as? String == "base64")
        #expect(source["media_type"] as? String == "application/pdf")
        #expect((source["data"] as? String)?.isEmpty == false)
    }

    @Test func openAI_emitsFilePart() throws {
        let p = OpenAICompatibleProvider(
            configuration: Self.config("builtin.openai"),
            provider: try Self.provider("builtin.openai", type: .openAICompatible, endpoint: "https://api.openai.com/v1"),
            readAPIKey: Self.dummyKey
        )
        let body = p.buildRequestBody(messages: [.user("read this", images: [], documents: [Self.pdf])], tools: [])
        let messages = try #require(body["messages"] as? [[String: Any]])
        let content = try #require(messages.first?["content"] as? [[String: Any]])
        let file = try #require(content.first { $0["type"] as? String == "file" })
        let fileObj = try #require(file["file"] as? [String: Any])
        #expect((fileObj["file_data"] as? String)?.hasPrefix("data:application/pdf;base64,") == true)
        #expect(fileObj["filename"] as? String == "doc.pdf")
    }

    @Test func gemini_emitsInlineData() throws {
        let p = GeminiProvider(
            configuration: Self.config("builtin.gemini"),
            provider: try Self.provider("builtin.gemini", type: .gemini, endpoint: "https://generativelanguage.googleapis.com/v1beta"),
            readAPIKey: Self.dummyKey
        )
        let body = try p.buildRequestBody(messages: [.user("read this", images: [], documents: [Self.pdf])], tools: [])
        let contents = try #require(body["contents"] as? [[String: Any]])
        let parts = try #require(contents.first?["parts"] as? [[String: Any]])
        let inline = try #require(parts.compactMap { $0["inlineData"] as? [String: Any] }.first)
        #expect(inline["mimeType"] as? String == "application/pdf")
        #expect((inline["data"] as? String)?.isEmpty == false)
    }

    @Test func documentFactory_storesDocuments() {
        let message = LLMMessage.user("hi", images: [], documents: [Self.pdf])
        #expect(message.documents?.count == 1)
        #expect(message.documents?.first?.mimeType == "application/pdf")
        #expect(message.images == nil)
    }
}
