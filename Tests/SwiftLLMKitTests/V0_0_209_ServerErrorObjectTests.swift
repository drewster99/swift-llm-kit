import Foundation
import Testing
@testable import SwiftLLMKit

/// 0.0.209 — an OpenAI-shaped body that carries an `error` object instead of `choices` is thrown
/// as the typed `responseFailed(code:message:)`, not `malformedResponse`. oMLX's prefill memory
/// guard answers HTTP 200 with exactly that shape; reported as a parse failure it lost the server's
/// code and reason, so no caller could tell "out of memory, send less" from a truncated body.
@Suite("0.0.209: server error objects on HTTP 2xx")
struct V0_0_209_ServerErrorObjectTests {

    private static func provider() throws -> OpenAICompatibleProvider {
        OpenAICompatibleProvider(
            configuration: ModelConfiguration(name: "t", providerID: "p", modelID: "m"),
            provider: ModelProvider(
                id: "p", name: "p",
                apiType: .openAICompatible,
                endpoint: try #require(URL(string: "http://example.invalid/v1"))
            ),
            readAPIKey: { "" }
        )
    }

    /// Verbatim body captured from oMLX on 2026-09-23 (HTTP 200).
    private static let omlxPrefillBody = """
    {"error": {"message": "oMLX prefill memory guard rejected this prompt: Prefill would require ~53.28 GB peak (current 46.76 GB + KV+SDPA 6.52 GB) but dynamic ceiling is 51.58 GB. Close other apps to free RAM.", "type": "invalid_request_error", "param": null, "code": "prefill_memory_exceeded", "estimated_bytes": 57212722732, "limit_bytes": 55386454718, "omlx_code": "prefill_memory_exceeded"}, "type": "error"}
    """

    private static func thrownError(parsing body: String) throws -> LLMProviderError? {
        let provider = try provider()
        let data = try #require(body.data(using: .utf8))
        do {
            _ = try provider.parseResponse(data: data)
            return nil
        } catch let error as LLMProviderError {
            return error
        }
    }

    @Test func omlxPrefillGuardIsATypedResponseFailure() throws {
        let error = try #require(try Self.thrownError(parsing: Self.omlxPrefillBody))
        guard case .responseFailed(let code, let message) = error else {
            Issue.record("expected responseFailed, got \(error)")
            return
        }
        #expect(code == "prefill_memory_exceeded")
        #expect(message.hasPrefix("oMLX prefill memory guard rejected this prompt"))
        #expect(error.serverMemoryExhaustion == .omlxPrefillMemoryExceeded)
        #expect(error.contentPolicyRefusal == nil)
    }

    @Test func stringErrorIsAResponseFailureWithNoCode() throws {
        let error = try #require(try Self.thrownError(parsing: #"{"error": "model is loading"}"#))
        guard case .responseFailed(let code, let message) = error else {
            Issue.record("expected responseFailed, got \(error)")
            return
        }
        #expect(code == nil)
        #expect(message == "model is loading")
        #expect(error.serverMemoryExhaustion == nil)
    }

    @Test func numericCodeIsRenderedAsItsDecimalString() throws {
        let error = try #require(try Self.thrownError(parsing: #"{"error": {"code": 503, "message": "busy"}}"#))
        guard case .responseFailed(let code, let message) = error else {
            Issue.record("expected responseFailed, got \(error)")
            return
        }
        #expect(code == "503")
        #expect(message == "busy")
    }

    @Test func errorObjectWithoutMessageKeepsTheWholeObject() throws {
        let error = try #require(try Self.thrownError(parsing: #"{"error": {"code": "x", "detail": "d"}}"#))
        guard case .responseFailed(_, let message) = error else {
            Issue.record("expected responseFailed, got \(error)")
            return
        }
        #expect(message == #"{"code":"x","detail":"d"}"#)
    }

    @Test func nullErrorBesideMissingChoicesIsStillMalformed() throws {
        let error = try #require(try Self.thrownError(parsing: #"{"error": null, "id": "x"}"#))
        guard case .malformedResponse = error else {
            Issue.record("expected malformedResponse, got \(error)")
            return
        }
    }

    /// A usable choice wins: an `error` key beside a real message must not turn a success into a failure.
    @Test func validChoiceBesideAnErrorKeyStillParses() throws {
        let provider = try Self.provider()
        let body = #"{"error": null, "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}"#
        let response = try provider.parseResponse(data: try #require(body.data(using: .utf8)))
        #expect(response.text == "hi")
    }

    @Test func httpErrorBodyCarryingTheSameCodeIsAlsoMemoryExhaustion() throws {
        let error = LLMProviderError.httpError(statusCode: 400, body: Self.omlxPrefillBody)
        #expect(error.serverMemoryExhaustion == .omlxPrefillMemoryExceeded)
    }

    @Test func unrelatedCodesAreNotMemoryExhaustion() {
        #expect(LLMProviderError.responseFailed(code: "server_error", message: "x").serverMemoryExhaustion == nil)
        #expect(LLMProviderError.responseFailed(code: nil, message: "x").serverMemoryExhaustion == nil)
        #expect(LLMProviderError.httpError(statusCode: 500, body: "not json").serverMemoryExhaustion == nil)
        #expect(LLMProviderError.malformedResponse(detail: "prefill_memory_exceeded").serverMemoryExhaustion == nil)
    }
}
