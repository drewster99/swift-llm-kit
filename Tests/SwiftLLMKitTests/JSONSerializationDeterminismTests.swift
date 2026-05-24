import Foundation
import Testing
@testable import SwiftLLMKit

/// Regression coverage for the byte-determinism guarantee that prompt caches
/// depend on. Providers serialize their request body with `[.sortedKeys]` so
/// dictionary key order doesn't perturb the wire bytes across requests with
/// otherwise identical content.
///
/// The tests below don't drive a real provider — that would require API keys
/// and network. Instead they verify the underlying invariant: when the same
/// nested dictionary is built with different insertion orders, the
/// `[.sortedKeys]` serialization produces byte-identical output. If a provider
/// ever drops the `[.sortedKeys]` option, these tests would still pass — so
/// each provider also gets a direct assertion that its outbound serialization
/// uses sorted keys (we check the keyword in source).
@Suite("JSONSerialization determinism")
struct JSONSerializationDeterminismTests {

    @Test func sortedKeys_normalizesDifferentInsertionOrders() throws {
        let dictA: [String: Any] = [
            "model": "claude",
            "max_tokens": 4096,
            "messages": [["role": "user", "content": "hi"]] as [[String: Any]],
            "temperature": 0.7
        ]
        let dictB: [String: Any] = [
            "temperature": 0.7,
            "messages": [["content": "hi", "role": "user"]] as [[String: Any]],
            "max_tokens": 4096,
            "model": "claude"
        ]
        let a = try JSONSerialization.data(withJSONObject: dictA, options: [.sortedKeys])
        let b = try JSONSerialization.data(withJSONObject: dictB, options: [.sortedKeys])
        #expect(a == b)
    }

    @Test func sortedKeys_normalizesNestedDictOrder() throws {
        // Tool parameters live in nested dicts; the cache invariant fails
        // unless nested keys also sort.
        let toolA: [String: Any] = [
            "name": "f",
            "input_schema": [
                "type": "object",
                "properties": [
                    "city": ["type": "string", "description": "city"] as [String: Any],
                    "unit": ["type": "string", "enum": ["c", "f"]] as [String: Any]
                ] as [String: Any]
            ] as [String: Any]
        ]
        let toolB: [String: Any] = [
            "input_schema": [
                "properties": [
                    "unit": ["enum": ["c", "f"], "type": "string"] as [String: Any],
                    "city": ["description": "city", "type": "string"] as [String: Any]
                ] as [String: Any],
                "type": "object"
            ] as [String: Any],
            "name": "f"
        ]
        let a = try JSONSerialization.data(withJSONObject: toolA, options: [.sortedKeys])
        let b = try JSONSerialization.data(withJSONObject: toolB, options: [.sortedKeys])
        #expect(a == b)
    }

    /// Source-level audit: every provider that hits the wire must use
    /// `.sortedKeys` when encoding the request body. If a future change drops
    /// the option, this test catches it without needing a live provider.
    @Test func everyProviderSerializesRequestBodyWithSortedKeys() throws {
        let providersDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/SwiftLLMKit/Providers")

        let files = [
            "AnthropicProvider.swift",
            "OpenAICompatibleProvider.swift",
            "GeminiProvider.swift",
            "OllamaProvider.swift"
        ]
        for file in files {
            let url = providersDir.appendingPathComponent(file)
            let source = try String(contentsOf: url, encoding: .utf8)
            // We only require sorted-keys on the request-body line — the
            // response-decoding `JSONSerialization.jsonObject` calls don't need it.
            let bodyLines = source.split(separator: "\n").filter {
                $0.contains("JSONSerialization.data(withJSONObject: body")
            }
            #expect(!bodyLines.isEmpty, "\(file): expected at least one request-body serialization")
            for line in bodyLines {
                #expect(line.contains(".sortedKeys"),
                        "\(file): request-body serialization missing .sortedKeys: \(line)")
            }
        }
    }
}
