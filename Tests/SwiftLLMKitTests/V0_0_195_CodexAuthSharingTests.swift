import Foundation
import Testing
@testable import SwiftLLMKit

/// The single-flight in `CodexAuthCoordinator` protects nothing unless every caller reading the
/// same `auth.json` holds the SAME coordinator.
///
/// It did not: `CodexResponsesProvider.init` defaulted `auth` to a fresh `CodexAuthCoordinator()`
/// and the factory passed nothing, so each agent role got its own — five roles noticing expiry at
/// once fired five refreshes and raced each other writing the file, and the loser persisted a
/// refresh token the server had rotated away, signing the user out. The default argument IS the
/// protection, which is why the last test below asserts on what the FACTORY builds rather than on
/// the registry alone.
///
/// Nothing here reads or writes a credential: coordinators are only minted, never asked for tokens,
/// so the developer's real `~/.codex/auth.json` is untouched. `CODEX_HOME` is deliberately NOT set
/// — tests run in parallel and `setenv` is process-global.
@Suite("Codex auth coordinator sharing")
struct CodexAuthCoordinatorSharingTests {

    @Test("One coordinator per store path, and spelling does not mint a second one")
    func oneCoordinatorPerPath() {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("codex-share-\(UUID().uuidString)")
        let direct = dir.appendingPathComponent("auth.json")
        let roundabout = dir.appendingPathComponent("nested/../auth.json")
        let other = dir.appendingPathComponent("other/auth.json")

        #expect(CodexAuthCoordinator.shared(forStoreAt: direct)
                === CodexAuthCoordinator.shared(forStoreAt: direct))
        #expect(CodexAuthCoordinator.shared(forStoreAt: roundabout)
                === CodexAuthCoordinator.shared(forStoreAt: direct),
                "two spellings of one file must not become two writers of it")
        #expect(CodexAuthCoordinator.shared(forStoreAt: other)
                !== CodexAuthCoordinator.shared(forStoreAt: direct),
                "different files are different credentials — sharing would be the real bug")
    }

    @Test("Concurrent lookups of one path settle on a single instance")
    func concurrentLookupsAgree() async {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("codex-share-\(UUID().uuidString)/auth.json")
        let all = await withTaskGroup(of: CodexAuthCoordinator.self) { group in
            for _ in 0..<32 { group.addTask { CodexAuthCoordinator.shared(forStoreAt: url) } }
            var seen: [CodexAuthCoordinator] = []
            for await coordinator in group { seen.append(coordinator) }
            return seen
        }
        #expect(all.count == 32)
        #expect(all.allSatisfy { $0 === all[0] })
    }

    @Test("Every provider the factory builds shares ONE coordinator")
    @MainActor
    func providersShareOneCoordinator() throws {
        // A unique app identifier so this gets its own storage directory rather than the
        // developer's real one.
        let kit = LLMKitManager(
            appIdentifier: "test.codex.share.\(UUID().uuidString)",
            keychainServicePrefix: "test.codex.share")
        let provider = ModelProvider(
            id: "builtin.codex-chatgpt", name: "ChatGPT Subscription (Codex)",
            apiType: .codexChatGPT,
            endpoint: try #require(URL(string: "https://chatgpt.com/backend-api/codex")))

        // Five providers, the shape Agent Smith actually produces: one per agent role.
        let coordinators: [CodexAuthCoordinator] = try (0..<5).map { index in
            let built = kit.makeProvider(
                configuration: ModelConfiguration(
                    name: "role-\(index)", providerID: provider.id, modelID: "gpt-5.5"),
                provider: provider)
            return try #require(built as? CodexResponsesProvider).auth
        }
        for coordinator in coordinators {
            #expect(coordinator === coordinators[0],
                    "per-provider coordinators defeat the single-flight and race on auth.json")
        }
        #expect(coordinators[0] === CodexAuthCoordinator.sharedForDefaultStore,
                "the factory must land on the default store's coordinator, not a private one")
    }
}
