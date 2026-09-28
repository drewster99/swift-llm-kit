import Foundation
import Security
import Testing
@testable import SwiftLLMKit

/// Unit coverage for `KeychainService.nextStep(afterAddStatus:)` — the pure
/// decision `saveImpl` makes after `SecItemAdd` runs following an
/// `errSecItemNotFound` update.
///
/// `saveImpl` calls `SecItemUpdate` first, and only falls through to
/// `SecItemAdd` when that update reports the item doesn't exist. If another
/// `save` (same provider, another thread or process) creates the item in the
/// gap between those two calls, `SecItemAdd` returns `errSecDuplicateItem`
/// even though a retried update would now succeed. `nextStep` is what decides
/// to retry instead of failing that save. The real race isn't practical to
/// drive through the Keychain in CI, so this tests the decision in isolation.
@Suite("KeychainService.nextStep(afterAddStatus:)")
struct KeychainServiceAddStepTests {

    @Test func success_needsNoFurtherAction() {
        #expect(KeychainService.nextStep(afterAddStatus: errSecSuccess) == .succeeded)
    }

    @Test func duplicateItem_retriesTheUpdateInsteadOfFailing() {
        // The race from the issue: another writer won SecItemAdd first.
        #expect(KeychainService.nextStep(afterAddStatus: errSecDuplicateItem) == .retryUpdate)
    }

    @Test func otherFailure_isSurfacedAsIs() {
        #expect(KeychainService.nextStep(afterAddStatus: errSecParam) == .failed(errSecParam))
        #expect(KeychainService.nextStep(afterAddStatus: errSecMissingEntitlement) == .failed(errSecMissingEntitlement))
    }
}
