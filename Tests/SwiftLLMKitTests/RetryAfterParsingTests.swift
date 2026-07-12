import Testing
import Foundation
@testable import SwiftLLMKit

/// `Retry-After` header parsing: the header is either delta-seconds or an HTTP-date, and the
/// value flows onto `LLMProviderError.httpError` so callers can honor server-directed pacing.
@Suite("Retry-After parsing")
struct RetryAfterParsingTests {

    @Test("delta-seconds parses to that many seconds")
    func deltaSeconds() {
        #expect(LLMProviderError.parseRetryAfter("120") == 120)
        #expect(LLMProviderError.parseRetryAfter("  0 ") == 0)
        #expect(LLMProviderError.parseRetryAfter("3600") == 3600)
    }

    @Test("HTTP-date parses to a delay from now, clamped at 0 when past")
    func httpDate() {
        let now = Date(timeIntervalSince1970: 1_000_000_000) // fixed reference
        let future = "Sun, 09 Sep 2001 01:50:00 GMT" // 1_000_000_200 → +200s
        let past = "Sun, 09 Sep 2001 01:40:00 GMT"    // 1_000_000_000 - 400 → clamped 0

        let futureDelay = LLMProviderError.parseRetryAfter(future, now: now)
        #expect(futureDelay != nil)
        if let futureDelay { #expect(abs(futureDelay - 200) < 1) }
        #expect(LLMProviderError.parseRetryAfter(past, now: now) == 0)
    }

    @Test("absent or unparseable values return nil")
    func unparseable() {
        #expect(LLMProviderError.parseRetryAfter(nil) == nil)
        #expect(LLMProviderError.parseRetryAfter("") == nil)
        #expect(LLMProviderError.parseRetryAfter("   ") == nil)
        #expect(LLMProviderError.parseRetryAfter("soon") == nil)
        #expect(LLMProviderError.parseRetryAfter("-5") == nil) // not a valid delta-seconds or date
    }

    @Test("retryAfterSeconds surfaces the value from an httpError")
    func surfacedFromError() {
        let withDelay = LLMProviderError.httpError(statusCode: 429, body: "rate limited", url: nil, retryAfter: 42)
        #expect(withDelay.retryAfterSeconds == 42)

        let withoutDelay = LLMProviderError.httpError(statusCode: 500, body: "boom")
        #expect(withoutDelay.retryAfterSeconds == nil)

        #expect(LLMProviderError.invalidResponse.retryAfterSeconds == nil)
    }
}
