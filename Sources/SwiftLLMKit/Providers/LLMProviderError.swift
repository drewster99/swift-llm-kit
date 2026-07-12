import Foundation

/// Errors from LLM provider API calls.
public enum LLMProviderError: Error, LocalizedError {
    case invalidResponse
    /// An HTTP error response. `retryAfter` carries the server's `Retry-After` delay in seconds
    /// when the response provided one (e.g. on a 429 rate limit), so callers can honor the
    /// server's pacing instead of guessing a backoff.
    case httpError(statusCode: Int, body: String, url: URL? = nil, retryAfter: TimeInterval? = nil)
    case malformedResponse(detail: String)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Response was not a valid HTTP response"
        case .httpError(let code, let body, let url, _):
            let detail = body.isEmpty ? (url?.absoluteString ?? "empty body") : body
            return "HTTP \(code): \(detail)"
        case .malformedResponse(let detail):
            return "Could not parse LLM response: \(detail)"
        }
    }

    /// The server-requested retry delay in seconds, if this is an HTTP error whose response
    /// carried a parseable `Retry-After` header. nil otherwise.
    public var retryAfterSeconds: TimeInterval? {
        guard case .httpError(_, _, _, let retryAfter) = self else { return nil }
        return retryAfter
    }

    /// Parses an HTTP `Retry-After` header value into a delay in seconds. The header is either
    /// delta-seconds (a non-negative integer) or an HTTP-date (RFC 7231 IMF-fixdate); a date is
    /// converted to a delay from `now` and clamped to 0 if already past. Returns nil when the
    /// value is absent or unparseable.
    public static func parseRetryAfter(_ headerValue: String?, now: Date = Date()) -> TimeInterval? {
        guard let raw = headerValue?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        // Delta-seconds: a non-negative number. A negative value is malformed — return nil so
        // the caller falls back to its own backoff rather than retrying immediately.
        if let seconds = TimeInterval(raw) { return seconds >= 0 ? seconds : nil }
        if let date = httpDate(raw) { return max(0, date.timeIntervalSince(now)) }
        return nil
    }

    /// RFC 7231 HTTP-date, all three forms recipients must accept: the preferred IMF-fixdate
    /// plus the obsolete RFC 850 and ANSI C `asctime` formats.
    private static let httpDateFormats = [
        "EEE, dd MMM yyyy HH:mm:ss zzz",   // IMF-fixdate  — Sun, 06 Nov 1994 08:49:37 GMT
        "EEEE, dd-MMM-yy HH:mm:ss zzz",    // RFC 850      — Sunday, 06-Nov-94 08:49:37 GMT
        "EEE MMM d HH:mm:ss yyyy"          // asctime      — Sun Nov  6 08:49:37 1994
    ]

    private static func httpDate(_ raw: String) -> Date? {
        // asctime space-pads a single-digit day ("Nov  6"); collapse runs of spaces so one
        // format string matches both single- and double-digit days.
        let normalized = raw.replacingOccurrences(of: " +", with: " ", options: .regularExpression)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "GMT")
        for format in httpDateFormats {
            formatter.dateFormat = format
            if let date = formatter.date(from: normalized) { return date }
        }
        return nil
    }

    /// If this error is a backend rejection reporting the model's maximum output-token
    /// limit, returns that limit so the caller can clamp and retry instead of failing the
    /// turn. Recognizes the OpenAI-compatible / Ollama Cloud shape, e.g.
    /// `max_tokens (256000) exceeds model's maximum output tokens (131072)`.
    ///
    /// Only the *model's* limit is extracted — the parenthesized number that follows
    /// "maximum output tokens". The leading offending value (the 256000) is deliberately
    /// ignored. Returns nil for any error that isn't a recognizable max-output rejection.
    public var reportedMaxOutputTokenLimit: Int? {
        guard case .httpError(_, let body, _, _) = self else { return nil }
        return Self.reportedMaxOutputTokenLimit(inBody: body)
    }

    /// Extracts the model's maximum output-token limit from a backend error body, if the
    /// body reports one. Exposed for callers that hold the raw body rather than an
    /// `LLMProviderError`. Matches "maximum output tokens (N)" and "maximum output tokens
    /// is N" (case-insensitive), returning the first such N.
    public static func reportedMaxOutputTokenLimit(inBody body: String) -> Int? {
        guard body.range(of: "maximum output tokens", options: .caseInsensitive) != nil else {
            return nil
        }
        // "... maximum output tokens (131072)" or "... maximum output tokens is 131072"
        let pattern = "maximum output tokens(?:\\s+is)?\\s*\\(?\\s*(\\d+)"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return nil
        }
        let range = NSRange(body.startIndex..<body.endIndex, in: body)
        guard let match = regex.firstMatch(in: body, options: [], range: range),
              match.numberOfRanges >= 2,
              let captureRange = Range(match.range(at: 1), in: body) else {
            return nil
        }
        return Int(body[captureRange])
    }
}
