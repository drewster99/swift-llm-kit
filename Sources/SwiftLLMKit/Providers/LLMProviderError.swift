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

    /// Extracts the model's maximum output-token limit from a backend error body, if the body
    /// reports one. Exposed for callers that hold the raw body rather than an `LLMProviderError`.
    ///
    /// Providers phrase this two ways, and both are matched (case-insensitive), first hit wins:
    /// - number AFTER the phrase — OpenAI-style "maximum output tokens (131072)" / "... is 131072";
    /// - number BEFORE it — Anthropic-style "max_tokens: 100000000 > 64000, which is the maximum
    ///   allowed number of output tokens for <model>". The earlier regex required the number to
    ///   follow "maximum output tokens" and so silently missed every Anthropic 400.
    public static func reportedMaxOutputTokenLimit(inBody body: String) -> Int? {
        let patterns = [
            // Anthropic: "... > 64000, which is the maximum allowed number of output tokens ..."
            "> *(\\d+),? *which is the maximum",
            // OpenAI (current): "This model supports at most 128000 completion tokens"
            "supports at most *(\\d+) *(?:completion|output) tokens",
            // z.ai / GLM: "max_completion_tokens is limited to 16384 for glm-5.2"
            "max_(?:completion_)?tokens (?:is )?limited to *(\\d+)",
            // Alibaba Cloud (dashscope): "Range of max_tokens should be [1, 65536]" — the top of
            // a max_tokens-specific range IS the output ceiling. 212 unparsed occurrences in the
            // 2026-07-18 log audit, each escalating to a full binary search that tripped the
            // account rate limit mid-run.
            "range of max_(?:completion_)?tokens should be *\\[ *\\d+ *, *(\\d+) *\\]",
            // HuggingFace router: "max_tokens (current value: 100000000) must be between 0 and
            // 65536" — some builds backtick-quote the parameter name.
            "max_(?:completion_)?tokens`? *\\(current value: *\\d+\\) *must be between *\\d+ *and *(\\d+)",
            // OpenAI-style (older): "... maximum output tokens (131072)" / "... maximum output tokens is 131072"
            "maximum(?: allowed(?: number of)?)? output tokens(?: is)? *\\(? *(\\d+)"
        ]
        return Self.firstCapture(in: body, patterns: patterns)
    }

    /// A nearby token limit the body reveals even when it isn't the exact max-output figure — a
    /// context length, or the upper bound of an allowed range. Used to shrink a max-output binary
    /// search from `[512, 100M]` to `[512, hint]`: it's only a bound (context ≥ output, and the
    /// range's top may exceed the true output cap), so the caller still bisects for the real value,
    /// but from a handful of calls instead of ~24.
    ///
    /// Kept separate from ``reportedMaxOutputTokenLimit(inBody:)`` on purpose: this is not
    /// necessarily the output ceiling, so it must never be reported AS the output ceiling.
    public static func reportedLimitHint(inBody body: String) -> Int? {
        let patterns = [
            // z.ai: "The max_tokens parameter is illegal.：限制数值范围[1,131072]". Lower bound not
            // hard-coded — HuggingFace-style ranges start at 0.
            "范围 *\\[ *\\d+ *, *(\\d+) *\\]",
            "range *\\[ *\\d+ *, *(\\d+) *\\]"
        ]
        return Self.firstCapture(in: body, patterns: patterns)
    }

    /// A CONTEXT-LENGTH bound the body reveals ("This endpoint's maximum context length is 202752
    /// tokens"), kept apart from ``reportedLimitHint(inBody:)`` since the 2026-07-18 audit: proxy
    /// endpoints (OpenRouter, the HuggingFace router) often bound `max_tokens` ONLY by context
    /// length, so a binary search "converges" to context-minus-input — a number that overstates
    /// what a production request with real input can use and must never be recorded as the
    /// model's output ceiling. Callers use this to recognize that situation and decline to
    /// search rather than to tighten one.
    public static func reportedContextLengthBound(inBody body: String) -> Int? {
        Self.firstCapture(in: body, patterns: ["context length is *(\\d+)"])
    }

    /// First capture group of the first matching pattern (case-insensitive), as an Int.
    private static func firstCapture(in body: String, patterns: [String]) -> Int? {
        let range = NSRange(body.startIndex..<body.endIndex, in: body)
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
                  let match = regex.firstMatch(in: body, options: [], range: range),
                  match.numberOfRanges >= 2,
                  let captureRange = Range(match.range(at: 1), in: body) else { continue }
            return Int(body[captureRange])
        }
        return nil
    }
}
