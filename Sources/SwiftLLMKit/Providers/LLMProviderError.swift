import Foundation

/// Errors from LLM provider API calls.
public enum LLMProviderError: Error, LocalizedError {
    case invalidResponse
    case httpError(statusCode: Int, body: String, url: URL? = nil)
    case malformedResponse(detail: String)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Response was not a valid HTTP response"
        case .httpError(let code, let body, let url):
            let detail = body.isEmpty ? (url?.absoluteString ?? "empty body") : body
            return "HTTP \(code): \(detail)"
        case .malformedResponse(let detail):
            return "Could not parse LLM response: \(detail)"
        }
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
        guard case .httpError(_, let body, _) = self else { return nil }
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
