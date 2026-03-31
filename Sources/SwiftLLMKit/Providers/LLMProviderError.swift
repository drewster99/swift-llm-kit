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
}
