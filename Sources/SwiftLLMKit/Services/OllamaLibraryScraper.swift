import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Reads a cloud Ollama model's parameter size and context window from its `ollama.com/library`
/// page — data the cloud `/api/tags` endpoint omits (it reports neither `parameter_size` nor
/// `context_length` for cloud models).
///
/// This is a CURATION tool, not a live enrichment step: the intended flow is to run it
/// occasionally and bake the results into the downloaded-overrides layer, so the app reads
/// persisted, force-applied overrides rather than scraping on every fetch/probe. Everything here
/// is fail-soft — a 404, timeout, or unrecognized page yields nil, never an error — and the pure
/// parse is unit-tested against a captured fixture so an ollama.com redesign fails a test rather
/// than silently mis-parsing.
public enum OllamaLibraryScraper {

    /// What a library page yields. Either field may be nil if the page didn't state it.
    public struct Facts: Sendable, Equatable {
        public var contextTokens: Int?
        public var sizeLabel: String?
        public init(contextTokens: Int? = nil, sizeLabel: String? = nil) {
            self.contextTokens = contextTokens
            self.sizeLabel = sizeLabel
        }
        public var isEmpty: Bool { contextTokens == nil && sizeLabel == nil }
    }

    /// The library URL for a cloud model, derived from its `/api/tags` name. Ollama tags the cloud
    /// build differently depending on whether the name already carries a size tag:
    /// - `qwen3.5:397b` (has a `:` tag)  → `qwen3.5:397b-cloud`
    /// - `deepseek-v4-pro` (no tag)      → `deepseek-v4-pro:cloud`
    /// Verified across the fleet 2026-07-18.
    public static func libraryURL(forModelID modelID: String) -> URL? {
        let suffix = modelID.contains(":") ? "-cloud" : ":cloud"
        // The colon in a tag is legal in the path; percent-encode defensively for anything else.
        let path = (modelID + suffix)
        var components = URLComponents()
        components.scheme = "https"
        components.host = "ollama.com"
        components.path = "/library/" + path
        return components.url
    }

    /// Extracts context + size from a library page's HTML. The card renders each stat as a label
    /// div (`>Context</div>`, `>Size</div>`) immediately followed by a value span; we match the
    /// nearest span after each label. PURE — no IO — so it is testable against a saved fixture.
    public static func parse(html: String) -> Facts {
        func value(after label: String) -> String? {
            let pattern = ">\(NSRegularExpression.escapedPattern(for: label))</div>.*?<span[^>]*>([^<]+)</span>"
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]),
                  let match = regex.firstMatch(in: html, range: NSRange(html.startIndex..., in: html)),
                  let range = Range(match.range(at: 1), in: html) else { return nil }
            return html[range].trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return Facts(
            contextTokens: value(after: "Context").flatMap(contextTokens(from:)),
            sizeLabel: value(after: "Size")
        )
    }

    /// Parses a rounded human context label ("256K", "1M", "976K", "1.5M") into a token count.
    /// Ollama displays context divided by 1024 and rounded, so we read K = 1024, M = 1024², T =
    /// 1024³ to invert it. The result is APPROXIMATE by construction — the page never states the
    /// exact figure — which is fine for context-window pruning (a few thousand tokens' slack).
    public static func contextTokens(from label: String) -> Int? {
        let text = label.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        let units: [(suffix: String, multiplier: Double)] = [
            ("T", 1024 * 1024 * 1024),
            ("M", 1024 * 1024),
            ("K", 1024),
        ]
        for unit in units where text.hasSuffix(unit.suffix) {
            guard let number = Double(text.dropLast()) else { return nil }
            return Int((number * unit.multiplier).rounded())
        }
        // A bare number (no suffix) is already a token count.
        return Int(text.replacingOccurrences(of: ",", with: ""))
    }

    /// Fetches and parses a model's library page. Fail-soft: any network/HTTP/parse failure returns
    /// an empty `Facts`, never throws — a curation tool must not abort a batch on one bad model.
    public static func scrape(
        modelID: String,
        session: URLSession = .shared,
        timeout: TimeInterval = 20
    ) async -> Facts {
        guard let url = libraryURL(forModelID: modelID) else { return Facts() }
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        // Ollama serves the model card only to a browser-like UA.
        request.setValue("Mozilla/5.0 (compatible; SwiftLLMKit)", forHTTPHeaderField: "User-Agent")
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
                  let html = String(data: data, encoding: .utf8) else { return Facts() }
            return parse(html: html)
        } catch {
            return Facts()
        }
    }
}
