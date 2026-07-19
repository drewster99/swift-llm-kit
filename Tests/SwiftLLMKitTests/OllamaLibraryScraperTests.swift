import Testing
import Foundation
@testable import SwiftLLMKit

/// The Ollama library scraper: URL derivation, human-context parsing, and HTML extraction pinned
/// against the real page structure (2026-07-18) so an ollama.com redesign trips a test.
@Suite("Ollama library scraper")
struct OllamaLibraryScraperTests {

    @Test("Library URL derivation: tagged names get -cloud, untagged get :cloud")
    func urlDerivation() {
        #expect(OllamaLibraryScraper.libraryURL(forModelID: "qwen3.5:397b")?.absoluteString
                == "https://ollama.com/library/qwen3.5:397b-cloud")
        #expect(OllamaLibraryScraper.libraryURL(forModelID: "gpt-oss:120b")?.absoluteString
                == "https://ollama.com/library/gpt-oss:120b-cloud")
        #expect(OllamaLibraryScraper.libraryURL(forModelID: "deepseek-v4-pro")?.absoluteString
                == "https://ollama.com/library/deepseek-v4-pro:cloud")
        #expect(OllamaLibraryScraper.libraryURL(forModelID: "glm-5.2")?.absoluteString
                == "https://ollama.com/library/glm-5.2:cloud")
    }

    @Test("Human context labels invert to token counts (K=1024, M=1024², rounded)")
    func contextParsing() {
        #expect(OllamaLibraryScraper.contextTokens(from: "256K") == 262144)
        #expect(OllamaLibraryScraper.contextTokens(from: "128K") == 131072)
        #expect(OllamaLibraryScraper.contextTokens(from: "1M") == 1048576)
        #expect(OllamaLibraryScraper.contextTokens(from: "976K") == 999424)   // ≈ a real 1,000,000
        #expect(OllamaLibraryScraper.contextTokens(from: "512K") == 524288)
        #expect(OllamaLibraryScraper.contextTokens(from: "8192") == 8192)     // bare number passes through
        #expect(OllamaLibraryScraper.contextTokens(from: "nonsense") == nil)
        // Malformed values must degrade to nil, never trap on Int overflow (agy review).
        #expect(OllamaLibraryScraper.contextTokens(from: "99999999999999999999T") == nil)
        #expect(OllamaLibraryScraper.contextTokens(from: "infK") == nil)
        #expect(OllamaLibraryScraper.contextTokens(from: "nanM") == nil)
        // Exactly 2^63 after the ×1024 must be rejected (Int(2^63) traps), not admitted.
        #expect(OllamaLibraryScraper.contextTokens(from: "9007199254740992K") == nil)
    }

    /// The real card structure: a `>Context</div>` / `>Size</div>` label div immediately followed
    /// by a value span. Verbatim shape from qwen3.5:397b-cloud on 2026-07-18.
    @Test("Parses context and size from the library card HTML")
    func parseCard() {
        let html = """
        <div class="text-[13px] font-medium text-neutral-500">Context</div>
        <div class="mt-3 flex min-w-0 flex-col gap-1">
          <span class="shrink-0 text-xl font-medium leading-none text-black">256K</span>
          <span class="min-w-0 break-words text-[13px] leading-tight">tokens</span>
        </div>
        <div class="text-[13px] font-medium text-neutral-500">Size</div>
        <div class="mt-3 flex min-w-0 flex-col gap-1">
          <span class="shrink-0 text-xl font-medium leading-none text-black">397B</span>
          <span class="min-w-0 break-words text-[13px] leading-tight">parameters</span>
        </div>
        """
        let facts = OllamaLibraryScraper.parse(html: html)
        #expect(facts.contextTokens == 262144)
        #expect(facts.sizeLabel == "397B")
    }

    @Test("A page missing the stats yields empty facts, never a crash")
    func parseMissing() {
        let facts = OllamaLibraryScraper.parse(html: "<html><body>no model card here</body></html>")
        #expect(facts.isEmpty)
    }
}
