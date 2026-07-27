import Foundation
import os

private let loggerOS = Logger(subsystem: "SwiftLLMKit", category: "LLMRequestLogger")

/// Utility for verbose request/response file logging.
///
/// When called, full JSON request and response bodies are written to
/// `$TMPDIR/<logDirectoryName>/` with timestamped filenames.
/// Callers decide whether to log based on their own `verboseLogging` flag.
public enum LLMRequestLogger {
    /// Name of the subdirectory under `$TMPDIR` where log files are written.
    /// Set this early at app launch (before any providers are created) to customize.
    public nonisolated(unsafe) static var logDirectoryName = "SwiftLLMKit-Logs"

    // MARK: - Log directory

    /// Lazily-created log directory. Uses `logDirectoryName` at first access.
    /// `static let` guarantees thread-safe one-time initialization.
    /// Set `logDirectoryName` before any provider sends a request.
    private static let logDirectory: URL = {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(logDirectoryName)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            loggerOS.warning("Failed to create log directory \(dir.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
        return dir
    }()

    // MARK: - Shared helpers

    /// Filename-safe local timestamp, date first so names sort chronologically and a
    /// multi-day log directory stays unambiguous.
    ///
    /// The date used to be omitted. Log directories live in `$TMPDIR` and routinely span
    /// several days, so `09-47-51.303_OpenAI_response.json` could be any of them — grouping
    /// failures "by hour" across such a directory silently mixes days, and the file's mtime
    /// was the only way back to the real date. `en_US_POSIX` + a fixed format keeps the name
    /// stable regardless of the user's locale or 12/24-hour setting.
    static func timestamp() -> String {
        // DateFormatter is not thread-safe — create per-call to avoid data races
        // when multiple providers log concurrently.
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd_HH-mm-ss.SSS"
        return f.string(from: Date())
    }

    /// Monotonic per-process counter behind `uniqueStamp()`. A lock rather than a bare `static var`
    /// because every provider logs from whatever task it happens to run on.
    private static let sequence = OSAllocatedUnfairLock(initialState: 0)

    /// A `timestamp()` followed by a monotonic sequence number — unique for the life of the process.
    ///
    /// The timestamp alone is millisecond-resolution and providers log CONCURRENTLY: a parallel
    /// tool-call batch fires several evaluations that land in the same millisecond. Those computed
    /// identical filenames, and `write(to:atomically:)` let the last one win — so the log silently
    /// destroyed exactly the requests most worth comparing against each other, and the console
    /// showed two identical summary lines that looked like double-logging rather than two calls.
    /// Both the filename and the summary carry the sequence, so neither is ambiguous now.
    ///
    /// A counter rather than random characters because it also records the ORDER the colliding
    /// calls were made, which a random tie-break throws away. It never wraps — uniqueness is the
    /// entire point — so past 99999 calls the field simply widens.
    static func uniqueStamp() -> String {
        let ordinal = sequence.withLock { (count: inout Int) -> Int in
            count += 1
            return count
        }
        return "\(timestamp())-\(String(format: "%05d", ordinal))"
    }

    // MARK: - Request/response correlation

    /// Handed back by `logRequest` / `logBodylessRequest` and passed to `logResponse` so a response
    /// can be tied to the request that produced it.
    ///
    /// Without it the two files were uncorrelated: each drew its own stamp, and once several calls
    /// are in flight — which is the normal case, not the exception — nothing pairs them. Matching by
    /// time doesn't work either, since responses arrive out of order.
    ///
    /// Construction is deliberately internal. A token is evidence that a request was actually
    /// logged, so the only way to get one is to log a request.
    public struct RequestLogToken: Sendable {
        /// The unique stamp the request file was named with. The response reuses it verbatim, so
        /// the pair shares a filename prefix and sorts adjacent in the log directory.
        public let stamp: String
        /// Filename-safe model, when the request had one. Carried so the response filename names the
        /// model too — it never did, which made same-millisecond responses from different models
        /// indistinguishable by name.
        public let modelSegment: String?
        /// When the request went out. Becomes the `+NNNms` on the response summary — the latency was
        /// previously only recoverable by subtracting two filenames by hand.
        public let sentAt: Date
    }

    /// The filename prefix a request and its response share. Both sides compose the name through
    /// here so a change to the scheme cannot desynchronize the pair.
    static func fileStem(stamp: String, label: String, modelSegment: String?) -> String {
        guard let modelSegment, !modelSegment.isEmpty else { return "\(stamp)_\(label)" }
        return "\(stamp)_\(label)_\(modelSegment)"
    }

    /// Logs a full request body to a JSON file and prints a console summary.
    ///
    /// - Returns: a token to hand to `logResponse` so the two files pair up.
    @discardableResult
    public static func logRequest(
        label: String,
        url: URL,
        model: String,
        body: [String: Any],
        rawData: Data
    ) -> RequestLogToken {
        let stamp = uniqueStamp()
        let safeModel = model.replacingOccurrences(of: "/", with: "_")
        let prefix = fileStem(stamp: stamp, label: label, modelSegment: safeModel)

        let toolCount: Int = {
            // OpenAI/Anthropic use "tools", Gemini wraps in functionDeclarations
            if let tools = body["tools"] as? [[String: Any]] {
                // Gemini nests tools inside one or more functionDeclarations arrays
                let geminiCount = tools.reduce(0) { sum, entry in
                    sum + ((entry["functionDeclarations"] as? [[String: Any]])?.count ?? 0)
                }
                return geminiCount > 0 ? geminiCount : tools.count
            }
            return 0
        }()
        // OpenAI/Anthropic use "messages", Gemini uses "contents"
        let messageCount = (body["messages"] as? [[String: Any]])?.count
            ?? (body["contents"] as? [[String: Any]])?.count
            ?? 0
        print("[\(label)] REQUEST \(stamp) → \(url.absoluteString) model=\(model) messages=\(messageCount) tools=\(toolCount)")

        if let pretty = try? JSONSerialization.data(withJSONObject: body, options: [.prettyPrinted, .sortedKeys]),
           let prettyString = String(data: pretty, encoding: .utf8) {
            let file = logDirectory.appendingPathComponent("\(prefix)_request.json")
            do {
                try prettyString.write(to: file, atomically: true, encoding: .utf8)
                print("[\(label)]   Full request logged to \(file.path)")
            } catch {
                loggerOS.warning("Failed to write request log to \(file.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        }

        return RequestLogToken(stamp: stamp, modelSegment: safeModel, sentAt: Date())
    }

    /// Logs a request that carries no JSON body — a `GET /models`, say.
    ///
    /// Exists so bodyless calls land in the same directory and timeline as chat traffic. Model
    /// fetches previously logged to a directory of their own, which meant that in a session full
    /// of chat logs the model calls looked absent rather than merely elsewhere.
    ///
    /// Headers are deliberately not recorded: for these requests the only interesting one is
    /// `Authorization`, and writing API keys to `$TMPDIR` is not a trade worth making.
    ///
    /// - Returns: a token to hand to `logResponse` so the two files pair up.
    @discardableResult
    public static func logBodylessRequest(
        label: String,
        method: String,
        url: URL
    ) -> RequestLogToken {
        let stamp = uniqueStamp()
        print("[\(label)] REQUEST \(stamp) → \(method) \(url.absoluteString)")

        let file = logDirectory.appendingPathComponent("\(fileStem(stamp: stamp, label: label, modelSegment: nil))_request.txt")
        do {
            try "\(method) \(url.absoluteString)\n".write(to: file, atomically: true, encoding: .utf8)
        } catch {
            loggerOS.warning("Failed to write request log to \(file.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }

        return RequestLogToken(stamp: stamp, modelSegment: nil, sentAt: Date())
    }

    /// Logs a full response body to a JSON file and prints a console summary.
    ///
    /// - Parameter token: the token `logRequest` returned for the call being answered. Supplying it
    ///   names the response file with the REQUEST's stamp, so the pair shares a prefix and sorts
    ///   adjacent, and prints the round-trip latency. Passing `nil` means what it says — no request
    ///   was logged for this response — and the file falls back to a stamp of its own, unpaired.
    public static func logResponse(
        label: String,
        statusCode: Int,
        data: Data,
        for token: RequestLogToken? = nil
    ) {
        let stamp = token?.stamp ?? uniqueStamp()
        let latency = token.map { " +\(Int(Date().timeIntervalSince($0.sentAt) * 1000))ms" } ?? ""
        let size = data.count
        var summary = "status=\(statusCode) bytes=\(size)"

        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // OpenAI/Anthropic-style: choices[0].message or content blocks
            if let choices = json["choices"] as? [[String: Any]],
               let message = choices.first?["message"] as? [String: Any] {
                let hasContent = (message["content"] as? String).map { !$0.isEmpty } ?? false
                let toolCallCount = (message["tool_calls"] as? [[String: Any]])?.count ?? 0
                summary += " hasContent=\(hasContent) toolCalls=\(toolCallCount)"
            } else if let message = json["message"] as? [String: Any] {
                // Ollama-style
                let hasContent = (message["content"] as? String).map { !$0.isEmpty } ?? false
                let toolCallCount = (message["tool_calls"] as? [[String: Any]])?.count ?? 0
                summary += " hasContent=\(hasContent) toolCalls=\(toolCallCount)"
            } else if let contentBlocks = json["content"] as? [[String: Any]] {
                // Anthropic-style
                let textBlocks = contentBlocks.filter { ($0["type"] as? String) == "text" }.count
                let toolBlocks = contentBlocks.filter { ($0["type"] as? String) == "tool_use" }.count
                summary += " textBlocks=\(textBlocks) toolUseBlocks=\(toolBlocks)"
            }
        }
        print("[\(label)] RESPONSE \(stamp)\(latency) \(summary)")

        let stem = fileStem(stamp: stamp, label: label, modelSegment: token?.modelSegment)
        let file = logDirectory.appendingPathComponent("\(stem)_response.json")
        if let parsed = try? JSONSerialization.jsonObject(with: data),
           let pretty = try? JSONSerialization.data(withJSONObject: parsed, options: [.prettyPrinted, .sortedKeys]),
           let prettyString = String(data: pretty, encoding: .utf8) {
            do {
                try prettyString.write(to: file, atomically: true, encoding: .utf8)
                print("[\(label)]   Full response logged to \(file.path)")
            } catch {
                loggerOS.warning("Failed to write response log to \(file.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        } else {
            do {
                try data.write(to: file)
                print("[\(label)]   Raw response logged to \(file.path)")
            } catch {
                loggerOS.warning("Failed to write raw response log to \(file.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
            }
        }
    }
}
