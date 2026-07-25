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

    /// Logs a full request body to a JSON file and prints a console summary.
    public static func logRequest(
        label: String,
        url: URL,
        model: String,
        body: [String: Any],
        rawData: Data
    ) {
        let stamp = timestamp()
        let safeModel = model.replacingOccurrences(of: "/", with: "_")
        let prefix = "\(stamp)_\(label)_\(safeModel)"

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
    }

    /// Logs a request that carries no JSON body — a `GET /models`, say.
    ///
    /// Exists so bodyless calls land in the same directory and timeline as chat traffic. Model
    /// fetches previously logged to a directory of their own, which meant that in a session full
    /// of chat logs the model calls looked absent rather than merely elsewhere.
    ///
    /// Headers are deliberately not recorded: for these requests the only interesting one is
    /// `Authorization`, and writing API keys to `$TMPDIR` is not a trade worth making.
    public static func logBodylessRequest(
        label: String,
        method: String,
        url: URL
    ) {
        let stamp = timestamp()
        print("[\(label)] REQUEST \(stamp) → \(method) \(url.absoluteString)")

        let file = logDirectory.appendingPathComponent("\(stamp)_\(label)_request.txt")
        do {
            try "\(method) \(url.absoluteString)\n".write(to: file, atomically: true, encoding: .utf8)
        } catch {
            loggerOS.warning("Failed to write request log to \(file.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Logs a full response body to a JSON file and prints a console summary.
    public static func logResponse(
        label: String,
        statusCode: Int,
        data: Data
    ) {
        let stamp = timestamp()
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
        print("[\(label)] RESPONSE \(stamp) \(summary)")

        let file = logDirectory.appendingPathComponent("\(stamp)_\(label)_response.json")
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
