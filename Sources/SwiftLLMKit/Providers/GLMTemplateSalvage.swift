import Foundation
import os

private let logger = Logger(subsystem: "SwiftLLMKit", category: "GLMTemplateSalvage")

/// Provider-agnostic salvage for GLM models' chat-template leakage.
///
/// GLM-4 / GLM-5 (Zhipu AI) emit tool calls in a native chat-template form that
/// not every adapter translates back to the OpenAI / Ollama tool-call schema:
///
///     <tool_call>name
///     <arg_key>k</arg_key><arg_value>v</arg_value>
///     </tool_call>
///
/// They also leak chat-template control tokens (`<|observation|>`, `<|assistant|>`,
/// `<tool_response>...</tool_response>`) into the assistant `content`. When stored
/// in conversation history and replayed on the next request, those tokens drive the
/// model into a recursive template-echo loop — the observable failure mode is a
/// 400+-second response that is just thousands of repeated `<tool_response>Tool error:
/// Missing required argument: ...</tool_response>` blocks.
///
/// The same leakage shows up regardless of which provider routes the request:
/// z.ai direct, OpenRouter, Ollama (cloud), OpenAI-compatible adapters in front of
/// GLM hosting, etc. So this helper is provider-agnostic and gates entirely on
/// content shape rather than the routing `ProviderAPIType`.
enum GLMTemplateSalvage {

    // MARK: - Detection

    /// True if `content` contains any GLM chat-template marker we know how to parse
    /// or strip. Cheap early-exit for the common case where there's nothing to do.
    static func contentLooksGLMTemplated(_ content: String) -> Bool {
        content.contains("<arg_key>")
            || content.contains("<tool_call>")
            || content.contains("<tool_response>")
            || content.contains("<|")
    }

    // MARK: - Salvaged tool call (used by providers that don't get any structured
    // tool_calls back at all, e.g. Ollama (cloud) relaying GLM raw output).

    /// A tool call recovered entirely from `content` text — name and args both came
    /// out of `<tool_call>...<arg_key>/<arg_value></tool_call>` blocks.
    struct SalvagedCall {
        let name: String
        /// JSON-encoded args string, ready to drop into `LLMToolCall.arguments`.
        let arguments: String
    }

    /// Parses every `<tool_call>NAME ... <arg_key>K</arg_key><arg_value>V</arg_value> ... </tool_call>`
    /// block in `content` into structured calls. Returns calls in document order.
    /// `arg_value` is parsed as JSON first (so `["…"]` stays an array); on parse
    /// failure it falls back to bare string. Skips blocks where the name is missing.
    static func extractFullCalls(content: String) -> [SalvagedCall] {
        var result: [SalvagedCall] = []
        for block in toolCallBlocks(content) {
            guard let name = leadingToolName(in: block) else { continue }
            let pairs = argPairs(block)
            guard !pairs.isEmpty else { continue }
            if let argsJSON = jsonArgs(from: pairs) {
                result.append(SalvagedCall(name: name, arguments: argsJSON))
            }
        }
        return result
    }

    // MARK: - Args-only salvage (used when the adapter populated tool_calls with
    // names + ids but left arguments empty / `{}`).

    /// Returns the salvaged JSON args strings in document order — slot N corresponds
    /// to the Nth `<tool_call>` block found in `content`. Tool calls whose args
    /// couldn't be parsed are absent from the result, so callers should match by
    /// position into their list of empty-args tool_calls.
    static func extractArgsByPosition(content: String) -> [String] {
        var result: [String] = []
        for block in toolCallBlocks(content) {
            let pairs = argPairs(block)
            guard !pairs.isEmpty else { continue }
            if let argsJSON = jsonArgs(from: pairs) {
                result.append(argsJSON)
            }
        }
        return result
    }

    // MARK: - Stripping

    /// Strips GLM chat-template control tokens from `content` so they don't poison
    /// the next request when fed back through conversation history. Removes:
    ///
    ///   - `<|...|>` control tokens (e.g. `<|observation|>`, `<|assistant|>`).
    ///   - `<tool_call>...</tool_call>` blocks (already consumed by salvage).
    ///   - `<tool_response>...</tool_response>` blocks (echoes of past tool results).
    ///
    /// Idempotent on content without those markers — safe to call unconditionally.
    /// Unmatched opens (truncated runaway responses) drop through end-of-string so a
    /// half-formed echo loop can't survive into stored history.
    static func strip(_ content: String) -> String {
        var working = content
        working = removeBlocks(in: working, openTag: "<tool_call>", closeTag: "</tool_call>")
        working = removeBlocks(in: working, openTag: "<tool_response>", closeTag: "</tool_response>")
        working = removeControlPipeTokens(working)
        return working.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Patching helpers for provider parsers

    /// Patches missing args onto an existing array of tool calls using positional
    /// salvage from `content`. Logs how many calls were patched. Returns the new
    /// array (or the original if nothing to patch).
    static func patchEmptyArgs(_ toolCalls: [LLMToolCall], content: String) -> [LLMToolCall] {
        let hasEmpty = toolCalls.contains(where: { $0.arguments.isEmpty || $0.arguments == "{}" })
        guard hasEmpty, content.contains("<arg_key>") else { return toolCalls }

        let salvaged = extractArgsByPosition(content: content)
        guard !salvaged.isEmpty else { return toolCalls }

        var slot = 0
        var patched = 0
        var rebuilt: [LLMToolCall] = []
        rebuilt.reserveCapacity(toolCalls.count)
        for call in toolCalls {
            if call.arguments.isEmpty || call.arguments == "{}" {
                if slot < salvaged.count {
                    rebuilt.append(LLMToolCall(id: call.id, name: call.name, arguments: salvaged[slot]))
                    slot += 1
                    patched += 1
                } else {
                    rebuilt.append(call)
                }
            } else {
                rebuilt.append(call)
            }
        }
        if patched > 0 {
            logger.info("GLM template salvage: recovered args for \(patched) of \(toolCalls.count) tool call(s) from message content")
            return rebuilt
        }
        return toolCalls
    }

    // MARK: - Internal parsers

    /// Splits `content` into the inner bodies of `<tool_call>...</tool_call>` blocks.
    /// If no opening tag is present but `<arg_key>` is, treats the whole content as
    /// a single block — older GLM template variants emit only the inner pair list.
    private static func toolCallBlocks(_ content: String) -> [String] {
        var blocks: [String] = []
        var cursor = content.startIndex
        while let openRange = content.range(of: "<tool_call>", range: cursor..<content.endIndex) {
            let afterOpen = openRange.upperBound
            let closeRange = content.range(of: "</tool_call>", range: afterOpen..<content.endIndex)
            let blockEnd = closeRange?.lowerBound ?? content.endIndex
            blocks.append(String(content[afterOpen..<blockEnd]))
            cursor = closeRange?.upperBound ?? content.endIndex
        }
        if blocks.isEmpty, content.contains("<arg_key>") {
            blocks.append(content)
        }
        return blocks
    }

    /// Returns the tool name from a block whose first non-empty line is the name
    /// (the GLM `<tool_call>NAME\n<arg_key>...` shape). Returns nil if there's no
    /// non-empty leading line (e.g. blocks that begin directly with `<arg_key>`).
    private static func leadingToolName(in block: String) -> String? {
        let lines = block.split(whereSeparator: { $0 == "\n" || $0 == "\r" })
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            // The name line precedes any <arg_key> tag. If the first non-empty line
            // already contains `<arg_key>` there's no leading name to recover.
            if trimmed.hasPrefix("<arg_key>") { return nil }
            // Strip any trailing tag fragments so a malformed line like "name<arg_key>k</arg_key>..."
            // (no newline between name and pairs) still yields a valid name.
            if let tagStart = trimmed.range(of: "<") {
                let candidate = String(trimmed[trimmed.startIndex..<tagStart.lowerBound])
                    .trimmingCharacters(in: .whitespaces)
                return candidate.isEmpty ? nil : candidate
            }
            return trimmed
        }
        return nil
    }

    /// Extracts (key, value) pairs of the form `<arg_key>K</arg_key><arg_value>V</arg_value>`
    /// from a block, in document order. Whitespace around the captured strings is trimmed.
    private static func argPairs(_ block: String) -> [(String, String)] {
        var pairs: [(String, String)] = []
        var cursor = block.startIndex
        while let keyOpen = block.range(of: "<arg_key>", range: cursor..<block.endIndex),
              let keyClose = block.range(of: "</arg_key>", range: keyOpen.upperBound..<block.endIndex),
              let valOpen = block.range(of: "<arg_value>", range: keyClose.upperBound..<block.endIndex),
              let valClose = block.range(of: "</arg_value>", range: valOpen.upperBound..<block.endIndex) {
            let key = String(block[keyOpen.upperBound..<keyClose.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let value = String(block[valOpen.upperBound..<valClose.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !key.isEmpty {
                pairs.append((key, value))
            }
            cursor = valClose.upperBound
        }
        return pairs
    }

    /// Builds a JSON args string from key/value pairs. Each value is parsed as JSON
    /// first (so `["UUID"]` stays an array, `42` stays a number) and falls back to
    /// bare string on parse failure. Returns nil if serialization fails.
    private static func jsonArgs(from pairs: [(String, String)]) -> String? {
        var args: [String: Any] = [:]
        for (key, valueText) in pairs {
            if let valData = valueText.data(using: .utf8),
               let parsed = try? JSONSerialization.jsonObject(with: valData, options: [.fragmentsAllowed]) {
                args[key] = parsed
            } else {
                args[key] = valueText
            }
        }
        guard let argsData = try? JSONSerialization.data(withJSONObject: args),
              let argsString = String(data: argsData, encoding: .utf8) else {
            return nil
        }
        return argsString
    }

    /// Removes every `openTag…closeTag` block in `content`. Unmatched opens (no
    /// corresponding close) are dropped through end-of-string so a half-formed echo
    /// loop can't survive past the strip.
    private static func removeBlocks(in content: String, openTag: String, closeTag: String) -> String {
        var result = ""
        var cursor = content.startIndex
        while let openRange = content.range(of: openTag, range: cursor..<content.endIndex) {
            result.append(contentsOf: content[cursor..<openRange.lowerBound])
            if let closeRange = content.range(of: closeTag, range: openRange.upperBound..<content.endIndex) {
                cursor = closeRange.upperBound
            } else {
                cursor = content.endIndex
            }
        }
        result.append(contentsOf: content[cursor..<content.endIndex])
        return result
    }

    /// Strips `<|...|>` control tokens (e.g. `<|observation|>`, `<|assistant|>`).
    /// Greedy on the inner segment so a multi-character token name is consumed in one go.
    private static func removeControlPipeTokens(_ content: String) -> String {
        var result = ""
        var cursor = content.startIndex
        while let openRange = content.range(of: "<|", range: cursor..<content.endIndex) {
            result.append(contentsOf: content[cursor..<openRange.lowerBound])
            if let closeRange = content.range(of: "|>", range: openRange.upperBound..<content.endIndex) {
                cursor = closeRange.upperBound
            } else {
                cursor = content.endIndex
            }
        }
        result.append(contentsOf: content[cursor..<content.endIndex])
        return result
    }
}
