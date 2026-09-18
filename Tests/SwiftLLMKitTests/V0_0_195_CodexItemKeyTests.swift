import Foundation
import Testing
@testable import SwiftLLMKit

/// The item key is the ITEM'S ID, and nothing else.
///
/// `response.output_item.added` carries it as `item.id`; every delta carries it as `item_id`.
/// `output_index` is the item's POSITION — a different identity — and was previously a fallback on
/// BOTH sides. Because the two sides prefer different fields, either side can fall through alone,
/// and then one side keys by id while the other keys by position: the arguments never join their
/// call. That is the same defect class that shipped in 0.0.187 with every tool call arriving `{}`.
///
/// These tests pin the outcome: an event that cannot name its item is DROPPED and logged, never
/// filed under a position where it may or may not meet the other half by luck.
@Suite("Codex item keying")
struct CodexItemKeyTests {

    /// The crossing, in the direction the old code hit when the ADDED side fell through:
    /// added has no `item.id` (so it keyed by position), deltas carry `item_id` (so they keyed by
    /// id). The old parser emitted the call anyway — with `{}` for arguments the model did send.
    @Test("A function_call item with no id is dropped, not emitted with empty arguments")
    func unkeyedAddedItemIsDropped() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,\
            "item":{"type":"function_call","call_id":"call_xyz","name":"get_weather"}}
            data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,\
            "delta":"{\\"city\\":\\"Paris\\"}"}
            data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,\
            "arguments":"{\\"city\\":\\"Paris\\"}"}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        #expect(response.toolCalls.isEmpty, """
            an item that cannot name itself must not be filed under its position: joining by \
            position is how arguments reached the wrong call, and emitting the call with `{}` \
            executes a tool the model meant to call with real arguments
            """)
    }

    /// The accepted regression, pinned deliberately rather than left to be discovered.
    ///
    /// When NEITHER side carries an id, the old code keyed both by `output_index` and the join
    /// worked — by accident, on an identity it had no right to use. It is now dropped. The trade is
    /// deliberate: a dropped call is logged and bounded (the agent loop sees a turn with no call),
    /// where a positional join is a silent guess that is wrong the moment the shapes diverge.
    @Test("With no id on either side, the call is dropped rather than joined by position")
    func positionalJoinNoLongerHappens() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,\
            "item":{"type":"function_call","call_id":"c1","name":"f"}}
            data: {"type":"response.function_call_arguments.done","output_index":0,\
            "arguments":"{\\"a\\":1}"}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        #expect(response.toolCalls.isEmpty)
    }

    /// The other direction: the added side names its item, the deltas do not. The call is real and
    /// survives; only its arguments are unjoinable. `{}` is the honest result — and the orphaned
    /// argument text must NOT be swept into some other item's bucket.
    @Test("Unkeyed argument deltas never reach a call they cannot name")
    func unkeyedDeltasDoNotCross() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,\
            "item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"f"}}
            data: {"type":"response.function_call_arguments.done","output_index":0,\
            "arguments":"{\\"a\\":1}"}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        let call = try #require(response.toolCalls.first)
        #expect(call.id == "c1")
        #expect(call.arguments == "{}")
    }

    /// One well-formed call is unaffected by a malformed sibling. The failure is per-item, not
    /// per-stream: a stream that drops one call must still deliver the others intact.
    @Test("A dropped item does not disturb a well-formed sibling")
    func siblingCallSurvives() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,\
            "item":{"type":"function_call","id":"fc_A","call_id":"cA","name":"alpha"}}
            data: {"type":"response.output_item.added","output_index":1,\
            "item":{"type":"function_call","call_id":"cB","name":"beta"}}
            data: {"type":"response.function_call_arguments.done","item_id":"fc_A","output_index":0,\
            "arguments":"{\\"a\\":1}"}
            data: {"type":"response.function_call_arguments.done","item_id":"fc_B","output_index":1,\
            "arguments":"{\\"b\\":2}"}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        #expect(response.toolCalls.count == 1)
        let call = try #require(response.toolCalls.first)
        #expect(call.name == "alpha")
        #expect(call.arguments == #"{"a":1}"#)
    }

    /// TEXT is the case the strictness must NOT reach. A text bucket is only grouped and ordered —
    /// it is never matched against an item registered by another event — so there is no second
    /// identity for a position to cross with, and dropping the delta would silently lose the
    /// answer with nothing downstream to notice.
    @Test("Text deltas with only an output_index still land in the answer")
    func positionalTextStillLands() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}
            data: {"type":"response.output_text.delta","output_index":0,"delta":"partial "}
            data: {"type":"response.output_text.delta","item_id":"m","delta":"answer"}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        #expect(response.text == "partial answer", """
            losing the assistant's text is a silent wrong answer; unlike a tool call there is no \
            join for a position to corrupt here
            """)
    }
}

/// `response.output_item.done` is the terminal, SELF-CONTAINED form of an output item: id,
/// `call_id`, `name` and the complete `arguments` string in one event. The live endpoint sends it
/// for every function_call item and the parser ignored it entirely — so a stream whose deltas
/// could not be joined produced `{}` while the whole call was sitting in an event being skipped.
@Suite("Codex terminal output item")
struct CodexTerminalItemTests {

    @Test("The terminal item rescues a call whose argument deltas could not be keyed")
    func terminalItemSuppliesArguments() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,\
            "item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"f","arguments":"","status":"in_progress"}}
            data: {"type":"response.function_call_arguments.done","output_index":0,\
            "arguments":"{\\"a\\":1}"}
            data: {"type":"response.output_item.done","output_index":0,\
            "item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"f",\
            "arguments":"{\\"a\\":1}","status":"completed"}}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        let call = try #require(response.toolCalls.first)
        #expect(call.id == "c1")
        #expect(call.arguments == #"{"a":1}"#, """
            the terminal item carries the arguments outright; needing the deltas to join is what \
            made an unjoinable key silently mean "no arguments"
            """)
    }

    @Test("The terminal item completes the call it already registered, it does not duplicate it")
    func terminalItemDoesNotDuplicate() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,\
            "item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"f","arguments":"","status":"in_progress"}}
            data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,\
            "delta":"{\\"a\\":1}"}
            data: {"type":"response.output_item.done","output_index":0,\
            "item":{"type":"function_call","id":"fc_1","call_id":"c1","name":"f",\
            "arguments":"{\\"a\\":1}","status":"completed"}}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        #expect(response.toolCalls.count == 1)
        #expect(response.toolCalls.first?.arguments == #"{"a":1}"#)
    }

    @Test("A terminal message item is not mistaken for a call")
    func terminalMessageItemIgnored() throws {
        let sse = """
            data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m"}}
            data: {"type":"response.output_text.delta","item_id":"m","delta":"hi"}
            data: {"type":"response.output_item.done","output_index":0,\
            "item":{"type":"message","id":"m","status":"completed"}}
            data: {"type":"response.completed","response":{}}
            """
        let response = try CodexResponsesProvider.parseStream(sse)
        #expect(response.toolCalls.isEmpty)
        #expect(response.text == "hi")
    }
}
