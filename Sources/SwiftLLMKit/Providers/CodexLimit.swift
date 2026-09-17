import Foundation

/// A limit the Codex backend reported, read from its own typed fields rather than from prose.
///
/// The distinction this type exists to preserve: **a rate limit and an exhausted allowance are not
/// the same failure.** One clears in seconds and is worth retrying; the other clears at a stated
/// time, or — for credits — not by waiting at all. Treating them alike either burns a retry budget
/// against a window that will not move for hours, or abandons work that would have succeeded on the
/// next attempt.
///
/// Every discriminator here is a field the backend sets (`type`, `rate_limit_reached_type`), never a
/// phrase matched out of a message. The messages are written for humans and get reworded; the codes
/// do not.
public struct CodexLimit: Sendable, Equatable {

    public enum Kind: Sendable, Equatable {
        /// Sending too fast. Clears on its own; `Retry-After` applies.
        case rateLimited
        /// The plan's usage window is spent. Clears at `resetsAt` — waiting is the correct response,
        /// and the only one.
        case usageWindowExhausted(resetsAt: Date?)
        /// The credit balance is zero. **Has no reset**: waiting may never help, because the remedy
        /// is buying more or being granted more. `userCanResolve` is false for a workspace MEMBER,
        /// who has to ask an owner — which changes what the UI should say, not just what it does.
        case creditsDepleted(userCanResolve: Bool)
        /// An administrative spending cap, not a usage window. Also has no reset.
        case spendControlReached
    }

    public let kind: Kind
    /// Subscription tier, for display.
    public let planType: String?
    /// The backend's own name for the limit that tripped (e.g. a per-model window).
    public let limitName: String?
    /// How full the window was, 0–100, when the backend said so.
    public let usedPercent: Double?

    public init(kind: Kind, planType: String? = nil, limitName: String? = nil, usedPercent: Double? = nil) {
        self.kind = kind
        self.planType = planType
        self.limitName = limitName
        self.usedPercent = usedPercent
    }

    /// Whether waiting can resolve this on its own. Credits and spend caps cannot.
    public var clearsByWaiting: Bool {
        switch kind {
        case .rateLimited, .usageWindowExhausted: return true
        case .creditsDepleted, .spendControlReached: return false
        }
    }

    /// When the limit lifts, when the backend stated it. `nil` for the kinds that never do.
    public var resetsAt: Date? {
        if case .usageWindowExhausted(let date) = kind { return date }
        return nil
    }

    // MARK: - Parsing

    /// Reads a limit out of an error response, or `nil` when the response is not about a limit.
    ///
    /// Tolerant of where the fields sit: the backend nests them under `error`, under `detail`, or at
    /// the top level depending on the path, so the whole payload is searched for the typed keys
    /// rather than one shape being assumed. A 429 with no recognisable payload still reports
    /// `.rateLimited`, since the status alone says that much.
    public static func parse(statusCode: Int, body: String) -> CodexLimit? {
        guard let data = body.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return statusCode == 429 ? CodexLimit(kind: .rateLimited) : nil }

        let fields = flatten(root)
        let reachedType = fields["rate_limit_reached_type"] as? String
        let errorType = (fields["type"] as? String) ?? (fields["code"] as? String)
        let planType = fields["plan_type"] as? String
        let limitName = (fields["limit_name"] as? String) ?? (fields["metered_feature"] as? String)
        let usedPercent = number(fields["used_percent"]) ?? number(fields["percent_used"])

        // `spend_control_reached` is a flag, not a type, and outranks the rest: an administrative
        // cap does not lift when the window rolls over.
        if fields["spend_control_reached"] as? Bool == true {
            return CodexLimit(kind: .spendControlReached, planType: planType,
                              limitName: limitName, usedPercent: usedPercent)
        }

        let kind: Kind?
        switch reachedType {
        case "workspace_owner_credits_depleted":
            kind = .creditsDepleted(userCanResolve: true)
        case "workspace_member_credits_depleted":
            // A member cannot top up the workspace; telling them to buy credits would be wrong.
            kind = .creditsDepleted(userCanResolve: false)
        case "workspace_owner_usage_limit_reached", "workspace_member_usage_limit_reached":
            kind = .usageWindowExhausted(resetsAt: resetDate(from: fields))
        case "rate_limit_reached":
            kind = .rateLimited
        default:
            switch errorType {
            case "usage_limit_reached", "usage_limit_exceeded":
                kind = .usageWindowExhausted(resetsAt: resetDate(from: fields))
            case "rate_limit_exceeded":
                kind = .rateLimited
            default:
                // Nothing typed said "limit". A 429 still means one, and nothing else does — a
                // guess from the message text is exactly what this type exists to avoid.
                kind = statusCode == 429 ? .rateLimited : nil
            }
        }
        guard let kind else { return nil }
        return CodexLimit(kind: kind, planType: planType, limitName: limitName, usedPercent: usedPercent)
    }

    /// The reset instant, from whichever form the payload used.
    ///
    /// Three spellings are accepted because three are in circulation: an absolute `resets_at` (epoch
    /// seconds or ISO-8601) and a relative `resets_in_seconds`. A relative value is resolved against
    /// `now` at parse time, which is correct — it was measured when the response was produced.
    static func resetDate(from fields: [String: Any], now: Date = Date()) -> Date? {
        if let seconds = number(fields["resets_in_seconds"]) {
            return now.addingTimeInterval(seconds)
        }
        for key in ["resets_at", "resetsAt", "reset_at"] {
            guard let value = fields[key] else { continue }
            if let epoch = number(value) {
                // Milliseconds are distinguishable from seconds by magnitude: a seconds-epoch this
                // large would be year 33658, which no reset window reaches.
                return Date(timeIntervalSince1970: epoch > 4_000_000_000 ? epoch / 1000 : epoch)
            }
            if let text = value as? String {
                let withFraction = ISO8601DateFormatter()
                withFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                if let date = withFraction.date(from: text) ?? ISO8601DateFormatter().date(from: text) {
                    return date
                }
            }
        }
        return nil
    }

    /// Every scalar in the payload, keyed by its own name, nested objects included.
    ///
    /// Shallower keys win, so a top-level `type` is not shadowed by one buried in an unrelated
    /// sub-object. Arrays are walked because `additional_rate_limits` is one.
    static func flatten(_ object: [String: Any]) -> [String: Any] {
        var out: [String: Any] = [:]
        var queue: [[String: Any]] = [object]
        while !queue.isEmpty {
            let level = queue.removeFirst()
            for (key, value) in level {
                if out[key] == nil { out[key] = value }
                if let nested = value as? [String: Any] {
                    queue.append(nested)
                } else if let array = value as? [Any] {
                    queue.append(contentsOf: array.compactMap { $0 as? [String: Any] })
                }
            }
        }
        return out
    }

    private static func number(_ value: Any?) -> Double? {
        if let double = value as? Double { return double }
        if let int = value as? Int { return Double(int) }
        if let string = value as? String { return Double(string) }
        return nil
    }
}
