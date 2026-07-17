import Foundation

/// Third-party benchmark scores a provider surfaces for a model — OpenRouter's `benchmarks` block.
/// Pure reference metadata for model selection (which model is strongest at coding, agentic work,
/// design, …); it never drives behavior. Every field is optional because OpenRouter populates
/// different subsets per model, and many models carry none.
public struct ModelBenchmarks: Codable, Sendable, Equatable {
    /// Artificial Analysis composite indices (roughly 0–100; higher is better).
    public struct ArtificialAnalysis: Codable, Sendable, Equatable {
        public var intelligenceIndex: Double?
        public var codingIndex: Double?
        public var agenticIndex: Double?

        public init(intelligenceIndex: Double? = nil, codingIndex: Double? = nil, agenticIndex: Double? = nil) {
            self.intelligenceIndex = intelligenceIndex
            self.codingIndex = codingIndex
            self.agenticIndex = agenticIndex
        }

        enum CodingKeys: String, CodingKey {
            case intelligenceIndex = "intelligence_index"
            case codingIndex = "coding_index"
            case agenticIndex = "agentic_index"
        }

        public var isEmpty: Bool { intelligenceIndex == nil && codingIndex == nil && agenticIndex == nil }
    }

    /// One row of OpenRouter's Design Arena — a head-to-head leaderboard giving each model an
    /// Elo rating (chess-style), a rank, and a win rate within an arena/category.
    public struct DesignArenaEntry: Codable, Sendable, Equatable {
        public var arena: String?
        public var category: String?
        public var elo: Double?
        public var rank: Int?
        public var winRate: Double?

        public init(arena: String? = nil, category: String? = nil, elo: Double? = nil, rank: Int? = nil, winRate: Double? = nil) {
            self.arena = arena
            self.category = category
            self.elo = elo
            self.rank = rank
            self.winRate = winRate
        }

        enum CodingKeys: String, CodingKey {
            case arena, category, elo, rank
            case winRate = "win_rate"
        }
    }

    public var artificialAnalysis: ArtificialAnalysis?
    public var designArena: [DesignArenaEntry]?

    public init(artificialAnalysis: ArtificialAnalysis? = nil, designArena: [DesignArenaEntry]? = nil) {
        self.artificialAnalysis = artificialAnalysis
        self.designArena = designArena
    }

    enum CodingKeys: String, CodingKey {
        case artificialAnalysis = "artificial_analysis"
        case designArena = "design_arena"
    }

    /// Whether nothing is populated — used to store `nil` rather than an empty benchmarks block.
    public var isEmpty: Bool {
        (artificialAnalysis?.isEmpty ?? true) && (designArena?.isEmpty ?? true)
    }
}
