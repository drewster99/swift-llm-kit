import Foundation

/// Feature flags describing what a model supports.
public struct ModelCapabilities: Sendable, Equatable {
    public var toolUse: Bool
    public var vision: Bool
    public var reasoning: Bool
    public var codeExecution: Bool
    public var promptCaching: Bool
    public var computerUse: Bool
    public var audioInput: Bool
    public var audioOutput: Bool
    public var videoInput: Bool
    public var responseSchema: Bool
    public var parallelToolCalls: Bool
    public var pdfInput: Bool
    public var webSearch: Bool
    public var systemMessages: Bool
    public var assistantPrefill: Bool
    public var toolChoice: Bool
    /// Empirical-only: the model called a tool AND consumed the tool result (returned the probe's
    /// identifier). The half an agent actually depends on; no vendor publishes it.
    public var toolResultRoundTrip: Bool

    public init(
        toolUse: Bool = false,
        vision: Bool = false,
        reasoning: Bool = false,
        codeExecution: Bool = false,
        promptCaching: Bool = false,
        computerUse: Bool = false,
        audioInput: Bool = false,
        audioOutput: Bool = false,
        videoInput: Bool = false,
        responseSchema: Bool = false,
        parallelToolCalls: Bool = false,
        pdfInput: Bool = false,
        webSearch: Bool = false,
        systemMessages: Bool = false,
        assistantPrefill: Bool = false,
        toolChoice: Bool = false,
        toolResultRoundTrip: Bool = false
    ) {
        self.toolUse = toolUse
        self.vision = vision
        self.reasoning = reasoning
        self.codeExecution = codeExecution
        self.promptCaching = promptCaching
        self.computerUse = computerUse
        self.audioInput = audioInput
        self.audioOutput = audioOutput
        self.videoInput = videoInput
        self.responseSchema = responseSchema
        self.parallelToolCalls = parallelToolCalls
        self.pdfInput = pdfInput
        self.webSearch = webSearch
        self.systemMessages = systemMessages
        self.assistantPrefill = assistantPrefill
        self.toolChoice = toolChoice
        self.toolResultRoundTrip = toolResultRoundTrip
    }

    /// Human-readable labels for capabilities that are enabled.
    public var enabledLabels: [String] {
        var labels: [String] = []
        if toolUse { labels.append("Tools") }
        if vision { labels.append("Vision") }
        if reasoning { labels.append("Reasoning") }
        if codeExecution { labels.append("Code Exec") }
        if promptCaching { labels.append("Caching") }
        if computerUse { labels.append("Computer Use") }
        if audioInput { labels.append("Audio In") }
        if audioOutput { labels.append("Audio Out") }
        if videoInput { labels.append("Video In") }
        if responseSchema { labels.append("Schema") }
        if parallelToolCalls { labels.append("Parallel Tools") }
        if pdfInput { labels.append("PDF") }
        if webSearch { labels.append("Web Search") }
        if systemMessages { labels.append("System Msgs") }
        if assistantPrefill { labels.append("Prefill") }
        if toolChoice { labels.append("Tool Choice") }
        if toolResultRoundTrip { labels.append("Tool Round-Trip") }
        return labels
    }
}

// MARK: - Codable (backward-compatible)

extension ModelCapabilities: Codable {
    private enum CodingKeys: String, CodingKey {
        case toolUse, vision, reasoning, codeExecution, promptCaching
        case computerUse, audioInput, audioOutput, videoInput
        case responseSchema, parallelToolCalls
        case pdfInput, webSearch, systemMessages, assistantPrefill, toolChoice
        case toolResultRoundTrip
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        toolUse = try container.decodeIfPresent(Bool.self, forKey: .toolUse) ?? false
        vision = try container.decodeIfPresent(Bool.self, forKey: .vision) ?? false
        reasoning = try container.decodeIfPresent(Bool.self, forKey: .reasoning) ?? false
        codeExecution = try container.decodeIfPresent(Bool.self, forKey: .codeExecution) ?? false
        promptCaching = try container.decodeIfPresent(Bool.self, forKey: .promptCaching) ?? false
        computerUse = try container.decodeIfPresent(Bool.self, forKey: .computerUse) ?? false
        audioInput = try container.decodeIfPresent(Bool.self, forKey: .audioInput) ?? false
        audioOutput = try container.decodeIfPresent(Bool.self, forKey: .audioOutput) ?? false
        videoInput = try container.decodeIfPresent(Bool.self, forKey: .videoInput) ?? false
        responseSchema = try container.decodeIfPresent(Bool.self, forKey: .responseSchema) ?? false
        parallelToolCalls = try container.decodeIfPresent(Bool.self, forKey: .parallelToolCalls) ?? false
        pdfInput = try container.decodeIfPresent(Bool.self, forKey: .pdfInput) ?? false
        webSearch = try container.decodeIfPresent(Bool.self, forKey: .webSearch) ?? false
        systemMessages = try container.decodeIfPresent(Bool.self, forKey: .systemMessages) ?? false
        assistantPrefill = try container.decodeIfPresent(Bool.self, forKey: .assistantPrefill) ?? false
        toolChoice = try container.decodeIfPresent(Bool.self, forKey: .toolChoice) ?? false
        toolResultRoundTrip = try container.decodeIfPresent(Bool.self, forKey: .toolResultRoundTrip) ?? false
    }
}
