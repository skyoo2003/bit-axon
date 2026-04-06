import Foundation
import MLX
import MLXNN

// MARK: - Mock Tokenizer (v1 placeholder)

struct MockTokenizer: Sendable {
    let eosTokenId = 2
    private let vocabSize = 32_000

    func encode(_ text: String) -> [Int] {
        text.utf8.map { Int($0) % vocabSize }
    }

    func decode(_ ids: [Int]) -> String {
        String(
            ids.compactMap { id -> Character? in
                let byte = id % 128
                return byte >= 32 ? Character(UnicodeScalar(UInt8(byte))) : nil
            })
    }
}

// MARK: - ChatViewModel

@Observable
final class ChatViewModel: @unchecked Sendable {

    struct Message: Identifiable {
        let id = UUID()
        let role: String
        var content: String
    }

    var messages: [Message] = []
    var isGenerating = false
    var prompt = ""
    var tokensPerSecond: Double = 0.0
    var timeToFirstTokenMs: Double?
    var errorMessage: String?

    private let modelService: ModelService
    private let tokenizer = MockTokenizer()

    init(modelService: ModelService) {
        self.modelService = modelService
    }

    // MARK: - Generation

    func generate() async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let userText = trimmed

        messages.append(Message(role: "user", content: userText))
        let assistantIndex = messages.count
        messages.append(Message(role: "assistant", content: ""))
        prompt = ""
        isGenerating = true
        errorMessage = nil

        guard let model = await modelService.model else {
            errorMessage = "Model not loaded — tap Load Model first."
            isGenerating = false
            return
        }

        let tokenIds = tokenizer.encode(userText)
        guard !tokenIds.isEmpty else {
            errorMessage = "Tokenization produced no tokens."
            isGenerating = false
            return
        }

        await Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            nonisolated(unsafe) let modelRef = model

            let inputIds = MLXArray(tokenIds).reshaped([1, tokenIds.count])

            let startTime = Date()
            var (logits, caches) = modelRef(inputIds)
            eval(logits)

            let ttft = Date().timeIntervalSince(startTime) * 1000.0
            await MainActor.run { self.timeToFirstTokenMs = ttft }

            // Decode loop
            let maxTokens = 512
            var generatedIds: [Int] = []

            for _ in 0..<maxTokens {
                let nextLogits = logits[0..., -1]
                let nextId = Int(nextLogits.argMax().item(Int32.self))

                if nextId == self.tokenizer.eosTokenId { break }
                generatedIds.append(nextId)

                let partialText = self.tokenizer.decode(generatedIds)
                let elapsed = Date().timeIntervalSince(startTime)
                let tps = Double(generatedIds.count) / Swift.max(elapsed, 0.001)

                await MainActor.run {
                    self.messages[assistantIndex].content = partialText
                    self.tokensPerSecond = tps
                }

                // Autoregressive step — pass caches forward
                let nextInput = MLXArray([nextId]).reshaped([1, 1])
                (logits, caches) = modelRef(nextInput, cache: caches)
                eval(logits)
            }

            await MainActor.run {
                self.isGenerating = false
            }
        }.value
    }

    // MARK: - Helpers

    func clear() {
        messages.removeAll()
        prompt = ""
        errorMessage = nil
        tokensPerSecond = 0.0
        timeToFirstTokenMs = nil
    }

    func loadModel() async {
        do {
            try await modelService.loadDefault()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
