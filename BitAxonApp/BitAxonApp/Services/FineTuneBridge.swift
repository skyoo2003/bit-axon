import Foundation

@Observable
final class FineTuneBridge: @unchecked Sendable {

    enum TrainingState: String {
        case idle, running, completed, failed, cancelled
    }

    @MainActor var state: TrainingState = .idle
    @MainActor var logLines: [String] = []
    @MainActor var currentStep: Int = 0
    @MainActor var currentLoss: Double?
    @MainActor var progress: Double = 0.0
    @MainActor var errorMessage: String?

    var dataPath: String = ""
    var modelWeightsPath: String = ""
    var tokenizerPath: String = "Qwen/Qwen2.5-3B"
    var learningRate: Double = 1e-4
    var maxSteps: Int = 10_000
    var loraRank: Int = 8
    var batchSize: Int = 1
    var gradAccumSteps: Int = 4
    var maxSeqLen: Int = 2048
    var outputDir: String = "checkpoints"
    var useDoRA: Bool = true
    var enableThermal: Bool = true

    private var process: Process?

    // MARK: - CLI Discovery

    private func findBitAxonCLI() -> String? {
        if let found = findInPATH() { return found }
        let fallbacks: [String] = [
            "/usr/local/bin/bit-axon",
            URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent(".local/bin/bit-axon").path,
        ]
        for path in fallbacks {
            if FileManager.default.isExecutableFile(atPath: path) { return path }
        }
        return nil
    }

    private func findInPATH() -> String? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        proc.arguments = ["bit-axon"]

        let pipe = Pipe()
        proc.standardOutput = pipe.fileHandleForReading
        proc.standardError = FileHandle.nullDevice

        do {
            try proc.run()
            proc.waitUntilExit()
            guard proc.terminationStatus == 0 else { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return nil
        }
    }

    // MARK: - Training

    func startTraining() async {
        await MainActor.run {
            state = .running
            logLines = []
            currentStep = 0
            currentLoss = nil
            progress = 0.0
            errorMessage = nil
        }

        guard let cliPath = findBitAxonCLI() else {
            await MainActor.run {
                state = .failed
                errorMessage = "bit-axon CLI not found. Install with: pip install bit-axon"
                logLines = ["ERROR: bit-axon CLI not found."]
            }
            return
        }

        let dataPath = self.dataPath
        let modelWeightsPath = self.modelWeightsPath
        let tokenizerPath = self.tokenizerPath
        let learningRate = self.learningRate
        let maxSteps = self.maxSteps
        let loraRank = self.loraRank
        let batchSize = self.batchSize
        let gradAccumSteps = self.gradAccumSteps
        let maxSeqLen = self.maxSeqLen
        let outputDir = self.outputDir
        let useDoRA = self.useDoRA
        let enableThermal = self.enableThermal

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: cliPath)
        proc.arguments = buildArguments(
            dataPath: dataPath,
            modelWeightsPath: modelWeightsPath,
            tokenizerPath: tokenizerPath,
            learningRate: learningRate,
            maxSteps: maxSteps,
            loraRank: loraRank,
            batchSize: batchSize,
            gradAccumSteps: gradAccumSteps,
            maxSeqLen: maxSeqLen,
            outputDir: outputDir,
            useDoRA: useDoRA,
            enableThermal: enableThermal
        )

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        proc.standardOutput = stdoutPipe.fileHandleForWriting
        proc.standardError = stderrPipe.fileHandleForWriting

        self.process = proc

        do {
            try proc.run()
        } catch {
            await MainActor.run {
                state = .failed
                errorMessage = "Failed to launch bit-axon: \(error.localizedDescription)"
            }
            return
        }

        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                await self.streamOutput(from: stdoutPipe.fileHandleForReading)
            }
            group.addTask {
                await self.streamOutput(from: stderrPipe.fileHandleForReading)
            }
        }

        proc.waitUntilExit()

        await MainActor.run {
            if proc.terminationStatus == 0 {
                state = .completed
                progress = 1.0
            } else if proc.terminationStatus == SIGTERM || proc.terminationStatus == SIGKILL {
                state = .cancelled
            } else {
                state = .failed
                errorMessage = "Training exited with code \(proc.terminationStatus)"
            }
            self.process = nil
        }
    }

    private func buildArguments(
        dataPath: String,
        modelWeightsPath: String,
        tokenizerPath: String,
        learningRate: Double,
        maxSteps: Int,
        loraRank: Int,
        batchSize: Int,
        gradAccumSteps: Int,
        maxSeqLen: Int,
        outputDir: String,
        useDoRA: Bool,
        enableThermal: Bool
    ) -> [String] {
        var args: [String] = [
            "train", dataPath,
            "--model-weights", modelWeightsPath,
            "--tokenizer", tokenizerPath,
            "--learning-rate", String(learningRate),
            "--max-steps", String(maxSteps),
            "--lora-rank", String(loraRank),
            "--batch-size", String(batchSize),
            "--grad-accum-steps", String(gradAccumSteps),
            "--max-seq-len", String(maxSeqLen),
            "--output-dir", outputDir,
        ]
        if !useDoRA { args.append("--no-dora") }
        if !enableThermal { args.append("--no-thermal") }
        return args
    }

    private func streamOutput(from handle: FileHandle) async {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else {
                    continuation.resume()
                    return
                }
                var buffer = ""
                let maxLogLines = 5000

                handle.readabilityHandler = { fh in
                    let data = fh.availableData
                    if data.isEmpty {
                        fh.readabilityHandler = nil
                        continuation.resume()
                        return
                    }
                    guard let str = String(data: data, encoding: .utf8) else { return }
                    buffer += str

                    while let nlRange = buffer.rangeOfCharacter(from: .newlines) {
                        let line = String(buffer[buffer.startIndex..<nlRange.lowerBound])
                        buffer = String(buffer[buffer.index(after: nlRange.lowerBound)...])
                        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !trimmed.isEmpty else { continue }

                        let (step, loss) = self.parseStepAndLoss(from: trimmed)

                        Task { @MainActor in
                            if self.logLines.count >= maxLogLines {
                                self.logLines.removeFirst(self.logLines.count - maxLogLines + 100)
                            }
                            self.logLines.append(trimmed)
                            if let step { self.currentStep = step }
                            if let loss { self.currentLoss = loss }
                            if let step, self.maxSteps > 0 {
                                self.progress = min(Double(step) / Double(self.maxSteps), 1.0)
                            }
                        }
                    }
                }
            }
        }
    }

    // Parses "Step 42/10000", "Loss: 2.345", "loss=2.345", "step: 42"
    private func parseStepAndLoss(from line: String) -> (step: Int?, loss: Double?) {
        var foundStep: Int?
        var foundLoss: Double?

        let stepLabels = ["Step ", "step: "]
        for label in stepLabels {
            if let labelRange = line.range(of: label, options: [.caseInsensitive]),
               let slashRange = line.range(of: "/", range: labelRange.upperBound..<line.endIndex) {
                let stepStr = line[labelRange.upperBound..<slashRange.lowerBound]
                    .trimmingCharacters(in: .whitespaces)
                foundStep = Int(stepStr)
                break
            }
        }

        let lossLabels = ["Loss: ", "loss=", "loss "]
        for label in lossLabels {
            if let labelRange = line.range(of: label, options: [.caseInsensitive]) {
                let valueStart = labelRange.upperBound
                let remaining = line[valueStart...]
                    .trimmingCharacters(in: .whitespaces)
                let numericEnd = remaining.prefix(while: { $0.isNumber || $0 == "." || $0 == "-" || $0 == "e" || $0 == "E" })
                foundLoss = Double(numericEnd)
                break
            }
        }

        return (foundStep, foundLoss)
    }

    func stopTraining() {
        process?.terminate()
    }

    func clear() {
        Task { @MainActor in
            logLines = []
            currentStep = 0
            currentLoss = nil
            progress = 0.0
            errorMessage = nil
            if state != .running {
                state = .idle
            }
        }
    }
}
