import Foundation
import MLX

@Observable
final class DeviceStat: @unchecked Sendable {
    @MainActor
    var gpuMemoryUsed: Float = 0
    @MainActor
    var gpuMemoryCache: Float = 0
    @MainActor
    var gpuMemoryPeak: Float = 0
    @MainActor
    var gpuMemoryLimit: Float = 0
    @MainActor
    var socTemperature: String = "N/A"

    private var timer: Timer?

    func startPolling(interval: TimeInterval = 2.0) {
        stopPolling()
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.updateStats()
        }
    }

    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }

    private func updateStats() {
        let snapshot = GPU.snapshot()
        let mb: Float = 1.0 / (1024 * 1024)

        let used = Float(snapshot.activeMemory) * mb
        let cache = Float(snapshot.cacheMemory) * mb
        let peak = Float(snapshot.peakMemory) * mb
        let limit = Float(GPU.memoryLimit) * mb

        let temp = readTemperature()

        DispatchQueue.main.async { [weak self] in
            self?.gpuMemoryUsed = used
            self?.gpuMemoryCache = cache
            self?.gpuMemoryPeak = peak
            self?.gpuMemoryLimit = limit
            self?.socTemperature = temp
        }
    }

    private func readTemperature() -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/powermetrics")
        process.arguments = ["--samplers", "thermal_pressure", "-i", "1", "-n", "1"]

        let pipe = Pipe()
        process.standardOutput = pipe.fileHandleForReading
        process.standardError = Pipe().fileHandleForWriting

        do {
            try process.run()
            process.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8),
               let range = output.range(of: "die temperature: ") {
                let remainder = output[range.upperBound...]
                let value = remainder.prefix(while: { $0.isNumber || $0 == "." })
                    .trimmingCharacters(in: .whitespaces)
                if let temp = Double(value) {
                    return String(format: "%.1f°C", temp)
                }
            }
        } catch {}
        return "N/A (requires sudo)"
    }

    deinit {
        stopPolling()
    }
}
