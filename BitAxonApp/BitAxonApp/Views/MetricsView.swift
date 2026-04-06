import SwiftUI

struct MetricsView: View {
    let tokensPerSecond: Double
    let ttftMs: Double?
    let deviceStat: DeviceStat

    var body: some View {
        HStack(spacing: 12) {
            MetricCard(
                title: "tok/s",
                value: String(format: "%.1f", tokensPerSecond),
                icon: "speedometer"
            )
            if let ttft = ttftMs {
                MetricCard(
                    title: "TTFT",
                    value: "\(Int(ttft))ms",
                    icon: "clock"
                )
            }
            MetricCard(
                title: "GPU",
                value: String(format: "%.1f GB", deviceStat.gpuMemoryUsed),
                icon: "memorychip"
            )
            MetricCard(
                title: "Temp",
                value: deviceStat.socTemperature,
                icon: "thermometer"
            )
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(value)
                    .font(.caption.monospacedDigit())
                Text(title)
                    .font(.caption2)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}
