import SwiftUI

struct FineTuneView: View {
    @State private var bridge = FineTuneBridge()
    @State private var isExpanded = true
    @FocusState private var focusedField: Field?

    private enum Field: Hashable {
        case dataPath, modelWeights, tokenizer, outputDir
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    stateIndicator
                    dataDropZone
                    parametersSection
                    actionButtons
                    logSection
                }
                .padding()
            }
        }
        .navigationTitle("Fine-Tune")
    }

    // MARK: - State

    @ViewBuilder
    private var stateIndicator: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(stateColor)
                .frame(width: 8, height: 8)
            Text(bridge.state.rawValue.capitalized)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            if bridge.state == .running, bridge.currentLoss != nil {
                Text("Step \(bridge.currentStep) — Loss \(String(format: "%.4f", bridge.currentLoss!))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private var stateColor: Color {
        switch bridge.state {
        case .idle: .secondary
        case .running: .green
        case .completed: .blue
        case .failed: .red
        case .cancelled: .orange
        }
    }

    // MARK: - Data Drop Zone

    private var dataDropZone: some View {
        VStack(spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [6, 4]))
                    .foregroundStyle(.tertiary)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))

                if bridge.dataPath.isEmpty {
                    VStack(spacing: 4) {
                        Image(systemName: "doc.on.doc")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                        Text("Drop JSONL training data here")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, minHeight: 60)
                }
            }
            .overlay(alignment: .leading) {
                if !bridge.dataPath.isEmpty {
                    HStack {
                        Image(systemName: "doc.text")
                            .foregroundStyle(.secondary)
                        Text(URL(fileURLWithPath: bridge.dataPath).lastPathComponent)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button {
                            bridge.dataPath = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.tertiary)
                        }
                        .buttonStyle(.borderless)
                    }
                    .font(.caption)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                }
            }
            .onDrop(of: [.fileURL], isTargeted: nil) { providers in
                handleDrop(providers: providers)
            }
        }
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        provider.loadItem(forTypeIdentifier: "public.file-url", options: nil) { item, _ in
            guard let data = item as? Data,
                  let urlString = String(data: data, encoding: .utf8),
                  let url = URL(string: urlString) else { return }
            let ext = url.pathExtension.lowercased()
            if ext == "jsonl" || ext == "json" {
                Task { @MainActor in
                    bridge.dataPath = url.path
                }
            }
        }
        return true
    }

    // MARK: - Parameters

    private var parametersSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Parameters")
                        .font(.headline)
                    Spacer()
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) { isExpanded.toggle() }
                    } label: {
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.borderless)
                }

                if isExpanded {
                    Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 12, verticalSpacing: 8) {
                        pathRow(label: "Model Weights", value: $bridge.modelWeightsPath, placeholder: "/path/to/model", field: .modelWeights)
                        pathRow(label: "Tokenizer", value: $bridge.tokenizerPath, placeholder: "Qwen/Qwen2.5-3B", field: .tokenizer)
                        pathRow(label: "Output Dir", value: $bridge.outputDir, placeholder: "checkpoints", field: .outputDir)

                        GridRow {
                            Text("Learning Rate")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            TextField("", text: Binding(
                                get: { String(format: "%.1e", bridge.learningRate) },
                                set: { if let v = Double($0) { bridge.learningRate = v } }
                            ))
                                .textFieldStyle(.roundedBorder)
                                .font(.caption.monospacedDigit())
                                .frame(maxWidth: 160)
                        }

                        GridRow {
                            Text("Max Steps")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            HStack {
                                Stepper("", value: $bridge.maxSteps, in: 1...1_000_000, step: 100)
                                Text("\(bridge.maxSteps)")
                                    .font(.caption.monospacedDigit())
                                    .frame(width: 60, alignment: .trailing)
                            }
                        }

                        GridRow {
                            Text("LoRA Rank")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            HStack {
                                Stepper("", value: $bridge.loraRank, in: 1...128)
                                Text("\(bridge.loraRank)")
                                    .font(.caption.monospacedDigit())
                                    .frame(width: 40, alignment: .trailing)
                            }
                        }

                        GridRow {
                            Text("Batch Size")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            HStack {
                                Stepper("", value: $bridge.batchSize, in: 1...64)
                                Text("\(bridge.batchSize)")
                                    .font(.caption.monospacedDigit())
                                    .frame(width: 40, alignment: .trailing)
                            }
                        }

                        GridRow {
                            Text("Grad Accum")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            HStack {
                                Stepper("", value: $bridge.gradAccumSteps, in: 1...64)
                                Text("\(bridge.gradAccumSteps)")
                                    .font(.caption.monospacedDigit())
                                    .frame(width: 40, alignment: .trailing)
                            }
                        }

                        GridRow {
                            Text("Max Seq Len")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            HStack {
                                Stepper("", value: $bridge.maxSeqLen, in: 256...65_536, step: 256)
                                Text("\(bridge.maxSeqLen)")
                                    .font(.caption.monospacedDigit())
                                    .frame(width: 60, alignment: .trailing)
                            }
                        }

                        Divider()

                        GridRow {
                            Toggle("DoRA", isOn: $bridge.useDoRA)
                                .font(.caption)
                                .toggleStyle(.switch)
                                .gridColumnAlignment(.leading)
                        }

                        GridRow {
                            Toggle("Thermal Monitoring", isOn: $bridge.enableThermal)
                                .font(.caption)
                                .toggleStyle(.switch)
                                .gridColumnAlignment(.leading)
                        }
                    }
                }
            }
            .padding(12)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func pathRow(label: String, value: Binding<String>, placeholder: String, field: Field) -> some View {
        GridRow {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(placeholder, text: value)
                .textFieldStyle(.roundedBorder)
                .font(.caption)
                .focused($focusedField, equals: field)
        }
    }

    // MARK: - Actions

    private var actionButtons: some View {
        HStack(spacing: 12) {
            let canStart = bridge.state != .running
                && !bridge.dataPath.isEmpty
                && !bridge.modelWeightsPath.isEmpty

            Button {
                Task { await bridge.startTraining() }
            } label: {
                Label("Start Training", systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .disabled(!canStart)
            .buttonStyle(.borderedProminent)
            .tint(.green)

            if bridge.state == .running {
                Button {
                    bridge.stopTraining()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.red)
            }

            if bridge.state != .running {
                Button {
                    bridge.clear()
                } label: {
                    Label("Clear", systemImage: "trash")
                }
                .buttonStyle(.bordered)
                .disabled(bridge.state == .idle && bridge.logLines.isEmpty)
            }
        }
    }

    // MARK: - Logs

    private var logSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Output")
                    .font(.headline)
                Spacer()
                Text("\(bridge.logLines.count) lines")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(bridge.logLines.enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(logColor(for: line))
                                .textSelection(.enabled)
                        }
                    }
                    .padding(8)
                }
                .frame(minHeight: 200, maxHeight: 300)
                .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                .onChange(of: bridge.logLines.count) {
                    if let lastLine = bridge.logLines.indices.last {
                        withAnimation {
                            proxy.scrollTo(lastLine, anchor: .bottom)
                        }
                    }
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private func logColor(for line: String) -> Color {
        let lower = line.lowercased()
        if lower.contains("error") || lower.contains("traceback") || lower.contains("exception") {
            return .red
        }
        if lower.contains("warning") || lower.contains("warn") {
            return .orange
        }
        if lower.contains("step") || lower.contains("loss") {
            return .primary
        }
        return .secondary
    }
}
