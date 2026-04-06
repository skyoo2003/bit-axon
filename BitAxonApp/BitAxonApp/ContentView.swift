import SwiftUI
import MLX

struct ContentView: View {
    @State private var chatViewModel = ChatViewModel(modelService: ModelService())
    @State private var deviceStat = DeviceStat()
    @State private var showMetrics = false

    var body: some View {
        NavigationSplitView {
            List {
                Section("Model") {
                    Button {
                        Task { await chatViewModel.loadModel() }
                    } label: {
                        Label("Load Model", systemImage: "square.and.arrow.down")
                    }
                    Button {
                        chatViewModel.clear()
                    } label: {
                        Label("Clear Chat", systemImage: "trash")
                    }
                }
                Section("Fine-Tune") {
                    NavigationLink {
                        FineTuneView()
                    } label: {
                        Label("Fine-Tune", systemImage: "slider.horizontal.3")
                    }
                }
                Section("Monitoring") {
                    Toggle("Show Metrics", isOn: $showMetrics)
                }
            }
            .listStyle(.sidebar)
            .navigationTitle("Bit-Axon")
        } detail: {
            ChatView(viewModel: chatViewModel)
                .toolbar {
                    if showMetrics {
                        ToolbarItem(placement: .automatic) {
                            MetricsView(
                                tokensPerSecond: chatViewModel.tokensPerSecond,
                                ttftMs: chatViewModel.timeToFirstTokenMs,
                                deviceStat: deviceStat
                            )
                        }
                    }
                }
        }
        .task {
            deviceStat.startPolling()
        }
        .onDisappear {
            deviceStat.stopPolling()
        }
    }
}
