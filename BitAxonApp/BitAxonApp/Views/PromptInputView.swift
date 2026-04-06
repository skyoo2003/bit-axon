import SwiftUI

struct PromptInputView: View {
    @Bindable var viewModel: ChatViewModel

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Type a message…", text: $viewModel.prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...5)
                .disabled(viewModel.isGenerating)
                .onSubmit { Task { await viewModel.generate() } }

            Button {
                Task { await viewModel.generate() }
            } label: {
                Image(
                    systemName: viewModel.isGenerating
                        ? "stop.circle.fill" : "arrow.up.circle.fill"
                )
                .font(.title2)
            }
            .disabled(viewModel.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            .buttonStyle(.borderless)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
}
