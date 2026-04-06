import SwiftUI

struct MessageRowView: View {
    let message: ChatViewModel.Message

    var body: some View {
        HStack {
            if message.role == "user" {
                Spacer()
            }
            Text(message.content)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .foregroundStyle(message.role == "user" ? .primary : .primary)
                .background(
                    message.role == "user"
                        ? Color.accentColor.opacity(0.15)
                        : Color.secondary.opacity(0.1),
                    in: RoundedRectangle(cornerRadius: 12)
                )
            if message.role == "assistant" {
                Spacer()
            }
        }
        .padding(.horizontal)
    }
}
