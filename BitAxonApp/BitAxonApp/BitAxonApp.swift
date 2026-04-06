import SwiftUI
import MLX

@main
struct BitAxonApp: App {
    init() {
        GPU.set(cacheLimit: 20 * 1024 * 1024 * 1024)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
