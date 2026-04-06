import Foundation
import MLX
import MLXNN

@Observable
final class ModelService: @unchecked Sendable {

    enum ModelState: Equatable {
        case idle
        case loading(progress: Double)
        case ready
        case failed(String)
    }

    @MainActor var state: ModelState = .idle
    @MainActor private(set) var model: BitAxonModel?
    @MainActor private(set) var config: BitAxonConfig?

    @MainActor
    func loadFromDirectory(_ url: URL) async throws {
        state = .loading(progress: 0.0)

        let configFile = url.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configFile)
        let decoded = try JSONDecoder().decode(BitAxonConfig.self, from: configData)

        self.config = decoded
        state = .loading(progress: 0.3)

        let builtModel = BitAxonModel(config: decoded)
        self.model = builtModel
        state = .loading(progress: 1.0)

        evaluate(model: builtModel)
        state = .ready
    }

    @MainActor
    func loadDefault() async throws {
        state = .loading(progress: 0.0)
        let cfg = BitAxonModelRegistry.defaultConfig
        self.config = cfg

        let builtModel = BitAxonModel(config: cfg)
        self.model = builtModel
        state = .loading(progress: 1.0)

        evaluate(model: builtModel)
        state = .ready
    }

    @MainActor
    func unload() {
        model = nil
        config = nil
        state = .idle
    }

    private func evaluate(model: BitAxonModel) {
        MLX.eval(model.parameters())
    }
}
