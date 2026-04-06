import Foundation

struct BitAxonModelRegistry {
    static func createModel(config: BitAxonConfig) -> BitAxonModel {
        BitAxonModel(config: config)
    }

    static var defaultConfig: BitAxonConfig {
        BitAxonConfig()
    }
}
