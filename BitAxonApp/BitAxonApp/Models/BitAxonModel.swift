import Foundation
import MLX
import MLXNN

enum LayerCache {
    case ssm([MLXArray]?)
    case swa(BitAxonKVCache?)
}

class BitAxonModel: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    @ModuleInfo(key: "layers") var layers: [Module]

    let layerTypes: [String]
    let config: BitAxonConfig

    init(config: BitAxonConfig) {
        self.config = config
        var builtLayers: [Module] = []
        var types: [String] = []

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.dSourceModel
        )
        self._inputProj.wrappedValue = Linear(
            config.dSourceModel, config.hiddenDim, bias: false
        )
        self._outputProj.wrappedValue = Linear(
            config.hiddenDim, config.dSourceModel, bias: false
        )
        self._lmHead.wrappedValue = Linear(
            config.dSourceModel, config.vocabSize, bias: false
        )

        for i in 0..<config.numLayers {
            let layerType = Self.getLayerType(layerIdx: i, totalLayers: config.numLayers)
            types.append(layerType)

            switch layerType {
            case "ssm":
                builtLayers.append(AxonSSMBlock(config: config))
            case "swa_moe":
                builtLayers.append(AxonSWAMoEBlock(config: config))
            case "ssm_moe":
                builtLayers.append(AxonSSMMoEBlock(config: config))
            default:
                fatalError("Unknown layer type: \(layerType)")
            }
        }

        self.layerTypes = types
        self._layers.wrappedValue = builtLayers

        super.init()
    }

    static func getLayerType(layerIdx: Int, totalLayers: Int) -> String {
        let third = totalLayers / 3
        if layerIdx < third { return "ssm" }
        if layerIdx < 2 * third { return "swa_moe" }
        return "ssm_moe"
    }

    func createCaches() -> [LayerCache] {
        layerTypes.map { type in
            switch type {
            case "swa_moe":
                return .swa(BitAxonKVCache())
            default:
                return .ssm(nil)
            }
        }
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [LayerCache]? = nil) -> (MLXArray, [LayerCache]) {
        var x = embedTokens(inputIds)
        x = inputProj(x)

        var newCaches: [LayerCache] = []
        newCaches.reserveCapacity(layers.count)

        for i in 0..<layers.count {
            let layer = layers[i]
            let layerCache = cache?[i]

            switch layerTypes[i] {
            case "ssm":
                let block = layer as! AxonSSMBlock
                let ssmCache: [MLXArray]?
                if case .ssm(let c) = layerCache { ssmCache = c } else { ssmCache = nil }
                let (out, newSsmCache) = block(x, cache: ssmCache)
                x = out
                newCaches.append(.ssm(newSsmCache))

            case "swa_moe":
                let block = layer as! AxonSWAMoEBlock
                let swaCache: BitAxonKVCache?
                if case .swa(let c) = layerCache { swaCache = c } else { swaCache = nil }
                let (out, newSwaCache) = block(x, cache: swaCache)
                x = out
                newCaches.append(.swa(newSwaCache))

            case "ssm_moe":
                let block = layer as! AxonSSMMoEBlock
                let ssmCache: [MLXArray]?
                if case .ssm(let c) = layerCache { ssmCache = c } else { ssmCache = nil }
                let (out, newSsmCache) = block(x, cache: ssmCache)
                x = out
                newCaches.append(.ssm(newSsmCache))

            default:
                fatalError("Unknown layer type at index \(i): \(layerTypes[i])")
            }
        }

        x = outputProj(x)
        let logits = lmHead(x)
        return (logits, newCaches)
    }
}
