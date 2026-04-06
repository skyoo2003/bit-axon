import MLX
import MLXNN

class AxonRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}
