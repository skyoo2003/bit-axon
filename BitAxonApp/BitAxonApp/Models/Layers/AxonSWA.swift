import MLX
import MLXNN

class AxonSWA: Module, UnaryLayer {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let numHeads: Int
    let headDim: Int
    let windowSize: Int
    let scale: Float

    init(hiddenDim: Int, numHeads: Int, windowSize: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenDim / numHeads
        self.windowSize = windowSize
        self.scale = 1.0 / Float(headDim).squareRoot()

        self._qProj.wrappedValue = Linear(hiddenDim, hiddenDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenDim, hiddenDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenDim, hiddenDim, bias: false)
        self._oProj.wrappedValue = Linear(hiddenDim, hiddenDim, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (output, _) = callAsFunction(x, mask: nil, cache: nil)
        return output
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: BitAxonKVCache? = nil) -> (MLXArray, BitAxonKVCache?) {
        let B = x.shape[0]
        let L = x.shape[1]
        let D = x.shape[2]

        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)

        q = q.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

        if let cache = cache {
            let (updatedK, updatedV) = cache.updateAndFetch(keys: k, values: v)
            k = updatedK
            v = updatedV
        }

        let kvLen = k.shape[2]

        var scores = matmul(q * self.scale, k.transposed(0, 1, 3, 2))

        if let mask = mask {
            scores = scores + mask
        } else {
            let swMask = makeSlidingWindowMask(seqLen: L, kvLen: kvLen)
            scores = scores + swMask
        }

        let attn = softmax(scores, axis: -1)
        var out = matmul(attn, v)

        out = out.transposed(0, 2, 1, 3).reshaped(B, L, D)
        out = oProj(out)

        return (out, cache)
    }

    func makeSlidingWindowMask(seqLen: Int, kvLen: Int) -> MLXArray {
        let qPos = MLXArray(0 ..< seqLen)
        let kPos = MLXArray(0 ..< kvLen)
        let causalOffset = kvLen - seqLen

        let qExpanded = qPos.expandedDimensions(axis: 1)
        let kExpanded = kPos.expandedDimensions(axis: 0)
        let causalMask = kExpanded .<= (qExpanded + causalOffset)

        let windowMask = ((qExpanded + causalOffset) - kExpanded) .< windowSize

        let causalScore = which(causalMask, MLXArray(0.0), MLXArray(-Float.infinity))
        let windowScore = which(windowMask, MLXArray(0.0), MLXArray(-Float.infinity))
        let mask = causalScore + windowScore

        return mask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    }
}
