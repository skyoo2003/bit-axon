import MLX
import MLXNN

/// Mamba-style State Space Model layer.
///
/// 1:1 port of the Python `AxonSSM` from `bit_axon.layers.axon_ssm`.
/// Weight keys (`in_proj`, `conv1d`, `x_proj`, `dt_proj`, `out_proj`, `A_log`, `D`)
/// match the Python implementation for cross-framework weight loading.
class AxonSSM: Module {

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "x_proj") var xProj: Linear
    @ModuleInfo(key: "dt_proj") var dtProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "D") var dParam: MLXArray

    let dConv: Int
    let dState: Int
    let E: Int

    init(config: BitAxonConfig) {
        let D = config.hiddenDim
        let E = config.ssmIntermediateDim
        let dState = config.ssmDState
        let dConv = config.ssmDConv

        self.dConv = dConv
        self.dState = dState
        self.E = E

        self._inProj.wrappedValue = Linear(D, 2 * E, bias: false)
        self._conv1d.wrappedValue = Conv1d(
            inputChannels: E, outputChannels: E, kernelSize: dConv,
            stride: 1, padding: 0, groups: E, bias: true)
        self._xProj.wrappedValue = Linear(E, dState * 2 + 1, bias: false)
        self._dtProj.wrappedValue = Linear(1, E, bias: true)
        self._outProj.wrappedValue = Linear(E, D, bias: false)

        let aRange = MLXArray((1...dState).map { Float($0) })
            .expandedDimensions(axis: 0)
        self._aLog.wrappedValue = MLX.log(tiled(aRange, repetitions: [E, 1]))

        self._dParam.wrappedValue = MLXArray.ones([E])

        super.init()
    }

    private func causalConv1d(_ x: MLXArray, convCache: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let kMinus1 = dConv - 1
        let xPadded: MLXArray

        if let convCache = convCache {
            xPadded = concatenated([convCache, x], axis: 1)
        } else {
            xPadded = padded(x, widths: [0, [kMinus1, 0], 0])
        }

        let cacheStart = xPadded.dim(1) - kMinus1
        let newConvCache = xPadded[0..., cacheStart...]

        let result = conv1d(xPadded)
        return (result, newConvCache)
    }

    /// Sequential SSM recurrence:
    ///   h_t = exp(dt * A) * h_{t-1} + dt * B * x_t
    ///   y_t = sum(h_t * C, axis=-1) + D * x_t
    private func ssmScan(
        _ x: MLXArray, dt: MLXArray, bIn: MLXArray, cIn: MLXArray,
        ssmState: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let L = x.dim(1)
        let a = -MLX.exp(aLog)  // [E, d_state]

        var h: MLXArray
        if let ssmState = ssmState {
            h = ssmState
        } else {
            h = MLXArray.zeros([x.dim(0), E, dState])
        }

        var ys = [MLXArray]()
        ys.reserveCapacity(L)

        for t in 0..<L {
            let xT = x[0..., t]       // [B, E]
            let dtT = dt[0..., t]     // [B, E]
            let bT = bIn[0..., t]     // [B, d_state]
            let cT = cIn[0..., t]     // [B, d_state]

            let dA = MLX.exp(
                dtT.expandedDimensions(axis: -1) * a.expandedDimensions(axis: 0))
            let dB = dtT.expandedDimensions(axis: -1)
                * bT.expandedDimensions(axis: 1)

            h = dA * h + dB * xT.expandedDimensions(axis: -1)

            let y = (h * cT.expandedDimensions(axis: 1)).sum(axis: -1)
                + dParam * xT
            ys.append(y)
        }

        let yOut = concatenated(
            ys.map { $0.expandedDimensions(axis: 1) }, axis: 1)
        return (yOut, h)
    }

    func callAsFunction(_ x: MLXArray, cache: [MLXArray]? = nil) -> (MLXArray, [MLXArray]) {
        let convCache = cache.map { $0[0] }
        let ssmState = cache.map { $0[1] }

        let (xBranch, zBranch) = inProj(x).split(axis: -1)

        let (xConv, newConvCache) = causalConv1d(xBranch, convCache: convCache)
        let xConvAct = silu(xConv)

        let bcDt = xProj(xConvAct).split(indices: [dState, 2 * dState], axis: -1)
        let bSSM = bcDt[0]
        let cSSM = bcDt[1]
        let dt = bcDt[2]

        let dtFinal = clip(
            softplus(dtProj(dt)), min: Float(1e-4), max: Float(100.0))

        let (y, newSSMState) = ssmScan(
            xConvAct, dt: dtFinal, bIn: bSSM, cIn: cSSM, ssmState: ssmState)

        let output = outProj(y * silu(zBranch))

        return (output, [newConvCache, newSSMState])
    }
}
