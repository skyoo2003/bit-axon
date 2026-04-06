import MLX
import MLXNN

// MARK: - AxonSSMBlock

/// Pure SSM block: RMSNorm → SSM → residual add.
///
/// Used in layers 0–7 (first third of the model) for linear-complexity
/// context absorption with O(1) memory per token.
class AxonSSMBlock: Module {
    @ModuleInfo(key: "input_norm") var inputNorm: AxonRMSNorm
    @ModuleInfo(key: "ssm") var ssm: AxonSSM

    init(config: BitAxonConfig) {
        self._inputNorm.wrappedValue = AxonRMSNorm(config.hiddenDim, eps: config.rmsNormEps)
        self._ssm.wrappedValue = AxonSSM(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: [MLXArray]? = nil) -> (MLXArray, [MLXArray]) {
        let residual = x
        let normed = inputNorm(x)
        let (ssmOut, newCache) = ssm(normed, cache: cache)
        return (residual + ssmOut, newCache)
    }
}

// MARK: - AxonSWAMoEBlock

/// SWA + MoE block: RMSNorm → SWA → residual → RMSNorm → MoE → residual.
///
/// Used in layers 8–15 (middle third) for attention-based deep reasoning
/// combined with sparse expert activation.
class AxonSWAMoEBlock: Module {
    @ModuleInfo(key: "input_norm") var inputNorm: AxonRMSNorm
    @ModuleInfo(key: "attention") var attention: AxonSWA
    @ModuleInfo(key: "post_attention_norm") var postAttentionNorm: AxonRMSNorm
    @ModuleInfo(key: "moe") var moe: AxonSharedExpertMoE

    init(config: BitAxonConfig) {
        self._inputNorm.wrappedValue = AxonRMSNorm(config.hiddenDim, eps: config.rmsNormEps)
        self._attention.wrappedValue = AxonSWA(
            hiddenDim: config.hiddenDim,
            numHeads: config.numHeads,
            windowSize: config.swaWindowSize
        )
        self._postAttentionNorm.wrappedValue = AxonRMSNorm(config.hiddenDim, eps: config.rmsNormEps)
        self._moe.wrappedValue = AxonSharedExpertMoE(
            dim: config.hiddenDim,
            intermediateDim: config.moeIntermediateDim,
            numExperts: config.moeNumExperts,
            topK: config.moeTopK
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: BitAxonKVCache? = nil) -> (MLXArray, BitAxonKVCache?) {
        let residual = x
        let normed = inputNorm(x)
        let (attnOut, newCache) = attention(normed, cache: cache)
        var out = residual + attnOut

        let residual2 = out
        out = postAttentionNorm(out)
        out = residual2 + moe(out)

        return (out, newCache)
    }
}

// MARK: - AxonSSMMoEBlock

/// SSM + MoE block: RMSNorm → SSM → residual → RMSNorm → MoE → residual.
///
/// Used in layers 16–23 (final third) for linear output synthesis
/// combined with sparse expert activation.
class AxonSSMMoEBlock: Module {
    @ModuleInfo(key: "input_norm") var inputNorm: AxonRMSNorm
    @ModuleInfo(key: "ssm") var ssm: AxonSSM
    @ModuleInfo(key: "post_ssm_norm") var postSSMNorm: AxonRMSNorm
    @ModuleInfo(key: "moe") var moe: AxonSharedExpertMoE

    init(config: BitAxonConfig) {
        self._inputNorm.wrappedValue = AxonRMSNorm(config.hiddenDim, eps: config.rmsNormEps)
        self._ssm.wrappedValue = AxonSSM(config: config)
        self._postSSMNorm.wrappedValue = AxonRMSNorm(config.hiddenDim, eps: config.rmsNormEps)
        self._moe.wrappedValue = AxonSharedExpertMoE(
            dim: config.hiddenDim,
            intermediateDim: config.moeIntermediateDim,
            numExperts: config.moeNumExperts,
            topK: config.moeTopK
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: [MLXArray]? = nil) -> (MLXArray, [MLXArray]) {
        let residual = x
        let normed = inputNorm(x)
        let (ssmOut, ssmCache) = ssm(normed, cache: cache)
        var out = residual + ssmOut

        let residual2 = out
        out = postSSMNorm(out)
        out = residual2 + moe(out)

        return (out, ssmCache)
    }
}
