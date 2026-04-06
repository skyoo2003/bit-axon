import MLX
import MLXNN

// MARK: - SwiGLU Activation

/// SwiGLU activation: silu(gate) * x
func swiglu(_ x: MLXArray, gate: MLXArray) -> MLXArray {
    silu(gate) * x
}

// MARK: - MLP (Shared Expert Feed-Forward)

/// Standard SwiGLU MLP used as the shared expert in MoE.
class MoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dim: Int, intermediateDim: Int, bias: Bool = false) {
        self._gateProj.wrappedValue = Linear(dim, intermediateDim, bias: bias)
        self._upProj.wrappedValue = Linear(dim, intermediateDim, bias: bias)
        self._downProj.wrappedValue = Linear(intermediateDim, dim, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(swiglu(upProj(x), gate: gateProj(x)))
    }
}

// MARK: - SharedExpertMoE

/// Mixture of Experts layer with shared expert and top-k routing.
///
/// Simplified v1 implementation: computes all expert outputs, then selects
/// via boolean masking. Correctness over efficiency.
///
/// Architecture:
/// - Router gate: Linear(dim, numExperts) -> softmax -> top-k
/// - Routed experts: N independent SwiGLU MLPs, selected by top-k
/// - Shared expert: single SwiGLU MLP with learned sigmoid gate
class AxonSharedExpertMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear
    @ModuleInfo(key: "shared_expert") var sharedExpert: MoEMLP
    @ModuleInfo(key: "experts") var experts: [MoEMLP]

    let topK: Int
    let numExperts: Int

    init(dim: Int, intermediateDim: Int, numExperts: Int = 8, topK: Int = 2, bias: Bool = false) {
        self.topK = topK
        self.numExperts = numExperts

        self._gate.wrappedValue = Linear(dim, numExperts, bias: false)
        self._sharedExpertGate.wrappedValue = Linear(dim, 1, bias: false)
        self._sharedExpert.wrappedValue = MoEMLP(
            dim: dim, intermediateDim: intermediateDim, bias: bias)

        self._experts.wrappedValue = (0..<numExperts).map { _ in
            MoEMLP(dim: dim, intermediateDim: intermediateDim, bias: bias)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let D = x.dim(2)

        let gates = softmax(gate(x), axis: -1)  // (B, L, numExperts)

        // argPartition gives top-k indices in O(n) without full sort
        let inds = argPartition(-gates, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]  // (B, L, topK)
        let scores = takeAlong(gates, inds, axis: -1)  // (B, L, topK)

        var expertOutputs = [MLXArray]()
        for e in 0..<numExperts {
            expertOutputs.append(experts[e](x))  // (B, L, D) each
        }

        var y = MLXArray.zeros([B, L, D])
        for k in 0..<topK {
            let expertIdx = inds[.ellipsis, k]  // (B, L)
            let score = scores[.ellipsis, k, .newAxis]  // (B, L, 1)

            var selected = MLXArray.zeros([B, L, D])
            for e in 0..<numExperts {
                let mask = (expertIdx .== MLXArray(e)).expandedDimensions(axis: -1)  // (B, L, 1)
                selected = selected + mask * expertOutputs[e]
            }

            y = y + selected * score
        }

        let sharedOut = sharedExpert(x)
        let sharedGateVal = sigmoid(sharedExpertGate(x))
        return y + sharedGateVal * sharedOut
    }
}
