import Foundation

struct BitAxonConfig: Codable, Sendable, Equatable {
    var vocabSize: Int = 32_000
    var hiddenDim: Int = 2_560
    var numLayers: Int = 24
    var numHeads: Int = 32
    var dSourceModel: Int = 2048
    var ssmDState: Int = 16
    var ssmDConv: Int = 4
    var ssmExpand: Int = 3
    var swaWindowSize: Int = 4_096
    var moeNumExperts: Int = 8
    var moeTopK: Int = 2
    var moeIntermediateDim: Int = 4_096
    var moeSharedExpert: Bool = true
    var weightTying: Bool = true
    var maxSeqLen: Int = 65_536
    var rmsNormEps: Float = 1e-6

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenDim = "hidden_dim"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case dSourceModel = "d_source_model"
        case ssmDState = "ssm_d_state"
        case ssmDConv = "ssm_d_conv"
        case ssmExpand = "ssm_expand"
        case swaWindowSize = "swa_window_size"
        case moeNumExperts = "moe_num_experts"
        case moeTopK = "moe_top_k"
        case moeIntermediateDim = "moe_intermediate_dim"
        case moeSharedExpert = "moe_shared_expert"
        case weightTying = "weight_tying"
        case maxSeqLen = "max_seq_len"
        case rmsNormEps = "rms_norm_eps"
    }

    var headDim: Int {
        hiddenDim / numHeads
    }

    var ssmIntermediateDim: Int {
        hiddenDim * ssmExpand
    }
}
