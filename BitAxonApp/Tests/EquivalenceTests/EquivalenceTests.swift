import XCTest
import Foundation
import MLX
import MLXNN
@testable import BitAxonApp

/// Cross-language numerical equivalence tests.
///
/// Reference tensors are exported by `export_reference.py` and stored as JSON resources.
/// Each test reconstructs a Swift layer with reference weights, runs the same forward pass,
/// and asserts output equality within tolerance.
final class EquivalenceTests: XCTestCase {

    private func loadJSON(_ name: String) throws -> [String: Any] {
        guard let url = Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "reference") else {
            XCTFail("Resource reference/\(name).json not found in Bundle.module")
            return [:]
        }
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    }

    private func toMLXArray(_ value: Any) -> MLXArray {
        if let nested = value as? [Any] {
            if let first = nested.first as? [Any] {
                if let firstFirst = first.first as? [Any] {
                    // 3D: [[Float]]
                    let d0 = nested.count
                    let d1 = first.count
                    let d2 = firstFirst.count
                    var flat = [Float]()
                    flat.reserveCapacity(d0 * d1 * d2)
                    for row in nested {
                        for col in row as! [Any] {
                            for val in col as! [Any] {
                                flat.append(Float(val as! Double))
                            }
                        }
                    }
                    return MLXArray(flat).reshaped([d0, d1, d2])
                } else {
                    let d0 = nested.count
                    let d1 = first.count
                    var flat = [Float]()
                    flat.reserveCapacity(d0 * d1)
                    for row in nested {
                        for val in row as! [Any] {
                            flat.append(Float(val as! Double))
                        }
                    }
                    return MLXArray(flat).reshaped([d0, d1])
                }
            } else {
                let flat = nested.map { Float($0 as! Double) }
                return MLXArray(flat)
            }
        }
        fatalError("Unexpected JSON type: \(type(of: value))")
    }

    private func assertMLXEqual(
        _ a: MLXArray, _ b: MLXArray, tolerance: Float = 1e-3,
        file: StaticString = #file, line: UInt = #line
    ) {
        XCTAssertEqual(a.shape, b.shape, "Shape mismatch: \(a.shape) vs \(b.shape)", file: file, line: line)
        let diff = abs(a - b)
        let maxDiff = Float(diff.max().item(Float.self))
        let meanDiff = Float(diff.mean().item(Float.self))
        let aNorm = Float(sqrt((a * a).mean()).item(Float.self))
        let relError = aNorm > 0 ? meanDiff / aNorm : meanDiff
        XCTAssertLessThan(
            maxDiff, tolerance,
            "Max abs diff \(maxDiff) >= \(tolerance). Mean diff: \(meanDiff), Rel error: \(relError)",
            file: file, line: line
        )
        XCTAssertLessThan(meanDiff, tolerance * 10, file: file, line: line)
    }

    private func linearParams(weight: MLXArray, bias: MLXArray? = nil) -> NestedItem<String, MLXArray> {
        var items: [String: NestedItem<String, MLXArray>] = ["weight": .value(weight)]
        if let bias { items["bias"] = .value(bias) }
        return .dictionary(items)
    }

    private func mlpParams(gateW: MLXArray, upW: MLXArray, downW: MLXArray) -> NestedItem<String, MLXArray> {
        .dictionary([
            "gate_proj": linearParams(weight: gateW),
            "up_proj": linearParams(weight: upW),
            "down_proj": linearParams(weight: downW),
        ])
    }

    // MARK: - RMSNorm

    func testRMSNorm() throws {
        let json = try loadJSON("rms_norm")
        let weight = toMLXArray(json["weight"]!)
        let input = toMLXArray(json["input"]!)
        let expected = toMLXArray(json["output"]!)
        let dims = json["hidden_dim"] as! Int
        let eps = Float(json["eps"] as! Double)

        let layer = AxonRMSNorm(dims, eps: eps)
        layer.update(parameters: ModuleParameters(values: ["weight": .value(weight)]))

        let output = layer(input)
        assertMLXEqual(output, expected)
    }

    // MARK: - AxonSSM

    func testAxonSSM() throws {
        let json = try loadJSON("axon_ssm")
        guard let config = json["config"] as? [String: Any],
              let weights = json["weights"] as? [String: Any]
        else {
            XCTFail("Missing config or weights in axon_ssm.json")
            return
        }

        let hiddenDim = config["hidden_dim"] as! Int
        let ssmIntermediateDim = config["ssm_intermediate_dim"] as! Int
        let ssmDState = config["ssm_d_state"] as! Int
        let ssmDConv = config["ssm_d_conv"] as! Int

        let bitConfig = BitAxonConfig(
            vocabSize: 512,
            hiddenDim: hiddenDim,
            numLayers: 1,
            numHeads: 2,
            dSourceModel: 32,
            ssmDState: ssmDState,
            ssmDConv: ssmDConv,
            ssmExpand: ssmIntermediateDim / hiddenDim
        )

        let layer = AxonSSM(config: bitConfig)
        let w = weights
        layer.update(parameters: ModuleParameters(values: [
            "in_proj": linearParams(weight: toMLXArray(w["in_proj_weight"]!)),
            "conv1d": linearParams(weight: toMLXArray(w["conv1d_weight"]!), bias: toMLXArray(w["conv1d_bias"]!)),
            "x_proj": linearParams(weight: toMLXArray(w["x_proj_weight"]!)),
            "dt_proj": linearParams(weight: toMLXArray(w["dt_proj_weight"]!), bias: toMLXArray(w["dt_proj_bias"]!)),
            "out_proj": linearParams(weight: toMLXArray(w["out_proj_weight"]!)),
            "A_log": .value(toMLXArray(w["A_log"]!)),
            "D": .value(toMLXArray(w["D"]!)),
        ]))

        let input = toMLXArray(json["input"]!)
        let expected = toMLXArray(json["output"]!)
        let (output, cache) = layer(input)

        assertMLXEqual(output, expected, tolerance: 1e-2)

        guard let cacheShapes = json["cache_shapes"] as? [String: [Int]],
              let cacheData = json["cache"] as? [String: Any]
        else { return }

        let expectedConvCache = toMLXArray(cacheData["conv_cache"]!)
        let expectedSSMState = toMLXArray(cacheData["ssm_state"]!)

        XCTAssertEqual(
            [cache[0].dim(0), cache[0].dim(1), cache[0].dim(2)],
            cacheShapes["conv_cache"]!,
            "Conv cache shape mismatch"
        )
        XCTAssertEqual(
            [cache[1].dim(0), cache[1].dim(1), cache[1].dim(2)],
            cacheShapes["ssm_state"]!,
            "SSM state shape mismatch"
        )
        assertMLXEqual(cache[0], expectedConvCache, tolerance: 1e-2)
        assertMLXEqual(cache[1], expectedSSMState, tolerance: 1e-2)
    }

    // MARK: - AxonSWA

    func testAxonSWA() throws {
        let json = try loadJSON("axon_swa")
        guard let config = json["config"] as? [String: Any],
              let weights = json["weights"] as? [String: Any]
        else {
            XCTFail("Missing config or weights in axon_swa.json")
            return
        }

        let hiddenDim = config["hidden_dim"] as! Int
        let numHeads = config["num_heads"] as! Int
        let windowSize = config["window_size"] as! Int

        let layer = AxonSWA(hiddenDim: hiddenDim, numHeads: numHeads, windowSize: windowSize)
        let w = weights
        layer.update(parameters: ModuleParameters(values: [
            "q_proj": linearParams(weight: toMLXArray(w["q_proj_weight"]!)),
            "k_proj": linearParams(weight: toMLXArray(w["k_proj_weight"]!)),
            "v_proj": linearParams(weight: toMLXArray(w["v_proj_weight"]!)),
            "o_proj": linearParams(weight: toMLXArray(w["o_proj_weight"]!)),
        ]))

        let input = toMLXArray(json["input"]!)
        let expected = toMLXArray(json["output"]!)
        let (output, _) = layer(input, mask: nil, cache: nil)

        assertMLXEqual(output, expected)
    }

    // MARK: - AxonMoE

    func testAxonMoE() throws {
        let json = try loadJSON("axon_moe")
        guard let config = json["config"] as? [String: Any],
              let weights = json["weights"] as? [String: Any]
        else {
            XCTFail("Missing config or weights in axon_moe.json")
            return
        }

        let hiddenDim = config["hidden_dim"] as! Int
        let intermediateDim = config["intermediate_dim"] as! Int
        let numExperts = config["num_experts"] as! Int
        let topK = config["top_k"] as! Int

        let layer = AxonSharedExpertMoE(
            dim: hiddenDim,
            intermediateDim: intermediateDim,
            numExperts: numExperts,
            topK: topK
        )

        // Python SwitchLinear stores weights as (num_experts, out, in);
        // Swift [MoEMLP] stores each expert as standard (out, in).
        let switchGateW = toMLXArray(weights["switch_gate_proj_weight"]!)
        let switchUpW = toMLXArray(weights["switch_up_proj_weight"]!)
        let switchDownW = toMLXArray(weights["switch_down_proj_weight"]!)

        var expertItems: [NestedItem<String, MLXArray>] = []
        for e in 0..<numExperts {
            expertItems.append(mlpParams(
                gateW: switchGateW[e].squeezed(axis: 0),
                upW: switchUpW[e].squeezed(axis: 0),
                downW: switchDownW[e].squeezed(axis: 0)
            ))
        }

        let w = weights
        layer.update(parameters: ModuleParameters(values: [
            "gate": linearParams(weight: toMLXArray(w["gate_weight"]!)),
            "shared_expert_gate": linearParams(weight: toMLXArray(w["shared_expert_gate_weight"]!)),
            "shared_expert": mlpParams(
                gateW: toMLXArray(w["shared_expert_gate_proj_weight"]!),
                upW: toMLXArray(w["shared_expert_up_proj_weight"]!),
                downW: toMLXArray(w["shared_expert_down_proj_weight"]!)
            ),
            "experts": .array(expertItems),
        ]))

        let input = toMLXArray(json["input"]!)
        let expected = toMLXArray(json["output"]!)
        let output = layer(input)

        assertMLXEqual(output, expected, tolerance: 5e-2)
    }
}
