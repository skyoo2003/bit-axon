"""Tests for weight mapping functions: embedding extraction and RMSNorm padding."""

from __future__ import annotations

import mlx.core as mx
import pytest

from bit_axon.porting.mapper import extract_embeddings, init_routed_experts, pad_rms_norm, project_mlp_to_shared_expert

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_VOCAB_SIZE = 100
MOCK_HIDDEN_DIM = 32
MOCK_TARGET_VOCAB = 50


@pytest.fixture()
def mock_embedding() -> mx.array:
    return mx.random.normal((MOCK_VOCAB_SIZE, MOCK_HIDDEN_DIM))


@pytest.fixture()
def vocab_mapping() -> dict[int, int]:
    return {i: i for i in range(MOCK_TARGET_VOCAB)}


@pytest.fixture()
def qwen_weights(mock_embedding: mx.array) -> dict[str, mx.array]:
    return {"model.embed_tokens.weight": mock_embedding}


@pytest.fixture()
def mock_norm() -> mx.array:
    return mx.random.normal((MOCK_HIDDEN_DIM,))


# ---------------------------------------------------------------------------
# extract_embeddings
# ---------------------------------------------------------------------------


class TestExtractEmbeddings:
    def test_extract_shape(self, qwen_weights: dict[str, mx.array], vocab_mapping: dict[int, int]) -> None:
        output = extract_embeddings(qwen_weights, vocab_mapping, target_vocab_size=MOCK_TARGET_VOCAB, source_hidden_dim=MOCK_HIDDEN_DIM)
        assert output.shape == (MOCK_TARGET_VOCAB, MOCK_HIDDEN_DIM)

    def test_extract_dtype(self, qwen_weights: dict[str, mx.array], vocab_mapping: dict[int, int]) -> None:
        output = extract_embeddings(qwen_weights, vocab_mapping, target_vocab_size=MOCK_TARGET_VOCAB, source_hidden_dim=MOCK_HIDDEN_DIM)
        assert output.dtype == mx.float32

    def test_extract_preserves_rows(self, qwen_weights: dict[str, mx.array], vocab_mapping: dict[int, int]) -> None:
        source = qwen_weights["model.embed_tokens.weight"]
        output = extract_embeddings(qwen_weights, vocab_mapping, target_vocab_size=MOCK_TARGET_VOCAB, source_hidden_dim=MOCK_HIDDEN_DIM)
        for old_id, new_id in vocab_mapping.items():
            assert mx.array_equal(output[new_id], source[old_id])

    def test_extract_no_nan(self, qwen_weights: dict[str, mx.array], vocab_mapping: dict[int, int]) -> None:
        output = extract_embeddings(qwen_weights, vocab_mapping, target_vocab_size=MOCK_TARGET_VOCAB, source_hidden_dim=MOCK_HIDDEN_DIM)
        assert not mx.any(mx.isnan(output)).item()

    def test_extract_unmapped_zeros(self, qwen_weights: dict[str, mx.array], vocab_mapping: dict[int, int]) -> None:
        target_size = MOCK_TARGET_VOCAB + 20
        output = extract_embeddings(qwen_weights, vocab_mapping, target_vocab_size=target_size, source_hidden_dim=MOCK_HIDDEN_DIM)
        unmapped_ids = [i for i in range(target_size) if i not in vocab_mapping.values()]
        assert len(unmapped_ids) == 20
        for uid in unmapped_ids:
            assert mx.array_equal(output[uid], mx.zeros((MOCK_HIDDEN_DIM,)))

    def test_extract_with_mock_weights(self) -> None:
        small_embedding = mx.random.normal((100, 32))
        mapping = {i: i for i in range(50)}
        weights = {"model.embed_tokens.weight": small_embedding}
        output = extract_embeddings(weights, mapping, target_vocab_size=50, source_hidden_dim=32)
        assert output.shape == (50, 32)
        assert mx.array_equal(output[0], small_embedding[0])
        assert mx.array_equal(output[49], small_embedding[49])


# ---------------------------------------------------------------------------
# pad_rms_norm
# ---------------------------------------------------------------------------


class TestPadRmsNorm:
    def test_pad_shape(self, mock_norm: mx.array) -> None:
        target_dim = 64
        output = pad_rms_norm(mock_norm, target_dim=target_dim)
        assert output.shape == (target_dim,)

    def test_pad_preserves_original(self, mock_norm: mx.array) -> None:
        target_dim = 64
        source_dim = mock_norm.shape[0]
        output = pad_rms_norm(mock_norm, target_dim=target_dim)
        assert mx.array_equal(output[:source_dim], mock_norm)

    def test_pad_values_ones(self, mock_norm: mx.array) -> None:
        target_dim = 64
        source_dim = mock_norm.shape[0]
        output = pad_rms_norm(mock_norm, target_dim=target_dim)
        pad_size = target_dim - source_dim
        assert mx.array_equal(output[source_dim:], mx.ones((pad_size,)))

    def test_pad_no_nan(self, mock_norm: mx.array) -> None:
        output = pad_rms_norm(mock_norm, target_dim=64)
        assert not mx.any(mx.isnan(output)).item()

    def test_pad_with_mock_norm(self) -> None:
        norm = mx.random.normal((32,))
        output = pad_rms_norm(norm, target_dim=64)
        assert output.shape == (64,)
        assert mx.array_equal(output[:32], norm)
        assert mx.array_equal(output[32:], mx.ones((32,)))


MOE_SOURCE_HIDDEN = 32
MOE_SOURCE_INTERMEDIATE = 64
MOE_TARGET_HIDDEN = 48
MOE_TARGET_INTERMEDIATE = 32


@pytest.fixture()
def moe_inputs() -> tuple[mx.array, mx.array, mx.array]:
    qwen_gate = mx.random.normal((MOE_SOURCE_INTERMEDIATE, MOE_SOURCE_HIDDEN))
    qwen_up = mx.random.normal((MOE_SOURCE_INTERMEDIATE, MOE_SOURCE_HIDDEN))
    qwen_down = mx.random.normal((MOE_SOURCE_HIDDEN, MOE_SOURCE_INTERMEDIATE))
    return qwen_gate, qwen_up, qwen_down


@pytest.fixture()
def shared_expert(
    moe_inputs: tuple[mx.array, mx.array, mx.array],
) -> tuple[mx.array, mx.array, mx.array]:
    return project_mlp_to_shared_expert(
        *moe_inputs,
        target_intermediate=MOE_TARGET_INTERMEDIATE,
        target_hidden=MOE_TARGET_HIDDEN,
        source_hidden=MOE_SOURCE_HIDDEN,
    )


class TestProjectMlpToSharedExpert:
    def test_shared_gate_shape(self, moe_inputs: tuple[mx.array, mx.array, mx.array]) -> None:
        gate, _, _ = project_mlp_to_shared_expert(
            *moe_inputs,
            target_intermediate=MOE_TARGET_INTERMEDIATE,
            target_hidden=MOE_TARGET_HIDDEN,
            source_hidden=MOE_SOURCE_HIDDEN,
        )
        assert gate.shape == (MOE_TARGET_INTERMEDIATE, MOE_TARGET_HIDDEN)

    def test_shared_down_shape(self, moe_inputs: tuple[mx.array, mx.array, mx.array]) -> None:
        _, _, down = project_mlp_to_shared_expert(
            *moe_inputs,
            target_intermediate=MOE_TARGET_INTERMEDIATE,
            target_hidden=MOE_TARGET_HIDDEN,
            source_hidden=MOE_SOURCE_HIDDEN,
        )
        assert down.shape == (MOE_TARGET_HIDDEN, MOE_TARGET_INTERMEDIATE)

    def test_shared_preserves_subspace(self, moe_inputs: tuple[mx.array, mx.array, mx.array]) -> None:
        qwen_gate = moe_inputs[0]
        gate, _, _ = project_mlp_to_shared_expert(
            *moe_inputs,
            target_intermediate=MOE_TARGET_INTERMEDIATE,
            target_hidden=MOE_TARGET_HIDDEN,
            source_hidden=MOE_SOURCE_HIDDEN,
        )
        assert mx.array_equal(gate[:, :MOE_SOURCE_HIDDEN], qwen_gate[:MOE_TARGET_INTERMEDIATE, :])

    def test_shared_padding_zeros(self, moe_inputs: tuple[mx.array, mx.array, mx.array]) -> None:
        gate, _, _ = project_mlp_to_shared_expert(
            *moe_inputs,
            target_intermediate=MOE_TARGET_INTERMEDIATE,
            target_hidden=MOE_TARGET_HIDDEN,
            source_hidden=MOE_SOURCE_HIDDEN,
        )
        assert mx.array_equal(gate[:, MOE_SOURCE_HIDDEN:], mx.zeros((MOE_TARGET_INTERMEDIATE, MOE_TARGET_HIDDEN - MOE_SOURCE_HIDDEN)))

    def test_down_preserves_subspace(self, moe_inputs: tuple[mx.array, mx.array, mx.array]) -> None:
        qwen_down = moe_inputs[2]
        _, _, down = project_mlp_to_shared_expert(
            *moe_inputs,
            target_intermediate=MOE_TARGET_INTERMEDIATE,
            target_hidden=MOE_TARGET_HIDDEN,
            source_hidden=MOE_SOURCE_HIDDEN,
        )
        assert mx.array_equal(down[:MOE_SOURCE_HIDDEN, :], qwen_down[:, :MOE_TARGET_INTERMEDIATE])

    def test_down_padding_zeros(self, moe_inputs: tuple[mx.array, mx.array, mx.array]) -> None:
        _, _, down = project_mlp_to_shared_expert(
            *moe_inputs,
            target_intermediate=MOE_TARGET_INTERMEDIATE,
            target_hidden=MOE_TARGET_HIDDEN,
            source_hidden=MOE_SOURCE_HIDDEN,
        )
        assert mx.array_equal(down[MOE_SOURCE_HIDDEN:, :], mx.zeros((MOE_TARGET_HIDDEN - MOE_SOURCE_HIDDEN, MOE_TARGET_INTERMEDIATE)))

    def test_shared_no_nan(self, moe_inputs: tuple[mx.array, mx.array, mx.array]) -> None:
        gate, up, down = project_mlp_to_shared_expert(
            *moe_inputs,
            target_intermediate=MOE_TARGET_INTERMEDIATE,
            target_hidden=MOE_TARGET_HIDDEN,
            source_hidden=MOE_SOURCE_HIDDEN,
        )
        assert not mx.any(mx.isnan(gate)).item()
        assert not mx.any(mx.isnan(up)).item()
        assert not mx.any(mx.isnan(down)).item()


NUM_EXPERTS = 8
PERTURBATION_STD = 0.02


@pytest.fixture()
def routed_experts(
    shared_expert: tuple[mx.array, mx.array, mx.array],
) -> tuple[mx.array, mx.array, mx.array]:
    return init_routed_experts(*shared_expert, num_experts=NUM_EXPERTS, perturbation_std=PERTURBATION_STD)


class TestInitRoutedExperts:
    def test_routed_gate_shape(self, routed_experts: tuple[mx.array, mx.array, mx.array]) -> None:
        gate, up, down = routed_experts
        assert gate.shape == (NUM_EXPERTS, MOE_TARGET_INTERMEDIATE, MOE_TARGET_HIDDEN)
        assert up.shape == (NUM_EXPERTS, MOE_TARGET_INTERMEDIATE, MOE_TARGET_HIDDEN)
        assert down.shape == (NUM_EXPERTS, MOE_TARGET_HIDDEN, MOE_TARGET_INTERMEDIATE)

    def test_expert_0_is_copy(self, routed_experts: tuple[mx.array, mx.array, mx.array], shared_expert: tuple[mx.array, mx.array, mx.array]) -> None:
        r_gate, r_up, r_down = routed_experts
        s_gate, s_up, s_down = shared_expert
        assert mx.array_equal(r_gate[0], s_gate)
        assert mx.array_equal(r_up[0], s_up)
        assert mx.array_equal(r_down[0], s_down)

    def test_experts_1_plus_perturbed(self, routed_experts: tuple[mx.array, mx.array, mx.array], shared_expert: tuple[mx.array, mx.array, mx.array]) -> None:
        r_gate, r_up, r_down = routed_experts
        s_gate, s_up, s_down = shared_expert
        assert not mx.array_equal(r_gate[1], s_gate)
        assert not mx.array_equal(r_up[1], s_up)
        assert not mx.array_equal(r_down[1], s_down)

    def test_perturbation_magnitude(self, routed_experts: tuple[mx.array, mx.array, mx.array], shared_expert: tuple[mx.array, mx.array, mx.array]) -> None:
        r_gate = routed_experts[0]
        s_gate = shared_expert[0]
        diff = r_gate[1] - s_gate
        mean_abs = float(mx.mean(mx.abs(diff)).item())
        expected = PERTURBATION_STD * (2.0 / 3.141592653589793) ** 0.5
        assert 0.5 * expected < mean_abs < 1.5 * expected

    def test_routed_no_nan(self, routed_experts: tuple[mx.array, mx.array, mx.array]) -> None:
        gate, up, down = routed_experts
        assert not mx.any(mx.isnan(gate)).item()
        assert not mx.any(mx.isnan(up)).item()
        assert not mx.any(mx.isnan(down)).item()

    def test_routed_small_dims(self) -> None:
        small_gate = mx.random.normal((16, 8))
        small_up = mx.random.normal((16, 8))
        small_down = mx.random.normal((8, 16))
        r_gate, r_up, r_down = init_routed_experts(small_gate, small_up, small_down, num_experts=4, perturbation_std=0.01)
        assert r_gate.shape == (4, 16, 8)
        assert r_up.shape == (4, 16, 8)
        assert r_down.shape == (4, 8, 16)
        assert mx.array_equal(r_gate[0], small_gate)
        assert not mx.any(mx.isnan(r_gate)).item()
