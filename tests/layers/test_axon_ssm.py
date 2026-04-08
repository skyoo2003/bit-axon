import mlx.core as mx
import mlx.nn as nn
import pytest

from bit_axon.config import BitAxonConfig
from bit_axon.layers.axon_ssm import AxonSSM, _compute_dt


def _get_ssm_inputs(model, x):
    xz = model.in_proj(x)
    x_branch, z_branch = mx.split(xz, 2, axis=-1)
    x_conv, _ = model._causal_conv1d(x_branch)
    x_conv = nn.silu(x_conv)
    x_proj_out = model.x_proj(x_conv)
    d_state = model.d_state
    BC_dt = mx.split(x_proj_out, [d_state, 2 * d_state], axis=-1)
    B_ssm = BC_dt[0]
    C_ssm = BC_dt[1]
    dt_raw = BC_dt[2]
    dt_raw = model.dt_proj(dt_raw)
    dt = _compute_dt(dt_raw, mx.zeros_like(dt_raw), 1e-4, 100.0)
    return x_conv, dt, B_ssm, C_ssm


@pytest.fixture
def default_config() -> BitAxonConfig:
    return BitAxonConfig()


@pytest.fixture
def small_config() -> BitAxonConfig:
    return BitAxonConfig(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        d_source_model=128,
        vocab_size=1024,
        ssm_d_state=4,
        ssm_d_conv=2,
        ssm_expand=2,
        swa_window_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_dim=512,
    )


class TestAxonSSM:
    def test_output_shape(self, default_config):
        model = AxonSSM(default_config)
        x = mx.random.normal((1, 128, default_config.hidden_dim))
        output, cache = model(x)
        assert output.shape == (1, 128, default_config.hidden_dim)

    def test_output_no_nan(self, small_config):
        model = AxonSSM(small_config)
        x = mx.random.normal((1, 64, small_config.hidden_dim))
        output, _ = model(x)
        mx.eval(output)
        assert mx.all(mx.isfinite(output))

    def test_cache_shapes(self, default_config):
        model = AxonSSM(default_config)
        E = default_config.ssm_intermediate_dim
        x = mx.random.normal((1, 8, default_config.hidden_dim))
        _, cache = model(x)
        conv_state, ssm_state = cache
        assert conv_state.shape == (1, default_config.ssm_d_conv - 1, E)
        assert ssm_state.shape == (1, E, default_config.ssm_d_state)

    def test_cache_incremental(self, small_config):
        model = AxonSSM(small_config)
        x1 = mx.random.normal((1, 16, small_config.hidden_dim))
        output1, cache = model(x1)
        mx.eval(output1, cache[0], cache[1])

        x2 = mx.random.normal((1, 4, small_config.hidden_dim))
        output2, new_cache = model(x2, cache=cache)
        mx.eval(output2, new_cache[0], new_cache[1])
        assert output2.shape == (1, 4, small_config.hidden_dim)
        assert mx.all(mx.isfinite(output2))

    def test_gradient_flow(self, small_config):
        model = AxonSSM(small_config)
        x = mx.random.normal((1, 8, small_config.hidden_dim))

        def loss_fn(model):
            output, _ = model(x)
            return output.sum()

        grads = mx.grad(loss_fn)(model)

        def check_grads(d):
            for v in d.values():
                if isinstance(v, dict):
                    check_grads(v)
                elif v is not None:
                    mx.eval(v)
                    assert mx.all(mx.isfinite(v))

        check_grads(grads)

    def test_small_config(self, small_config):
        model = AxonSSM(small_config)
        E = small_config.ssm_intermediate_dim
        x = mx.random.normal((1, 32, small_config.hidden_dim))
        output, cache = model(x)
        mx.eval(output, cache[0], cache[1])
        assert output.shape == (1, 32, small_config.hidden_dim)
        assert cache[0].shape == (1, small_config.ssm_d_conv - 1, E)
        assert cache[1].shape == (1, E, small_config.ssm_d_state)


class TestAxonSSMParallelScan:
    def test_parallel_matches_sequential_small(self, small_config):
        model = AxonSSM(small_config)
        x = mx.random.normal((1, 32, small_config.hidden_dim))
        x_conv, dt, B_ssm, C_ssm = _get_ssm_inputs(model, x)

        y_seq, h_seq = model._ssm_scan(x_conv, dt, B_ssm, C_ssm)
        y_par, h_par = model._ssm_scan_parallel(x_conv, dt, B_ssm, C_ssm)
        mx.eval(y_seq, h_seq, y_par, h_par)

        assert mx.allclose(y_seq, y_par, atol=1e-3)
        assert mx.allclose(h_seq, h_par, atol=1e-3)

    def test_parallel_matches_sequential_medium(self):
        config = BitAxonConfig(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            d_source_model=64,
            vocab_size=512,
            ssm_d_state=4,
            ssm_d_conv=2,
            ssm_expand=2,
            swa_window_size=32,
            moe_num_experts=4,
            moe_top_k=2,
            moe_intermediate_dim=256,
        )
        model = AxonSSM(config)
        x = mx.random.normal((1, 128, config.hidden_dim))
        x_conv, dt, B_ssm, C_ssm = _get_ssm_inputs(model, x)

        y_seq, h_seq = model._ssm_scan(x_conv, dt, B_ssm, C_ssm)
        y_par, h_par = model._ssm_scan_parallel(x_conv, dt, B_ssm, C_ssm)
        mx.eval(y_seq, h_seq, y_par, h_par)

        assert mx.allclose(y_seq, y_par, atol=1e-3)
        assert mx.allclose(h_seq, h_par, atol=1e-3)

    def test_parallel_with_cache(self, small_config):
        model = AxonSSM(small_config)
        x_prefill = mx.random.normal((1, 16, small_config.hidden_dim))
        output_prefill, cache = model(x_prefill)
        mx.eval(output_prefill, cache[0], cache[1])

        x_decode = mx.random.normal((1, 4, small_config.hidden_dim))
        output_decode, new_cache = model(x_decode, cache=cache)
        mx.eval(output_decode, new_cache[0], new_cache[1])

        assert output_decode.shape == (1, 4, small_config.hidden_dim)
        assert mx.all(mx.isfinite(output_decode))
        assert new_cache[1].shape == (1, small_config.ssm_intermediate_dim, small_config.ssm_d_state)

    def test_parallel_gradient_flow(self, small_config):
        model = AxonSSM(small_config)
        x = mx.random.normal((1, 8, small_config.hidden_dim))

        def loss_fn(model):
            output, _ = model(x)
            return output.sum()

        grads = mx.grad(loss_fn)(model)

        def check_grads(d):
            for v in d.values():
                if isinstance(v, dict):
                    check_grads(v)
                elif v is not None:
                    mx.eval(v)
                    assert mx.all(mx.isfinite(v))

        check_grads(grads)

    def test_parallel_edge_cases(self, small_config):
        model = AxonSSM(small_config)

        for L in [1, 3, 65]:
            x = mx.random.normal((1, L, small_config.hidden_dim))
            x_conv, dt, B_ssm, C_ssm = _get_ssm_inputs(model, x)

            y_seq, h_seq = model._ssm_scan(x_conv, dt, B_ssm, C_ssm)
            y_par, h_par = model._ssm_scan_parallel(x_conv, dt, B_ssm, C_ssm)
            mx.eval(y_seq, h_seq, y_par, h_par)

            assert mx.allclose(y_seq, y_par, atol=1e-3), f"L={L} output mismatch"
            assert mx.allclose(h_seq, h_par, atol=1e-3), f"L={L} state mismatch"

    def test_parallel_state_shapes(self, small_config):
        model = AxonSSM(small_config)
        x = mx.random.normal((2, 32, small_config.hidden_dim))
        x_conv, dt, B_ssm, C_ssm = _get_ssm_inputs(model, x)

        _, h = model._ssm_scan_parallel(x_conv, dt, B_ssm, C_ssm)
        mx.eval(h)

        assert h.shape == (2, small_config.ssm_intermediate_dim, small_config.ssm_d_state)
