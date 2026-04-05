import mlx.core as mx
import pytest

from bit_axon.config import BitAxonConfig
from bit_axon.layers.axon_ssm import AxonSSM


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
