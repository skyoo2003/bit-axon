import mlx.core as mx

from bit_axon.config import BitAxonConfig
from bit_axon.layers.moe import (
    MLP,
    SharedExpertMoE,
    SwitchGLU,
    SwitchLinear,
    swiglu,
)


class TestSwiGLU:
    def test_output_shape(self):
        x = mx.random.normal((2, 8, 16))
        gate = mx.random.normal((2, 8, 16))
        out = swiglu(x, gate)
        assert out.shape == (2, 8, 16)

    def test_values(self):
        x = mx.ones((2, 4))
        gate = mx.zeros((2, 4))
        out = swiglu(x, gate)
        mx.eval(out)
        assert mx.allclose(out, mx.zeros((2, 4)))


class TestSwitchLinear:
    def test_switch_linear_shape(self):
        B, L, K, D, H, E = 2, 4, 2, 16, 32, 4
        layer = SwitchLinear(D, H, E, bias=True)
        x = mx.random.normal((B, L, D))
        indices = mx.random.randint(0, E, shape=(B, L, K))
        out = layer(x, indices)
        mx.eval(out)
        assert out.shape == (B, L, K, H)

    def test_no_bias(self):
        B, L, K, D, H, E = 1, 4, 2, 8, 16, 4
        layer = SwitchLinear(D, H, E, bias=False)
        assert layer.bias is None
        x = mx.random.normal((B, L, D))
        indices = mx.random.randint(0, E, shape=(B, L, K))
        out = layer(x, indices)
        mx.eval(out)
        assert out.shape == (B, L, K, H)

    def test_4d_input(self):
        B, L, K, D, H, E = 1, 4, 2, 8, 16, 4
        layer = SwitchLinear(D, H, E, bias=False)
        x = mx.random.normal((B, L, K, D))
        indices = mx.random.randint(0, E, shape=(B, L, K))
        out = layer(x, indices)
        mx.eval(out)
        assert out.shape == (B, L, K, H)


class TestSwitchGLU:
    def test_switch_glu_shape(self):
        B, L, K, D, H, E = 1, 8, 2, 16, 32, 4
        layer = SwitchGLU(D, H, E, bias=False)
        x = mx.random.normal((B, L, D))
        indices = mx.random.randint(0, E, shape=(B, L, K))
        out = layer(x, indices)
        mx.eval(out)
        assert out.shape == (B, L, K, D)

    def test_sort_path(self):
        B, L, K, D, H, E = 1, 32, 2, 8, 16, 4
        layer = SwitchGLU(D, H, E, bias=False)
        x = mx.random.normal((B, L, D))
        indices = mx.random.randint(0, E, shape=(B, L, K))
        assert indices.size >= 64
        out = layer(x, indices)
        mx.eval(out)
        assert out.shape == (B, L, K, D)


class TestMLP:
    def test_mlp_shape(self):
        B, L, D, H = 2, 8, 16, 32
        mlp = MLP(D, H, bias=False)
        x = mx.random.normal((B, L, D))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (B, L, D)


class TestSharedExpertMoE:
    def test_moe_output_shape(self, default_config: BitAxonConfig):
        cfg = default_config
        moe = SharedExpertMoE(cfg.hidden_dim, cfg.moe_intermediate_dim, cfg.moe_num_experts, cfg.moe_top_k)
        x = mx.random.normal((1, 128, cfg.hidden_dim))
        out = moe(x)
        mx.eval(out)
        assert out.shape == (1, 128, cfg.hidden_dim)

    def test_routing_indices(self, default_config: BitAxonConfig):
        cfg = default_config
        moe = SharedExpertMoE(cfg.hidden_dim, cfg.moe_intermediate_dim, cfg.moe_num_experts, cfg.moe_top_k)
        x = mx.random.normal((1, 128, cfg.hidden_dim))
        gates = moe.gate(x)
        gates = mx.softmax(gates, axis=-1)
        k = cfg.moe_top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        mx.eval(inds)
        assert inds.shape == (1, 128, cfg.moe_top_k)
        assert inds.min() >= 0
        assert inds.max() < cfg.moe_num_experts

    def test_shared_expert_active(self, default_config: BitAxonConfig):
        cfg = default_config
        moe = SharedExpertMoE(cfg.hidden_dim, cfg.moe_intermediate_dim, cfg.moe_num_experts, cfg.moe_top_k)
        x = mx.random.normal((1, 8, cfg.hidden_dim))
        shared_out = moe.shared_expert(x)
        shared_gate = mx.sigmoid(moe.shared_expert_gate(x))
        gated = shared_out * shared_gate
        mx.eval(gated)
        assert gated.shape == (1, 8, cfg.hidden_dim)
        assert not mx.allclose(gated, mx.zeros_like(gated))

    def test_no_numpy_in_forward(self):
        import pathlib

        moe_src = pathlib.Path(__file__).resolve().parent.parent.parent / "src" / "bit_axon" / "layers" / "moe.py"
        content = moe_src.read_text()
        lines = content.split("\n")
        forward_lines = []
        in_forward = False
        for line in lines:
            stripped = line.strip()
            if "def __call__" in stripped:
                in_forward = True
            elif in_forward and stripped.startswith("def ") and "__call__" not in stripped:
                in_forward = False
            if in_forward:
                forward_lines.append(line)
        forward_code = "\n".join(forward_lines)
        assert "np." not in forward_code

    def test_gradient_flow(self, default_config: BitAxonConfig):
        cfg = default_config
        moe = SharedExpertMoE(cfg.hidden_dim, cfg.moe_intermediate_dim, cfg.moe_num_experts, cfg.moe_top_k)
        x = mx.random.normal((1, 8, cfg.hidden_dim))

        def loss_fn(m, x):
            return m(x).sum()

        loss, grads = mx.value_and_grad(loss_fn)(moe, x)
        mx.eval(loss, grads)
        assert loss.shape == ()
        assert len(grads) > 0
        for g in grads.values():
            if hasattr(g, "shape"):
                assert g.size > 0

    def test_small_config(self, small_config: BitAxonConfig):
        cfg = small_config
        moe = SharedExpertMoE(cfg.hidden_dim, cfg.moe_intermediate_dim, cfg.moe_num_experts, cfg.moe_top_k)
        x = mx.random.normal((1, 8, cfg.hidden_dim))
        out = moe(x)
        mx.eval(out)
        assert out.shape == (1, 8, cfg.hidden_dim)
