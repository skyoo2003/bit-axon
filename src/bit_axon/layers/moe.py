import mlx.core as mx
import mlx.nn as nn


@mx.compile
def swiglu(x: mx.array, gate: mx.array) -> mx.array:
    return nn.silu(gate) * x


class SwitchLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True):
        super().__init__()
        scale = (1.0 / input_dims) ** 0.5
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(num_experts, output_dims, input_dims))
        self.bias = mx.zeros((num_experts, output_dims)) if bias else None

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        B, L, K = indices.shape
        D = x.shape[-1]
        flat_idx = indices.reshape(-1)

        if x.ndim == 3:
            x = mx.broadcast_to(mx.expand_dims(x, 2), (B, L, K, D))
        x_flat = x.reshape(-1, 1, D)

        w_t = self.weight.swapaxes(-1, -2)
        out = mx.gather_mm(x_flat, w_t, rhs_indices=flat_idx, sorted_indices=sorted_indices)
        out = out.squeeze(-2)
        out = out.reshape(B, L, K, -1)
        if self.bias is not None:
            out = out + self.bias[indices]
        return out


def _gather_sort(x, indices):
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.reshape(-1, x.shape[-1])[order], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class SwitchGLU(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int, bias: bool = False):
        super().__init__()
        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        B, L, K = indices.shape
        D = x.shape[-1]

        do_sort = indices.size >= 64
        inv_order = None
        if do_sort:
            x_exp = mx.broadcast_to(mx.expand_dims(x, 2), (B, L, K, D))
            x_sorted, idx_flat, inv_order = _gather_sort(x_exp, indices)
            x_in = x_sorted.reshape(B, L, K, D)
            idx = idx_flat.reshape(B, L, K)
        else:
            x_in = x
            idx = indices

        x_up = self.up_proj(x_in, idx)
        x_gate = self.gate_proj(x_in, idx)
        h = swiglu(x_up, x_gate)
        y = self.down_proj(h, idx)

        if do_sort:
            y_flat = y.reshape(-1, y.shape[-1])
            y = _scatter_unsort(y_flat, inv_order, shape=(B, L, K))

        return y


class MLP(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.up_proj(x), self.gate_proj(x)))


class SharedExpertMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_dim, num_experts)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)
        self.shared_expert = MLP(dim, intermediate_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_out = self.shared_expert(x)
        shared_out = mx.sigmoid(self.shared_expert_gate(x)) * shared_out
        return y + shared_out
