import mlx.core as mx
import mlx.nn as nn

from bit_axon.config import BitAxonConfig


@mx.compile
def _ssm_fma(a: mx.array, b: mx.array, c: mx.array) -> mx.array:
    return a * b + c


@mx.compile
def _compute_dt(dt: mx.array, dt_bias: mx.array, lo: float, hi: float) -> mx.array:
    return mx.clip(nn.softplus(dt + dt_bias), lo, hi)


class AxonSSM(nn.Module):
    """Mamba-style state space model layer with causal convolution.

    Implements selective SSM with hardware-aware scan, causal conv1d prefix,
    and a gating branch (SiLU). The SSM expansion replaces the traditional
    FFN/MLP role in the block.

    Attributes:
        in_proj: Projects input to 2x intermediate dim (x and z branches).
        conv1d: Depthwise causal 1D convolution.
        x_proj: Projects conv output to B, C, dt parameters.
        dt_proj: Projects raw dt to per-channel step sizes.
        out_proj: Projects SSM output back to hidden dim.
        A_log: Log of the diagonal SSM state matrix (learnable).
        D: Skip connection parameter per channel.
    """

    def __init__(self, config: BitAxonConfig):
        """Initialize the AxonSSM layer.

        Args:
            config: BitAxonConfig with ssm_intermediate_dim, ssm_d_state,
                ssm_d_conv, and hidden_dim settings.
        """
        super().__init__()
        D = config.hidden_dim
        E = config.ssm_intermediate_dim
        d_state = config.ssm_d_state
        d_conv = config.ssm_d_conv

        self.in_proj = nn.Linear(D, 2 * E, bias=False)
        self.conv1d = nn.Conv1d(E, E, kernel_size=d_conv, groups=E, bias=True)
        self.x_proj = nn.Linear(E, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, E, bias=True)
        self.out_proj = nn.Linear(E, D, bias=False)
        self.d_conv = d_conv
        self.d_state = d_state
        self.E = E

        A = mx.repeat(
            mx.arange(1, d_state + 1).astype(mx.float32)[None, :],
            repeats=E,
            axis=0,
        )
        self.A_log = mx.log(A)
        self.D = mx.ones((E,))

    def _causal_conv1d(self, x, conv_cache=None):
        K = self.d_conv
        if conv_cache is not None:
            x = mx.concatenate([conv_cache, x], axis=1)
            new_conv_cache = x[:, -(K - 1) :, :]
        else:
            x = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])
            new_conv_cache = x[:, -(K - 1) :, :]
        x = self.conv1d(x)
        return x, new_conv_cache

    def _ssm_scan(self, x, dt, B_in, C_in, ssm_state=None):
        B_batch, L, E = x.shape
        d_state = self.d_state
        A = -mx.exp(self.A_log)

        if ssm_state is None:
            ssm_state = mx.zeros((B_batch, E, d_state))

        ys = []
        h = ssm_state
        for t in range(L):
            x_t = x[:, t, :]
            dt_t = dt[:, t, :]
            B_t = B_in[:, t, :]
            C_t = C_in[:, t, :]

            dA = mx.exp(dt_t[:, :, None] * A[None, :, :])
            dB = dt_t[:, :, None] * B_t[:, None, :]

            h = _ssm_fma(dA, h, dB * x_t[:, :, None])
            y = (h * C_t[:, None, :]).sum(axis=-1) + self.D * x_t
            ys.append(y)

        y = mx.stack(ys, axis=1)
        return y, h

    def __call__(self, x, cache=None):
        """Run the SSM forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            cache: Optional [conv_cache, ssm_state] from a previous step.

        Returns:
            Tuple of (output, new_cache). Output has shape (batch, seq_len, hidden_dim).
            new_cache is [updated_conv_cache, updated_ssm_state] for autoregressive decoding.
        """
        B_batch, L, D = x.shape

        if cache is not None:
            conv_cache, ssm_state = cache
        else:
            conv_cache = None
            ssm_state = None

        xz = self.in_proj(x)
        x_branch, z_branch = mx.split(xz, 2, axis=-1)

        x_conv, new_conv_cache = self._causal_conv1d(x_branch, conv_cache)
        x_conv = nn.silu(x_conv)

        x_proj_out = self.x_proj(x_conv)
        BC_dt = mx.split(x_proj_out, [self.d_state, 2 * self.d_state], axis=-1)
        B_ssm = BC_dt[0]
        C_ssm = BC_dt[1]
        dt = BC_dt[2]

        dt_raw = self.dt_proj(dt)
        dt = _compute_dt(dt_raw, mx.zeros_like(dt_raw), 1e-4, 100.0)

        y, new_ssm_state = self._ssm_scan(x_conv, dt, B_ssm, C_ssm, ssm_state)

        y = y * nn.silu(z_branch)
        output = self.out_proj(y)

        new_cache = [new_conv_cache, new_ssm_state]
        return output, new_cache
