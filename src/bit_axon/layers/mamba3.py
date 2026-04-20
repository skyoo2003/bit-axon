"""Pure-MLX Mamba-3 block (arXiv:2603.15569) for Bit-Axon.

Port strategy
-------------
The reference implementation (``reference_impl/mamba3.py``) is PyTorch + Triton
and fuses the entire SSM recurrence into a single kernel. MLX has no Triton;
we re-derive the recurrence in pure MLX using three facts from the paper:

1. **Appendix A.1.** The 3-term exponential-trapezoidal recurrence's SSD mask
   factors as ``L = L1 * L2`` where ``L1`` is the standard Mamba-2 scalar-decay
   mask (1-semiseparable) and ``L2`` is a width-2 convolution on the
   state-input. We therefore precompute ``v_t = gamma_t * B_tilde_t * x_t + beta_t * B_tilde_{t-1} * x_{t-1}``
   and hand it to a standard scalar-decay scan - no new scan algorithm is
   needed; the existing Mamba-2-style chunked scan is reused.

2. **Proposition 3/4 + Appendix B.3.** The complex SSM is expressed as a real
   SSM via the RoPE trick: cumulative 2x2 block rotations
   ``Q_t = Prod_{s<=t} R_s^T`` applied to B and C. After the change of
   variables ``h_tilde_t = Q_t * h_t``, the internal scan is rotation-free.
   Rotations are a pre-processing step on ``B, C`` - not baked into the
   recurrence.

3. **Appendix A.3.** ``lambda_t = sigmoid(trap_t)`` is the data-dependent
   trapezoidal gate. No reparameterization; the fixed-1/2 variant is worse.

SISO only in this file; MIMO is added in a sibling class (task #13).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class RMSNormGated(nn.Module):
    """RMSNorm over the last axis with a learnable per-feature gain."""

    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims

    def __call__(self, x: mx.array) -> mx.array:
        mean_sq = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(mean_sq + self.eps) * self.weight


def _apply_rotary(v: mx.array, theta: mx.array) -> mx.array:
    """Apply per-position 2x2 rotation by ``theta`` to the last two entries of ``v``."""
    c = mx.cos(theta)
    s = mx.sin(theta)
    v0 = v[..., 0]
    v1 = v[..., 1]
    return mx.stack([c * v0 - s * v1, s * v0 + c * v1], axis=-1)


def _init_dt_bias(nheads: int, dt_min: float, dt_max: float, dt_init_floor: float) -> mx.array:
    log_min = math.log(dt_min)
    log_max = math.log(dt_max)
    u = mx.random.uniform(shape=(nheads,))
    dt = mx.exp(u * (log_max - log_min) + log_min)
    dt = mx.maximum(dt, dt_init_floor)
    return dt + mx.log(-mx.expm1(-dt))


def _segsum(x: mx.array) -> mx.array:
    L = x.shape[-1]
    cs = mx.cumsum(x, axis=-1)
    diff = cs[..., :, None] - cs[..., None, :]
    mask = mx.tril(mx.ones((L, L), dtype=diff.dtype), -1)
    return diff * mask


def _scalar_decay_scan(
    v: mx.array,
    a_dt: mx.array,
    chunk_size: int,
    init_state: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Mamba-2 scalar-decay scan: h_t = exp(a_dt_t) · h_{t-1} + v_t.

    Args:
        v: (B, T, H, N) state-input (already includes the width-2 L2 conv).
        a_dt: (B, T, H) decay exponents Δ_t · A_t, expected non-positive.
        chunk_size: chunk size; paper recommends 64.
        init_state: optional (B, H, N).

    Returns:
        (h_seq, h_last) — (B, T, H, N) and (B, H, N).
    """
    B, T, H, N = v.shape
    if init_state is None:
        init_state = mx.zeros((B, H, N), dtype=v.dtype)

    if chunk_size >= T:
        h = init_state
        outs: list[mx.array] = []
        for t in range(T):
            alpha_t = mx.exp(a_dt[:, t, :, None])
            h = alpha_t * h + v[:, t, :, :]
            outs.append(h)
        return mx.stack(outs, axis=1), h

    n_chunks = (T + chunk_size - 1) // chunk_size
    outs_chunks: list[mx.array] = []
    h = init_state
    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, T)
        S = hi - lo
        v_c = v[:, lo:hi, :, :]
        a_c = a_dt[:, lo:hi, :]

        a_c_hl = a_c.transpose(0, 2, 1)  # (B, H, S)
        seg = _segsum(a_c_hl)  # (B, H, S, S)
        mask = mx.tril(mx.ones((S, S), dtype=v.dtype))
        decay = mx.exp(seg) * mask  # (B, H, S, S)
        v_c_bhsn = v_c.transpose(0, 2, 1, 3)  # (B, H, S, N)
        intra = mx.matmul(decay, v_c_bhsn)  # (B, H, S, N)

        cs = mx.cumsum(a_c_hl, axis=-1)
        carry = mx.exp(cs)[..., None] * h[:, :, None, :]
        h_chunk = intra + carry
        outs_chunks.append(h_chunk.transpose(0, 2, 1, 3))
        h = h_chunk[:, :, -1, :]

    return mx.concatenate(outs_chunks, axis=1), h


class Mamba3(nn.Module):
    """Pure-MLX Mamba-3 SISO block.

    Shapes:
        d_model       = hidden dim (D)
        d_state       = SSM state size (N)
        headdim       = P
        expand        = inner expansion  → d_inner = expand * d_model
        nheads        = d_inner / headdim
        num_bc_heads  = B/C groups (MVA-style sharing); SISO default = 1
        rope_fraction ∈ {0.5, 1.0}
        num_rope_angles = (d_state * rope_fraction) // 2
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        a_floor: float = 1e-4,
        d_conv: int = 4,
        chunk_size: int = 64,
        is_mimo: bool = False,
        mimo_rank: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.d_inner = expand * d_model
        if self.d_inner % headdim != 0:
            msg = f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"
            raise ValueError(msg)
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups
        self.chunk_size = chunk_size
        self.a_floor = a_floor
        self.d_conv = d_conv
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1

        if rope_fraction not in (0.5, 1.0):
            msg = f"rope_fraction must be 0.5 or 1.0, got {rope_fraction}"
            raise ValueError(msg)
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2
        if self.num_rope_angles <= 0:
            msg = f"num_rope_angles must be > 0 (d_state={d_state}, rope_fraction={rope_fraction})"
            raise ValueError(msg)

        # in_proj output: [z, x, B, C, dd_dt, dd_A, trap, angle].
        # B/C grow with mimo_rank via the MVA sharing (paper §3.3 + App C).
        d_in_proj = 2 * self.d_inner + 2 * d_state * self.num_bc_heads * self.mimo_rank + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
        )

        self.dt_bias = _init_dt_bias(self.nheads, dt_min, dt_max, dt_init_floor)
        # B/C biases carry a rank dim: (H, R, N). SISO collapses R to 1.
        self.B_bias = mx.ones((self.nheads, self.mimo_rank, d_state))
        self.C_bias = mx.ones((self.nheads, self.mimo_rank, d_state))

        self.B_norm = RMSNormGated(d_state, eps=1e-5)
        self.C_norm = RMSNormGated(d_state, eps=1e-5)

        self.D = mx.ones((self.nheads,))

        # MIMO rank-R extension projections (Appendix C). Shape (H, R, P).
        #   X_t = einsum('bthp, hrp -> bthrp', X'_t, mimo_x)
        #   Z_t = einsum('bthp, hrp -> bthrp', Z'_t, mimo_z)
        #   O'_t = einsum('bthrp, hrp -> bthp', Y'_t, mimo_o)
        # For SISO we still allocate rank=1 tensors so the forward pass is
        # parameterized uniformly; an R=1 projection is an identity-like
        # per-channel linear and adds only H*P parameters.
        init_x = mx.full((self.nheads, self.mimo_rank, headdim), 1.0 / self.mimo_rank)
        init_z = mx.ones((self.nheads, self.mimo_rank, headdim))
        init_o = mx.full((self.nheads, self.mimo_rank, headdim), 1.0 / self.mimo_rank)
        self.mimo_x = init_x
        self.mimo_z = init_z
        self.mimo_o = init_o

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _causal_conv(self, x: mx.array, conv_cache: mx.array | None) -> tuple[mx.array, mx.array]:
        K = self.d_conv
        if conv_cache is not None:
            x_cat = mx.concatenate([conv_cache, x], axis=1)
        else:
            x_cat = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])
        new_conv_cache = x_cat[:, -(K - 1) :, :]
        return self.conv1d(x_cat), new_conv_cache

    def __call__(
        self,
        u: mx.array,
        cache: list | None = None,
    ) -> tuple[mx.array, list]:
        B, T, _D_in = u.shape
        H = self.nheads
        N = self.d_state
        P = self.headdim
        S = self.num_rope_angles
        R = self.mimo_rank
        G = self.num_bc_heads

        if cache is not None:
            conv_cache, ssm_state, angle_state, last_Bx = cache
        else:
            conv_cache = None
            ssm_state = None
            angle_state = None
            last_Bx = None

        zxBCdtAtrap_angle = self.in_proj(u)
        sizes = [
            self.d_inner,  # z
            self.d_inner,  # x
            N * G * R,  # B
            N * G * R,  # C
            H,  # dd_dt
            H,  # dd_A
            H,  # trap
            S,  # angle
        ]
        z, x, B_proj, C_proj, dd_dt, dd_A, trap, angles = mx.split(
            zxBCdtAtrap_angle,
            indices_or_sections=self._split_points(sizes),
            axis=-1,
        )

        x, new_conv_cache = self._causal_conv(x, conv_cache)
        if x.shape[1] > T:
            x = x[:, -T:, :]
        x = nn.silu(x)
        x = x.reshape(B, T, H, P)
        z = z.reshape(B, T, H, P)

        # B/C: reshape to (B, T, R, G, N), normalize state-dim, broadcast G→H.
        B_ssm = B_proj.reshape(B, T, R, G, N)
        C_ssm = C_proj.reshape(B, T, R, G, N)
        B_ssm = self.B_norm(B_ssm)
        C_ssm = self.C_norm(C_ssm)
        B_ssm = mx.broadcast_to(B_ssm, (B, T, R, H, N))
        C_ssm = mx.broadcast_to(C_ssm, (B, T, R, H, N))
        # Transpose to (B, T, H, R, N) for per-head downstream ops.
        B_ssm = B_ssm.transpose(0, 1, 3, 2, 4)
        C_ssm = C_ssm.transpose(0, 1, 3, 2, 4)

        DT = nn.softplus(dd_dt + self.dt_bias)  # (B, T, H)
        A = -nn.softplus(dd_A)
        A = mx.minimum(A, -self.a_floor)  # (B, T, H)
        ADT = A * DT  # (B, T, H)
        lam = mx.sigmoid(trap)  # (B, T, H)

        # Cumulative rotary on the first 2S of the state dim.
        angles_step = DT[..., None] * angles[:, :, None, :]  # (B, T, H, S)
        if angle_state is not None:
            angles_cum = mx.cumsum(angles_step, axis=1) + angle_state[:, None, :, :]
        else:
            angles_cum = mx.cumsum(angles_step, axis=1)
        new_angle_state = angles_cum[:, -1, :, :]  # (B, H, S)

        # Apply rotation independently per rank (same angles, broadcast over R).
        two_S = 2 * S
        angles_cum_r = angles_cum[:, :, :, None, :]  # (B, T, H, 1, S)
        B_rot = B_ssm[..., :two_S].reshape(B, T, H, R, S, 2)
        C_rot = C_ssm[..., :two_S].reshape(B, T, H, R, S, 2)
        B_rot = _apply_rotary(B_rot, angles_cum_r).reshape(B, T, H, R, two_S)
        C_rot = _apply_rotary(C_rot, angles_cum_r).reshape(B, T, H, R, two_S)
        B_pass = B_ssm[..., two_S:]
        C_pass = C_ssm[..., two_S:]
        # B/C biases carry a rank dim: (H, R, N) → broadcast over (B, T).
        B_tilde = mx.concatenate([B_rot, B_pass], axis=-1) + self.B_bias[None, None, :, :, :]
        C_tilde = mx.concatenate([C_rot, C_pass], axis=-1) + self.C_bias[None, None, :, :, :]

        gamma = lam * DT  # (B, T, H)
        beta = (1.0 - lam) * DT * mx.exp(ADT)  # (B, T, H)

        # MIMO rank-R lifting of x (per Appendix C): X_t = mimo_x · x'_t  →  (B, T, H, R, P).
        # SISO (R=1) collapses this to a per-channel multiply and is identity-like
        # when mimo_x is initialized with 1/R = 1.
        X_rank = x[..., None, :] * self.mimo_x[None, None, :, :, :]  # (B, T, H, R, P)
        Z_rank = z[..., None, :] * self.mimo_z[None, None, :, :, :]  # (B, T, H, R, P)

        # State-input outer product across rank: (B, T, H, N, R) x (B, T, H, R, P) → (B, T, H, N, P).
        B_for_mm = B_tilde.transpose(0, 1, 2, 4, 3)  # (B, T, H, N, R)
        Bx = mx.matmul(B_for_mm, X_rank)  # (B, T, H, N, P)
        # β term uses the previous timestep's B̃·x. Pull it from the cache so
        # decode-path results equal the prefill of the same tokens.
        if last_Bx is None:
            prev_seed = mx.zeros_like(Bx[:, :1, :, :, :])
        else:
            prev_seed = last_Bx[:, None, :, :, :]
        if T > 1:
            Bx_prev = mx.concatenate([prev_seed, Bx[:, :-1, :, :, :]], axis=1)
        else:
            Bx_prev = prev_seed
        v = gamma[..., None, None] * Bx + beta[..., None, None] * Bx_prev  # (B, T, H, N, P)
        new_last_Bx = Bx[:, -1, :, :, :]  # (B, H, N, P)
        NP = N * P
        v_flat = v.reshape(B, T, H, NP)

        init = None if ssm_state is None else ssm_state.reshape(B, H, NP)
        h_seq_flat, h_last_flat = _scalar_decay_scan(v_flat, ADT, self.chunk_size, init_state=init)
        h_seq = h_seq_flat.reshape(B, T, H, N, P)
        new_ssm_state = h_last_flat.reshape(B, H, N, P)

        # Output: Y_t = H_t^T · C_t → per-rank (B, T, H, R, P), gated, then project R→1.
        # Use einsum: (B, T, H, R, N) · (B, T, H, N, P) → (B, T, H, R, P).
        Y_rank = mx.einsum("bthrn,bthnp->bthrp", C_tilde, h_seq)
        Y_rank = Y_rank * nn.silu(Z_rank)  # gate per-rank
        # Down-project R back to 1: sum over rank with mimo_o → (B, T, H, P).
        y = mx.einsum("bthrp,hrp->bthp", Y_rank, self.mimo_o)

        y = y + self.D[None, None, :, None] * x
        y = y.reshape(B, T, self.d_inner)
        out = self.out_proj(y)

        return out, [new_conv_cache, new_ssm_state, new_angle_state, new_last_Bx]

    @staticmethod
    def _split_points(sizes: list[int]) -> list[int]:
        pts: list[int] = []
        acc = 0
        for s in sizes[:-1]:
            acc += s
            pts.append(acc)
        return pts
