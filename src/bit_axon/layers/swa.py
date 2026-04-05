import math

import mlx.core as mx
import mlx.nn as nn


class SlidingWindowAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        kv_len = k.shape[2]

        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)

        if mask is not None:
            scores = scores + mask
        else:
            sw_mask = self._make_sliding_window_mask(L, kv_len)
            scores = scores + sw_mask

        attn = mx.softmax(scores, axis=-1)

        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)

        out = self.o_proj(out)

        new_cache = cache if cache is not None else None
        return out, new_cache

    def _make_sliding_window_mask(self, seq_len: int, kv_len: int):
        q_pos = mx.arange(seq_len)
        k_pos = mx.arange(kv_len)

        causal_offset = kv_len - seq_len
        causal_mask = k_pos[None, :] <= (q_pos[:, None] + causal_offset)
        window_mask = (q_pos[:, None] + causal_offset) - k_pos[None, :] < self.window_size

        mask = mx.where(causal_mask & window_mask, 0.0, -mx.inf)
        return mask
