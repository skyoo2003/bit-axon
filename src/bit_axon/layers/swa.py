import math

import mlx.core as mx
import mlx.nn as nn


class SlidingWindowAttention(nn.Module):
    """Multi-head attention with a sliding window mask.

    Restricts each token to attend only to tokens within a fixed window size,
    reducing attention from O(n²) to O(n * window_size).

    Attributes:
        q_proj: Query projection.
        k_proj: Key projection.
        v_proj: Value projection.
        o_proj: Output projection.
    """

    def __init__(self, hidden_dim: int, num_heads: int, window_size: int):
        """Initialize sliding window attention.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            window_size: Maximum attention distance per side.
        """
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
        """Forward pass for sliding window attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            mask: Optional attention mask to add to scores. If None, a sliding
                window causal mask is generated automatically.
            cache: Optional KVCache for autoregressive decoding.

        Returns:
            Tuple of (output, cache). Output has shape (batch, seq_len, hidden_dim).
            cache is the updated KVCache or None.
        """
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
