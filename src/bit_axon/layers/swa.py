from __future__ import annotations

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
        self.chunk_size = 512

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array, mask: mx.array | None = None, cache=None) -> tuple[mx.array, object | None]:
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

        kv_len_before = cache.k.shape[2] if cache is not None and cache.k is not None else 0
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        kv_len = k.shape[2]
        q_base_offset = kv_len_before

        out_chunks = []
        for i in range(0, L, self.chunk_size):
            j = min(i + self.chunk_size, L)
            q_chunk = q[:, :, i:j, :]

            scores = (q_chunk * self.scale) @ k.transpose(0, 1, 3, 2)

            if mask is not None:
                scores = scores + mask[:, :, i:j, :]
            else:
                chunk_mask = self._make_sliding_window_mask(j - i, kv_len, q_offset=q_base_offset + i)
                scores = scores + chunk_mask

            attn = mx.softmax(scores, axis=-1)
            out_chunk = attn @ v
            out_chunks.append(out_chunk)

        out = mx.concatenate(out_chunks, axis=2)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)

        out = self.o_proj(out)

        new_cache = cache if cache is not None else None
        return out, new_cache

    def _make_sliding_window_mask(self, seq_len: int, kv_len: int, q_offset: int = 0):
        q_pos = mx.arange(q_offset, q_offset + seq_len)
        k_pos = mx.arange(kv_len)

        causal_offset = kv_len - (q_offset + seq_len)
        causal_mask = k_pos[None, :] <= (q_pos[:, None] + causal_offset)
        window_mask = (q_pos[:, None] + causal_offset) - k_pos[None, :] < self.window_size

        mask = mx.where(causal_mask & window_mask, 0.0, -mx.inf)
        return mask
