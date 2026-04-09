from __future__ import annotations

import mlx.core as mx


class ArraysCache:
    """Fixed-size cache storing optional mx.array slots.

    Used as a generic container for layer-level intermediate arrays.
    """

    def __init__(self, size: int):
        self.cache: list[mx.array | None] = [None] * size

    def __getitem__(self, idx: int) -> mx.array | None:
        return self.cache[idx]

    def __setitem__(self, idx: int, value: mx.array | None):
        self.cache[idx] = value

    def update(self, values: list[mx.array]):
        for i, v in enumerate(values):
            self.cache[i] = v


class KVCache:
    """Key-value cache for autoregressive sliding-window attention.

    Concatenates new K/V tensors along the sequence dimension (axis 2)
    on each update step.  When *window_size* is set, the cache is trimmed
    to retain at most that many positions, preventing unbounded memory
    growth during long sessions.
    """

    def __init__(self, window_size: int | None = None):
        self.window_size = window_size
        self.k: mx.array | None = None
        self.v: mx.array | None = None

    def update_and_fetch(self, xk: mx.array, xv: mx.array) -> tuple[mx.array, mx.array]:
        if self.k is None:
            self.k = xk
            self.v = xv
        else:
            k = self.k
            v = self.v
            if k is None or v is None:
                raise RuntimeError("KVCache state is inconsistent")
            self.k = mx.concatenate([k, xk], axis=2)
            self.v = mx.concatenate([v, xv], axis=2)
        if self.window_size is not None:
            self.k = self.k[:, :, -self.window_size :]
            self.v = self.v[:, :, -self.window_size :]
        return self.k, self.v
