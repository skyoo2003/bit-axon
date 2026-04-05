from __future__ import annotations

import mlx.core as mx


class ArraysCache:
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
    def __init__(self):
        self.k: mx.array | None = None
        self.v: mx.array | None = None

    def update_and_fetch(self, xk: mx.array, xv: mx.array) -> tuple[mx.array, mx.array]:
        if self.k is None:
            self.k = xk
            self.v = xv
        else:
            k = self.k
            v = self.v
            assert k is not None and v is not None
            self.k = mx.concatenate([k, xk], axis=2)
            self.v = mx.concatenate([v, xv], axis=2)
        return self.k, self.v
