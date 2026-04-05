import math

import mlx.core as mx
import mlx.nn as nn

from bit_axon.config import BitAxonConfig
from bit_axon.evaluation.perplexity import compute_perplexity
from bit_axon.model import BitAxonModel


class PerfectModel(nn.Module):
    """Mock model that predicts the same token as the current input (identity prediction).

    Use with constant token sequences (e.g., all zeros) where next token == current token.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def __call__(self, input_ids: mx.array):
        batch, seq_len = input_ids.shape
        logits = mx.full((batch, seq_len, self.vocab_size), -100.0)
        input_ids_int = input_ids.astype(mx.int32)
        for b in range(batch):
            for t in range(seq_len):
                idx = input_ids_int[b, t].item()
                logits[b, t, idx] = 100.0
        return logits, []


class RandomModel(nn.Module):
    """Mock model that returns random logits."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def __call__(self, input_ids: mx.array):
        batch, seq_len = input_ids.shape
        logits = mx.random.normal((batch, seq_len, self.vocab_size))
        return logits, []


def test_perplexity_perfect_model():
    vocab_size = 1024
    model = PerfectModel(vocab_size)
    token_ids = mx.zeros((1, 32), dtype=mx.uint32)
    ppl, se = compute_perplexity(model, token_ids)
    assert ppl < 1.5, f"Perfect model PPL should be near 1.0, got {ppl}"
    assert se >= 0


def test_perplexity_random_logits():
    vocab_size = 1024
    model = RandomModel(vocab_size)
    token_ids = mx.random.randint(0, vocab_size, shape=(1, 64), dtype=mx.uint32)
    ppl, se = compute_perplexity(model, token_ids)
    assert ppl > 100, f"Random logits PPL should be high, got {ppl}"
    assert math.isfinite(ppl)
    assert math.isfinite(se)


def test_perplexity_shape_handling():
    vocab_size = 1024
    model = RandomModel(vocab_size)

    ids_1 = mx.random.randint(0, vocab_size, shape=(1, 32), dtype=mx.uint32)
    ppl_1, _ = compute_perplexity(model, ids_1)
    assert math.isfinite(ppl_1)

    ids_2 = mx.random.randint(0, vocab_size, shape=(2, 16), dtype=mx.uint32)
    ppl_2, _ = compute_perplexity(model, ids_2)
    assert math.isfinite(ppl_2)


def test_perplexity_finite():
    vocab_size = 1024
    model = RandomModel(vocab_size)
    token_ids = mx.random.randint(0, vocab_size, shape=(2, 32), dtype=mx.uint32)
    ppl, se = compute_perplexity(model, token_ids)
    assert math.isfinite(ppl), f"PPL is not finite: {ppl}"
    assert math.isfinite(se), f"SE is not finite: {se}"
    assert ppl > 0


def test_perplexity_basic(small_config: BitAxonConfig):
    model = BitAxonModel(small_config)
    token_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 16), dtype=mx.uint32)
    ppl, se = compute_perplexity(model, token_ids)
    assert math.isfinite(ppl), f"PPL is not finite: {ppl}"
    assert math.isfinite(se), f"SE is not finite: {se}"
    assert ppl > 0
