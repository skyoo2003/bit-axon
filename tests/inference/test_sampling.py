import mlx.core as mx

from bit_axon.inference.sampling import sample_logits


class TestSampleLogits:
    def test_greedy_returns_argmax(self):
        logits = mx.array([[1.0, 5.0, 2.0, 0.5]])
        result = sample_logits(logits, temperature=0.0)
        assert result.item() == 1

    def test_temperature_scaled(self):
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        mx.random.seed(42)
        result = sample_logits(logits, temperature=1.0)
        assert 0 <= result.item() < 4

    def test_top_k_limits_candidates(self):
        logits = mx.array([[10.0, 9.0, 1.0, 0.0, -5.0]])
        for _ in range(20):
            result = sample_logits(logits, top_k=2, temperature=1.0)
            assert result.item() in [0, 1]

    def test_top_p_limits_candidates(self):
        logits = mx.array([[10.0, 5.0, 1.0, 0.1, 0.0]])
        mx.random.seed(42)
        result = sample_logits(logits, top_p=0.5)
        assert result.item() in [0, 1]

    def test_combined_top_k_top_p(self):
        logits = mx.array([[10.0, 8.0, 5.0, 1.0, 0.0, -1.0, -2.0, -3.0]])
        for _ in range(20):
            result = sample_logits(logits, temperature=0.8, top_k=3, top_p=0.9)
            assert result.item() in [0, 1, 2]

    def test_single_logit(self):
        logits = mx.array([[3.0]])
        result = sample_logits(logits, temperature=0.0)
        assert result.item() == 0

    def test_all_equal_logits(self):
        logits = mx.array([[1.0, 1.0, 1.0, 1.0]])
        mx.random.seed(42)
        result = sample_logits(logits, temperature=1.0)
        assert 0 <= result.item() < 4

    def test_very_low_temperature(self):
        logits = mx.array([[1.0, 2.0, 3.0]])
        for _ in range(10):
            result = sample_logits(logits, temperature=0.01)
            assert result.item() == 2  # nearly greedy

    def test_batch_sampling(self):
        logits = mx.array([[10.0, 1.0], [1.0, 10.0]])
        results = sample_logits(logits, temperature=0.0)
        assert results.shape == (2,)
        assert results[0].item() == 0
        assert results[1].item() == 1

    def test_seed_reproducibility(self):
        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        mx.random.seed(123)
        r1 = sample_logits(logits, temperature=1.0).item()
        mx.random.seed(123)
        r2 = sample_logits(logits, temperature=1.0).item()
        assert r1 == r2

    def test_2d_logits_shape(self):
        """Test that 2D logits (batch, vocab) work correctly."""
        logits = mx.array([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]])
        results = sample_logits(logits, temperature=0.0)
        assert results.shape == (2,)
        assert results[0].item() == 1
        assert results[1].item() == 2
