import mlx.core as mx
import pytest

from bit_axon.training.loss import cross_entropy_loss


class TestCrossEntropyLoss:
    def test_perfect_prediction(self):
        V = 10
        logits = mx.zeros((1, V)) - 100.0
        logits = logits.at[0, 3].add(200.0)
        labels = mx.array([3])
        loss, ntoks = cross_entropy_loss(logits, labels)
        mx.eval(loss, ntoks)
        assert float(loss) < 0.01
        assert int(ntoks) == 1

    def test_random_prediction(self):
        mx.random.seed(42)
        V = 16
        logits = mx.random.normal(shape=(1, V))
        labels = mx.array([5])
        loss, ntoks = cross_entropy_loss(logits, labels)
        mx.eval(loss, ntoks)
        assert float(loss) > 0.0

    def test_ignore_index_masking(self):
        mx.random.seed(42)
        V = 16
        T = 8
        logits = mx.random.normal(shape=(1, T, V))
        labels = mx.array([0, 1, 2, -100, -100, 5, 6, -100])

        loss_all, ntoks_all = cross_entropy_loss(logits, labels)
        valid_logits = logits[:, [0, 1, 2, 5, 6], :]
        valid_labels = mx.array([0, 1, 2, 5, 6])
        loss_valid, ntoks_valid = cross_entropy_loss(valid_logits, valid_labels)
        mx.eval(loss_all, loss_valid, ntoks_all, ntoks_valid)
        assert float(mx.abs(loss_all - loss_valid)) < 1e-5
        assert int(ntoks_all) == 5

    def test_all_ignored(self):
        V = 16
        logits = mx.random.normal(shape=(1, 4, V))
        labels = mx.full((1, 4), -100)
        loss, ntoks = cross_entropy_loss(logits, labels)
        mx.eval(loss, ntoks)
        assert float(loss) == 0.0
        assert int(ntoks) == 0

    def test_2d_input(self):
        mx.random.seed(42)
        B, T, V = 2, 4, 16
        logits = mx.random.normal(shape=(B, T, V))
        labels = mx.array([[1, 2, -100, 4], [5, -100, 7, 8]])
        loss, ntoks = cross_entropy_loss(logits, labels)
        mx.eval(loss, ntoks)
        assert float(loss) > 0.0
        assert int(ntoks) == 6

    def test_1d_input(self):
        mx.random.seed(42)
        T, V = 8, 16
        logits = mx.random.normal(shape=(T, V))
        labels = mx.array([1, 2, -100, 4, 5, -100, 7, 8])
        loss, ntoks = cross_entropy_loss(logits, labels)
        mx.eval(loss, ntoks)
        assert float(loss) > 0.0
        assert int(ntoks) == 6

    def test_gradient_flow(self):
        mx.random.seed(42)
        V = 16
        logits = mx.random.normal(shape=(1, V))
        labels = mx.array([3])

        def fn(l):
            return cross_entropy_loss(l, labels)[0]

        grads = mx.grad(fn)(logits)
        mx.eval(grads)
        assert grads.shape == logits.shape
        assert mx.all(mx.isfinite(grads))
