import mlx.core as mx
import mlx.nn as nn

from bit_axon.layers.swa import SlidingWindowAttention
from bit_axon.utils.cache import ArraysCache, KVCache


def test_swa_output_shape(default_config):
    swa = SlidingWindowAttention(
        hidden_dim=default_config.hidden_dim,
        num_heads=default_config.num_heads,
        window_size=default_config.swa_window_size,
    )
    x = mx.random.normal(shape=(1, 128, default_config.hidden_dim))
    out, _ = swa(x)
    assert out.shape == (1, 128, default_config.hidden_dim)


def test_swa_small_window():
    hidden_dim, num_heads, window_size = 64, 2, 16
    swa = SlidingWindowAttention(hidden_dim, num_heads, window_size)
    x = mx.random.normal(shape=(1, 32, hidden_dim))
    out, _ = swa(x)
    assert out.shape == (1, 32, hidden_dim)

    mask = swa._make_sliding_window_mask(32, 32)
    mask_val = mx.where(mask == 0.0, 1.0, 0.0)
    for i in range(32):
        for j in range(32):
            if j > i or j < i - window_size + 1:
                assert mask_val[i, j].item() == 0.0, f"Position ({i},{j}) should be masked"
            else:
                assert mask_val[i, j].item() == 1.0, f"Position ({i},{j}) should be visible"


def test_kv_cache_update():
    cache = KVCache()
    k1 = mx.random.normal(shape=(1, 4, 8, 16))
    v1 = mx.random.normal(shape=(1, 4, 8, 16))
    k_out, v_out = cache.update_and_fetch(k1, v1)
    assert k_out.shape == (1, 4, 8, 16)
    assert v_out.shape == (1, 4, 8, 16)

    k2 = mx.random.normal(shape=(1, 4, 4, 16))
    v2 = mx.random.normal(shape=(1, 4, 4, 16))
    k_out, v_out = cache.update_and_fetch(k2, v2)
    assert k_out.shape == (1, 4, 12, 16)
    assert v_out.shape == (1, 4, 12, 16)


def test_kv_cache_incremental():
    hidden_dim, num_heads, window_size = 64, 2, 32
    swa = SlidingWindowAttention(hidden_dim, num_heads, window_size)
    cache = KVCache()

    x1 = mx.random.normal(shape=(1, 8, hidden_dim))
    out1, cache = swa(x1, cache=cache)
    mx.eval(out1)
    assert not mx.any(mx.isnan(out1)).item()

    x2 = mx.random.normal(shape=(1, 4, hidden_dim))
    out2, cache = swa(x2, cache=cache)
    mx.eval(out2)
    assert not mx.any(mx.isnan(out2)).item()
    assert out2.shape == (1, 4, hidden_dim)


def test_gradient_flow():
    hidden_dim, num_heads, window_size = 64, 2, 32
    swa = SlidingWindowAttention(hidden_dim, num_heads, window_size)

    x = mx.random.normal(shape=(1, 8, hidden_dim))

    def loss_fn(model):
        out, _ = model(x)
        return out.sum()

    loss, grads = nn.value_and_grad(swa, loss_fn)(swa)
    mx.eval(loss, grads)

    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        g = grads[proj_name]["weight"]
        assert not mx.any(mx.isnan(g)).item(), f"NaN gradient in {proj_name}.weight"


def test_window_mask_shape():
    hidden_dim, num_heads, window_size = 64, 2, 16
    swa = SlidingWindowAttention(hidden_dim, num_heads, window_size)
    mask = swa._make_sliding_window_mask(8, 8)
    assert mask.shape == (8, 8)

    mask_cached = swa._make_sliding_window_mask(4, 12)
    assert mask_cached.shape == (4, 12)


def test_window_mask_values():
    hidden_dim, num_heads, window_size = 64, 2, 8
    swa = SlidingWindowAttention(hidden_dim, num_heads, window_size)
    mask = swa._make_sliding_window_mask(8, 8)
    mask_np = mx.where(mask == 0.0, 1.0, 0.0)

    for i in range(8):
        for j in range(8):
            visible = j <= i and (i - j) < window_size
            if visible:
                assert mask_np[i, j].item() == 1.0
            else:
                assert mask_np[i, j].item() == 0.0


def test_small_config(small_config):
    swa = SlidingWindowAttention(
        hidden_dim=small_config.hidden_dim,
        num_heads=small_config.num_heads,
        window_size=small_config.swa_window_size,
    )
    x = mx.random.normal(shape=(1, 16, small_config.hidden_dim))
    out, _ = swa(x)
    mx.eval(out)
    assert out.shape == (1, 16, small_config.hidden_dim)
    assert not mx.any(mx.isnan(out)).item()


def test_arrays_cache():
    cache = ArraysCache(4)
    assert len(cache.cache) == 4
    assert cache[0] is None

    arr = mx.array([1.0, 2.0])
    cache[0] = arr
    cached = cache[0]
    assert cached is not None
    assert cached.shape == (2,)

    values = [mx.array([i]) for i in range(4)]
    cache.update(values)
    for i in range(4):
        v = cache[i]
        assert v is not None
        assert v.item() == i
