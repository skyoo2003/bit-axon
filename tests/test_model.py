import mlx.core as mx
import mlx.utils

from bit_axon.config import BitAxonConfig
from bit_axon.layers.block import AxonSSMBlock, AxonSSMMoEBlock, AxonSWAMoEBlock
from bit_axon.model import BitAxonModel


def test_forward_pass_shape(small_config):
    model = BitAxonModel(small_config)
    input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 128))
    logits, caches = model(input_ids)
    assert logits.shape == (1, 128, small_config.vocab_size)


def test_forward_pass_no_nan(small_config):
    model = BitAxonModel(small_config)
    input_ids = mx.random.randint(0, small_config.vocab_size, shape=(1, 128))
    logits, _ = model(input_ids)
    assert mx.all(mx.isfinite(logits))


def test_weight_tying(small_config):
    model = BitAxonModel(small_config)
    assert model.embed_tokens.weight is model.lm_head.weight


def test_layer_types(small_config):
    model = BitAxonModel(small_config)
    # small_config: num_layers=4, third=1
    # layer_0: ssm, layer_1: swa_moe, layer_2-3: ssm_moe
    assert isinstance(getattr(model, "layer_0"), AxonSSMBlock)
    assert isinstance(getattr(model, "layer_1"), AxonSWAMoEBlock)
    assert isinstance(getattr(model, "layer_2"), AxonSSMMoEBlock)
    assert isinstance(getattr(model, "layer_3"), AxonSSMMoEBlock)


def test_parameter_count(small_config):
    model = BitAxonModel(small_config)
    flat = mlx.utils.tree_flatten(model.parameters())
    leaves = [v for _, v in flat]
    total = sum(p.size for p in leaves)
    assert total > 0


def test_no_duplicate_params(small_config):
    model = BitAxonModel(small_config)
    flat = mlx.utils.tree_flatten(model.parameters())
    leaves = [v for _, v in flat]
    param_ids = set()
    for p in leaves:
        param_ids.add(id(p))
    # Weight tying shares embed_tokens.weight with lm_head.weight
    if small_config.weight_tying:
        assert len(param_ids) == len(leaves) - 1
    else:
        assert len(param_ids) == len(leaves)


def test_forward_pass_default_config():
    config = BitAxonConfig(num_layers=3)
    model = BitAxonModel(config)
    input_ids = mx.random.randint(0, config.vocab_size, shape=(1, 4))
    logits, caches = model(input_ids)
    assert logits.shape == (1, 4, config.vocab_size)


def test_cache_creation(small_config):
    model = BitAxonModel(small_config)
    caches = model._create_caches()
    assert len(caches) == small_config.num_layers
    # layer 0: ssm -> None
    assert caches[0] is None
    # layer 1: swa_moe -> KVCache
    assert isinstance(caches[1], type(caches[1]))
    assert hasattr(caches[1], "update_and_fetch")
    # layers 2-3: ssm_moe -> None
    assert caches[2] is None
    assert caches[3] is None
