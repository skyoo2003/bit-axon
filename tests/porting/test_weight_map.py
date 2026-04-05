from bit_axon.config import BitAxonConfig
from bit_axon.porting.weight_map import build_key_mappings, enumerate_target_keys


def test_enumerate_target_keys_count(small_config: BitAxonConfig):
    keys = enumerate_target_keys(small_config)
    # small_config: 4 layers (1 SSM + 1 SWA+MoE + 2 SSM+MoE), weight_tying=True
    # Top: 4, Layer 0 (SSM): 10, Layer 1 (SWA+MoE): 14, Layer 2 (SSM+MoE): 19, Layer 3 (SSM+MoE): 19
    assert len(keys) == 66


def test_enumerate_target_keys_no_duplicates(small_config: BitAxonConfig):
    keys = enumerate_target_keys(small_config)
    assert len(keys) == len(set(keys))


def test_enumerate_target_keys_includes_top_level(small_config: BitAxonConfig):
    keys = enumerate_target_keys(small_config)
    for expected in ("embed_tokens.weight", "input_proj.weight", "output_proj.weight", "lm_head.weight"):
        assert expected in keys


def test_build_key_mappings_portable_count(small_config: BitAxonConfig):
    mappings = build_key_mappings(small_config)
    portable = [m for m in mappings if m.source_key is not None]
    # embed_tokens, lm_head = 2
    # Layer 0 (SSM): input_norm = 1
    # Layer 1 (SWA+MoE): input_norm, post_attention_norm, shared_expert {gate,up,down} = 5
    # Layer 2 (SSM+MoE): input_norm, post_ssm_norm, shared_expert {gate,up,down} = 5
    # Layer 3 (SSM+MoE): input_norm, post_ssm_norm, shared_expert {gate,up,down} = 5
    # Total: 2 + 1 + 5 + 5 + 5 = 18
    assert len(portable) == 18


def test_build_key_mappings_ssm_layers_no_source(small_config: BitAxonConfig):
    mappings = build_key_mappings(small_config)
    ssm_params = [m for m in mappings if m.target_key.startswith("layer_0.ssm.") and m.target_key != "layer_0.input_norm.weight"]
    assert len(ssm_params) > 0
    for m in ssm_params:
        assert m.source_key is None
        assert m.transform == "default"


def test_build_key_mappings_swa_layers_no_attn_source(small_config: BitAxonConfig):
    mappings = build_key_mappings(small_config)
    attn_params = [m for m in mappings if m.target_key.startswith("layer_1.attention.")]
    assert len(attn_params) == 4  # q, k, v, o projections
    for m in attn_params:
        assert m.source_key is None
        assert m.transform == "default"


def test_build_key_mappings_moe_shared_has_source(small_config: BitAxonConfig):
    mappings = build_key_mappings(small_config)
    shared_expert_params = [
        m for m in mappings if "shared_expert." in m.target_key and "shared_expert_gate" not in m.target_key and m.target_key.startswith("layer_")
    ]
    assert len(shared_expert_params) >= 6  # 3 per MoE layer, at least 2 MoE layers
    for m in shared_expert_params:
        assert m.source_key is not None
        assert m.transform == "moe_project"
        assert "model.layers." in m.source_key
        assert ".mlp." in m.source_key


def test_build_key_mappings_moe_routed_is_copy_perturb(small_config: BitAxonConfig):
    mappings = build_key_mappings(small_config)
    switch_params = [m for m in mappings if "switch_mlp." in m.target_key]
    assert len(switch_params) >= 6  # 3 per MoE layer, at least 2 MoE layers
    for m in switch_params:
        assert m.source_key is None
        assert m.transform == "copy_perturb"
