# Porting

::: bit_axon.porting.vocab_map
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
        - build_vocab_mapping
        - load_truncated_tokenizer

::: bit_axon.porting.pipeline
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
        - initialize_from_qwen_weights
        - save_ported_model
