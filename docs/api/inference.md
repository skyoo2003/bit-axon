# Inference

::: bit_axon.inference
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
        - generate
        - GenerateConfig
        - GenerateResult
        - load_model
        - sample_logits

## Model Loading & Vocab Resize

::: bit_axon.inference.loader.load_model
    handler: python
    options:
      show_source: false

::: bit_axon.inference.loader.resize_model_vocab
    handler: python
    options:
      show_source: false
