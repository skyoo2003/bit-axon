# Training

::: bit_axon.training
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
        - TrainingConfig
        - Trainer
        - ORPOTrainer
        - apply_lora_to_model
        - LoRALinear
        - DoRALinear
        - SFTDataset
        - ORPODataset
        - AlpacaDataset
        - CoolingScheduler

## Checkpointing

::: bit_axon.training.checkpoint.save_checkpoint
    handler: python
    options:
      show_source: false

::: bit_axon.training.checkpoint.load_checkpoint
    handler: python
    options:
      show_source: false

::: bit_axon.training.checkpoint.get_latest_checkpoint
    handler: python
    options:
      show_source: false

::: bit_axon.training.checkpoint.save_adapter_only
    handler: python
    options:
      show_source: false

## Adapter Merging & Export

::: bit_axon.training.merging.merge_adapters
    handler: python
    options:
      show_source: false

::: bit_axon.training.merging.dequantize_model
    handler: python
    options:
      show_source: false

::: bit_axon.training.merging.quantize_model
    handler: python
    options:
      show_source: false

::: bit_axon.training.merging.save_merged_model
    handler: python
    options:
      show_source: false

::: bit_axon.training.merging.load_and_merge
    handler: python
    options:
      show_source: false

## Learning Rate Schedule

::: bit_axon.training.scheduler.build_lr_schedule
    handler: python
    options:
      show_source: false

## Data Pipeline

::: bit_axon.training.collate.iterate_batches
    handler: python
    options:
      show_source: false

::: bit_axon.training.collate.BatchCollator
    handler: python
    options:
      show_source: false

::: bit_axon.training.packing.SequencePacker
    handler: python
    options:
      show_source: false

::: bit_axon.training.packing.PackedBatch
    handler: python
    options:
      show_source: false
