from __future__ import annotations

from collections.abc import Iterator, Sequence

import mlx.core as mx
import numpy as np

from bit_axon.training.packing import PackedBatch


class BatchCollator:
    """Collates PackedBatch objects into mx.array training batches.

    Applies shift-by-one for next-token prediction:
        input_ids = token_ids[:, :-1]
        labels    = token_ids[:, 1:]  with -100 at ignored positions
    """

    def __init__(
        self,
        batch_size: int = 1,
        max_seq_len: int = 2048,
        ignore_index: int = -100,
    ) -> None:
        """Args:
        batch_size: Number of sequences per batch.
        max_seq_len: Sequence length (after packing).
        ignore_index: Label value for ignored positions (default: -100).
        """
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

    def collate(self, batches: Sequence[PackedBatch]) -> tuple[mx.array, mx.array]:
        """Convert a sequence of PackedBatch into (input_ids, labels) mx.array pair.

        Args:
            batches: Sequence of PackedBatch objects (length = batch_size).

        Returns:
            input_ids: mx.array of shape (B, T-1) with dtype mx.int32.
            labels: mx.array of shape (B, T-1) with dtype mx.int32.
                    Positions where loss_mask=0 are set to ignore_index.
        """
        stacked_ids = np.array([b.token_ids for b in batches], dtype=np.int32)
        stacked_mask = np.array([b.loss_mask for b in batches], dtype=np.int32)

        input_ids = stacked_ids[:, :-1]
        shifted_ids = stacked_ids[:, 1:]
        shifted_mask = stacked_mask[:, 1:]

        labels = np.where(shifted_mask == 1, shifted_ids, self.ignore_index)

        return mx.array(input_ids), mx.array(labels)


def iterate_batches(
    dataset,
    batch_size: int = 1,
    max_seq_len: int = 2048,
    shuffle: bool = True,
    loop: bool = False,
    seed: int | None = None,
    eos_token_id: int = 151645,
) -> Iterator[tuple[mx.array, mx.array]]:
    """Iterate over a dataset, yielding (input_ids, labels) training batches.

    This function orchestrates the full data pipeline:
    1. Optionally shuffle dataset indices
    2. Iterate through examples
    3. Pack examples into fixed-length sequences via SequencePacker
    4. Collate packed batches into mx.array tensors via BatchCollator

    Args:
        dataset: A dataset yielding (token_ids, loss_mask) tuples.
                 Can be SFTDataset, AlpacaDataset, or CacheDataset.
        batch_size: Number of sequences per batch (default: 1).
        max_seq_len: Fixed sequence length for packing (default: 2048).
        shuffle: If True, shuffle examples each epoch (default: True).
        loop: If True, loop infinitely (default: False).
        seed: Random seed for reproducibility (default: None).
        eos_token_id: EOS token ID for sequence packing (default: 151645 for Qwen2.5).

    Yields:
        (input_ids, labels) tuples where:
        - input_ids: mx.array of shape (B, T-1), dtype mx.int32
        - labels: mx.array of shape (B, T-1), dtype mx.int32, -100 for ignored positions
    """
    from bit_axon.training.packing import SequencePacker

    collator = BatchCollator(batch_size=batch_size, max_seq_len=max_seq_len)

    while True:
        indices = list(range(len(dataset)))
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)

        packer = SequencePacker(max_seq_len=max_seq_len, eos_token_id=eos_token_id)
        batch_buffer: list[PackedBatch] = []

        for idx in indices:
            token_ids, loss_mask = dataset[idx]

            for pb in packer.add_example(token_ids, loss_mask):
                batch_buffer.append(pb)
                if len(batch_buffer) == batch_size:
                    yield collator.collate(batch_buffer)
                    batch_buffer = []

        # Flush remaining
        final = packer.flush()
        if final is not None:
            batch_buffer.append(final)
            if len(batch_buffer) == batch_size:
                yield collator.collate(batch_buffer)
                batch_buffer = []

        # Handle leftover partial batch
        if batch_buffer and not loop:
            while len(batch_buffer) < batch_size:
                batch_buffer.append(PackedBatch(token_ids=[0] * max_seq_len, loss_mask=[0] * max_seq_len))
            yield collator.collate(batch_buffer)

        if not loop:
            break
