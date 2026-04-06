"""Batch collation for ORPO preference optimization."""

from __future__ import annotations

import random
from collections.abc import Iterator

import mlx.core as mx

IGNORE_INDEX: int = -100


def pad_to_length(ids: list[int], length: int, pad_value: int = 0) -> list[int]:
    """Right-pad a sequence to target length.

    If the sequence is already >= length, truncate from the right.

    Args:
        ids: Token IDs to pad.
        length: Target length.
        pad_value: Padding value (default: 0).

    Returns:
        Padded or truncated list of exactly `length` elements.
    """
    if len(ids) >= length:
        return ids[:length]
    return ids + [pad_value] * (length - len(ids))


def create_labels(ids: list[int], mask: list[int], pad_length: int) -> list[int]:
    """Create labels with IGNORE_INDEX for prompt/pad tokens, actual IDs for response tokens.

    Args:
        ids: Token IDs.
        mask: Binary mask where 0 = prompt, 1 = response.
        pad_length: Target length after padding.

    Returns:
        Labels list of length `pad_length` with -100 at non-response positions.
    """
    labels = []
    for i in range(pad_length):
        if i < len(ids) and i < len(mask) and mask[i] == 1:
            labels.append(ids[i])
        else:
            labels.append(IGNORE_INDEX)
    return labels


def collate_orpo_batch(
    batch: list[tuple[list[int], list[int], list[int], list[int]]],
    max_seq_len: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Collate a list of 4-tuples into padded batch arrays.

    Each item in batch is: (chosen_ids, chosen_mask, rejected_ids, rejected_mask).
    Pairs are padded to the maximum pair length in the batch (capped at max_seq_len).

    Args:
        batch: List of (chosen_ids, chosen_mask, rejected_ids, rejected_mask) tuples.
        max_seq_len: Maximum sequence length (truncation cap).

    Returns:
        (chosen_ids, chosen_labels, rejected_ids, rejected_labels) as mx.array tensors
        of shape (B, batch_length).
    """
    item_lengths = []
    for chosen_ids, _cm, rejected_ids, _rm in batch:
        pair_length = min(max(len(chosen_ids), len(rejected_ids)), max_seq_len)
        item_lengths.append(pair_length)

    batch_length = max(item_lengths)

    all_chosen_ids: list[list[int]] = []
    all_chosen_labels: list[list[int]] = []
    all_rejected_ids: list[list[int]] = []
    all_rejected_labels: list[list[int]] = []

    for chosen_ids, chosen_mask, rejected_ids, rejected_mask in batch:
        padded_chosen = pad_to_length(chosen_ids, batch_length)
        chosen_labels = create_labels(chosen_ids, chosen_mask, batch_length)

        padded_rejected = pad_to_length(rejected_ids, batch_length)
        rejected_labels = create_labels(rejected_ids, rejected_mask, batch_length)

        all_chosen_ids.append(padded_chosen)
        all_chosen_labels.append(chosen_labels)
        all_rejected_ids.append(padded_rejected)
        all_rejected_labels.append(rejected_labels)

    return (
        mx.array(all_chosen_ids, dtype=mx.int32),
        mx.array(all_chosen_labels, dtype=mx.int32),
        mx.array(all_rejected_ids, dtype=mx.int32),
        mx.array(all_rejected_labels, dtype=mx.int32),
    )


def iterate_orpo_batches(
    dataset,
    batch_size: int = 1,
    max_seq_len: int = 2048,
    shuffle: bool = True,
    loop: bool = True,
    seed: int | None = None,
) -> Iterator[tuple[mx.array, mx.array, mx.array, mx.array]]:
    """Iterate over a dataset, yielding ORPO preference optimization batches.

    Follows the same iteration pattern as ``iterate_batches`` in collate.py but
    without sequence packing — each chosen/rejected pair is independent.

    Args:
        dataset: Dataset yielding (chosen_ids, chosen_mask, rejected_ids, rejected_mask) tuples.
        batch_size: Number of preference pairs per batch (default: 1).
        max_seq_len: Maximum sequence length (default: 2048).
        shuffle: If True, shuffle examples each epoch (default: True).
        loop: If True, loop infinitely (default: True).
        seed: Random seed for reproducibility (default: None).

    Yields:
        (chosen_ids, chosen_labels, rejected_ids, rejected_labels) tuples as mx.array.
    """
    while True:
        indices = list(range(len(dataset)))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_items = [dataset[i] for i in batch_indices]
            yield collate_orpo_batch(batch_items, max_seq_len)

        if not loop:
            break
