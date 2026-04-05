from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PackedBatch:
    """A packed sequence of token IDs with corresponding loss mask.

    Attributes:
        token_ids: Packed token IDs, length = max_seq_len.
        loss_mask: Binary mask, 1 = compute loss, 0 = ignore (padding/separators).
                   Same length as token_ids.
    """

    token_ids: list[int]
    loss_mask: list[int]


class SequencePacker:
    """Concatenates multiple tokenized examples into fixed-length sequences.

    This maximizes GPU utilization by filling each sequence to max_seq_len
    with multiple examples, inserting EOS tokens between them. Loss masks
    ensure that separator and padding tokens don't contribute to training loss.

    Usage:
        packer = SequencePacker(max_seq_len=2048, eos_token_id=151645)

        # Add examples one by one
        for token_ids, loss_mask in dataset:
            packed_batches = packer.add_example(token_ids, loss_mask)
            for batch in packed_batches:
                # batch is a PackedBatch ready for collation
                process(batch)

        # Don't forget remaining tokens
        final = packer.flush()
        if final is not None:
            process(final)
    """

    def __init__(
        self,
        max_seq_len: int = 2048,
        eos_token_id: int = 151645,
    ) -> None:
        """
        Args:
            max_seq_len: Fixed sequence length for packed output.
            eos_token_id: Token ID to insert between examples as separator.
        """
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self._buffer_ids: list[int] = []
        self._buffer_mask: list[int] = []

    def add_example(
        self,
        token_ids: list[int],
        loss_mask: list[int],
    ) -> list[PackedBatch]:
        """Add a tokenized example to the internal buffer.

        When the buffer reaches max_seq_len, one or more PackedBatch objects
        are yielded and the buffer is trimmed. If the buffer doesn't fill
        completely, the example remains in the buffer for the next call.

        Args:
            token_ids: Token IDs for this example.
            loss_mask: Binary loss mask (1=compute loss, 0=ignore).

        Returns:
            List of PackedBatch objects (may be empty if buffer not full).
        """
        # Truncate oversized examples
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
            loss_mask = loss_mask[: self.max_seq_len]

        # Insert EOS separator if buffer is not empty
        if self._buffer_ids:
            self._buffer_ids.append(self.eos_token_id)
            self._buffer_mask.append(0)

        self._buffer_ids.extend(token_ids)
        self._buffer_mask.extend(loss_mask)

        # Yield complete batches
        results: list[PackedBatch] = []
        while len(self._buffer_ids) >= self.max_seq_len:
            results.append(
                PackedBatch(
                    token_ids=self._buffer_ids[: self.max_seq_len],
                    loss_mask=self._buffer_mask[: self.max_seq_len],
                )
            )
            self._buffer_ids = self._buffer_ids[self.max_seq_len :]
            self._buffer_mask = self._buffer_mask[self.max_seq_len :]

        return results

    def flush(self) -> PackedBatch | None:
        """Flush remaining tokens from the buffer.

        Pads to max_seq_len with pad tokens (ID 0) and sets loss_mask to 0
        for padding positions. Returns None if buffer is empty.
        """
        if not self._buffer_ids:
            return None

        pad_len = self.max_seq_len - len(self._buffer_ids)
        padded_ids = self._buffer_ids + [0] * pad_len
        padded_mask = self._buffer_mask + [0] * pad_len
        return PackedBatch(token_ids=padded_ids, loss_mask=padded_mask)

    def reset(self) -> None:
        """Clear the internal buffer. Useful for starting a new epoch."""
        self._buffer_ids = []
        self._buffer_mask = []
