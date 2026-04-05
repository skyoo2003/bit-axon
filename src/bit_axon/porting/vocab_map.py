"""Tokenizer vocabulary mapping for Qwen2.5-3B → Bit-Axon 32K truncation."""

from __future__ import annotations

from collections import Counter

from tokenizers import Tokenizer, models


def build_vocab_mapping(
    tokenizer_name: str = "Qwen/Qwen2.5-3B",
    target_size: int = 32000,
    corpus_text: str | None = None,
) -> dict[int, int]:
    """Build a mapping from Qwen token IDs to truncated Bit-Axon token IDs.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier.
        target_size: Target vocabulary size for Bit-Axon.
        corpus_text: Optional text corpus for frequency-based selection.
            When None, the first `target_size` tokens in BPE order are used.

    Returns:
        Dictionary mapping old Qwen token IDs to new Bit-Axon token IDs.
    """
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.get_vocab_size()

    if target_size > vocab_size:
        msg = f"target_size ({target_size}) exceeds tokenizer vocab size ({vocab_size})"
        raise ValueError(msg)

    if corpus_text is not None:
        encoding = tokenizer.encode(corpus_text)
        freq = Counter(encoding.ids)
        # Sort by frequency descending, then by original ID for stable ordering
        sorted_ids = sorted(freq.keys(), key=lambda t: (-freq[t], t))
        selected_ids = sorted_ids[:target_size]
    else:
        # First-N strategy: BPE merge order ≈ frequency order
        selected_ids = list(range(target_size))

    return {old_id: new_id for new_id, old_id in enumerate(selected_ids)}


def load_truncated_tokenizer(
    tokenizer_name: str,
    vocab_mapping: dict[int, int],
) -> Tokenizer:
    """Load a tokenizer with only the mapped vocabulary.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier.
        vocab_mapping: Mapping from old token IDs to new IDs.

    Returns:
        Tokenizer with truncated vocabulary containing only mapped tokens.
    """
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    vocab = tokenizer.get_vocab()

    # Build reverse lookup: old_id -> token string
    id_to_token = {v: k for k, v in vocab.items()}

    # Build new vocab with remapped IDs
    new_vocab = {}
    for old_id, new_id in vocab_mapping.items():
        token_str = id_to_token.get(old_id)
        if token_str is not None:
            new_vocab[token_str] = new_id

    new_tokenizer = Tokenizer(models.BPE(vocab=new_vocab, merges=[]))
    return new_tokenizer
