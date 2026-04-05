"""WikiText dataset loader for perplexity evaluation."""

import mlx.core as mx

_FALLBACK_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning is a branch of artificial intelligence that focuses on building systems "
    "that learn from data. Deep learning uses neural networks with many layers to model complex patterns. "
    "Natural language processing enables computers to understand and generate human language. "
    "Transformers have become the dominant architecture for language models since their introduction. "
    "Attention mechanisms allow models to focus on relevant parts of the input sequence. "
    "State space models offer an alternative to attention with linear complexity. "
    "Quantization reduces the precision of model weights to decrease memory usage and inference time. "
    "Apple Silicon provides unified memory architecture that benefits machine learning workloads. "
    "The Bit-Axon model combines state space models with mixture of experts for efficient inference. "
    "Perplexity measures how well a language model predicts a sample of text. "
    "Lower perplexity indicates better predictive performance on the evaluation dataset. "
    "Cross entropy loss is the negative log probability assigned to the correct next token. "
    "Training language models requires large datasets and significant computational resources. "
    "Fine-tuning adapts a pretrained model to specific tasks or domains with additional training. "
    "Evaluation benchmarks help compare different models and track improvements over time. "
)


class WikiTextDataset:
    """Loads WikiText-103-raw test split for perplexity evaluation.

    Tries HuggingFace datasets library first. Falls back to built-in sample text if unavailable.
    Tokenizes at character level and splits into fixed-length chunks.
    """

    def __init__(self, split: str = "test", seq_length: int = 2048, max_tokens: int = 100_000):
        self.seq_length = seq_length
        raw_text = self._load_text(split)
        token_ids = self._tokenize(raw_text, max_tokens)
        self.chunks = self._chunk(token_ids, seq_length)

    def _load_text(self, split: str) -> str:
        try:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            return "\n".join(ds["text"])
        except (ImportError, OSError):
            return _FALLBACK_TEXT * 100

    def _tokenize(self, text: str, max_tokens: int) -> list[int]:
        token_ids = [ord(c) % 256 for c in text]
        return token_ids[:max_tokens]

    def _chunk(self, token_ids: list[int], seq_length: int) -> list[mx.array]:
        n_chunks = len(token_ids) // seq_length
        if n_chunks == 0:
            return [mx.array(token_ids)]
        chunks = []
        for i in range(n_chunks):
            start = i * seq_length
            chunk = mx.array(token_ids[start : start + seq_length])
            chunks.append(chunk)
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> mx.array:
        return self.chunks[idx]
