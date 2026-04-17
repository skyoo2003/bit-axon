from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer as HFTokenizer


class QwenTokenizerWrapper:
    """Lightweight Qwen2.5 tokenizer wrapper using the tokenizers library.

    Loads a tokenizer.json file (Qwen2.5 format) and provides:
    - encode/decode
    - Qwen2.5 chat template rendering (pure Python, no Jinja)
    - Special token properties
    """

    def __init__(self, path_or_name: str | Path) -> None:
        """Load tokenizer from local file path or HuggingFace Hub repo name.

        - If path is a local file that exists: HFTokenizer.from_file()
        - If path looks like a HuggingFace ID (contains '/'):
          download tokenizer.json via huggingface_hub.hf_hub_download,
          then load with HFTokenizer.from_file()
        """
        path = Path(path_or_name)
        if path.is_file():
            self._tokenizer = HFTokenizer.from_file(str(path))
        elif "/" in str(path_or_name):
            from huggingface_hub import hf_hub_download

            tokenizer_path = hf_hub_download(repo_id=str(path_or_name), filename="tokenizer.json")
            self._tokenizer = HFTokenizer.from_file(tokenizer_path)
        else:
            msg = f"Cannot load tokenizer from '{path_or_name}': not a local file and not a HuggingFace repo ID"
            raise FileNotFoundError(msg)

        self._vocab = self._tokenizer.get_vocab(with_added_tokens=True)
        self._path_or_name = str(path_or_name)

    def encode(self, text: str) -> list[int]:
        """Encode text to list of token IDs."""
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids: list[int] | mx.array, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text. Accepts list or mx.array."""
        if isinstance(token_ids, mx.array):
            raw = token_ids.tolist()
            ids = raw if isinstance(raw, list) else [raw]
        else:
            ids = token_ids
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> list[int]:
        """Apply Qwen2.5 chat template to messages.

        Template: <|im_start|>{role}\\n{content}<|im_end|>\\n
        If add_generation_prompt=True, appends: <|im_start|>assistant\\n

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            add_generation_prompt: Whether to append assistant prompt

        Returns: list of token IDs
        """
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")

        chat_text = "".join(parts)
        return self.encode(chat_text)

    @property
    def pad_token_id(self) -> int:
        """Return the pad token ID (endoftext, 151643 for Qwen2.5)."""
        token_id = self._vocab.get("<｜end▁of▁text｜>")  # noqa: RUF001
        if token_id is None:
            token_id = self._vocab.get("", 0)
        return token_id

    @property
    def eos_token_id(self) -> int:
        """Return the end-of-sequence token ID (im_end, 151645 for Qwen2.5)."""
        token_id = self._vocab.get("<|im_end|>")
        if token_id is None:
            token_id = self._vocab.get("</s>", 1)
        return token_id

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size including added tokens."""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)
