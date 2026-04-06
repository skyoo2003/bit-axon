"""Tests for autoregressive text generation."""

from __future__ import annotations

import mlx.core as mx
import pytest

from bit_axon.config import BitAxonConfig
from bit_axon.inference.generate import GenerateConfig, GenerateResult, generate
from bit_axon.model import BitAxonModel

SMALL_CONFIG = BitAxonConfig(
    hidden_dim=256,
    num_layers=4,
    num_heads=4,
    d_source_model=128,
    vocab_size=1024,
    ssm_d_state=4,
    ssm_d_conv=2,
    ssm_expand=2,
    swa_window_size=64,
    moe_num_experts=4,
    moe_top_k=2,
    moe_intermediate_dim=512,
)


class MockTokenizer:
    """Minimal tokenizer mock that satisfies the generate() interface."""

    def __init__(self, vocab_size: int = 1024):
        self._vocab_size = vocab_size
        self._eos_id = vocab_size - 1

    def encode(self, text: str) -> list[int]:
        # Simple byte-level encoding clamped to vocab range
        return [min(ord(c), self._vocab_size - 2) for c in text] or [0]

    def decode(self, token_ids: list[int] | mx.array, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, mx.array):
            raw = token_ids.tolist()
            ids = raw if isinstance(raw, list) else [raw]
        else:
            ids = token_ids
        return "".join(chr(int(i)) if 32 <= int(i) < 127 else "?" for i in ids)

    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> list[int]:
        parts: list[str] = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        return self.encode(text)

    @property
    def eos_token_id(self) -> int:
        return self._eos_id


@pytest.fixture
def model():
    m = BitAxonModel(SMALL_CONFIG)
    mx.eval(m.parameters())
    return m


@pytest.fixture
def tokenizer():
    return MockTokenizer(vocab_size=SMALL_CONFIG.vocab_size)


class TestGenerateConfig:
    def test_generate_config_defaults(self):
        cfg = GenerateConfig()
        assert cfg.max_tokens == 512
        assert cfg.temperature == 0.6
        assert cfg.top_k == 50
        assert cfg.top_p == 0.95
        assert cfg.repetition_penalty == 1.0
        assert cfg.seed is None


class TestGenerate:
    def test_generate_returns_string(self, model, tokenizer):
        result = generate(model, tokenizer, "hello", config=GenerateConfig(max_tokens=5))
        assert isinstance(result, GenerateResult)
        assert isinstance(result.text, str)

    def test_generate_respects_max_tokens(self, model, tokenizer):
        max_tokens = 3
        result = generate(model, tokenizer, "test", config=GenerateConfig(max_tokens=max_tokens))
        assert result.completion_tokens <= max_tokens
        assert len(result.token_ids) <= max_tokens

    def test_generate_result_has_metrics(self, model, tokenizer):
        result = generate(model, tokenizer, "hello world", config=GenerateConfig(max_tokens=5))
        assert isinstance(result, GenerateResult)
        assert isinstance(result.completion_tokens, int)
        assert isinstance(result.tokens_per_sec, float)
        assert result.tokens_per_sec >= 0
        assert isinstance(result.prompt_tokens, int)
        assert result.prompt_tokens > 0

    def test_generate_empty_prompt(self, model, tokenizer):
        # Empty string encodes to [0] in mock, should not crash
        result = generate(model, tokenizer, "", config=GenerateConfig(max_tokens=3))
        assert isinstance(result, GenerateResult)

    def test_generate_with_chat_messages(self, model, tokenizer):
        messages = [{"role": "user", "content": "hi"}]
        result = generate(model, tokenizer, "", config=GenerateConfig(max_tokens=3), messages=messages)
        assert isinstance(result, GenerateResult)
        # With messages, prompt_tokens should be longer than just encoding ""
        assert result.prompt_tokens > 1

    def test_generate_chat_flag(self, model, tokenizer):
        result = generate(model, tokenizer, "hello", config=GenerateConfig(max_tokens=3), chat=True)
        assert isinstance(result, GenerateResult)
        # chat=True wraps in template, so prompt is longer than raw encode
        raw_prompt_len = len(tokenizer.encode("hello"))
        assert result.prompt_tokens > raw_prompt_len

    def test_generate_default_config(self, model, tokenizer):
        # Should use default GenerateConfig when None is passed
        result = generate(model, tokenizer, "test")
        assert isinstance(result, GenerateResult)

    def test_generate_prompt_tokens_count(self, model, tokenizer):
        prompt = "hello"
        expected_prompt_tokens = len(tokenizer.encode(prompt))
        result = generate(model, tokenizer, prompt, config=GenerateConfig(max_tokens=3))
        assert result.prompt_tokens == expected_prompt_tokens

    def test_generate_time_to_first_token(self, model, tokenizer):
        result = generate(model, tokenizer, "hello", config=GenerateConfig(max_tokens=3))
        assert result.time_to_first_token_ms is not None
        assert result.time_to_first_token_ms > 0
