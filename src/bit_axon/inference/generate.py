"""Autoregressive text generation for Bit-Axon."""

from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass

import mlx.core as mx

from bit_axon.inference.sampling import sample_logits


@dataclass
class GenerateConfig:
    max_tokens: int = 512
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    seed: int | None = None


@dataclass
class GenerateResult:
    text: str
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float
    time_to_first_token_ms: float | None = None


def _encode_prompt(tokenizer, prompt, chat, messages):
    if messages is not None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if chat:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return tokenizer.encode(prompt)


def _prefill(model, token_ids):
    input_ids = mx.array([token_ids], dtype=mx.uint32)
    mx.synchronize()
    t_start = time.perf_counter()
    logits, caches = model(input_ids)
    mx.synchronize()
    ttft_ms = (time.perf_counter() - t_start) * 1000.0
    return logits, caches, t_start, ttft_ms


def generate(
    model,
    tokenizer,
    prompt: str,
    config: GenerateConfig | None = None,
    stream: bool = False,
    chat: bool = False,
    messages: list[dict[str, str]] | None = None,
) -> GenerateResult | Generator[str, None, GenerateResult]:
    cfg = config or GenerateConfig()

    token_ids = _encode_prompt(tokenizer, prompt, chat, messages)
    prompt_tokens = len(token_ids)
    logits, caches, t_start, ttft_ms = _prefill(model, token_ids)

    if stream:
        return _generate_stream(model, tokenizer, logits, caches, cfg, prompt_tokens, t_start, ttft_ms)

    return _generate_sync(model, tokenizer, logits, caches, cfg, prompt_tokens, t_start, ttft_ms)


def _generate_sync(model, tokenizer, logits, caches, cfg, prompt_tokens, t_start, ttft_ms):
    generated_ids: list[int] = []
    eos_id = tokenizer.eos_token_id

    for _ in range(cfg.max_tokens):
        next_logits = logits[:, -1, :]
        next_token = sample_logits(
            next_logits,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )
        tok_id = int(next_token.item())

        if tok_id == eos_id:
            break

        generated_ids.append(tok_id)
        next_input = mx.array([[tok_id]], dtype=mx.uint32)
        logits, caches = model(next_input, cache=caches)

    mx.synchronize()
    elapsed = time.perf_counter() - t_start
    completion_tokens = len(generated_ids)

    return GenerateResult(
        text=tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else "",
        token_ids=generated_ids,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_sec=completion_tokens / elapsed if elapsed > 0 else 0.0,
        time_to_first_token_ms=ttft_ms,
    )


def _generate_stream(model, tokenizer, logits, caches, cfg, prompt_tokens, t_start, ttft_ms):
    generated_ids: list[int] = []
    eos_id = tokenizer.eos_token_id

    for _ in range(cfg.max_tokens):
        next_logits = logits[:, -1, :]
        next_token = sample_logits(
            next_logits,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )
        tok_id = int(next_token.item())

        if tok_id == eos_id:
            break

        generated_ids.append(tok_id)
        yield tokenizer.decode(generated_ids, skip_special_tokens=True)

        next_input = mx.array([[tok_id]], dtype=mx.uint32)
        logits, caches = model(next_input, cache=caches)

    mx.synchronize()
    elapsed = time.perf_counter() - t_start
    completion_tokens = len(generated_ids)

    return GenerateResult(
        text=tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else "",
        token_ids=generated_ids,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_sec=completion_tokens / elapsed if elapsed > 0 else 0.0,
        time_to_first_token_ms=ttft_ms,
    )
