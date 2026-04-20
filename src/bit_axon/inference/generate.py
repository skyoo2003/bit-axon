"""Autoregressive text generation for Bit-Axon."""

from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass

import mlx.core as mx

from bit_axon.inference.sampling import sample_logits


@dataclass
class GenerateConfig:
    """Configuration for autoregressive text generation.

    Attributes:
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Higher values increase randomness.
        top_k: Number of top logits to keep during sampling. 0 disables filtering.
        top_p: Nucleus sampling probability threshold. 1.0 disables filtering.
        repetition_penalty: Penalty for repeated tokens. 1.0 disables penalty.
        seed: Optional random seed for reproducible generation.
    """

    max_tokens: int = 512
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    seed: int | None = None
    # Optional list of substrings that, if they appear in the decoded
    # output, halt generation (sync path only). Used by the benchmark
    # harness to stop GSM8K on few-shot delimiters like "\n\nQ:" and keep
    # greedy extraction from scooping up a hallucinated next question.
    stop_strings: list[str] | None = None


@dataclass
class GenerateResult:
    """Result of text generation.

    Attributes:
        text: Decoded output text.
        token_ids: Generated token IDs (excluding prompt).
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated.
        tokens_per_sec: Generation throughput in tokens per second.
        time_to_first_token_ms: Time from prefill start to first sampled token, in ms.
    """

    text: str
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float
    time_to_first_token_ms: float | None = None


def _encode_prompt(tokenizer, prompt: str, chat: bool, messages: list[dict[str, str]] | None) -> list[int]:
    if messages is not None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if chat:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return tokenizer.encode(prompt)


def _prefill(model, token_ids: list[int]) -> tuple[mx.array, list, float, float]:
    input_ids = mx.array([token_ids], dtype=mx.uint32)
    mx.synchronize()
    t_start = time.perf_counter()
    try:
        logits, caches = model(input_ids)
        mx.synchronize()
    except Exception as exc:
        max_id = max(token_ids) if token_ids else 0
        raise RuntimeError(
            f"Model forward pass failed during prefill. "
            f"Prompt tokens: {len(token_ids)}, max token ID: {max_id}. "
            f"Check that the model's vocab_size covers the tokenizer's range. "
            f"Original error: {exc}"
        ) from exc
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
    """Run autoregressive text generation.

    Prefills the model with the prompt, then generates tokens one at a time
    until max_tokens is reached or an EOS token is sampled.

    Args:
        model: BitAxonModel instance.
        tokenizer: Tokenizer with encode/decode and apply_chat_template methods.
        prompt: Input text prompt.
        config: Generation parameters. Defaults to GenerateConfig().
        stream: If True, yields partial text strings and returns GenerateResult.
        chat: If True, applies chat template to prompt.
        messages: Chat messages for apply_chat_template. Overrides prompt/chat.

    Returns:
        GenerateResult with generated text and stats, or a generator that
        yields decoded text strings and returns GenerateResult.
    """
    cfg = config or GenerateConfig()

    token_ids = _encode_prompt(tokenizer, prompt, chat, messages)
    prompt_tokens = len(token_ids)
    logits, caches, t_start, ttft_ms = _prefill(model, token_ids)

    if stream:
        return _generate_stream(model, tokenizer, logits, caches, cfg, prompt_tokens, t_start, ttft_ms)

    return _generate_sync(model, tokenizer, logits, caches, cfg, prompt_tokens, t_start, ttft_ms)


def _generate_tokens(model, logits, caches, cfg, eos_id) -> Generator[int, None, None]:
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

        yield tok_id

        next_input = mx.array([[tok_id]], dtype=mx.uint32)
        logits, caches = model(next_input, cache=caches)


def _generate_sync(model, tokenizer, logits, caches, cfg, prompt_tokens, t_start, ttft_ms) -> GenerateResult:
    eos_id = tokenizer.eos_token_id
    generated_ids: list[int] = []
    stop_strings = cfg.stop_strings or []

    for tok_id in _generate_tokens(model, logits, caches, cfg, eos_id):
        generated_ids.append(tok_id)
        if stop_strings:
            # Decoding every token is O(n²) in the worst case but
            # completions here are capped (≤512), and the benchmark
            # wall-time win from early-stopping dominates. If this
            # becomes a bottleneck, buffer and decode every K tokens.
            partial = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if any(s in partial for s in stop_strings):
                break

    mx.synchronize()
    elapsed = time.perf_counter() - t_start
    completion_tokens = len(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""
    if stop_strings and text:
        earliest = len(text)
        for s in stop_strings:
            idx = text.find(s)
            if idx != -1 and idx < earliest:
                earliest = idx
        if earliest < len(text):
            text = text[:earliest]

    return GenerateResult(
        text=text,
        token_ids=generated_ids,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_sec=completion_tokens / elapsed if elapsed > 0 else 0.0,
        time_to_first_token_ms=ttft_ms,
    )


def _generate_stream(model, tokenizer, logits, caches, cfg, prompt_tokens, t_start, ttft_ms) -> Generator[str, None, GenerateResult]:
    eos_id = tokenizer.eos_token_id
    generated_ids: list[int] = []

    for tok_id in _generate_tokens(model, logits, caches, cfg, eos_id):
        generated_ids.append(tok_id)
        yield tokenizer.decode([tok_id], skip_special_tokens=True)

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
