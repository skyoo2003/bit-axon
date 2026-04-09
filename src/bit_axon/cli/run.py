from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

from bit_axon.cli._console import print_error, print_info, print_success

console = Console()


def run_inference(
    prompt: str | None = None,
    model: str = "skyoo2003/bit-axon",
    tokenizer: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: int | None = None,
    config_small: bool = False,
    chat: bool = False,
    no_stream: bool = False,
) -> None:
    from bit_axon.config import BitAxonConfig
    from bit_axon.inference.generate import GenerateConfig
    from bit_axon.model import BitAxonModel

    tokenizer_path = tokenizer or model

    if config_small:
        config = BitAxonConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            d_source_model=128,
            vocab_size=1024,
        )
        with console.status("[bold green]Initializing small model...", spinner="dots"):
            model_obj = BitAxonModel(config)
            import mlx.core as mx

            mx.eval(model_obj.parameters())
        tok = _MockTokenizer()
        print_success(f"Small model ready (hidden_dim={config.hidden_dim}, layers={config.num_layers})")
    else:
        from bit_axon.inference.loader import load_model
        from bit_axon.tokenizer import QwenTokenizerWrapper

        with console.status(f"[bold green]Loading model from {model}...", spinner="dots"):
            model_obj = load_model(Path(model), quantize=True)
        with console.status(f"[bold green]Loading tokenizer from {tokenizer_path}...", spinner="dots"):
            tok = QwenTokenizerWrapper(tokenizer_path)
        print_success(f"Model loaded: {model}")

    gen_config = GenerateConfig(
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    if chat:
        _chat_loop(model_obj, tok, gen_config, no_stream)
    elif prompt:
        _single_prompt(model_obj, tok, prompt, gen_config, no_stream)
    else:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
            if prompt:
                _single_prompt(model_obj, tok, prompt, gen_config, no_stream)
            else:
                print_error("No input provided. Use --chat for interactive mode.")
                raise typer.Exit(1)
        else:
            print_error("No prompt provided. Use --chat for interactive mode or provide a prompt.")
            raise typer.Exit(1)


def _single_prompt(model, tokenizer, prompt: str, config, no_stream: bool) -> None:
    from bit_axon.inference.generate import GenerateResult, generate

    result = generate(model, tokenizer, prompt, config=config, stream=False)
    if not isinstance(result, GenerateResult):
        raise TypeError(f"Expected GenerateResult, got {type(result).__name__}")
    console.print(result.text)
    console.print(
        f"\n[dim]─── {result.completion_tokens} tokens · {result.tokens_per_sec:.1f} tok/s"
        + (f" · TTFT {result.time_to_first_token_ms:.0f}ms" if result.time_to_first_token_ms else "")
        + " ───[/dim]"
    )


def _chat_loop(model, tokenizer, config, no_stream: bool) -> None:
    print_info("Chat mode. Type 'exit' or Ctrl+C to quit.\n")

    messages: list[dict[str, str]] = []

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        messages.append({"role": "user", "content": user_input})

        from bit_axon.inference.generate import GenerateResult, generate

        result = generate(model, tokenizer, "", config=config, stream=False, messages=messages)
        if not isinstance(result, GenerateResult):
            raise TypeError(f"Expected GenerateResult, got {type(result).__name__}")

        console.print(f"[bold green]Assistant:[/bold green] {result.text}")
        console.print(f"[dim]─── {result.completion_tokens} tokens · {result.tokens_per_sec:.1f} tok/s ───[/dim]\n")

        messages.append({"role": "assistant", "content": result.text})


class _MockTokenizer:
    @property
    def eos_token_id(self) -> int:
        return 1

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text[:64]] if text else [0]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(t % 256) for t in token_ids)

    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> list[int]:
        parts = []
        for msg in messages:
            parts.extend(self.encode(msg["content"]))
        if add_generation_prompt:
            parts.append(0)
        return parts
