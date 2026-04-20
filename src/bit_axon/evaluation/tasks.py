"""LLM benchmark task definitions for evaluation.

Provides prompt templates, data loading, answer extraction, and scoring
for standard benchmarks: MMLU, GSM8K, ARC-Challenge, ARC-Easy, HellaSwag, WinoGrande.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bit_axon.utils import _require_datasets

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkItem:
    """A single benchmark evaluation item."""

    id: str
    prompt: str
    answer: str
    category: str = ""


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    benchmarks: list[str] = field(default_factory=lambda: ["mmlu", "gsm8k", "arc_challenge", "arc_easy", "hellaswag", "winogrande"])
    limit: int | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int | None = None
    num_few_shot: int | None = None
    scoring_method: str = "generate"  # "generate" or "logprob"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

_LETTERS = ["A", "B", "C", "D"]


def _load_with_retry(load_fn, *args: Any, max_retries: int = 5, base_delay: float = 2.0, **kwargs: Any) -> Any:
    """Call a HuggingFace load function with exponential backoff on 429 rate limits."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return load_fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "429" in err_str or "too many requests" in err_str or "rate limit" in err_str:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                import sys

                print(
                    f"[yellow]HF Hub rate limit hit (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {delay:.1f}s... Set HF_TOKEN for higher limits.[/yellow]",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                raise
    raise last_err or RuntimeError("Unexpected retry loop exit without raising")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BenchmarkTask:
    """Base class for benchmark evaluation tasks."""

    name: str = ""
    default_few_shot: int = 0

    def load_data(
        self, limit: int | None = None, *, num_few_shot: int | None = None, status_callback: Callable[[str], None] | None = None
    ) -> list[BenchmarkItem]:
        msg = "Subclasses must implement load_data"
        raise NotImplementedError(msg)

    def format_prompt(self, item: BenchmarkItem) -> str:
        msg = "Subclasses must implement format_prompt"
        raise NotImplementedError(msg)

    def extract_answer(self, response: str) -> str:
        # Base models rarely emit a bare choice letter at column 0; they tend
        # to prefix with "Answer:", "The answer is ", " A.", "(A)" etc. Scan
        # the full response for the first standalone A/B/C/D instead of the
        # stricter "^[A-D]" on the first line, which produced sub-random
        # accuracy on MMLU/ARC baselines.
        m = re.search(r"\b([A-D])\b", response)
        return m.group(1) if m else ""

    def check_answer(self, predicted: str, expected: str) -> bool:
        return predicted.strip().upper() == expected.strip().upper()

    def score_by_logprobs(self, model, tokenizer, item: BenchmarkItem) -> str:
        """Score multiple-choice item by log-probability of each option.

        Uses KV cache: processes prompt once, then incremental forward for each choice.
        Returns the predicted answer letter/string.
        """
        import mlx.core as mx

        prompt = item.prompt
        choices = _LETTERS[:4]
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            return choices[0]
        prompt_arr = mx.array([prompt_ids], dtype=mx.uint32)
        logits, caches = model(prompt_arr)

        logprobs: list[float] = []
        for choice in choices:
            choice_ids = tokenizer.encode(f" {choice}")
            if not choice_ids:
                logprobs.append(-float("inf"))
                continue
            for cid in choice_ids:
                token_arr = mx.array([[cid]], dtype=mx.uint32)
                logits, caches = model(token_arr, cache=caches)
            logits_f = logits.astype(mx.float32)
            last_logits = logits_f[0, -1, :]
            log_probs = mx.log_softmax(last_logits)
            logprobs.append(float(log_probs[0]))
        best_idx = int(max(range(len(logprobs)), key=lambda i: logprobs[i]))
        return choices[best_idx]


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------


class MMLUTask(BenchmarkTask):
    """MMLU: Massive Multitask Language Understanding (57 subjects, 5-shot)."""

    name = "mmlu"
    default_few_shot = 5

    def __init__(self) -> None:
        self._few_shot_cache: dict[str, list[dict[str, Any]]] = {}

    def _get_few_shot(self, subject: str, dev_data: list[dict[str, Any]], n: int = 5) -> list[dict[str, Any]]:
        if subject not in self._few_shot_cache:
            rng = random.Random(42)
            n_avail = min(n, len(dev_data))
            self._few_shot_cache[subject] = rng.sample(dev_data, n_avail)
        return self._few_shot_cache[subject]

    @staticmethod
    def _format_mc_question(question: str, choices: list[str], answer_letter: str | None = None) -> str:
        lines = [question]
        for i, choice in enumerate(choices):
            lines.append(f"{_LETTERS[i]}. {choice}")
        if answer_letter is not None:
            lines.append(f"Answer: {answer_letter}")
        return "\n".join(lines)

    def load_data(
        self, limit: int | None = None, *, num_few_shot: int | None = None, status_callback: Callable[[str], None] | None = None
    ) -> list[BenchmarkItem]:
        datasets = _require_datasets()
        items: list[BenchmarkItem] = []
        n_shot = num_few_shot if num_few_shot is not None else self.default_few_shot
        for i, subject in enumerate(_MMLU_SUBJECTS):
            if limit is not None and len(items) >= limit:
                if status_callback is not None:
                    status_callback(f"MMLU: loaded {len(items)} items (limit reached)")
                break
            if status_callback is not None:
                status_callback(f"Loading MMLU: {subject} ({i + 1}/{len(_MMLU_SUBJECTS)})...")
            test_ds = _load_with_retry(datasets.load_dataset, "cais/mmlu", subject, split="test")
            dev_ds = _load_with_retry(datasets.load_dataset, "cais/mmlu", subject, split="dev")
            dev_rows = [dev_ds[i] for i in range(len(dev_ds))]
            few_shot = self._get_few_shot(subject, dev_rows, n=n_shot)
            for idx in range(len(test_ds)):
                row = test_ds[idx]
                answer_letter = _LETTERS[row["answer"]]
                shot_lines = [f"The following are multiple choice questions (with answers) about {subject}.\n"]
                for ex in few_shot:
                    shot_lines.append(self._format_mc_question(ex["question"], ex["choices"], _LETTERS[ex["answer"]]))
                    shot_lines.append("")
                shot_lines.append(self._format_mc_question(row["question"], row["choices"]))
                prompt = "\n".join(shot_lines)
                items.append(BenchmarkItem(id=f"mmlu_{subject}_{idx}", prompt=prompt, answer=answer_letter, category=subject))
                if limit is not None and len(items) >= limit:
                    return items
        return items

    def format_prompt(self, item: BenchmarkItem) -> str:
        return item.prompt


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


class GSM8KTask(BenchmarkTask):
    """GSM8K: Grade School Math 8K (8-shot chain-of-thought)."""

    name = "gsm8k"
    default_few_shot = 8

    def __init__(self) -> None:
        self._few_shot: list[dict[str, Any]] | None = None

    def _get_few_shot(self, train_data: list[dict[str, Any]], n: int = 8) -> list[dict[str, Any]]:
        if self._few_shot is None:
            rng = random.Random(42)
            n_avail = min(n, len(train_data))
            self._few_shot = rng.sample(train_data, n_avail)
        return self._few_shot

    def load_data(
        self, limit: int | None = None, *, num_few_shot: int | None = None, status_callback: Callable[[str], None] | None = None
    ) -> list[BenchmarkItem]:
        datasets = _require_datasets()
        if status_callback is not None:
            status_callback(f"Loading {self.name}...")
        test_ds = _load_with_retry(datasets.load_dataset, "openai/gsm8k", "main", split="test")
        train_ds = _load_with_retry(datasets.load_dataset, "openai/gsm8k", "main", split="train")
        train_rows = [train_ds[i] for i in range(len(train_ds))]
        n_shot = num_few_shot if num_few_shot is not None else self.default_few_shot
        few_shot = self._get_few_shot(train_rows, n=n_shot)

        items: list[BenchmarkItem] = []
        for idx in range(len(test_ds)):
            row = test_ds[idx]
            answer_text = row["answer"]
            numeric_match = re.search(r"####\s*(-?[0-9\.,]+)", answer_text)
            numeric_answer = numeric_match.group(1).replace(",", "").strip() if numeric_match else ""

            # Build prompt with few-shot examples
            parts: list[str] = []
            for ex in few_shot:
                parts.append(f"Q: {ex['question']}\n\n  A: {ex['answer']}\n\n")
            parts.append(f"Q: {row['question']}\n\n  A:")
            prompt = "".join(parts)

            items.append(BenchmarkItem(id=f"gsm8k_{idx}", prompt=prompt, answer=numeric_answer))
            if limit is not None and len(items) >= limit:
                return items
        return items

    def format_prompt(self, item: BenchmarkItem) -> str:
        return item.prompt

    def extract_answer(self, response: str) -> str:
        # Stage 0: canonical GSM8K format "#### <num>". The few-shot
        # exemplars all end with this sentinel, so a properly-primed base
        # model emits it too. Preferred over any fallback.
        m0 = re.search(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", response)
        if m0:
            return m0.group(1).replace(",", "")
        # Stage 1: "The answer is X"
        m = re.search(r"The answer is (-?[0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)", response)
        if m:
            return m.group(1).replace(",", "")
        # Stage 2: last number-like pattern
        matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", response)
        if matches:
            last = matches[-1]
            candidate = last[0] or last[1]
            return candidate.replace(",", "").replace("$", "").strip()
        return ""

    def _normalize_numeric(self, s: str) -> str:
        return s.replace(",", "").replace("$", "").strip()

    def check_answer(self, predicted: str, expected: str) -> bool:
        return self._normalize_numeric(predicted) == self._normalize_numeric(expected)


# ---------------------------------------------------------------------------
# ARC (Challenge + Easy)
# ---------------------------------------------------------------------------


class _ARCTask(BenchmarkTask):
    """Shared base for ARC-Challenge and ARC-Easy."""

    name = ""
    _config: str = ""
    default_few_shot = 25

    def __init__(self) -> None:
        self._few_shot: list[dict[str, Any]] | None = None

    def _get_few_shot(self, train_data: list[dict[str, Any]], n: int = 25) -> list[dict[str, Any]]:
        if self._few_shot is None:
            rng = random.Random(42)
            n_avail = min(n, len(train_data))
            self._few_shot = rng.sample(train_data, n_avail)
        return self._few_shot

    @staticmethod
    def _format_arc_question(question: str, choices_text: list[str], answer_key: str | None = None) -> str:
        lines = [question]
        for i, text in enumerate(choices_text):
            lines.append(f"{_LETTERS[i]}. {text}")
        if answer_key is not None:
            lines.append(f"Answer: {answer_key}")
        return "\n".join(lines)

    def load_data(
        self, limit: int | None = None, *, num_few_shot: int | None = None, status_callback: Callable[[str], None] | None = None
    ) -> list[BenchmarkItem]:
        datasets = _require_datasets()
        if status_callback is not None:
            status_callback(f"Loading {self.name}...")
        test_ds = _load_with_retry(datasets.load_dataset, "allenai/ai2_arc", self._config, split="test")
        train_ds = _load_with_retry(datasets.load_dataset, "allenai/ai2_arc", self._config, split="train")
        train_rows = [train_ds[i] for i in range(len(train_ds))]
        n_shot = num_few_shot if num_few_shot is not None else self.default_few_shot
        few_shot = self._get_few_shot(train_rows, n=n_shot)

        items: list[BenchmarkItem] = []
        for idx in range(len(test_ds)):
            row = test_ds[idx]
            choices_text = row["choices"]["text"]
            answer_key = row["answerKey"]

            parts: list[str] = []
            for ex in few_shot:
                parts.append(self._format_arc_question(ex["question"], ex["choices"]["text"], ex["answerKey"]))
                parts.append("\n\n")
            parts.append(self._format_arc_question(row["question"], choices_text))
            prompt = "".join(parts)

            items.append(BenchmarkItem(id=f"{self.name}_{idx}", prompt=prompt, answer=answer_key))
            if limit is not None and len(items) >= limit:
                return items
        return items

    def format_prompt(self, item: BenchmarkItem) -> str:
        return item.prompt


class ARCChallengeTask(_ARCTask):
    """ARC-Challenge: AI2 Reasoning Challenge (25-shot)."""

    name = "arc_challenge"
    _config = "ARC-Challenge"


class ARCEasyTask(_ARCTask):
    """ARC-Easy: AI2 Reasoning Challenge Easy (25-shot)."""

    name = "arc_easy"
    _config = "ARC-Easy"


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------


def _preprocess(text: str) -> str:
    """Preprocess HellaSwag text (from lm-eval-harness)."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


class HellaSwagTask(BenchmarkTask):
    """HellaSwag: Commonsense NLI (0-shot)."""

    name = "hellaswag"

    def load_data(
        self, limit: int | None = None, *, num_few_shot: int | None = None, status_callback: Callable[[str], None] | None = None
    ) -> list[BenchmarkItem]:
        datasets = _require_datasets()
        if status_callback is not None:
            status_callback(f"Loading {self.name}...")
        ds = _load_with_retry(datasets.load_dataset, "Rowan/hellaswag", split="validation")
        items: list[BenchmarkItem] = []
        for idx in range(len(ds)):
            row = ds[idx]
            ctx_a = _preprocess(row["ctx_a"])
            ctx_b = _preprocess(row["ctx_b"])
            endings = row["endings"]
            label = int(row["label"])
            activity_label = row["activity_label"]

            ctx_b_cap = ctx_b[0].upper() + ctx_b[1:] if ctx_b else ""
            prompt_parts = [f"{activity_label}: {ctx_a} {ctx_b_cap}"]
            for i, ending in enumerate(endings):
                prompt_parts.append(f"{_LETTERS[i]}. {_preprocess(ending)}")
            prompt_parts.append("Answer:")
            prompt = "\n".join(prompt_parts)

            answer_letter = _LETTERS[label]
            items.append(BenchmarkItem(id=f"hellaswag_{idx}", prompt=prompt, answer=answer_letter, category=activity_label))
            if limit is not None and len(items) >= limit:
                return items
        return items

    def format_prompt(self, item: BenchmarkItem) -> str:
        return item.prompt


# ---------------------------------------------------------------------------
# WinoGrande
# ---------------------------------------------------------------------------


class WinoGrandeTask(BenchmarkTask):
    """WinoGrande: Winograd Schema Challenge (0-shot)."""

    name = "winogrande"

    def load_data(
        self, limit: int | None = None, *, num_few_shot: int | None = None, status_callback: Callable[[str], None] | None = None
    ) -> list[BenchmarkItem]:
        datasets = _require_datasets()
        if status_callback is not None:
            status_callback(f"Loading {self.name}...")
        ds = _load_with_retry(datasets.load_dataset, "allenai/winogrande", "winogrande_debiased", split="validation")
        items: list[BenchmarkItem] = []
        for idx in range(len(ds)):
            row = ds[idx]
            sentence = row["sentence"]
            option1 = row["option1"]
            option2 = row["option2"]
            answer = row["answer"]

            prompt = f"{sentence}\n1. {option1}\n2. {option2}\nAnswer:"
            items.append(BenchmarkItem(id=f"winogrande_{idx}", prompt=prompt, answer=answer))
            if limit is not None and len(items) >= limit:
                return items
        return items

    def format_prompt(self, item: BenchmarkItem) -> str:
        return item.prompt

    def extract_answer(self, response: str) -> str:
        first_line = response.strip().split("\n")[0].strip()
        m = re.match(r"^[12]", first_line)
        return m.group(0) if m else ""

    def check_answer(self, predicted: str, expected: str) -> bool:
        return predicted.strip() == expected.strip()

    def score_by_logprobs(self, model, tokenizer, item: BenchmarkItem) -> str:
        import mlx.core as mx

        prompt = item.prompt
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            return "1"
        prompt_arr = mx.array([prompt_ids], dtype=mx.uint32)
        logits, caches = model(prompt_arr)

        logprobs: list[float] = []
        for choice in ("1", "2"):
            choice_ids = tokenizer.encode(f" {choice}")
            if not choice_ids:
                logprobs.append(-float("inf"))
                continue
            for cid in choice_ids:
                token_arr = mx.array([[cid]], dtype=mx.uint32)
                logits, caches = model(token_arr, cache=caches)
            logits_f = logits.astype(mx.float32)
            last_logits = logits_f[0, -1, :]
            log_probs = mx.log_softmax(last_logits)
            logprobs.append(float(log_probs[0]))
        return "1" if logprobs[0] >= logprobs[1] else "2"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: dict[str, type[BenchmarkTask]] = {
    cls.name: cls  # type: ignore[attr-defined]
    for cls in [MMLUTask, GSM8KTask, ARCChallengeTask, ARCEasyTask, HellaSwagTask, WinoGrandeTask]
}
