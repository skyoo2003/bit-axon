"""Unit tests for benchmark task definitions."""

from unittest.mock import MagicMock, patch

import pytest

from bit_axon.evaluation.tasks import (
    BENCHMARK_REGISTRY,
    ARCChallengeTask,
    ARCEasyTask,
    BenchmarkConfig,
    BenchmarkItem,
    GSM8KTask,
    HellaSwagTask,
    MMLUTask,
    WinoGrandeTask,
)

# ---------------------------------------------------------------------------
# Mock dataset helpers
# ---------------------------------------------------------------------------


def _make_ds(rows: list[dict[str, object]]) -> MagicMock:
    ds = MagicMock()
    ds.__len__ = lambda self: len(rows)
    ds.__getitem__ = lambda self, idx: rows[idx]
    ds.__iter__ = lambda self: iter(rows)
    return ds


def _mock_mmlu_data():
    return {
        "test": [{"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}],
        "dev": [
            {"question": "What is 1+1?", "choices": ["1", "2", "3", "4"], "answer": 1},
            {"question": "What is 3+3?", "choices": ["5", "6", "7", "8"], "answer": 1},
            {"question": "What is 0+0?", "choices": ["0", "1", "2", "3"], "answer": 0},
            {"question": "What is 4+4?", "choices": ["7", "8", "9", "10"], "answer": 1},
            {"question": "What is 5+5?", "choices": ["9", "10", "11", "12"], "answer": 1},
        ],
    }


def _mock_gsm8k_data():
    return {
        "test": [
            {"question": "Natalia sold 48 clips in April and half in May. Total?", "answer": "She sold 48/2=24 in May.\n48+24=72\n#### 72"},
        ],
        "train": [
            {"question": "Janet has 5 eggs.", "answer": "5 eggs.\n#### 5"},
            {"question": "Tom has 3 apples.", "answer": "3 apples.\n#### 3"},
            {"question": "Mary read 10 pages.", "answer": "10 pages.\n#### 10"},
            {"question": "Bob ran 4 miles.", "answer": "4 miles.\n#### 4"},
            {"question": "Sue baked 6 pies.", "answer": "6 pies.\n#### 6"},
            {"question": "Jim drove 20 miles.", "answer": "20 miles.\n#### 20"},
            {"question": "Ann drew 8 shapes.", "answer": "8 shapes.\n#### 8"},
            {"question": "Roy lifted 15 lbs.", "answer": "15 lbs.\n#### 15"},
        ],
    }


def _mock_arc_data():
    return {
        "test": [{"question": "Which is a gas?", "choices": {"text": ["oxygen", "water", "iron", "salt"], "label": ["A", "B", "C", "D"]}, "answerKey": "A"}],
        "train": [
            {"question": f"Q{i}?", "choices": {"text": [f"a{i}", f"b{i}", f"c{i}", f"d{i}"], "label": ["A", "B", "C", "D"]}, "answerKey": "A"}
            for i in range(30)
        ],
    }


def _mock_hellaswag_data():
    return [
        {
            "ctx_a": "The man",
            "ctx_b": "picked up the ball",
            "endings": ["and threw it", "and dropped it", "and ate it", "and sat on it"],
            "label": 0,
            "activity_label": "Playing sports",
        }
    ]


def _mock_winogrande_data():
    return [{"sentence": "The trophy didn't fit because _ was too large.", "option1": "the trophy", "option2": "the suitcase", "answer": "1"}]


def _patch_load_dataset(side_effect_fn):
    return patch("bit_axon.evaluation.tasks._require_datasets", return_value=MagicMock(load_dataset=side_effect_fn))


def _patch_load_with_retry(side_effect_fn):
    return patch("bit_axon.evaluation.tasks._load_with_retry", side_effect=side_effect_fn)


# ---------------------------------------------------------------------------
# BenchmarkConfig tests
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    def test_default_benchmarks(self):
        cfg = BenchmarkConfig()
        assert len(cfg.benchmarks) == 6

    def test_custom_limit(self):
        cfg = BenchmarkConfig(limit=100)
        assert cfg.limit == 100


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_tasks_registered(self):
        assert set(BENCHMARK_REGISTRY.keys()) == {"mmlu", "gsm8k", "arc_challenge", "arc_easy", "hellaswag", "winogrande"}

    def test_registry_instantiation(self):
        for name, cls in BENCHMARK_REGISTRY.items():
            task = cls()
            assert task.name == name


# ---------------------------------------------------------------------------
# MMLU tests
# ---------------------------------------------------------------------------


class TestMMLUTask:
    def _load(self, limit=None):
        data = _mock_mmlu_data()

        def load_fn(path, config=None, split=None):
            assert split is not None
            return _make_ds(data[split])

        with _patch_load_dataset(load_fn) as _:
            return MMLUTask().load_data(limit=limit)

    def test_load_data_returns_items(self):
        items = self._load()
        assert len(items) > 0
        assert all(isinstance(i, BenchmarkItem) for i in items)

    def test_answer_is_letter(self):
        items = self._load()
        for item in items:
            assert item.answer in {"A", "B", "C", "D"}

    def test_category_is_subject(self):
        items = self._load()
        assert items[0].category == "abstract_algebra"

    def test_limit(self):
        items = self._load(limit=1)
        assert len(items) == 1

    def test_prompt_includes_choices(self):
        items = self._load()
        assert "A. " in items[0].prompt
        assert "D. " in items[0].prompt

    def test_prompt_has_few_shot_but_no_final_answer(self):
        items = self._load()
        prompt = items[0].prompt
        assert prompt.count("Answer:") >= 5
        lines = prompt.strip().split("\n")
        assert lines[-1] != "Answer:"

    def test_extract_answer_standard(self):
        task = MMLUTask()
        assert task.extract_answer("B") == "B"
        assert task.extract_answer("A\nbecause...") == "A"

    def test_extract_answer_with_whitespace(self):
        task = MMLUTask()
        assert task.extract_answer(" B\n") == "B"

    def test_extract_answer_empty(self):
        task = MMLUTask()
        assert task.extract_answer("") == ""

    def test_check_answer_case_insensitive(self):
        task = MMLUTask()
        assert task.check_answer("a", "A") is True
        assert task.check_answer("B", "b") is True
        assert task.check_answer("A", "B") is False

    def test_import_error(self):
        with patch("bit_axon.evaluation.tasks._require_datasets", side_effect=ImportError("no datasets")), pytest.raises(ImportError):
            MMLUTask().load_data()


# ---------------------------------------------------------------------------
# GSM8K tests
# ---------------------------------------------------------------------------


class TestGSM8KTask:
    def _load(self, limit=None):
        data = _mock_gsm8k_data()

        def load_fn(path, config=None, split=None):
            assert split is not None
            return _make_ds(data[split])

        with _patch_load_dataset(load_fn) as _:
            return GSM8KTask().load_data(limit=limit)

    def test_load_data_returns_items(self):
        items = self._load()
        assert len(items) > 0
        assert isinstance(items[0], BenchmarkItem)

    def test_answer_is_numeric(self):
        items = self._load()
        assert items[0].answer == "72"

    def test_limit(self):
        items = self._load(limit=1)
        assert len(items) == 1

    def test_prompt_has_few_shot(self):
        items = self._load()
        assert "Q: " in items[0].prompt
        assert items[0].prompt.count("Q: ") >= 9

    def test_prompt_no_final_answer(self):
        items = self._load()
        assert not items[0].prompt.rstrip().endswith("#### 72")

    def test_extract_answer_the_answer_is(self):
        task = GSM8KTask()
        assert task.extract_answer("The answer is 42.") == "42"

    def test_extract_answer_negative(self):
        task = GSM8KTask()
        assert task.extract_answer("The answer is -3.") == "-3"

    def test_extract_answer_with_commas(self):
        task = GSM8KTask()
        assert task.extract_answer("The answer is 1,234.") == "1234"

    def test_extract_answer_fallback_last_number(self):
        task = GSM8KTask()
        result = task.extract_answer("We computed 10 + 20 = 30\nSo 30")
        assert result == "30"

    def test_extract_answer_empty(self):
        task = GSM8KTask()
        assert task.extract_answer("") == ""

    def test_check_answer_normalizes_commas(self):
        task = GSM8KTask()
        assert task.check_answer("1234", "1,234") is True

    def test_check_answer_wrong(self):
        task = GSM8KTask()
        assert task.check_answer("42", "72") is False


# ---------------------------------------------------------------------------
# ARC tests
# ---------------------------------------------------------------------------


class TestARCChallengeTask:
    def _load(self, limit=None):
        data = _mock_arc_data()

        def load_fn(path, config=None, split=None):
            return _make_ds(data[split])

        with _patch_load_dataset(load_fn) as _:
            return ARCChallengeTask().load_data(limit=limit)

    def test_load_data(self):
        items = self._load()
        assert len(items) > 0
        assert items[0].answer == "A"

    def test_limit(self):
        assert len(self._load(limit=1)) == 1

    def test_prompt_has_few_shot(self):
        items = self._load()
        assert items[0].prompt.count("Answer:") >= 25

    def test_extract_answer(self):
        task = ARCChallengeTask()
        assert task.extract_answer("B") == "B"
        assert task.extract_answer(" C\n") == "C"
        assert task.extract_answer("") == ""

    def test_check_answer(self):
        task = ARCChallengeTask()
        assert task.check_answer("a", "A") is True
        assert task.check_answer("A", "B") is False


class TestARCEasyTask:
    def _load(self, limit=None):
        data = _mock_arc_data()

        def load_fn(path, config=None, split=None):
            return _make_ds(data[split])

        with _patch_load_dataset(load_fn) as _:
            return ARCEasyTask().load_data(limit=limit)

    def test_load_data(self):
        items = self._load()
        assert len(items) > 0

    def test_name(self):
        assert ARCEasyTask().name == "arc_easy"


# ---------------------------------------------------------------------------
# HellaSwag tests
# ---------------------------------------------------------------------------


class TestHellaSwagTask:
    def _load(self, limit=None):
        data = _mock_hellaswag_data()

        def load_fn(path, config=None, split=None):
            return _make_ds(data)

        with _patch_load_dataset(load_fn) as _:
            return HellaSwagTask().load_data(limit=limit)

    def test_load_data(self):
        items = self._load()
        assert len(items) > 0
        assert items[0].answer == "A"

    def test_category(self):
        items = self._load()
        assert items[0].category == "Playing sports"

    def test_limit(self):
        assert len(self._load(limit=1)) == 1

    def test_prompt_format(self):
        items = self._load()
        assert "Playing sports:" in items[0].prompt
        assert "A. " in items[0].prompt
        assert "Answer:" in items[0].prompt

    def test_extract_answer(self):
        task = HellaSwagTask()
        assert task.extract_answer("B") == "B"
        assert task.extract_answer(" D\nexplanation") == "D"
        assert task.extract_answer("") == ""

    def test_check_answer(self):
        task = HellaSwagTask()
        assert task.check_answer("a", "A") is True
        assert task.check_answer("A", "B") is False


# ---------------------------------------------------------------------------
# WinoGrande tests
# ---------------------------------------------------------------------------


class TestWinoGrandeTask:
    def _load(self, limit=None):
        data = _mock_winogrande_data()

        def load_fn(path, config=None, split=None):
            return _make_ds(data)

        with _patch_load_dataset(load_fn) as _:
            return WinoGrandeTask().load_data(limit=limit)

    def test_load_data(self):
        items = self._load()
        assert len(items) > 0
        assert items[0].answer == "1"

    def test_prompt_format(self):
        items = self._load()
        assert "1. the trophy" in items[0].prompt
        assert "2. the suitcase" in items[0].prompt
        assert items[0].prompt.endswith("Answer:")

    def test_limit(self):
        assert len(self._load(limit=1)) == 1

    def test_extract_answer(self):
        task = WinoGrandeTask()
        assert task.extract_answer("1") == "1"
        assert task.extract_answer(" 2\n") == "2"
        assert task.extract_answer("") == ""
        assert task.extract_answer("X") == ""

    def test_check_answer(self):
        task = WinoGrandeTask()
        assert task.check_answer("1", "1") is True
        assert task.check_answer("2", "1") is False


# ---------------------------------------------------------------------------
# Retry and load_with_retry tests
# ---------------------------------------------------------------------------


class TestLoadWithRetry:
    def test_succeeds_on_first_try(self):
        from bit_axon.evaluation.tasks import _load_with_retry

        mock_fn = MagicMock(return_value="ok")
        result = _load_with_retry(mock_fn, "a", "b")
        assert result == "ok"
        assert mock_fn.call_count == 1

    def test_retries_on_429(self):
        from bit_axon.evaluation.tasks import _load_with_retry

        mock_fn = MagicMock(side_effect=[Exception("429 Too Many Requests"), Exception("429 Too Many Requests"), "ok"])
        with patch("bit_axon.evaluation.tasks.time.sleep") as mock_sleep:
            result = _load_with_retry(mock_fn, "a", max_retries=5, base_delay=0.01)
        assert result == "ok"
        assert mock_fn.call_count == 3
        assert mock_sleep.call_count == 2

    def test_raises_after_max_retries(self):
        from bit_axon.evaluation.tasks import _load_with_retry

        mock_fn = MagicMock(side_effect=Exception("429 Too Many Requests"))
        with pytest.raises(Exception, match="429"):
            _load_with_retry(mock_fn, "a", max_retries=3, base_delay=0.01)

    def test_raises_non_retryable_immediately(self):
        from bit_axon.evaluation.tasks import _load_with_retry

        mock_fn = MagicMock(side_effect=FileNotFoundError("not found"))
        with pytest.raises(FileNotFoundError):
            _load_with_retry(mock_fn, "a")

    def test_retries_on_generic_rate_limit_error(self):
        from bit_axon.evaluation.tasks import _load_with_retry

        mock_fn = MagicMock(side_effect=[Exception("Rate limit exceeded"), "ok"])
        with patch("bit_axon.evaluation.tasks.time.sleep"):
            result = _load_with_retry(mock_fn, "a", max_retries=5, base_delay=0.01)
        assert result == "ok"
        assert mock_fn.call_count == 2


class TestMMLUEarlyTermination:
    def test_limit_stops_before_loading_all_subjects(self):
        data = _mock_mmlu_data()
        call_count = 0

        def load_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_ds(data[kwargs["split"]])

        with _patch_load_with_retry(load_fn):
            items = MMLUTask().load_data(limit=1)
        assert len(items) == 1
        assert call_count <= 2

    def test_limit_zero_loads_all(self):
        data = _mock_mmlu_data()

        def load_fn(*args, **kwargs):
            return _make_ds(data[kwargs["split"]])

        with _patch_load_with_retry(load_fn):
            items = MMLUTask().load_data(limit=None)
        assert len(items) > 0
