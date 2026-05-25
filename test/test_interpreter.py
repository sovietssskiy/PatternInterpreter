"""
tests/test_interpreter.py
Тесты interpreter.py — без реальных вызовов API (моки).
"""

import json
import os
import sys
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.interpreter import (
    SCHEMA,
    SYSTEM_PROMPT,
    _build_prompt,
    _build_pattern_prompt,
    _call,
    _clusters_for_pattern,
    format_results,
    interpret_all,
)


# ═══════════════════════════════════════════════════════════════════════════
# ФИКСТУРЫ
# ═══════════════════════════════════════════════════════════════════════════

MOCK_RESPONSE = {
    "classification":  "типичный",
    "confidence":      "высокая",
    "description":     "Пользователи просматривают каталог и переходят к товарам.",
    "problems":        ["Отсутствует кнопка возврата в каталог"],
    "recommendations": ["Проверить навигацию со страницы товара обратно в каталог"],
}


def _make_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    seqs = [
        ["/", "/catalog", "/products/{id}", "/cart", "/checkout"],
        ["/", "/catalog", "/products/{id}"],
        ["/", "/search", "/products/{id}", "/cart"],
    ]
    for i in range(n):
        seq = seqs[i % len(seqs)]
        rows.append({
            "session_id":       f"s_{i}",
            "ip_hash":          f"ip_{i % 5}",
            "request_count":    float(rng.integers(2, 10)),
            "duration_sec":     float(rng.integers(30, 300)),
            "unique_pages":     float(len(set(seq))),
            "error_rate":       float(rng.uniform(0, 0.1)),
            "avg_bytes":        2000.0,
            "page_sequence":    seq,
            "page_set":         list(set(seq)),
            "cluster_id":       i % 3,
            "anomaly_flag":     i >= n - 3,
            "anomaly_priority": i >= n - 2,
        })
    return pd.DataFrame(rows)


def _make_patterns() -> list:
    return [
        {
            "pattern":     ["/", "/catalog", "/products/{id}"],
            "pattern_str": "/ → /catalog → /products/{id}",
            "support_abs": 20,
            "support_pct": 66.7,
            "length":      3,
        },
        {
            "pattern":     ["/", "/search", "/products/{id}"],
            "pattern_str": "/ → /search → /products/{id}",
            "support_abs": 10,
            "support_pct": 33.3,
            "length":      3,
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════
# _build_prompt  (кластерный промпт — без изменений)
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPrompt:

    def test_contains_cluster_id(self):
        prompt = _build_prompt(0, {"size": 10}, [], {})
        assert "0" in prompt

    def test_contains_stats(self):
        prompt = _build_prompt(1, {"size": 42, "pct": 30.0}, [], {})
        assert "42" in prompt

    def test_anomaly_flag_high(self):
        prompt = _build_prompt(0, {}, [], {"priority_pct": 25.0})
        assert "АНОМАЛЬНЫХ" in prompt.upper()

    def test_no_anomaly_flag_when_low(self):
        prompt = _build_prompt(0, {}, [], {"priority_pct": 5.0})
        assert "ВЫСОКАЯ ДОЛЯ" not in prompt

    def test_pattern_included(self):
        prompt = _build_prompt(0, {}, _make_patterns(), {})
        assert "/catalog" in prompt

    def test_no_patterns_fallback(self):
        prompt = _build_prompt(0, {}, [], {})
        assert "не обнаружено" in prompt.lower()

    def test_returns_string(self):
        assert isinstance(_build_prompt(0, {}, [], {}), str)


# ═══════════════════════════════════════════════════════════════════════════
# _build_pattern_prompt  (новый промпт для анализа паттерна)
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPatternPrompt:

    @pytest.fixture
    def pattern(self):
        return _make_patterns()[0]

    def test_returns_string(self, pattern):
        assert isinstance(_build_pattern_prompt(pattern, []), str)

    def test_contains_each_step(self, pattern):
        prompt = _build_pattern_prompt(pattern, [])
        for step in pattern["pattern"]:
            assert step in prompt

    def test_contains_support_metrics(self, pattern):
        prompt = _build_pattern_prompt(pattern, [])
        assert "20" in prompt       # support_abs
        assert "66.7" in prompt     # support_pct

    def test_contains_length(self, pattern):
        prompt = _build_pattern_prompt(pattern, [])
        assert "3" in prompt

    def test_step_numbering(self, pattern):
        prompt = _build_pattern_prompt(pattern, [])
        assert "Шаг 1" in prompt
        assert "Шаг 3" in prompt

    def test_cluster_stats_included(self, pattern):
        stats = [{"cluster_id": 0, "size": 10, "med_requests": 5.0, "med_errors": 0.02}]
        prompt = _build_pattern_prompt(pattern, stats)
        assert "Кластер 0" in prompt
        assert "10" in prompt

    def test_no_cluster_stats(self, pattern):
        prompt = _build_pattern_prompt(pattern, [])
        assert "Кластер" not in prompt

    def test_contains_task_instructions(self, pattern):
        prompt = _build_pattern_prompt(pattern, [])
        # Промпт должен явно просить тест-кейсы / UX-анализ
        assert any(kw in prompt for kw in ("тест", "проверить", "шаг", "Задача"))


# ═══════════════════════════════════════════════════════════════════════════
# _clusters_for_pattern
# ═══════════════════════════════════════════════════════════════════════════

class TestClustersForPattern:

    @pytest.fixture
    def df(self):
        return _make_df(30)

    def test_returns_only_matching_clusters(self, df):
        pattern = _make_patterns()[0]   # / → /catalog → /products/{id}
        stats = [
            {"cluster_id": 0, "size": 10, "med_requests": 4.0, "med_errors": 0.0},
            {"cluster_id": 1, "size": 10, "med_requests": 4.0, "med_errors": 0.0},
            {"cluster_id": 2, "size": 10, "med_requests": 4.0, "med_errors": 0.0},
        ]
        result = _clusters_for_pattern(df, pattern, stats)
        ids = {c["cluster_id"] for c in result}
        # кластер 0 содержит seqs[0] и seqs[1], которые включают паттерн
        assert 0 in ids

    def test_empty_when_no_match(self, df):
        pattern = {
            "pattern": ["/nonexistent", "/also-nonexistent"],
            "pattern_str": "/nonexistent → /also-nonexistent",
        }
        result = _clusters_for_pattern(df, pattern, [{"cluster_id": 0, "size": 5,
                                                       "med_requests": 2.0, "med_errors": 0.0}])
        assert result == []

    def test_returns_list_of_dicts(self, df):
        result = _clusters_for_pattern(df, _make_patterns()[0],
                                       [{"cluster_id": 0, "size": 5,
                                         "med_requests": 2.0, "med_errors": 0.0}])
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)


# ═══════════════════════════════════════════════════════════════════════════
# interpret_all
# ═══════════════════════════════════════════════════════════════════════════

class TestInterpretAll:

    @pytest.fixture
    def df(self):
        return _make_df(30)

    @pytest.fixture
    def patterns(self):
        return _make_patterns()

    # ── структура возвращаемого словаря ───────────────────────────────────

    @patch("pipeline.interpreter._call")
    def test_returns_dict_with_clusters_and_patterns(self, mock_call, df, patterns):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all(df=df, patterns=patterns, api_key="sk-test")
        assert "clusters" in result
        assert "patterns" in result

    @patch("pipeline.interpreter._call")
    def test_cluster_keys_are_ints(self, mock_call, df, patterns):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all(df=df, patterns=patterns, api_key="sk-test", max_clusters=3)
        for k in result["clusters"]:
            assert isinstance(k, int)

    @patch("pipeline.interpreter._call")
    def test_patterns_list_length(self, mock_call, df, patterns):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all(df=df, patterns=patterns, api_key="sk-test",
                               max_patterns=1)
        assert len(result["patterns"]) == 1

    @patch("pipeline.interpreter._call")
    def test_max_patterns_zero(self, mock_call, df, patterns):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all(df=df, patterns=patterns, api_key="sk-test",
                               max_patterns=0)
        assert result["patterns"] == []

    # ── содержимое кластерной интерпретации ───────────────────────────────

    @patch("pipeline.interpreter._call")
    def test_cluster_result_has_required_fields(self, mock_call, df, patterns):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all(df=df, patterns=patterns, api_key="sk-test", max_clusters=1)
        cid = next(iter(result["clusters"]))
        for field in ("classification", "confidence", "description",
                      "problems", "recommendations"):
            assert field in result["clusters"][cid]

    # ── содержимое паттерновой интерпретации ──────────────────────────────

    @patch("pipeline.interpreter._call")
    def test_pattern_entry_has_required_fields(self, mock_call, df, patterns):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all(df=df, patterns=patterns, api_key="sk-test",
                               max_clusters=1, max_patterns=2)
        for entry in result["patterns"]:
            for field in ("pattern", "pattern_str", "support_abs",
                          "support_pct", "length", "interpretation"):
                assert field in entry

    @patch("pipeline.interpreter._call")
    def test_pattern_interpretation_has_schema_fields(self, mock_call, df, patterns):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all(df=df, patterns=patterns, api_key="sk-test",
                               max_clusters=1, max_patterns=2)
        for entry in result["patterns"]:
            interp = entry["interpretation"]
            if "error" not in interp:
                for field in ("classification", "confidence", "description",
                              "problems", "recommendations"):
                    assert field in interp

    # ── вызов API ─────────────────────────────────────────────────────────

    @patch("pipeline.interpreter._call")
    def test_call_count_clusters_plus_patterns(self, mock_call, df, patterns):
        mock_call.return_value = MOCK_RESPONSE
        n_clusters, n_patterns = 2, 2
        interpret_all(df=df, patterns=patterns, api_key="sk-test",
                      max_clusters=n_clusters, max_patterns=n_patterns)
        # один вызов на кластер + один на паттерн
        assert mock_call.call_count == n_clusters + n_patterns

    @patch("pipeline.interpreter._call")
    def test_pattern_prompt_differs_from_cluster_prompt(self, mock_call, df, patterns):
        """Промпт для паттерна должен содержать шаги, а не «Кластер N»."""
        captured = []
        def _capture(*args, **kw):
            # _call(prompt, api_key, retries=3)
            captured.append(args[0] if args else kw.get("prompt", ""))
            return MOCK_RESPONSE
        mock_call.side_effect = _capture
        interpret_all(df=df, patterns=patterns, api_key="sk-test",
                      max_clusters=1, max_patterns=1)
        cluster_prompt = captured[0]
        pattern_prompt = captured[1]
        assert "Кластер" in cluster_prompt
        assert "Шаг 1" in pattern_prompt

    # ── обработка ошибок ──────────────────────────────────────────────────

    @patch("pipeline.interpreter._call")
    def test_api_error_in_cluster_handled(self, mock_call, df):
        mock_call.side_effect = RuntimeError("timeout")
        result = interpret_all(df=df, patterns=[], api_key="sk-test", max_clusters=1)
        cid = next(iter(result["clusters"]))
        assert "error" in result["clusters"][cid]

    @patch("pipeline.interpreter._call")
    def test_api_error_in_pattern_handled(self, mock_call, df, patterns):
        mock_call.side_effect = RuntimeError("timeout")
        result = interpret_all(df=df, patterns=patterns, api_key="sk-test",
                               max_clusters=0, max_patterns=1)
        assert "error" in result["patterns"][0]["interpretation"]

    @patch("pipeline.interpreter._call")
    def test_no_cluster_id_returns_empty(self, mock_call):
        mock_call.return_value = MOCK_RESPONSE
        df = _make_df(10).drop(columns=["cluster_id"])
        result = interpret_all(df=df, patterns=[], api_key="sk-test")
        assert result == {"clusters": {}, "patterns": []}

    # ── progress callback ─────────────────────────────────────────────────

    @patch("pipeline.interpreter._call")
    def test_progress_callback_called_for_clusters_and_patterns(self, mock_call, df, patterns):
        mock_call.return_value = MOCK_RESPONSE
        calls = []
        interpret_all(df=df, patterns=patterns, api_key="sk-test",
                      max_clusters=2, max_patterns=2,
                      progress_cb=lambda cur, total, label: calls.append(label))
        labels = calls
        assert any("Кластер" in l for l in labels)
        assert any("Паттерн" in l or "→" in l for l in labels)


# ═══════════════════════════════════════════════════════════════════════════
# format_results
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatResults:

    def _full_results(self):
        return {
            "clusters": {
                0: MOCK_RESPONSE,
                1: {"error": "Timeout"},
            },
            "patterns": [
                {
                    "pattern":     ["/", "/catalog"],
                    "pattern_str": "/ → /catalog",
                    "support_abs": 15,
                    "support_pct": 50.0,
                    "length":      2,
                    "interpretation": MOCK_RESPONSE,
                },
                {
                    "pattern":     ["/login", "/dashboard"],
                    "pattern_str": "/login → /dashboard",
                    "support_abs": 8,
                    "support_pct": 26.7,
                    "length":      2,
                    "interpretation": {"error": "API timeout"},
                },
            ],
        }

    def test_returns_string(self):
        assert isinstance(format_results(self._full_results()), str)

    def test_contains_cluster_header(self):
        assert "КЛАСТЕР 0" in format_results(self._full_results())

    def test_contains_pattern_section_header(self):
        text = format_results(self._full_results())
        assert "ПАТТЕРН" in text

    def test_contains_pattern_str(self):
        text = format_results(self._full_results())
        assert "/ → /catalog" in text

    def test_contains_support_info(self):
        text = format_results(self._full_results())
        assert "15" in text
        assert "50.0" in text

    def test_cluster_error_shown(self):
        text = format_results(self._full_results())
        assert "Timeout" in text

    def test_pattern_error_shown(self):
        text = format_results(self._full_results())
        assert "API timeout" in text

    def test_classification_shown(self):
        assert "типичный" in format_results(self._full_results())

    def test_recommendation_shown(self):
        assert "навигацию" in format_results(self._full_results()).lower()

    def test_empty_results(self):
        text = format_results({"clusters": {}, "patterns": []})
        assert "нет" in text.lower()

    def test_backward_compat_old_format(self):
        """format_results должен принимать старый формат Dict[int, Dict]."""
        old = {0: MOCK_RESPONSE, 1: {"error": "x"}}
        text = format_results(old)
        assert "КЛАСТЕР 0" in text

    def test_clusters_before_patterns(self):
        text = format_results(self._full_results())
        assert text.index("КЛАСТЕР") < text.index("ПАТТЕРН")


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA и SYSTEM_PROMPT
# ═══════════════════════════════════════════════════════════════════════════

class TestSchema:

    def test_required_fields(self):
        for f in ("classification", "confidence", "description",
                  "problems", "recommendations"):
            assert f in SCHEMA["required"]

    def test_classification_enum(self):
        enum = SCHEMA["properties"]["classification"]["enum"]
        assert "типичный" in enum and "нетипичный" in enum

    def test_confidence_enum(self):
        enum = SCHEMA["properties"]["confidence"]["enum"]
        assert {"высокая", "средняя", "низкая"} == set(enum)

    def test_arrays(self):
        for f in ("problems", "recommendations"):
            assert SCHEMA["properties"][f]["type"] == "array"


class TestSystemPrompt:

    def test_is_string(self):
        assert isinstance(SYSTEM_PROMPT, str)

    def test_mentions_json(self):
        assert "JSON" in SYSTEM_PROMPT

    def test_mentions_russian(self):
        assert "русский" in SYSTEM_PROMPT.lower() or "Язык" in SYSTEM_PROMPT

    def test_mentions_classification(self):
        assert "типичный" in SYSTEM_PROMPT or "классифицир" in SYSTEM_PROMPT.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
