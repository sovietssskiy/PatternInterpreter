"""
tests/test_interpreter.py
Тесты модуля interpreter.py — без реальных вызовов API (используются моки).

Запуск:
    cd web_log_analyzer
    python -m pytest tests/test_interpreter.py -v
"""

import json
import os
import sys
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.interpreter import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    _anonymize_sessions,
    _pick_representative_sessions,
    build_prompt,
    format_interpretations_text,
    interpret_all_patterns,
    interpret_pattern,
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


def _make_df(n: int = 20) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        seq = ["/", "/catalog", "/products/{id}"]
        rows.append({
            "session_id":    f"s_{i}",
            "ip_hash":       f"ip_{i % 5}",
            "request_count": float(rng.integers(2, 10)),
            "duration_sec":  float(rng.integers(30, 300)),
            "unique_pages":  3.0,
            "error_rate":    0.0,
            "avg_bytes":     2000.0,
            "page_sequence": seq,
            "page_set":      list(set(seq)),
            "cluster_id":    i % 3,
            "anomaly_flag":  i >= n - 3,
            "anomaly_priority": i >= n - 2,
            "if_score":      float(i) / n,
            "methods_triggered": 2 if i >= n - 2 else 0,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# _anonymize_sessions
# ═══════════════════════════════════════════════════════════════════════════

class TestAnonymizeSessions:

    def test_ip_replaced(self):
        sessions = [
            {"ip_hash": "abc123", "request_count": 5},
            {"ip_hash": "def456", "request_count": 3},
        ]
        result = _anonymize_sessions(sessions)
        for r in result:
            assert "ip_hash" not in r
            assert r["user_id"].startswith("user_")

    def test_same_ip_same_alias(self):
        sessions = [
            {"ip_hash": "abc", "request_count": 1},
            {"ip_hash": "abc", "request_count": 2},
        ]
        result = _anonymize_sessions(sessions)
        assert result[0]["user_id"] == result[1]["user_id"]

    def test_different_ips_different_alias(self):
        sessions = [
            {"ip_hash": "aaa", "request_count": 1},
            {"ip_hash": "bbb", "request_count": 2},
        ]
        result = _anonymize_sessions(sessions)
        assert result[0]["user_id"] != result[1]["user_id"]

    def test_safe_keys_preserved(self):
        sessions = [{
            "ip_hash": "x",
            "request_count": 5,
            "duration_sec": 120.0,
            "page_sequence": ["/", "/catalog"],
            "secret_field": "should_be_removed",
        }]
        result = _anonymize_sessions(sessions)
        assert "request_count" in result[0]
        assert "duration_sec" in result[0]
        assert "page_sequence" in result[0]
        assert "secret_field" not in result[0]

    def test_empty_list(self):
        assert _anonymize_sessions([]) == []


# ═══════════════════════════════════════════════════════════════════════════
# _pick_representative_sessions
# ═══════════════════════════════════════════════════════════════════════════

class TestPickRepresentativeSessions:

    def test_respects_n_limit(self):
        df = _make_df(30)
        result = _pick_representative_sessions(df, n=5)
        assert len(result) <= 5

    def test_returns_list_of_dicts(self):
        df = _make_df(10)
        result = _pick_representative_sessions(df, n=5)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_page_sequence_truncated(self):
        import pandas as pd
        df = pd.DataFrame([{
            "session_id": "s0",
            "ip_hash": "x",
            "page_sequence": [f"/page/{i}" for i in range(20)],
        }])
        result = _pick_representative_sessions(df, n=5)
        assert len(result[0]["page_sequence"]) <= 10

    def test_prefer_high_score(self):
        df = _make_df(20)
        result_high = _pick_representative_sessions(df, n=5, prefer_high_score=True)
        result_normal = _pick_representative_sessions(df, n=5, prefer_high_score=False)
        # Они могут отличаться — просто проверяем, что оба возвращают данные
        assert len(result_high) > 0
        assert len(result_normal) > 0

    def test_empty_df(self):
        result = _pick_representative_sessions(pd.DataFrame(), n=5)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# build_prompt
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPrompt:

    def _make_sessions(self):
        return [{"ip_hash": "aaa", "request_count": 5, "page_sequence": ["/", "/catalog"]}]

    def test_contains_pattern_type(self):
        prompt = build_prompt("Кластер 0", {}, self._make_sessions())
        assert "Кластер 0" in prompt

    def test_contains_stats(self):
        stats = {"size": 42, "median_requests": 5.0}
        prompt = build_prompt("test", stats, self._make_sessions())
        assert "42" in prompt
        assert "median_requests" in prompt

    def test_anomaly_flag_noted(self):
        prompt = build_prompt("test", {}, self._make_sessions(), anomaly_flag=True)
        assert "НЕТИПИЧНЫЙ" in prompt.upper() or "АНОМАЛ" in prompt.upper()

    def test_no_anomaly_flag_when_false(self):
        prompt = build_prompt("test", {}, self._make_sessions(), anomaly_flag=False)
        assert "⚠️" not in prompt

    def test_extra_context_included(self):
        prompt = build_prompt("test", {}, self._make_sessions(),
                              extra_context="/ → /catalog → /checkout")
        assert "/checkout" in prompt

    def test_ip_not_in_prompt(self):
        sessions = [{"ip_hash": "192.168.1.100", "request_count": 3}]
        prompt = build_prompt("test", {}, sessions)
        assert "192.168.1.100" not in prompt

    def test_returns_string(self):
        prompt = build_prompt("test", {}, self._make_sessions())
        assert isinstance(prompt, str)
        assert len(prompt) > 50


# ═══════════════════════════════════════════════════════════════════════════
# interpret_pattern (с моком API)
# ═══════════════════════════════════════════════════════════════════════════

class TestInterpretPattern:

    def _mock_response(self, data: dict = None):
        """Создаёт мок объекта ответа OpenAI."""
        resp_data = data or MOCK_RESPONSE
        msg = MagicMock()
        msg.content = json.dumps(resp_data, ensure_ascii=False)
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        return response

    @patch("pipeline.interpreter._call_api")
    def test_returns_dict(self, mock_call):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_pattern(
            pattern_type="Кластер 0",
            stats={"size": 10},
            sessions=[{"ip_hash": "x", "request_count": 3}],
            api_key="sk-test",
        )
        assert isinstance(result, dict)

    @patch("pipeline.interpreter._call_api")
    def test_result_has_required_fields(self, mock_call):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_pattern(
            pattern_type="Кластер 0",
            stats={},
            sessions=[],
            api_key="sk-test",
        )
        for field in ("classification", "confidence", "description",
                      "problems", "recommendations"):
            assert field in result

    @patch("pipeline.interpreter._call_api")
    def test_classification_valid(self, mock_call):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_pattern("test", {}, [], api_key="sk-test")
        assert result["classification"] in ("типичный", "нетипичный")

    @patch("pipeline.interpreter._call_api")
    def test_api_called_once(self, mock_call):
        mock_call.return_value = MOCK_RESPONSE
        interpret_pattern("test", {}, [], api_key="sk-test")
        assert mock_call.call_count == 1

    @patch("pipeline.interpreter._call_api")
    def test_api_key_passed(self, mock_call):
        mock_call.return_value = MOCK_RESPONSE
        interpret_pattern("test", {}, [], api_key="sk-mykey-123")
        _, kwargs = mock_call.call_args
        assert kwargs.get("api_key") == "sk-mykey-123"


# ═══════════════════════════════════════════════════════════════════════════
# interpret_all_patterns (с моком API)
# ═══════════════════════════════════════════════════════════════════════════

class TestInterpretAllPatterns:

    @pytest.fixture
    def df(self):
        return _make_df(30)

    @pytest.fixture
    def seq_patterns(self):
        return [
            {
                "pattern":     ["/", "/catalog", "/products/{id}"],
                "support_abs": 15,
                "support_pct": 75.0,
                "length":      3,
                "pattern_str": "/ → /catalog → /products/{id}",
            }
        ]

    @patch("pipeline.interpreter._call_api")
    def test_returns_dict_with_keys(self, mock_call, df, seq_patterns):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all_patterns(
            df=df,
            seq_patterns=seq_patterns,
            rules_df=pd.DataFrame(),
            api_key="sk-test",
            max_clusters=2,
            max_patterns=1,
            max_anomaly_groups=1,
        )
        assert "clusters" in result
        assert "patterns" in result
        assert "anomalies" in result

    @patch("pipeline.interpreter._call_api")
    def test_clusters_interpreted(self, mock_call, df, seq_patterns):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all_patterns(
            df=df, seq_patterns=[], rules_df=pd.DataFrame(),
            api_key="sk-test", max_clusters=3, max_patterns=0,
            max_anomaly_groups=0,
        )
        assert len(result["clusters"]) <= 3

    @patch("pipeline.interpreter._call_api")
    def test_patterns_interpreted(self, mock_call, df, seq_patterns):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all_patterns(
            df=df, seq_patterns=seq_patterns, rules_df=pd.DataFrame(),
            api_key="sk-test", max_clusters=0, max_patterns=1,
            max_anomaly_groups=0,
        )
        assert len(result["patterns"]) <= 1

    @patch("pipeline.interpreter._call_api")
    def test_api_error_handled_gracefully(self, mock_call, df, seq_patterns):
        """Ошибка API не роняет весь конвейер."""
        mock_call.side_effect = RuntimeError("API недоступен")
        result = interpret_all_patterns(
            df=df, seq_patterns=seq_patterns, rules_df=pd.DataFrame(),
            api_key="sk-test", max_clusters=1, max_patterns=1,
            max_anomaly_groups=0,
        )
        # Результаты содержат поле error, но структура сохранена
        assert "clusters" in result
        assert "patterns" in result

    @patch("pipeline.interpreter._call_api")
    def test_progress_callback_called(self, mock_call, df, seq_patterns):
        mock_call.return_value = MOCK_RESPONSE
        calls = []
        def _cb(cur, total, label):
            calls.append((cur, total, label))

        interpret_all_patterns(
            df=df, seq_patterns=seq_patterns, rules_df=pd.DataFrame(),
            api_key="sk-test", max_clusters=2, max_patterns=1,
            max_anomaly_groups=0, progress_callback=_cb,
        )
        assert len(calls) > 0

    @patch("pipeline.interpreter._call_api")
    def test_empty_df(self, mock_call):
        mock_call.return_value = MOCK_RESPONSE
        result = interpret_all_patterns(
            df=pd.DataFrame(), seq_patterns=[], rules_df=pd.DataFrame(),
            api_key="sk-test",
        )
        assert result["clusters"] == {}
        assert result["patterns"] == []


# ═══════════════════════════════════════════════════════════════════════════
# format_interpretations_text
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatInterpretationsText:

    def _make_results(self):
        return {
            "clusters": {
                0: MOCK_RESPONSE,
                1: {"error": "Timeout"},
            },
            "patterns": [
                {
                    "pattern_str": "/ → /catalog",
                    "support_abs": 10,
                    "support_pct": 50.0,
                    "interpretation": MOCK_RESPONSE,
                },
            ],
            "anomalies": [
                {
                    "group": "высокоприоритетные аномалии",
                    "size": 5,
                    "interpretation": {
                        **MOCK_RESPONSE,
                        "classification": "нетипичный",
                    },
                }
            ],
        }

    def test_returns_string(self):
        text = format_interpretations_text(self._make_results())
        assert isinstance(text, str)

    def test_contains_cluster_info(self):
        text = format_interpretations_text(self._make_results())
        assert "КЛАСТЕР 0" in text

    def test_contains_error_for_cluster_1(self):
        text = format_interpretations_text(self._make_results())
        assert "Ошибка" in text

    def test_contains_pattern(self):
        text = format_interpretations_text(self._make_results())
        assert "/catalog" in text

    def test_contains_anomaly(self):
        text = format_interpretations_text(self._make_results())
        assert "АНОМАЛЬНАЯ" in text

    def test_contains_recommendations(self):
        text = format_interpretations_text(self._make_results())
        assert "навигацию" in text.lower() or "каталог" in text.lower()

    def test_empty_results(self):
        text = format_interpretations_text({"clusters": {}, "patterns": [], "anomalies": []})
        assert "нет" in text.lower() or len(text) < 30

    def test_classification_shown(self):
        text = format_interpretations_text(self._make_results())
        assert "типичный" in text


# ═══════════════════════════════════════════════════════════════════════════
# Проверка JSON Schema
# ═══════════════════════════════════════════════════════════════════════════

class TestResponseSchema:

    def test_required_fields_present(self):
        required = RESPONSE_SCHEMA.get("required", [])
        for field in ("classification", "confidence", "description",
                      "problems", "recommendations"):
            assert field in required

    def test_classification_enum(self):
        enum = RESPONSE_SCHEMA["properties"]["classification"]["enum"]
        assert "типичный" in enum
        assert "нетипичный" in enum

    def test_confidence_enum(self):
        enum = RESPONSE_SCHEMA["properties"]["confidence"]["enum"]
        assert "высокая" in enum
        assert "средняя" in enum
        assert "низкая" in enum

    def test_problems_is_array(self):
        assert RESPONSE_SCHEMA["properties"]["problems"]["type"] == "array"

    def test_recommendations_is_array(self):
        assert RESPONSE_SCHEMA["properties"]["recommendations"]["type"] == "array"


# ═══════════════════════════════════════════════════════════════════════════
# Системный промпт
# ═══════════════════════════════════════════════════════════════════════════

class TestSystemPrompt:

    def test_is_string(self):
        assert isinstance(SYSTEM_PROMPT, str)

    def test_contains_key_instructions(self):
        assert "русский" in SYSTEM_PROMPT.lower() or "Язык" in SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT
        assert "тестировщик" in SYSTEM_PROMPT.lower() or "тестирован" in SYSTEM_PROMPT.lower()

    def test_mentions_classification(self):
        assert "типичный" in SYSTEM_PROMPT or "классифицир" in SYSTEM_PROMPT.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
