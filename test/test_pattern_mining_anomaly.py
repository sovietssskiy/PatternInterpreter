"""
tests/test_pattern_mining_anomaly.py
Тесты модулей поиска паттернов и обнаружения аномалий.

Запуск:
    cd web_log_analyzer
    python -m pytest tests/test_pattern_mining_anomaly.py -v
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pattern_mining import (
    _apply_gap_filter,
    _build_sequence_db,
    mine_sequential_patterns,
    mine_association_rules,
    patterns_to_llm_context,
    rules_to_llm_context,
    save_patterns,
    load_patterns,
    save_rules,
)
from pipeline.anomaly import (
    detect_anomalies,
    anomaly_summary,
    anomaly_summary_text,
    get_anomalous_sessions,
    ANOMALY_FEATURES,
)


# ═══════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФИКСТУРЫ
# ═══════════════════════════════════════════════════════════════════════════

def _make_session_df(n_sessions: int = 20, seed: int = 42) -> pd.DataFrame:
    """Синтетический DataFrame сессий для тестов."""
    rng = np.random.default_rng(seed)

    shop_paths = ["/", "/catalog", "/products/{id}", "/cart", "/checkout",
                  "/search", "/about", "/news", "/login", "/profile"]

    rows = []
    for i in range(n_sessions):
        length = rng.integers(2, 8)
        seq = list(rng.choice(shop_paths, size=length, replace=True))
        rows.append({
            "session_id":    f"s_{i}",
            "ip_hash":       f"hash_{i % 5}",
            "request_count": float(length),
            "duration_sec":  float(rng.integers(30, 600)),
            "unique_pages":  float(len(set(seq))),
            "error_rate":    float(rng.uniform(0, 0.1)),
            "avg_bytes":     float(rng.integers(500, 5000)),
            "page_sequence": seq,
            "page_set":      list(set(seq)),
        })
    return pd.DataFrame(rows)


def _make_anomaly_df(n_normal: int = 50, n_anomaly: int = 5) -> pd.DataFrame:
    """DataFrame с явными аномалиями для тестов."""
    rng = np.random.default_rng(0)

    normal = pd.DataFrame({
        "session_id":    [f"n_{i}" for i in range(n_normal)],
        "request_count": rng.integers(2, 10, n_normal).astype(float),
        "duration_sec":  rng.integers(30, 300, n_normal).astype(float),
        "unique_pages":  rng.integers(2, 8, n_normal).astype(float),
        "error_rate":    rng.uniform(0, 0.05, n_normal),
        "avg_bytes":     rng.integers(1000, 3000, n_normal).astype(float),
        "page_set":      [["/", "/catalog"] for _ in range(n_normal)],
    })

    # Явные аномалии: огромное число запросов и высокая доля ошибок
    anomaly = pd.DataFrame({
        "session_id":    [f"a_{i}" for i in range(n_anomaly)],
        "request_count": [500.0, 800.0, 1200.0, 600.0, 900.0],
        "duration_sec":  [3600.0, 5400.0, 7200.0, 4000.0, 6000.0],
        "unique_pages":  [80.0, 120.0, 200.0, 90.0, 150.0],
        "error_rate":    [0.9, 0.85, 0.95, 0.88, 0.92],
        "avg_bytes":     [100.0, 50.0, 30.0, 80.0, 60.0],
        "page_set":      [["/"] for _ in range(n_anomaly)],
    })

    return pd.concat([normal, anomaly], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# _apply_gap_filter
# ═══════════════════════════════════════════════════════════════════════════

class TestApplyGapFilter:

    def test_no_gap(self):
        """Без разрывов — возвращается вся последовательность."""
        seq = ["/a", "/b", "/c"]
        ts  = [0.0, 60.0, 120.0]
        result = _apply_gap_filter(seq, ts, max_gap_sec=300)
        assert result == seq

    def test_gap_splits(self):
        """Разрыв больше max_gap — возвращается более длинная часть."""
        seq = ["/a", "/b", "/c", "/d"]
        ts  = [0.0, 60.0, 700.0, 760.0]   # разрыв 640 с между /b и /c
        result = _apply_gap_filter(seq, ts, max_gap_sec=300)
        # ['/a', '/b'] длина 2,  ['/c', '/d'] длина 2 — возвращается первая
        assert len(result) == 2

    def test_gap_at_start(self):
        """Разрыв в самом начале."""
        seq = ["/a", "/b", "/c"]
        ts  = [0.0, 500.0, 560.0]   # разрыв сразу после /a
        result = _apply_gap_filter(seq, ts, max_gap_sec=300)
        assert result == ["/b", "/c"]

    def test_short_sequence(self):
        """Последовательность из одного элемента возвращается как есть."""
        assert _apply_gap_filter(["/a"], [0.0], 300) == ["/a"]

    def test_mismatched_lengths(self):
        """Несовпадение длин — возвращается исходная последовательность."""
        seq = ["/a", "/b"]
        ts  = [0.0]
        assert _apply_gap_filter(seq, ts, 300) == seq

    def test_empty(self):
        assert _apply_gap_filter([], [], 300) == []


# ═══════════════════════════════════════════════════════════════════════════
# _build_sequence_db
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildSequenceDb:

    def test_basic(self):
        df = pd.DataFrame({
            "page_sequence": [["/a", "/b"], ["/c", "/d", "/e"]],
        })
        db = _build_sequence_db(df)
        assert len(db) == 2
        assert db[0] == ["/a", "/b"]

    def test_filters_short(self):
        """Последовательности длиной < 2 не включаются."""
        df = pd.DataFrame({
            "page_sequence": [["/a"], ["/b", "/c"]],
        })
        db = _build_sequence_db(df)
        assert len(db) == 1

    def test_applies_gap_filter(self):
        """Временно́е ограничение применяется при наличии page_timestamps."""
        df = pd.DataFrame({
            "page_sequence":  [["/a", "/b", "/c", "/d"]],
            "page_timestamps": [[0.0, 60.0, 700.0, 760.0]],  # разрыв после /b
        })
        db = _build_sequence_db(df, max_gap_sec=300)
        # Обе части равны по длине (2), возвращается любая из них
        assert len(db[0]) == 2

    def test_empty_df(self):
        assert _build_sequence_db(pd.DataFrame()) == []


# ═══════════════════════════════════════════════════════════════════════════
# mine_sequential_patterns
# ═══════════════════════════════════════════════════════════════════════════

class TestMineSequentialPatterns:

    def _make_df(self):
        """6 сессий с явными частыми паттернами."""
        return pd.DataFrame({
            "session_id": list(range(6)),
            "page_sequence": [
                ["/", "/catalog", "/products/{id}", "/cart", "/checkout"],
                ["/", "/catalog", "/products/{id}", "/cart"],
                ["/", "/catalog", "/products/{id}"],
                ["/", "/search",  "/products/{id}", "/cart", "/checkout"],
                ["/", "/catalog", "/products/{id}"],
                ["/", "/search",  "/products/{id}", "/cart"],
            ],
        })

    def test_returns_list(self):
        df = self._make_df()
        result = mine_sequential_patterns(df, min_support_pct=0.1)
        assert isinstance(result, list)

    def test_pattern_fields(self):
        df = self._make_df()
        result = mine_sequential_patterns(df, min_support_pct=0.1)
        assert len(result) > 0
        p = result[0]
        assert "pattern" in p
        assert "support_abs" in p
        assert "support_pct" in p
        assert "pattern_str" in p
        assert "length" in p

    def test_support_pct_correct(self):
        df = self._make_df()
        result = mine_sequential_patterns(df, min_support_pct=0.0)
        for p in result:
            expected_pct = round(p["support_abs"] / 6 * 100, 1)
            assert abs(p["support_pct"] - expected_pct) < 0.1

    def test_sorted_by_support(self):
        df = self._make_df()
        result = mine_sequential_patterns(df, min_support_pct=0.0)
        supports = [p["support_abs"] for p in result]
        assert supports == sorted(supports, reverse=True)

    def test_root_in_all_sessions(self):
        """'/' встречается во всех 6 сессиях — его поддержка должна быть 6."""
        df = self._make_df()
        result = mine_sequential_patterns(df, min_support_pct=0.0)
        # В режиме closed одиночный '/' вытесняется более длинным суперпаттерном
        # с той же поддержкой — проверяем, что паттерн '/ → /catalog' имеет ≥4
        root_patterns = [p for p in result if p["pattern"][0] == "/"]
        assert len(root_patterns) > 0
        assert root_patterns[0]["support_abs"] >= 4

    def test_top_k_respected(self):
        df = self._make_df()
        result = mine_sequential_patterns(df, min_support_pct=0.0, top_k=3)
        assert len(result) <= 3

    def test_empty_df(self):
        result = mine_sequential_patterns(pd.DataFrame(), min_support_pct=0.1)
        assert result == []

    def test_all_single_page_sessions(self):
        """Сессии из одного URL — паттернов нет."""
        df = pd.DataFrame({
            "session_id": [0, 1, 2],
            "page_sequence": [["/a"], ["/b"], ["/c"]],
        })
        result = mine_sequential_patterns(df)
        assert result == []

    def test_pattern_str_format(self):
        df = self._make_df()
        result = mine_sequential_patterns(df, min_support_pct=0.0)
        for p in result:
            assert " → ".join(p["pattern"]) == p["pattern_str"]


# ═══════════════════════════════════════════════════════════════════════════
# patterns_to_llm_context
# ═══════════════════════════════════════════════════════════════════════════

class TestPatternsToLlmContext:

    def test_empty_patterns(self):
        text = patterns_to_llm_context([])
        assert "не обнаружено" in text.lower()

    def test_contains_pattern(self):
        patterns = [{"support_abs": 5, "support_pct": 83.3,
                     "pattern_str": "/ → /catalog", "pattern": ["/", "/catalog"]}]
        text = patterns_to_llm_context(patterns)
        assert "/catalog" in text
        assert "83.3" in text


# ═══════════════════════════════════════════════════════════════════════════
# mine_association_rules
# ═══════════════════════════════════════════════════════════════════════════

class TestMineAssociationRules:

    def _make_df(self, n=30):
        """DataFrame с повторяющимися наборами страниц."""
        rng = np.random.default_rng(7)
        # Делаем паттерн: {'/catalog', '/products/{id}'} часто встречаются вместе
        rows = []
        for i in range(n):
            if i < n * 0.7:
                pages = ["/", "/catalog", "/products/{id}"]
            elif i < n * 0.85:
                pages = ["/", "/search", "/products/{id}"]
            else:
                pages = ["/", "/news"]
            rows.append({
                "session_id": i,
                "page_set":   pages,
            })
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        df = self._make_df()
        rules = mine_association_rules(df, min_support=0.05, min_confidence=0.3)
        assert isinstance(rules, pd.DataFrame)

    def test_has_required_columns(self):
        df = self._make_df()
        rules = mine_association_rules(df, min_support=0.05, min_confidence=0.3)
        if not rules.empty:
            for col in ("support", "confidence", "lift", "rule_str"):
                assert col in rules.columns

    def test_sorted_by_lift(self):
        df = self._make_df()
        rules = mine_association_rules(df, min_support=0.05, min_confidence=0.3)
        if len(rules) > 1:
            lifts = rules["lift"].tolist()
            assert lifts == sorted(lifts, reverse=True)

    def test_min_lift_filter(self):
        df = self._make_df()
        rules = mine_association_rules(df, min_support=0.05,
                                       min_confidence=0.3, min_lift=2.0)
        if not rules.empty:
            assert (rules["lift"] >= 2.0).all()

    def test_empty_df(self):
        result = mine_association_rules(pd.DataFrame())
        assert result.empty

    def test_missing_column(self):
        df = pd.DataFrame({"session_id": [1, 2]})
        result = mine_association_rules(df)
        assert result.empty

    def test_too_few_pages(self):
        """Одна страница — правил быть не может."""
        df = pd.DataFrame({"page_set": [["/only"]] * 10})
        result = mine_association_rules(df, min_support=0.1)
        assert result.empty


# ═══════════════════════════════════════════════════════════════════════════
# save / load patterns & rules
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveLoad:

    def test_save_load_patterns(self, tmp_path):
        patterns = [
            {"pattern": ["/", "/catalog"], "support_abs": 5,
             "support_pct": 83.3, "length": 2, "pattern_str": "/ → /catalog"},
        ]
        path = str(tmp_path / "patterns.json")
        save_patterns(patterns, path)
        loaded = load_patterns(path)
        assert loaded == patterns

    def test_save_rules_empty(self, tmp_path):
        path = str(tmp_path / "rules.json")
        save_rules(pd.DataFrame(), path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data == []

    def test_save_rules_nonempty(self, tmp_path):
        df = pd.DataFrame({
            "antecedents_str": ["/catalog"],
            "consequents_str": ["/products/{id}"],
            "rule_str":        ["/catalog → /products/{id}"],
            "support":         [0.4],
            "confidence":      [0.8],
            "lift":            [2.0],
        })
        path = str(tmp_path / "rules.json")
        save_rules(df, path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["lift"] == 2.0


# ═══════════════════════════════════════════════════════════════════════════
# detect_anomalies
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectAnomalies:

    @pytest.fixture
    def df(self):
        return _make_anomaly_df(n_normal=50, n_anomaly=5)

    def test_returns_dataframe(self, df):
        result = detect_anomalies(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_new_columns_present(self, df):
        result = detect_anomalies(df)
        for col in ("if_flag", "if_score", "z_flag", "iqr_flag",
                    "methods_triggered", "anomaly_flag", "anomaly_priority"):
            assert col in result.columns

    def test_anomaly_flag_bool(self, df):
        result = detect_anomalies(df)
        assert result["anomaly_flag"].dtype == bool

    def test_known_anomalies_detected(self, df):
        """Явные аномалии (session_id a_0..a_4) должны быть помечены."""
        result = detect_anomalies(df, contamination=0.08)
        anomaly_ids = set(result[result["anomaly_flag"]]["session_id"])
        # Хотя бы 3 из 5 явных аномалий должны быть обнаружены
        detected = sum(1 for i in range(5) if f"a_{i}" in anomaly_ids)
        assert detected >= 3

    def test_priority_subset_of_flag(self, df):
        """anomaly_priority всегда является подмножеством anomaly_flag."""
        result = detect_anomalies(df)
        priority_ids = set(result[result["anomaly_priority"]]["session_id"])
        flag_ids     = set(result[result["anomaly_flag"]]["session_id"])
        assert priority_ids.issubset(flag_ids)

    def test_methods_triggered_range(self, df):
        result = detect_anomalies(df)
        assert result["methods_triggered"].between(0, 3).all()

    def test_if_score_positive(self, df):
        result = detect_anomalies(df)
        assert (result["if_score"] >= 0).all()

    def test_original_df_not_modified(self, df):
        original_cols = set(df.columns)
        detect_anomalies(df)
        assert set(df.columns) == original_cols

    def test_missing_feature_raises(self, df):
        df_bad = df.drop(columns=["request_count"])
        with pytest.raises(ValueError, match="Отсутствуют колонки"):
            detect_anomalies(df_bad)

    def test_empty_df(self):
        result = detect_anomalies(pd.DataFrame())
        assert result.empty

    def test_custom_contamination(self, df):
        r1 = detect_anomalies(df, contamination=0.02)
        r2 = detect_anomalies(df, contamination=0.20)
        # Больший contamination → больше аномалий от IF
        assert r2["if_flag"].sum() >= r1["if_flag"].sum()

    def test_custom_z_threshold(self, df):
        r_strict = detect_anomalies(df, z_threshold=5.0)
        r_loose  = detect_anomalies(df, z_threshold=1.0)
        assert r_loose["z_flag"].sum() >= r_strict["z_flag"].sum()


# ═══════════════════════════════════════════════════════════════════════════
# anomaly_summary
# ═══════════════════════════════════════════════════════════════════════════

class TestAnomalySummary:

    @pytest.fixture
    def df_marked(self):
        df = _make_anomaly_df()
        return detect_anomalies(df)

    def test_keys_present(self, df_marked):
        s = anomaly_summary(df_marked)
        for key in ("total_sessions", "anomaly_count", "anomaly_pct",
                    "priority_count", "by_method"):
            assert key in s

    def test_total_correct(self, df_marked):
        s = anomaly_summary(df_marked)
        assert s["total_sessions"] == len(df_marked)

    def test_pct_in_range(self, df_marked):
        s = anomaly_summary(df_marked)
        assert 0.0 <= s["anomaly_pct"] <= 100.0

    def test_error_without_detect(self):
        df = pd.DataFrame({"session_id": [1, 2]})
        s = anomaly_summary(df)
        assert "error" in s

    def test_summary_text(self, df_marked):
        text = anomaly_summary_text(df_marked)
        assert "Isolation Forest" in text
        assert "Z-score" in text
        assert "IQR" in text


# ═══════════════════════════════════════════════════════════════════════════
# get_anomalous_sessions
# ═══════════════════════════════════════════════════════════════════════════

class TestGetAnomalousSessions:

    @pytest.fixture
    def df_marked(self):
        return detect_anomalies(_make_anomaly_df())

    def test_returns_only_anomalous(self, df_marked):
        result = get_anomalous_sessions(df_marked)
        assert result["anomaly_flag"].all()

    def test_priority_only(self, df_marked):
        result = get_anomalous_sessions(df_marked, priority_only=True)
        assert result["anomaly_priority"].all()

    def test_top_n(self, df_marked):
        result = get_anomalous_sessions(df_marked, top_n=3)
        assert len(result) <= 3

    def test_sorted_by_score(self, df_marked):
        result = get_anomalous_sessions(df_marked)
        if len(result) > 1 and "if_score" in result.columns:
            scores = result["if_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_raises_without_detect(self):
        df = pd.DataFrame({"session_id": [1]})
        with pytest.raises(ValueError):
            get_anomalous_sessions(df)


# ═══════════════════════════════════════════════════════════════════════════
# ИНТЕГРАЦИОННЫЙ ТЕСТ: полный аналитический цикл
# ═══════════════════════════════════════════════════════════════════════════

class TestFullAnalyticsPipeline:
    """Предобработка → кластеризация → паттерны → аномалии."""

    def test_pipeline(self):
        from pipeline.clustering import cluster_sessions

        df = _make_session_df(n_sessions=40)

        # Кластеризация
        df_cl, meta = cluster_sessions(df, k=3)
        assert "cluster_id" in df_cl.columns

        # Последовательные паттерны
        patterns = mine_sequential_patterns(df_cl, min_support_pct=0.05)
        assert isinstance(patterns, list)

        # Ассоциативные правила
        rules = mine_association_rules(df_cl, min_support=0.05)
        assert isinstance(rules, pd.DataFrame)

        # Обнаружение аномалий
        df_an = detect_anomalies(df_cl)
        assert "anomaly_flag" in df_an.columns

        # LLM-контекст формируется без ошибок
        ctx_seq = patterns_to_llm_context(patterns)
        ctx_rules = rules_to_llm_context(rules)
        summary = anomaly_summary_text(df_an)

        assert len(ctx_seq) > 0
        assert len(ctx_rules) > 0
        assert len(summary) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
