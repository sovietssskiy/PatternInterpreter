"""
tests/test_pattern_mining_anomaly.py
Тесты модулей поиска паттернов и обнаружения аномалий.

Запуск:
    cd project
    python -m pytest test/test_pattern_mining_anomaly.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pattern_mining import (
    _gap_filter,
    mine_patterns,
    patterns_for_cluster,
    patterns_summary,
)
from pipeline.anomaly import (
    detect_anomalies,
    anomaly_report,
    FEATURES as ANOMALY_FEATURES,
)


# ═══════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФИКСТУРЫ
# ═══════════════════════════════════════════════════════════════════════════

def _make_session_df(n_sessions: int = 20, seed: int = 42) -> pd.DataFrame:
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
# _gap_filter
# ═══════════════════════════════════════════════════════════════════════════

class TestGapFilter:

    def test_no_gap(self):
        seq = ["/a", "/b", "/c"]
        ts  = [0.0, 60.0, 120.0]
        result = _gap_filter(seq, ts, max_gap=300)
        assert result == seq

    def test_gap_splits(self):
        """Разрыв больше max_gap — возвращается более длинная часть."""
        seq = ["/a", "/b", "/c", "/d"]
        ts  = [0.0, 60.0, 700.0, 760.0]
        result = _gap_filter(seq, ts, max_gap=300)
        assert len(result) == 2

    def test_gap_at_start(self):
        seq = ["/a", "/b", "/c"]
        ts  = [0.0, 500.0, 560.0]
        result = _gap_filter(seq, ts, max_gap=300)
        assert result == ["/b", "/c"]

    def test_short_sequence(self):
        """Последовательность из одного элемента возвращается как есть."""
        assert _gap_filter(["/a"], [0.0], 300) == ["/a"]

    def test_mismatched_lengths(self):
        """Несовпадение длин — возвращается исходная последовательность."""
        seq = ["/a", "/b"]
        ts  = [0.0]
        assert _gap_filter(seq, ts, 300) == seq

    def test_empty(self):
        assert _gap_filter([], [], 300) == []


# ═══════════════════════════════════════════════════════════════════════════
# mine_patterns
# ═══════════════════════════════════════════════════════════════════════════

class TestMinePatterns:

    def _make_df(self):
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
        result = mine_patterns(self._make_df(), min_support_pct=0.1)
        assert isinstance(result, list)

    def test_pattern_fields(self):
        result = mine_patterns(self._make_df(), min_support_pct=0.1)
        assert len(result) > 0
        p = result[0]
        for field in ("pattern", "support_abs", "support_pct", "pattern_str", "length"):
            assert field in p

    def test_support_pct_correct(self):
        result = mine_patterns(self._make_df(), min_support_pct=0.0)
        for p in result:
            expected = round(p["support_abs"] / 6 * 100, 1)
            assert abs(p["support_pct"] - expected) < 0.1

    def test_sorted_by_support(self):
        result = mine_patterns(self._make_df(), min_support_pct=0.0)
        supports = [p["support_abs"] for p in result]
        assert supports == sorted(supports, reverse=True)

    def test_root_in_all_sessions(self):
        result = mine_patterns(self._make_df(), min_support_pct=0.0)
        root_patterns = [p for p in result if p["pattern"][0] == "/"]
        assert len(root_patterns) > 0
        assert root_patterns[0]["support_abs"] >= 4

    def test_top_k_respected(self):
        result = mine_patterns(self._make_df(), min_support_pct=0.0, top_k=3)
        assert len(result) <= 3

    def test_empty_df(self):
        result = mine_patterns(pd.DataFrame(), min_support_pct=0.1)
        assert result == []

    def test_all_single_page_sessions(self):
        df = pd.DataFrame({
            "session_id": [0, 1, 2],
            "page_sequence": [["/a"], ["/b"], ["/c"]],
        })
        result = mine_patterns(df)
        assert result == []

    def test_pattern_str_format(self):
        result = mine_patterns(self._make_df(), min_support_pct=0.0)
        for p in result:
            assert " → ".join(p["pattern"]) == p["pattern_str"]


# ═══════════════════════════════════════════════════════════════════════════
# patterns_summary
# ═══════════════════════════════════════════════════════════════════════════

class TestPatternsSummary:

    def test_empty_patterns(self):
        text = patterns_summary([])
        assert "не обнаружено" in text.lower()

    def test_contains_pattern(self):
        patterns = [{
            "support_abs": 5, "support_pct": 83.3,
            "pattern_str": "/ → /catalog", "pattern": ["/", "/catalog"], "length": 2,
        }]
        text = patterns_summary(patterns)
        assert "/catalog" in text
        assert "83.3" in text


# ═══════════════════════════════════════════════════════════════════════════
# patterns_for_cluster
# ═══════════════════════════════════════════════════════════════════════════

class TestPatternsForCluster:

    def test_returns_matching_patterns(self):
        df = pd.DataFrame({
            "page_sequence": [["/", "/catalog", "/products/{id}"]],
        })
        all_patterns = [
            {"pattern": ["/", "/catalog"], "support_abs": 4, "support_pct": 66.0,
             "pattern_str": "/ → /catalog", "length": 2},
            {"pattern": ["/news"], "support_abs": 2, "support_pct": 33.0,
             "pattern_str": "/news", "length": 1},
        ]
        result = patterns_for_cluster(df, all_patterns)
        assert any(p["pattern"] == ["/", "/catalog"] for p in result)

    def test_max_8_returned(self):
        df = pd.DataFrame({
            "page_sequence": [["/a", "/b", "/c", "/d", "/e", "/f", "/g", "/h", "/i"]],
        })
        all_patterns = [
            {"pattern": ["/a", "/b"], "support_abs": 1, "support_pct": 10.0,
             "pattern_str": "/a → /b", "length": 2}
        ] * 20
        result = patterns_for_cluster(df, all_patterns)
        assert len(result) <= 8


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
        for col in ("if_flag", "if_score", "z_flag", "anomaly_flag", "anomaly_priority"):
            assert col in result.columns

    def test_anomaly_flag_bool(self, df):
        result = detect_anomalies(df)
        assert result["anomaly_flag"].dtype == bool

    def test_known_anomalies_detected(self, df):
        result = detect_anomalies(df, contamination=0.08)
        anomaly_ids = set(result[result["anomaly_flag"]]["session_id"])
        detected = sum(1 for i in range(5) if f"a_{i}" in anomaly_ids)
        assert detected >= 3

    def test_priority_subset_of_flag(self, df):
        result = detect_anomalies(df)
        priority_ids = set(result[result["anomaly_priority"]]["session_id"])
        flag_ids     = set(result[result["anomaly_flag"]]["session_id"])
        assert priority_ids.issubset(flag_ids)

    def test_if_score_positive(self, df):
        result = detect_anomalies(df)
        assert (result["if_score"] >= 0).all()

    def test_original_df_not_modified(self, df):
        original_cols = set(df.columns)
        detect_anomalies(df)
        assert set(df.columns) == original_cols

    def test_missing_feature_raises(self, df):
        df_bad = df.drop(columns=["request_count"])
        with pytest.raises(ValueError, match="Нет колонок"):
            detect_anomalies(df_bad)

    def test_empty_df(self):
        result = detect_anomalies(pd.DataFrame())
        assert result.empty

    def test_custom_contamination(self, df):
        r1 = detect_anomalies(df, contamination=0.02)
        r2 = detect_anomalies(df, contamination=0.20)
        assert r2["if_flag"].sum() >= r1["if_flag"].sum()

    def test_custom_z_threshold(self, df):
        r_strict = detect_anomalies(df, z_threshold=5.0)
        r_loose  = detect_anomalies(df, z_threshold=1.0)
        assert r_loose["z_flag"].sum() >= r_strict["z_flag"].sum()


# ═══════════════════════════════════════════════════════════════════════════
# anomaly_report
# ═══════════════════════════════════════════════════════════════════════════

class TestAnomalyReport:

    @pytest.fixture
    def df_marked(self):
        return detect_anomalies(_make_anomaly_df())

    def test_returns_string(self, df_marked):
        text = anomaly_report(df_marked)
        assert isinstance(text, str)

    def test_contains_session_count(self, df_marked):
        text = anomaly_report(df_marked)
        assert str(len(df_marked)) in text

    def test_contains_methods(self, df_marked):
        text = anomaly_report(df_marked)
        assert "Isolation Forest" in text
        assert "Z-score" in text

    def test_without_detect(self):
        df = pd.DataFrame({"session_id": [1, 2]})
        text = anomaly_report(df)
        assert "не выполнял" in text.lower() or "маркировка" in text.lower()

    def test_priority_in_report(self, df_marked):
        text = anomaly_report(df_marked)
        assert "риоритет" in text  # «Приоритетных»


# ═══════════════════════════════════════════════════════════════════════════
# ANOMALY_FEATURES константа
# ═══════════════════════════════════════════════════════════════════════════

class TestAnomalyFeatures:

    def test_is_list(self):
        assert isinstance(ANOMALY_FEATURES, list)

    def test_contains_required(self):
        for f in ("request_count", "duration_sec", "unique_pages", "error_rate"):
            assert f in ANOMALY_FEATURES


# ═══════════════════════════════════════════════════════════════════════════
# ИНТЕГРАЦИОННЫЙ ТЕСТ: полный аналитический цикл
# ═══════════════════════════════════════════════════════════════════════════

class TestFullAnalyticsPipeline:

    def test_pipeline(self):
        from pipeline.clustering import cluster_sessions

        df = _make_session_df(n_sessions=40)

        df_cl, meta = cluster_sessions(df, k=3)
        assert "cluster_id" in df_cl.columns

        patterns = mine_patterns(df_cl, min_support_pct=0.05)
        assert isinstance(patterns, list)

        df_an = detect_anomalies(df_cl)
        assert "anomaly_flag" in df_an.columns

        summary = patterns_summary(patterns)
        report  = anomaly_report(df_an)
        assert len(summary) > 0
        assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
