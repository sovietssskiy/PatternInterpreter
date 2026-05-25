"""
tests/test_preprocessor_clustering.py
Тесты модулей предобработки и кластеризации.

Запуск:
    cd project
    python -m pytest test/ -v
"""

import os
import sys
from datetime import datetime

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import (
    _hash,
    _parse_ts,
    identify_sessions,
    parse_log_file,
    preprocess,
)
from pipeline.clustering import (
    cluster_sessions,
    cluster_summary,
    find_optimal_k,
)
from utils.url_normalizer import normalize_url


# ═══════════════════════════════════════════════════════════════════════════
# URL NORMALIZER
# ═══════════════════════════════════════════════════════════════════════════

class TestUrlNormalizer:

    def test_numeric_id(self):
        assert normalize_url("/products/12345") == "/products/{id}"

    def test_uuid(self):
        uri = "/items/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert normalize_url(uri) == "/items/{uuid}"

    def test_query_string_removed(self):
        assert normalize_url("/search?q=test&page=2") == "/search"

    def test_anchor_removed(self):
        assert normalize_url("/page#section") == "/page"

    def test_html_extension_removed(self):
        assert normalize_url("/index.html") == "/index"
        assert normalize_url("/about.php") == "/about"

    def test_trailing_slash(self):
        assert normalize_url("/catalog/") == "/catalog"

    def test_root(self):
        assert normalize_url("/") == "/"

    def test_lowercase(self):
        assert normalize_url("/Catalog/Category") == "/catalog/category"

    def test_empty(self):
        assert normalize_url("") == "/"

    def test_nested_ids(self):
        assert normalize_url("/a/1/b/2") == "/a/{id}/b/{id}"


# ═══════════════════════════════════════════════════════════════════════════
# СТАТИЧЕСКИЕ РЕСУРСЫ: через STATIC frozenset в preprocessor
# ═══════════════════════════════════════════════════════════════════════════

class TestStaticDetection:
    """Проверяем фильтрацию статики через parse_log_file, не через внутренний хелпер."""

    @pytest.fixture
    def log_with_static(self, tmp_path):
        content = (
            '192.168.1.1 - - [01/Jun/2024:10:00:00] "GET /style.css HTTP/1.1" 200 512 "-" "Mozilla/5.0"\n'
            '192.168.1.1 - - [01/Jun/2024:10:00:10] "GET /logo.png HTTP/1.1" 200 1024 "-" "Mozilla/5.0"\n'
            '192.168.1.1 - - [01/Jun/2024:10:00:20] "GET /catalog HTTP/1.1" 200 2048 "-" "Mozilla/5.0"\n'
        )
        f = tmp_path / "access.log"
        f.write_text(content)
        return str(f)

    def test_static_counted(self, log_with_static):
        _, stats = parse_log_file(log_with_static)
        assert stats["static"] == 2

    def test_static_not_in_records(self, log_with_static):
        records, _ = parse_log_file(log_with_static)
        for r in records:
            assert not r["uri"].endswith(".css")
            assert not r["uri"].endswith(".png")

    def test_static_with_query(self, tmp_path):
        content = '1.1.1.1 - - [01/Jun/2024:10:00:00] "GET /image.jpg?v=2 HTTP/1.1" 200 512 "-" "Mozilla/5.0"\n'
        f = tmp_path / "a.log"
        f.write_text(content)
        records, stats = parse_log_file(str(f))
        assert stats["static"] == 1
        assert len(records) == 0

    def test_page_not_static(self, tmp_path):
        content = '1.1.1.1 - - [01/Jun/2024:10:00:00] "GET /products/123 HTTP/1.1" 200 512 "-" "Mozilla/5.0"\n'
        f = tmp_path / "b.log"
        f.write_text(content)
        records, stats = parse_log_file(str(f))
        assert stats["static"] == 0
        assert len(records) == 1


# ═══════════════════════════════════════════════════════════════════════════
# ПАРСЕР ВРЕМЕННОЙ МЕТКИ: _parse_ts
# ═══════════════════════════════════════════════════════════════════════════

class TestParseTs:

    def test_clf_standard(self):
        ts = _parse_ts("01/Jun/2024:12:30:00 +0300")
        assert ts is not None
        assert ts.hour == 12
        assert ts.month == 6

    def test_clf_no_tz(self):
        ts = _parse_ts("15/Jan/2023:08:00:00")
        assert ts is not None
        assert ts.day == 15

    def test_nasa_format(self):
        ts = _parse_ts("Thursday, 01-Jun-95 12:00:00 EDT")
        assert ts is not None
        assert ts.month == 6

    def test_invalid(self):
        assert _parse_ts("not-a-date") is None

    def test_empty(self):
        assert _parse_ts("") is None


# ═══════════════════════════════════════════════════════════════════════════
# ПАРСЕР: parse_log_file
# ═══════════════════════════════════════════════════════════════════════════

CLF_SAMPLE = """\
192.168.1.1 - - [01/Jun/2024:10:00:00] "GET / HTTP/1.1" 200 1024 "-" "Mozilla/5.0"
192.168.1.1 - - [01/Jun/2024:10:00:30] "GET /catalog HTTP/1.1" 200 2048 "-" "Mozilla/5.0"
192.168.1.1 - - [01/Jun/2024:10:01:00] "GET /products/123 HTTP/1.1" 200 4096 "-" "Mozilla/5.0"
192.168.1.2 - - [01/Jun/2024:10:00:00] "GET / HTTP/1.1" 200 1024 "-" "Googlebot/2.1"
192.168.1.3 - - [01/Jun/2024:10:00:00] "GET /style.css HTTP/1.1" 200 512 "-" "Mozilla/5.0"
192.168.1.3 - - [01/Jun/2024:10:00:10] "GET /catalog HTTP/1.1" 200 1024 "-" "Mozilla/5.0"
192.168.1.3 - - [01/Jun/2024:10:00:40] "GET /cart HTTP/1.1" 200 2048 "-" "Mozilla/5.0"
this is a bad line
"""


class TestParseLogFile:

    @pytest.fixture
    def log_file(self, tmp_path):
        f = tmp_path / "access.log"
        f.write_text(CLF_SAMPLE)
        return str(f)

    def test_stat_keys_present(self, log_file):
        _, stats = parse_log_file(log_file)
        for key in ("total", "errors", "static", "bots", "ok"):
            assert key in stats

    def test_parsed_count(self, log_file):
        records, stats = parse_log_file(log_file)
        # style.css отфильтрован (статика), Googlebot отфильтрован (бот),
        # bad line — ошибка парсинга
        assert stats["static"] == 1
        assert stats["bots"] == 1
        assert stats["errors"] == 1
        assert stats["ok"] == 5

    def test_url_normalized(self, log_file):
        records, _ = parse_log_file(log_file)
        uris = [r["uri"] for r in records]
        assert "/products/{id}" in uris

    def test_no_static_in_results(self, log_file):
        records, _ = parse_log_file(log_file)
        for r in records:
            assert not r["uri"].endswith(".css")

    def test_no_bots_in_results(self, log_file):
        records, _ = parse_log_file(log_file)
        for r in records:
            assert "googlebot" not in r["agent"].lower()

    def test_record_fields(self, log_file):
        records, _ = parse_log_file(log_file)
        r = records[0]
        for field in ("ip", "ts", "uri", "status", "bytes", "agent"):
            assert field in r


# ═══════════════════════════════════════════════════════════════════════════
# ИДЕНТИФИКАЦИЯ СЕССИЙ
# ═══════════════════════════════════════════════════════════════════════════

class TestIdentifySessions:

    def _make_records(self, ip_ts_uri):
        base = datetime(2024, 1, 1, 0, 0, 0)
        from datetime import timedelta
        return [
            {
                "ip": ip,
                "ts": base + timedelta(seconds=t),
                "uri": uri,
                "status": 200,
                "bytes": 1024,
                "agent": "Mozilla/5.0",
            }
            for ip, t, uri in ip_ts_uri
        ]

    def test_single_session(self):
        records = self._make_records([
            ("1.2.3.4", 0,  "/"),
            ("1.2.3.4", 30, "/catalog"),
            ("1.2.3.4", 90, "/item"),
        ])
        sessions = identify_sessions(records, timeout_sec=1800, min_req=2)
        assert len(sessions) == 1
        assert sessions[0]["request_count"] == 3

    def test_split_by_timeout(self):
        records = self._make_records([
            ("1.2.3.4", 0,    "/"),
            ("1.2.3.4", 60,   "/catalog"),
            ("1.2.3.4", 5000, "/news"),
            ("1.2.3.4", 5060, "/about"),
        ])
        sessions = identify_sessions(records, timeout_sec=1800)
        assert len(sessions) == 2

    def test_multiple_ips(self):
        records = self._make_records([
            ("1.1.1.1", 0,  "/"),
            ("1.1.1.1", 30, "/a"),
            ("2.2.2.2", 0,  "/"),
            ("2.2.2.2", 30, "/b"),
        ])
        sessions = identify_sessions(records, timeout_sec=1800)
        assert len(sessions) == 2

    def test_min_req_filter(self):
        records = self._make_records([
            ("1.1.1.1", 0, "/"),
            ("2.2.2.2", 0, "/"),
            ("2.2.2.2", 30, "/catalog"),
        ])
        sessions = identify_sessions(records, timeout_sec=1800, min_req=2)
        assert len(sessions) == 1

    def test_session_fields(self):
        records = self._make_records([
            ("1.1.1.1", 0,   "/"),
            ("1.1.1.1", 60,  "/catalog"),
            ("1.1.1.1", 120, "/item"),
        ])
        sessions = identify_sessions(records, timeout_sec=1800)
        s = sessions[0]
        assert s["request_count"] == 3
        assert s["duration_sec"] == 120
        assert s["unique_pages"] == 3
        assert s["error_rate"] == 0.0
        assert len(s["page_sequence"]) == 3
        assert len(s["page_set"]) == 3

    def test_ip_anonymized(self):
        records = self._make_records([
            ("192.168.0.1", 0,  "/"),
            ("192.168.0.1", 30, "/a"),
        ])
        sessions = identify_sessions(records)
        assert sessions[0]["ip_hash"] != "192.168.0.1"
        assert len(sessions[0]["ip_hash"]) == 12


# ═══════════════════════════════════════════════════════════════════════════
# ПОЛНЫЙ ЦИКЛ ПРЕДОБРАБОТКИ: preprocess
# ═══════════════════════════════════════════════════════════════════════════

class TestPreprocess:

    @pytest.fixture
    def log_file(self, tmp_path):
        f = tmp_path / "access.log"
        f.write_text(CLF_SAMPLE)
        return str(f)

    def test_returns_dataframe(self, log_file):
        df, stats = preprocess(log_file)
        assert isinstance(df, pd.DataFrame)
        assert "session_id" in df.columns
        assert "request_count" in df.columns

    def test_stats_keys(self, log_file):
        _, stats = preprocess(log_file)
        for key in ("total", "ok", "sessions"):
            assert key in stats

    def test_sessions_positive(self, log_file):
        df, stats = preprocess(log_file)
        assert stats["sessions"] > 0
        assert len(df) == stats["sessions"]


# ═══════════════════════════════════════════════════════════════════════════
# КЛАСТЕРИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════

def _make_cluster_df(n=60) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)

    cluster_a = pd.DataFrame({
        "session_id": [f"a_{i}" for i in range(n // 3)],
        "request_count": rng.integers(1, 3, n // 3).astype(float),
        "duration_sec":  rng.integers(10, 60, n // 3).astype(float),
        "unique_pages":  rng.integers(1, 3, n // 3).astype(float),
        "error_rate":    rng.uniform(0, 0.05, n // 3),
        "avg_bytes":     rng.uniform(500, 2000, n // 3),
        "page_set": [["/"] for _ in range(n // 3)],
    })
    cluster_b = pd.DataFrame({
        "session_id": [f"b_{i}" for i in range(n // 3)],
        "request_count": rng.integers(5, 15, n // 3).astype(float),
        "duration_sec":  rng.integers(200, 600, n // 3).astype(float),
        "unique_pages":  rng.integers(5, 10, n // 3).astype(float),
        "error_rate":    rng.uniform(0, 0.1, n // 3),
        "avg_bytes":     rng.uniform(2000, 5000, n // 3),
        "page_set": [["/catalog", "/products/{id}"] for _ in range(n // 3)],
    })
    cluster_c = pd.DataFrame({
        "session_id": [f"c_{i}" for i in range(n // 3)],
        "request_count": rng.integers(20, 50, n // 3).astype(float),
        "duration_sec":  rng.integers(1000, 3600, n // 3).astype(float),
        "unique_pages":  rng.integers(15, 30, n // 3).astype(float),
        "error_rate":    rng.uniform(0, 0.2, n // 3),
        "avg_bytes":     rng.uniform(1000, 3000, n // 3),
        "page_set": [["/news", "/catalog", "/checkout"] for _ in range(n // 3)],
    })
    return pd.concat([cluster_a, cluster_b, cluster_c], ignore_index=True)


class TestClustering:

    @pytest.fixture
    def df(self):
        return _make_cluster_df(60)

    def test_returns_dataframe_with_cluster_id(self, df):
        result, meta = cluster_sessions(df)
        assert "cluster_id" in result.columns
        assert result["cluster_id"].notna().all()

    def test_k_detected_correctly(self, df):
        result, meta = cluster_sessions(df, k_min=2, k_max=8)
        assert 2 <= meta["k"] <= 5

    def test_explicit_k(self, df):
        result, meta = cluster_sessions(df, k=4)
        assert meta["k"] == 4
        assert result["cluster_id"].nunique() == 4

    def test_all_rows_labeled(self, df):
        result, _ = cluster_sessions(df)
        assert len(result) == len(df)
        assert (result["cluster_id"] >= 0).all()

    def test_meta_structure(self, df):
        _, meta = cluster_sessions(df)
        assert "k" in meta
        assert "clusters" in meta
        assert len(meta["clusters"]) == meta["k"]
        for cid, info in meta["clusters"].items():
            assert "size" in info
            assert "pct" in info
            assert "top_pages" in info

    def test_missing_column_raises(self, df):
        df_bad = df.drop(columns=["request_count"])
        with pytest.raises(ValueError, match="Отсутствуют колонки"):
            cluster_sessions(df_bad)

    def test_empty_dataframe(self):
        result, meta = cluster_sessions(pd.DataFrame())
        assert result.empty
        assert meta["k"] == 0

    def test_summary_string(self, df):
        _, meta = cluster_sessions(df)
        summary = cluster_summary(meta)
        assert "Кластеризация" in summary
        assert "Кластер 0" in summary


# ═══════════════════════════════════════════════════════════════════════════
# ИНТЕГРАЦИОННЫЙ ТЕСТ
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Полный цикл: лог → DataFrame → кластеры."""

    MULTI_SESSION_LOG = """\
10.0.0.1 - - [01/Jun/2024:10:00:00] "GET / HTTP/1.1" 200 1024 "-" "Mozilla/5.0"
10.0.0.1 - - [01/Jun/2024:10:00:30] "GET /catalog HTTP/1.1" 200 2048 "-" "Mozilla/5.0"
10.0.0.1 - - [01/Jun/2024:10:01:00] "GET /products/1 HTTP/1.1" 200 4096 "-" "Mozilla/5.0"
10.0.0.2 - - [01/Jun/2024:10:00:00] "GET / HTTP/1.1" 200 1024 "-" "Mozilla/5.0"
10.0.0.2 - - [01/Jun/2024:10:00:10] "GET /news HTTP/1.1" 200 1024 "-" "Mozilla/5.0"
10.0.0.3 - - [01/Jun/2024:10:00:00] "GET / HTTP/1.1" 200 1024 "-" "Mozilla/5.0"
10.0.0.3 - - [01/Jun/2024:10:05:00] "GET /about HTTP/1.1" 200 1024 "-" "Mozilla/5.0"
""" * 10

    def test_full_pipeline(self, tmp_path):
        log_path = str(tmp_path / "test.log")
        with open(log_path, "w") as f:
            f.write(self.MULTI_SESSION_LOG)

        df, stats = preprocess(log_path)
        assert stats["sessions"] > 0

        df_clustered, meta = cluster_sessions(df, k=2)
        assert "cluster_id" in df_clustered.columns
        assert meta["k"] == 2

        summary = cluster_summary(meta)
        assert len(summary) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])