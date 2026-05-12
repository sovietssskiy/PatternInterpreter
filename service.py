"""
    → Gradio UI:  http://localhost:3000/ui
    → REST API:   http://localhost:3000/analyze
    → Swagger:    http://localhost:3000/docs
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import bentoml
import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.preprocessor  import preprocess
from pipeline.clustering     import cluster_sessions, cluster_summary
from pipeline.pattern_mining import mine_patterns, patterns_summary
from pipeline.anomaly        import detect_anomalies, anomaly_report
from pipeline.interpreter    import interpret_all, format_results

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def _run_pipeline(
    filepath: str,
    api_key: str,
    timeout_min: float = 30,
    min_support: float = 0.02,
    contamination: float = 0.05,
    progress=None,
) -> dict:

    def _prog(frac: float, desc: str):
        if progress is not None:
            progress(frac, desc=desc)

    # Предобработка
    _prog(0.05, "Предобработка логов...")
    df, stats = preprocess(filepath, timeout_sec=int(timeout_min * 60))
    if df.empty:
        raise ValueError("После фильтрации не осталось сессий. Проверьте формат файла.")

    # Сериализуем списки в JSON-строки для сохранения в CSV
    def _save_df(d: pd.DataFrame):
        out = d.copy()
        for col in ("page_sequence", "page_timestamps", "page_set"):
            if col in out.columns:
                out[col] = out[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x
                )
        out.to_csv(DATA_DIR / "sessions.csv", index=False)

    _save_df(df)

    #  Кластеризация
    _prog(0.25, "Кластеризация сессий (K-Means)...")
    df, cluster_meta = cluster_sessions(df)

    # Паттерны
    _prog(0.45, "Поиск паттернов (PrefixSpan)...")
    patterns = mine_patterns(df, min_support_pct=min_support)
    with open(DATA_DIR / "patterns.json", "w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)

    # Аномалии
    _prog(0.60, "Обнаружение аномалий (Isolation Forest + z-score)...")
    df = detect_anomalies(df, contamination=contamination)
    _save_df(df)

    # LLM-интерпретация
    def _prog_cb(i, total, label):
        _prog(0.65 + 0.30 * (i / max(total, 1)), f"LLM: {label}")

    _prog(0.65, "LLM-интерпретация кластеров...")
    interpretations = interpret_all(
        df=df,
        patterns=patterns,
        api_key=api_key.strip(),
        max_clusters=8,
        progress_cb=_prog_cb,
    )
    with open(DATA_DIR / "interpretations.json", "w", encoding="utf-8") as f:
        json.dump(interpretations, f, ensure_ascii=False, indent=2)

    _prog(1.0, "Готово!")
    return {
        "stats":          stats,
        "df":             df,
        "cluster_meta":   cluster_meta,
        "patterns":       patterns,
        "interpretations": interpretations,
    }



def _preproc_text(stats: dict, df: pd.DataFrame) -> str:
    lines = [
        "Предобработка",
        "─" * 36,
        f"Строк в файле:        {stats.get('total', '—')}",
        f"Успешно распознано:   {stats.get('ok', '—')}",
        f"Ошибки парсинга:      {stats.get('errors', '—')}",
        f"Статика (отфильтр.):  {stats.get('static', '—')}",
        f"Боты  (отфильтр.):    {stats.get('bots', '—')}",
        f"Итого сессий:         {stats.get('sessions', '—')}",
        f"Уникальных IP:        {stats.get('unique_ips', '—')}",
        f"Медиана запросов:     {stats.get('median_reqs', 0):.1f}",
        f"Медиана длит. (с):    {stats.get('median_dur', 0):.0f}",
        "",
        anomaly_report(df),
    ]
    return "\n".join(lines)


def _cluster_table(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    rows = []
    for cid, info in meta.get("clusters", {}).items():
        rows.append({
            "Кластер":          cid,
            "Сессий":           info["size"],
            "%":                info["pct"],
            #"Медиана запросов": info["med_reqs"],
            #"Медиана длит.(с)": info["med_dur"],
            #"Медиана страниц":  info["med_pages"],
            #"Топ страниц":      " | ".join(info.get("top_pages", [])[:3]),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _anomaly_table(df: pd.DataFrame) -> pd.DataFrame:
    if "anomaly_flag" not in df.columns:
        return pd.DataFrame()
    flag = df["anomaly_flag"]
    if flag.dtype == object:
        flag = flag.map(lambda x: str(x).strip().lower() == "true")
    adf = df[flag].copy()
    if adf.empty:
        return pd.DataFrame()
    if "if_score" in adf.columns:
        adf = adf.sort_values("if_score", ascending=False)
    cols = ["session_id", "request_count", "duration_sec",
            "unique_pages", "error_rate", "if_score",
            "anomaly_flag", "anomaly_priority"]
    available = [c for c in cols if c in adf.columns]
    result = adf[available].head(50).reset_index(drop=True)
    for col in ("anomaly_flag", "anomaly_priority"):
        if col in result.columns:
            result[col] = result[col].map(
                lambda x: "⚠️ да" if str(x).lower() in ("true", "1") else "нет"
            )
    return result

def _build_ui() -> gr.Blocks:

    def run_analysis(log_file, api_key, timeout_min, min_support, contamination, progress=gr.Progress()):
        if log_file is None:
            return ("Загрузите лог-файл",) + (None,) * 5
        if not api_key or len(api_key.strip()) < 10:
            return ("Введите OpenAI API ключ",) + (None,) * 5

        filepath = log_file.name if hasattr(log_file, "name") else str(log_file)
        try:
            result = _run_pipeline(
                filepath=filepath,
                api_key=api_key,
                timeout_min=timeout_min,
                min_support=min_support,
                contamination=contamination,
                progress=progress,
            )
        except Exception as exc:
            return (f"{exc}\n\n{traceback.format_exc()}",) + (None,) * 5

        df   = result["df"]
        meta = result["cluster_meta"]

        return (
            _preproc_text(result["stats"], df),
            _cluster_table(df, meta),
            cluster_summary(meta),
            patterns_summary(result["patterns"]),
            _anomaly_table(df),
            format_results(result["interpretations"]),
        )

    with gr.Blocks(title="WebLogAnalyzer") as demo:
        gr.Markdown(
            "PatternInterpreter\n"
        )

        with gr.Row():
            with gr.Column(scale=2):
                file_in = gr.File(label="Лог-файл (.log / .txt)", file_types=[".log", ".txt"])
            with gr.Column(scale=2):
                key_in = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")

        with gr.Accordion("Параметры", open=False):
            with gr.Row():
                timeout_sl = gr.Slider(10, 120, value=30, step=5,
                                       label="Тайм-аут сессии (мин)")
                minsup_sl  = gr.Slider(0.01, 0.15, value=0.02, step=0.01,
                                       label="Мин. поддержка паттернов")
                contam_sl  = gr.Slider(0.01, 0.20, value=0.05, step=0.01,
                                       label="Доля аномалий (contamination)")

        btn = gr.Button("Запустить анализ", variant="primary", size="lg")

        with gr.Tabs():
            with gr.Tab("Предобработка / Аномалии"):
                preproc_out = gr.Textbox(label="Статистика", lines=22, interactive=False)
            with gr.Tab("Кластеры"):
                cluster_tbl = gr.DataFrame(label="Кластеры", wrap=True)
                cluster_txt = gr.Textbox(label="Сводка", lines=14, interactive=False)
            with gr.Tab("Паттерны"):
                patterns_out = gr.Textbox(label="Паттерны PrefixSpan", lines=18, interactive=False)
            with gr.Tab("Аномальные сессии"):
                anomaly_tbl = gr.DataFrame(label="Топ-50 аномальных сессий", wrap=True)
            with gr.Tab("Рекомендации (LLM)"):
                recs_out = gr.Textbox(label="Интерпретации для тестировщиков",
                                      lines=30, interactive=False)
                gr.Markdown("_Результаты сохраняются в `data/interpretations.json`_")

        btn.click(
            fn=run_analysis,
            inputs=[file_in, key_in, timeout_sl, minsup_sl, contam_sl],
            outputs=[preproc_out, cluster_tbl, cluster_txt,
                     patterns_out, anomaly_tbl, recs_out],
        )

    return demo

_ui = _build_ui()

try:
    def _safe_get_api_info(self):
        return {"named_endpoints": {}, "unnamed_endpoints": {}}
    _ui.__class__.get_api_info = _safe_get_api_info
except Exception:
    pass


@bentoml.service(
    name="PatternInterpreter",
    resources={"cpu": "2"},
    traffic={"timeout": 600},
)
@bentoml.gradio.mount_gradio_app(_ui, path="/ui")
class WebLogAnalyzer:

    @bentoml.api
    def analyze(
        self,
        log_content: str,
        api_key: str,
        timeout_min: float = 30,
        min_support: float = 0.02,
        contamination: float = 0.05,
    ) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(log_content)
            tmp_path = tmp.name
        try:
            result = _run_pipeline(
                filepath=tmp_path,
                api_key=api_key,
                timeout_min=timeout_min,
                min_support=min_support,
                contamination=contamination,
            )
        finally:
            os.unlink(tmp_path)

        return json.dumps({
            "stats":           result["stats"],
            "cluster_meta":    result["cluster_meta"],
            "patterns":        result["patterns"],
            "interpretations": result["interpretations"],
        }, ensure_ascii=False)
