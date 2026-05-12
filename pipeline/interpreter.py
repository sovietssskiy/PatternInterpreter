import json, os, time
from typing import Dict, List, Optional
import pandas as pd

SYSTEM_PROMPT = """Ты — UX-аналитик и специалист по тестированию веб-приложений.
    Анализируй паттерны поведения пользователей веб-сайта по данным логов
    и формируй рекомендации для специалистов по тестированию.

    Правила:
    1. Классифицируй паттерн: «типичный» или «нетипичный».
    2. Описывай поведение кратко, без технического жаргона.
    3. В recommendations — конкретные сценарии проверки (что воспроизвести и проверить).
    4. Если проблем нет — problems: [].
    5. Язык ответа: русский.
    6. Отвечай строго в формате JSON по схеме."""

SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["типичный", "нетипичный"]},
        "confidence": {"type": "string", "enum": ["высокая", "средняя", "низкая"]},
        "description": {"type": "string"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["classification", "confidence", "description", "problems", "recommendations"],
    "additionalProperties": False,
}


def _build_prompt(
        cluster_id: int,
        stats: Dict,
        patterns: List[Dict],
        anomaly_info: Dict,
) -> str:

    flag = "ВЫСОКАЯ ДОЛЯ АНОМАЛЬНЫХ СЕССИЙ В КЛАСТЕРЕ\n" \
        if anomaly_info.get("priority_pct", 0) > 20 else ""

    pat_lines = "\n".join(
        f"  {p['support_pct']:.0f}%  {p['pattern_str']}"
        for p in patterns[:8]
    ) or "  паттернов не обнаружено"

    anom_lines = (
        f"  Аномальных сессий: {anomaly_info.get('anomaly_pct', 0):.1f}%\n"
        f"  Высокоприоритетных: {anomaly_info.get('priority_pct', 0):.1f}%"
    )

    return (
        f"Кластер {cluster_id}\n{flag}\n"
        f"Статистика:\n{json.dumps(stats, ensure_ascii=False, indent=2)}\n\n"
        f"Навигационные паттерны (PrefixSpan):\n{pat_lines}\n\n"
        f"Маркировка аномалий:\n{anom_lines}\n\n"
        f"Проанализируй и заполни JSON."
    )


def _call(prompt: str, api_key: str, retries: int = 3) -> Dict:
    try:
        from openai import OpenAI, RateLimitError
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=api_key)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pattern_interpretation",
                        "schema": SCHEMA,
                        "strict": True,
                    },
                },
                temperature=0.3,
                max_tokens=600,
            )
            return json.loads(resp.choices[0].message.content)
        except RateLimitError:
            time.sleep(2 ** attempt * 2)
        except Exception as e:
            return {"error": str(e)}
    return {"error": "API недоступен"}


def interpret_all(
        df: pd.DataFrame,
        patterns: List[Dict],
        api_key: str,
        max_clusters: int = 8,
        progress_cb=None,
) -> Dict[int, Dict]:
    from pipeline.pattern_mining import patterns_for_cluster

    results: Dict[int, Dict] = {}
    if "cluster_id" not in df.columns:
        return results

    cluster_ids = sorted(df["cluster_id"].unique())[:max_clusters]

    for i, cid in enumerate(cluster_ids):
        if progress_cb:
            progress_cb(i, len(cluster_ids), f"Кластер {cid}")

        cdf = df[df["cluster_id"] == cid]

        stats = {
            "cluster_id": int(cid),
            "size": len(cdf),
            "pct": round(len(cdf) / len(df) * 100, 1),
            "med_requests": float(cdf["request_count"].median()),
            "med_duration": float(cdf["duration_sec"].median()),
            "med_pages": float(cdf["unique_pages"].median()),
            "med_errors": float(cdf["error_rate"].median()),
        }

        cluster_patterns = patterns_for_cluster(cdf, patterns)

        anomaly_info = {"anomaly_pct": 0.0, "priority_pct": 0.0}
        if "anomaly_flag" in cdf.columns:
            n = len(cdf)
            anomaly_info = {
                "anomaly_pct": round(cdf["anomaly_flag"].sum() / n * 100, 1),
                "priority_pct": round(cdf["anomaly_priority"].sum() / n * 100, 1)
                if "anomaly_priority" in cdf.columns else 0.0,
            }

        prompt = _build_prompt(cid, stats, cluster_patterns, anomaly_info)

        try:
            results[int(cid)] = _call(prompt, api_key)
        except Exception as e:
            results[int(cid)] = {"error": str(e)}

    return results


def format_results(results: Dict[int, Dict]) -> str:
    if not results:
        return "Нет результатов."
    lines = []
    for cid, r in results.items():
        lines.append(f"{'═' * 52}")
        lines.append(f"КЛАСТЕР {cid}")
        if "error" in r:
            lines.append(f"  {r['error']}")
            continue
        cls = r.get("classification", "—")
        conf = r.get("confidence", "—")
        icon = "✅" if cls == "типичный" else "⚠️"
        lines.append(f"  {icon} {cls}  (уверенность: {conf})")
        lines.append(f"  {r.get('description', '')}")
        for prob in r.get("problems", []):
            lines.append(f"  ⚠  {prob}")
        for rec in r.get("recommendations", []):
            lines.append(f"  →  {rec}")
    return "\n".join(lines)
