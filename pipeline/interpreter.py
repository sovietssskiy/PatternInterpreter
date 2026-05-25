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
    4. Если если паттерн «типичный» — recommendations: [].
    5. Если проблем нет — problems: [].
    6. Язык ответа: русский.
    7. Отвечай строго в формате JSON по схеме."""

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


def _build_pattern_prompt(
        pattern: Dict,
        cluster_stats: List[Dict],
) -> str:
    """Промпт для анализа конкретного навигационного паттерна."""

    steps = pattern["pattern"]
    step_lines = "\n".join(
        f"  Шаг {i + 1}: {step}" for i, step in enumerate(steps)
    )

    clusters_block = ""
    if cluster_stats:
        rows = "\n".join(
            f"  Кластер {c['cluster_id']}: {c['size']} сессий "
            f"(med_requests={c['med_requests']}, med_errors={c['med_errors']:.2f})"
            for c in cluster_stats
        )
        clusters_block = f"\nКластеры, содержащие паттерн:\n{rows}\n"

    return (
        f"Навигационный паттерн (последовательность страниц):\n{step_lines}\n\n"
        f"Метрики паттерна:\n"
        f"  Встречается в {pattern['support_abs']} сессиях ({pattern['support_pct']:.1f}% от всех)\n"
        f"  Длина цепочки: {pattern['length']} шагов\n"
        f"{clusters_block}\n"
        f"Задача:\n"
        f"  1. Опиши, что делает пользователь на каждом шаге.\n"
        f"  2. Найди потенциальные UX-проблемы или неожиданные переходы.\n"
        f"  3. Сформулируй тест-кейсы — конкретно: с какой страницы, куда, что проверить.\n"
        f"\nЗаполни JSON."
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
        max_patterns: int = 10,
        progress_cb=None,
) -> Dict:
    """
    Возвращает словарь:
      {
        "clusters":  {cluster_id: interpretation, ...},
        "patterns":  [{"pattern": ..., "interpretation": ...}, ...],
      }
    """
    from pipeline.pattern_mining import patterns_for_cluster

    results: Dict = {"clusters": {}, "patterns": []}
    if "cluster_id" not in df.columns:
        return results

    cluster_ids = sorted(df["cluster_id"].unique())[:max_clusters]
    total_steps = len(cluster_ids) + min(max_patterns, len(patterns))
    step = 0

    # ── кластеры ──────────────────────────────────────────────
    cluster_stats_list: List[Dict] = []

    for i, cid in enumerate(cluster_ids):
        if progress_cb:
            progress_cb(step, total_steps, f"Кластер {cid}")
        step += 1

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
        cluster_stats_list.append(stats)

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
            results["clusters"][int(cid)] = _call(prompt, api_key)
        except Exception as e:
            results["clusters"][int(cid)] = {"error": str(e)}

    # ── паттерны ──────────────────────────────────────────────
    for pattern in patterns[:max_patterns]:
        if progress_cb:
            progress_cb(step, total_steps, f"Паттерн: {pattern['pattern_str']}")
        step += 1

        # передаём статистику только тех кластеров, где паттерн встречается
        relevant_clusters = _clusters_for_pattern(df, pattern, cluster_stats_list)
        prompt = _build_pattern_prompt(pattern, relevant_clusters)

        try:
            interpretation = _call(prompt, api_key)
        except Exception as e:
            interpretation = {"error": str(e)}

        results["patterns"].append({
            "pattern":        pattern["pattern"],
            "pattern_str":    pattern["pattern_str"],
            "support_abs":    pattern["support_abs"],
            "support_pct":    pattern["support_pct"],
            "length":         pattern["length"],
            "interpretation": interpretation,
        })

    return results


def _clusters_for_pattern(
        df: pd.DataFrame,
        pattern: Dict,
        cluster_stats: List[Dict],
) -> List[Dict]:
    """Возвращает статистику кластеров, в чьих сессиях встречается паттерн."""
    pat = pattern["pattern"]

    def _found(seq):
        if not isinstance(seq, list):
            return False
        it = iter(seq)
        return all(p in it for p in pat)

    matching_ids = set(
        df[df["page_sequence"].apply(_found)]["cluster_id"].unique()
    )
    return [c for c in cluster_stats if c["cluster_id"] in matching_ids]


def format_results(results: Dict) -> str:
    # обратная совместимость: старый код возвращал Dict[int, Dict]
    if results and isinstance(next(iter(results.values()), None), dict) \
            and "classification" in next(iter(results.values()), {}):
        clusters = results
        patterns_list = []
    else:
        clusters = results.get("clusters", {})
        patterns_list = results.get("patterns", [])

    if not clusters and not patterns_list:
        return "Нет результатов."

    lines = []

    # ── кластеры ──────────────────────────────────────────────
    for cid, r in clusters.items():
        lines.append(f"{'═' * 52}")
        lines.append(f"КЛАСТЕР {cid}")
        if "error" in r:
            lines.append(f"  Ошибка: {r['error']}")
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

    # ── паттерны ──────────────────────────────────────────────
    if patterns_list:
        lines.append(f"\n{'━' * 52}")
        lines.append("АНАЛИЗ НАВИГАЦИОННЫХ ПАТТЕРНОВ")
        lines.append(f"{'━' * 52}")

    for entry in patterns_list:
        lines.append(f"\n{'─' * 52}")
        lines.append(f"ПАТТЕРН  {entry['pattern_str']}")
        lines.append(
            f"  Встречается в {entry['support_abs']} сессиях"
            f" ({entry['support_pct']:.1f}%),  длина: {entry['length']}"
        )
        r = entry.get("interpretation", {})
        if "error" in r:
            lines.append(f"  Ошибка: {r['error']}")
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
