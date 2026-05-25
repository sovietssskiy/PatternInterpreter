from typing import Dict, List
import pandas as pd
from prefixspan import PrefixSpan


def _gap_filter(seq: List[str], ts: List[float], max_gap: float) -> List[str]:
    if len(seq) < 2 or len(seq) != len(ts):
        return seq
    best, cur = [], [seq[0]]
    for i in range(1, len(seq)):
        if ts[i] - ts[i-1] <= max_gap:
            cur.append(seq[i])
        else:
            if len(cur) > len(best): best = cur
            cur = [seq[i]]
    return cur if len(cur) > len(best) else best


def mine_patterns(
    df: pd.DataFrame,
    min_support_pct: float = 0.02,
    max_gap_sec: float = 300,
    top_k: int = 20,
) -> List[Dict]:
    has_ts = "page_timestamps" in df.columns
    db: List[List[str]] = []

    for _, row in df.iterrows():
        seq = row.get("page_sequence", [])
        if not isinstance(seq, list) or len(seq) < 2:
            continue
        if has_ts:
            tss = row.get("page_timestamps", [])
            if isinstance(tss, list) and len(tss) == len(seq):
                seq = _gap_filter(seq, tss, max_gap_sec)
        if len(seq) >= 2:
            db.append(seq)
    if len(db) < 2:
        return []
    min_sup = max(2, int(len(db) * min_support_pct))
    raw = PrefixSpan(db).topk(top_k, closed=True)

    return [
        {
            "pattern":     pat,
            "support_abs": sup,
            "support_pct": round(sup / len(db) * 100, 1),
            "length":      len(pat),
            "pattern_str": " → ".join(pat),
        }
        for sup, pat in sorted(raw, key=lambda x: -x[0])
        if sup >= min_sup
    ]


def patterns_for_cluster(
    cluster_df: pd.DataFrame,
    all_patterns: List[Dict],
) -> List[Dict]:
    seqs = [
        tuple(s) for s in cluster_df["page_sequence"]
        if isinstance(s, list)
    ]

    def _found(pat):
        for seq in seqs:
            it = iter(seq)
            if all(p in it for p in pat):
                return True
        return False

    return [p for p in all_patterns if _found(p["pattern"])][:8]


def patterns_summary(patterns: List[Dict], top_n: int = 10) -> str:
    if not patterns:
        return "Паттернов не обнаружено."
    lines = [
        f"Топ-{min(top_n, len(patterns))} навигационных паттернов:",
        f"  {'Поддержка':>12}   Паттерн",
        "  " + "─" * 56,
    ]
    for p in patterns[:top_n]:
        lines.append(f"  {p['support_abs']:>4} ({p['support_pct']:>5.1f}%)   {p['pattern_str']}")
    return "\n".join(lines)
