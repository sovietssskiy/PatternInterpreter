import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES: List[str] = [
    "request_count",
    "duration_sec",
    "unique_pages",
    "error_rate",
    "avg_bytes",
]

MINIBATCH_THRESHOLD = 50_000


def find_optimal_k(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 12,
) -> Tuple[int, List[float]]:
    k_range = range(k_min, min(k_max + 1, len(X_scaled)))
    if len(k_range) < 2:
        return k_min, []

    inertias: List[float] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    d1 = np.diff(inertias)
    d2 = np.diff(d1)

    if len(d2) == 0:
        return k_min, inertias

    elbow_idx = int(np.argmax(d2))
    optimal_k = list(k_range)[elbow_idx + 2]  # +2: компенсация двух diff()

    return optimal_k, inertias


def cluster_sessions(
    df: pd.DataFrame,
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 12,
    features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:

    if df.empty:
        return df, {"k": 0, "inertias": [], "clusters": {}}

    feat = features or CLUSTER_FEATURES

    missing = [f for f in feat if f not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки для кластеризации: {missing}")

    X_raw = df[feat].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    inertias: List[float] = []
    if k is None:
        k, inertias = find_optimal_k(X_scaled, k_min=k_min, k_max=k_max)

    if len(X_scaled) > MINIBATCH_THRESHOLD:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            batch_size=min(1024, len(X_scaled) // 10),
        )
    else:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            max_iter=300,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = model.fit_predict(X_scaled)

    df_result = df.copy()
    df_result["cluster_id"] = labels

    clusters: Dict[int, Dict] = {}
    for cid in range(k):
        mask = df_result["cluster_id"] == cid
        cdf = df_result[mask]

        all_pages: List[str] = []
        for pages in cdf["page_set"]:
            if isinstance(pages, list):
                all_pages.extend(pages)

        page_counts: Dict[str, int] = {}
        for pg in all_pages:
            page_counts[pg] = page_counts.get(pg, 0) + 1
        top_pages = sorted(page_counts, key=lambda x: -page_counts[x])[:5]

        clusters[cid] = {
            "size":              int(mask.sum()),
            "pct":               round(mask.sum() / len(df_result) * 100, 1),
            "median_requests":   float(cdf["request_count"].median()),
            "median_duration":   float(cdf["duration_sec"].median()),
            "median_pages":      float(cdf["unique_pages"].median()),
            "median_error_rate": float(cdf["error_rate"].median()),
            "median_bytes":      float(cdf["avg_bytes"].median()),
            "top_pages":         top_pages,
        }

    meta = {
        "k":        k,
        "inertias": inertias,
        "clusters": clusters,
        "features": feat,
    }

    return df_result, meta


def cluster_summary(meta: Dict) -> str:

    lines = [f"Кластеризация: k = {meta['k']} кластеров\n"]
    for cid, info in meta["clusters"].items():
        lines.append(
            f"Кластер {cid}  ({info['size']} сессий, {info['pct']}%)\n"
            f"  Медиана запросов:     {info['median_requests']:.0f}\n"
            f"  Медиана длит. (сек):  {info['median_duration']:.0f}\n"
            f"  Медиана страниц:      {info['median_pages']:.0f}\n"
            f"  Медиана доли ошибок:  {info['median_error_rate']:.3f}\n"
            f"  Топ страниц:          {', '.join(info['top_pages']) or '—'}\n"
        )
    return "\n".join(lines)
