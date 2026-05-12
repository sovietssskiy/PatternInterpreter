import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from typing import Dict

FEATURES = ["request_count", "duration_sec", "unique_pages", "error_rate"]


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    if df.empty:
        return df
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Нет колонок: {missing}")

    X = df[FEATURES].fillna(0).values
    result = df.copy()

    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    result["if_flag"]  = iso.fit_predict(X) == -1
    result["if_score"] = -iso.score_samples(X)

    z = np.abs(scipy_stats.zscore(X, nan_policy="omit"))
    result["z_flag"] = np.nan_to_num(z, nan=0.0).max(axis=1) > z_threshold

    result["anomaly_flag"]     = result["if_flag"] | result["z_flag"]
    result["anomaly_priority"] = result["if_flag"] & result["z_flag"]
    return result


def anomaly_report(df: pd.DataFrame) -> str:
    if "anomaly_flag" not in df.columns:
        return "Маркировка не выполнялась."
    n, nf, np_ = len(df), int(df["anomaly_flag"].sum()), int(df["anomaly_priority"].sum())
    return (
        f"Обнаружение аномалий\n{'─'*34}\n"
        f"Всего сессий:         {n}\n"
        f"Аномальных (≥1):      {nf} ({round(nf/n*100,1) if n else 0}%)\n"
        f"Приоритетных (оба):   {np_} ({round(np_/n*100,1) if n else 0}%)\n"
        f"  Isolation Forest:   {int(df['if_flag'].sum())}\n"
        f"  Z-score (|z|>3):    {int(df['z_flag'].sum())}\n"
    )
