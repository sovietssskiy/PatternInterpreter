from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class Session:
    session_id:    str
    ip_hash:       str
    start_dt:      datetime
    end_dt:        datetime
    duration_sec:  float
    request_count: int
    unique_pages:  int
    error_rate:    float
    avg_bytes:     float
    page_sequence: List[str]
    page_set:      List[str]
    user_agent:    str = ""
    anomaly_flag:  bool = False
    anomaly_score: float = 0.0
    cluster_id:    int = -1

    def to_dict(self) -> dict:
        return {
            "session_id":    self.session_id,
            "ip_hash":       self.ip_hash,
            "start_dt":      self.start_dt.isoformat(),
            "end_dt":        self.end_dt.isoformat(),
            "duration_sec":  self.duration_sec,
            "request_count": self.request_count,
            "unique_pages":  self.unique_pages,
            "error_rate":    round(self.error_rate, 4),
            "avg_bytes":     round(self.avg_bytes, 2),
            "page_sequence": self.page_sequence,
            "page_set":      self.page_set,
            "user_agent":    self.user_agent,
            "anomaly_flag":  self.anomaly_flag,
            "anomaly_score": round(self.anomaly_score, 4),
            "cluster_id":    self.cluster_id,
        }