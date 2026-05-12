import hashlib, re
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.url_normalizer import normalize_url

CLF_RE = re.compile(
    r'(?P<host>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<time>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<uri>\S+)\s+\S+"\s+'
    r'(?P<status>\d{3})\s+(?P<bytes>\S+)'
    r'(?:\s+"(?P<referer>[^"]*)")?'
    r'(?:\s+"(?P<agent>[^"]*)")?'
)
STATIC = frozenset([
    ".jpg",".jpeg",".png",".gif",".ico",".bmp",".webp",".svg",
    ".css",".js",".woff",".woff2",".ttf",".eot",
    ".pdf",".zip",".gz",".mp3",".mp4",".map",
])
BOT_RE = re.compile(r"bot|crawl|spider|slurp|wget|curl|python-requests|go-http", re.I)
TS_FMTS = [
    "%d/%b/%Y:%H:%M:%S %z", "%d/%b/%Y:%H:%M:%S",
    "%A, %d-%b-%y %H:%M:%S %Z", "%d-%b-%y %H:%M:%S %Z",
    "%A, %d-%b-%y %H:%M:%S",   "%d-%b-%y %H:%M:%S",
]

def _hash(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:12]

def _parse_ts(raw: str):
    raw = re.sub(r"\.\d+", "", raw.strip())
    for fmt in TS_FMTS:
        try: return datetime.strptime(raw, fmt).replace(tzinfo=None)
        except ValueError: pass
    stripped = re.sub(r"\s+[A-Z]{2,5}$", "", raw)
    if stripped != raw:
        for fmt in TS_FMTS:
            try: return datetime.strptime(stripped, fmt).replace(tzinfo=None)
            except ValueError: pass
    return None

def parse_log_file(filepath: str) -> Tuple[List[Dict], Dict]:
    stats = {"total":0,"errors":0,"static":0,"bots":0,"ok":0}
    records = []
    with open(filepath, "r", errors="replace", encoding="utf-8") as f:
        for line in f:
            stats["total"] += 1
            line = line.strip()
            if not line or line.startswith("#"): continue
            m = CLF_RE.match(line)
            if not m: stats["errors"] += 1; continue
            uri = m.group("uri")
            if not uri or uri == "-": stats["errors"] += 1; continue
            if any(uri.split("?")[0].lower().endswith(e) for e in STATIC):
                stats["static"] += 1; continue
            agent = m.group("agent") or ""
            if agent and BOT_RE.search(agent): stats["bots"] += 1; continue
            ts = _parse_ts(m.group("time"))
            if ts is None: stats["errors"] += 1; continue
            try: bv = int(m.group("bytes"))
            except: bv = 0
            records.append({"ip":m.group("host"),"ts":ts,"uri":normalize_url(uri),
                            "status":int(m.group("status")),"bytes":bv,"agent":agent})
            stats["ok"] += 1
    return records, stats

def identify_sessions(records: List[Dict], timeout_sec:int=1800, min_req:int=2) -> List[Dict]:
    if not records: return []
    records = sorted(records, key=lambda r:(r["ip"],r["ts"]))
    active, sessions = {}, []

    def _close(ip, data):
        reqs = data["reqs"]
        if len(reqs) < min_req: return None
        uris   = [r["uri"] for r in reqs]
        tss    = [(r["ts"]-reqs[0]["ts"]).total_seconds() for r in reqs]
        errors = sum(1 for r in reqs if r["status"]>=400)
        dur    = (reqs[-1]["ts"]-reqs[0]["ts"]).total_seconds()
        return {
            "session_id":    f"{_hash(ip)}_{data['seq']}",
            "ip_hash":       _hash(ip),
            "duration_sec":  dur,
            "request_count": len(reqs),
            "unique_pages":  len(set(uris)),
            "error_rate":    errors/len(reqs),
            "avg_bytes":     sum(r["bytes"] for r in reqs)/len(reqs),
            "page_sequence": uris,
            "page_timestamps": tss,
            "page_set":      list(set(uris)),
        }

    for rec in records:
        ip   = rec["ip"]
        data = active.setdefault(ip, {"reqs":[],"seq":0})
        if data["reqs"]:
            gap = (rec["ts"]-data["reqs"][-1]["ts"]).total_seconds()
            if gap > timeout_sec:
                s = _close(ip, data)
                if s: sessions.append(s)
                active[ip] = {"reqs":[],"seq":data["seq"]+1}
                data = active[ip]
        data["reqs"].append(rec)

    for ip, data in active.items():
        if data["reqs"]:
            s = _close(ip, data)
            if s: sessions.append(s)
    return sessions

def preprocess(filepath:str, timeout_sec:int=1800, min_req:int=2) -> Tuple[pd.DataFrame, Dict]:
    records, stats = parse_log_file(filepath)
    sessions = identify_sessions(records, timeout_sec, min_req)
    df = pd.DataFrame(sessions) if sessions else pd.DataFrame()
    stats.update({
        "sessions": len(sessions),
        "unique_ips": df["ip_hash"].nunique() if not df.empty else 0,
        "median_reqs": float(df["request_count"].median()) if not df.empty else 0,
        "median_dur":  float(df["duration_sec"].median())  if not df.empty else 0,
    })
    return df, stats
