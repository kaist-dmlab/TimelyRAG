import os
import json
import sys
from datetime import datetime

class TeeLogger:
    def __init__(self, logfile: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        self.log = open(logfile, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def dataset_tag(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def to_jsonable_list(arr):
    import numpy as np
    out = []
    for v in list(arr):
        try:
            if v is None:
                out.append(None)
            else:
                fv = float(v)
                out.append(None if not np.isfinite(fv) else fv)
        except Exception:
            out.append(None)
    return out


def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fmt_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, datetime) else None