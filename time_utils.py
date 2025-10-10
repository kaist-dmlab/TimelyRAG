import re
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, List

GRAN_ORDER = {"hour": 0, "day": 1, "month": 2, "year": 3}
GRAN_LIST  = ["hour", "day", "month", "year"]


def to_dt(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def infer_gran_from_query_text(qtext: str) -> str:
    # Heuristics for Korean/ISO patterns
    if re.search(r"\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", qtext) or re.search(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일\s*\d{1,2}\s*시", qtext):
        return "hour"
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", qtext) or re.search(r"\d{1,2}\s*일", qtext):
        return "day"
    if re.search(r"\b\d{4}-\d{2}\b", qtext) or re.search(r"\d{1,2}\s*월", qtext):
        return "month"
    if re.search(r"\b(19|20)\d{2}\b", qtext) or re.search(r"(19|20)\d{2}\s*년", qtext):
        return "year"
    return "month"


def infer_gran_from_docs(dits, dets, cand_idx, threshold: float = 0.7) -> str:
    def gather_flags(arr_dt):
        hrs, days, mons = [], [], []
        for i in cand_idx:
            dt = arr_dt[i]
            if dt is None:
                continue
            hrs.append(int(not (dt.hour == 0 and dt.minute == 0 and dt.second == 0)))
            days.append(int(dt.day != 1))
            mons.append(int(dt.month != 1))
        a = lambda x: (np.mean(x) if x else 0.0)
        return a(hrs), a(days), a(mons)

    h_dit, d_dit, m_dit = gather_flags(dits)
    h_det, d_det, m_det = gather_flags(dets)
    if max(h_dit, h_det) >= threshold:
        return "hour"
    if max(d_dit, d_det) >= threshold:
        return "day"
    if max(m_dit, m_det) >= threshold:
        return "month"
    return "year"


def combine_granularity(qg: str, dg: str) -> str:
    return GRAN_LIST[min(GRAN_ORDER.get(qg, 2), GRAN_ORDER.get(dg, 2))]


def stabilize_gran(gran: str, qg: str, dg: str) -> str:
    gap = abs(GRAN_ORDER.get(qg, 2) - GRAN_ORDER.get(dg, 2))
    if gap >= 2:
        hi = max(GRAN_ORDER.get(qg, 2), GRAN_ORDER.get(dg, 2))
        return GRAN_LIST[hi - 1]
    return gran


def unit_diff(a: datetime | None, b: datetime | None, gran: str) -> float | None:
    if a is None or b is None:
        return None
    if gran == "year":
        return float(abs(a.year - b.year))
    if gran == "month":
        return float(abs((a.year - b.year) * 12 + (a.month - b.month)))
    if gran == "day":
        return float(abs((a - b).days))
    if gran == "hour":
        return float(abs((a - b).total_seconds()) / 3600.0)
    return float(abs((a - b).days))


def compute_gap_avg(cand_idx, det_arr, dit_arr, gran: str) -> float:
    # Sort by D* then average adjacent gaps in chosen units
    seq: List[datetime] = []
    for i in cand_idx:
        det = det_arr[i]
        dit = dit_arr[i]
        dstar = det if det is not None else dit
        if dstar is not None:
            seq.append(dstar)
    if len(seq) <= 1:
        return 0.0
    seq.sort()
    gaps = []
    for a, b in zip(seq[:-1], seq[1:]):
        g = unit_diff(a, b, gran)
        if g is not None:
            gaps.append(g)
    if not gaps:
        return 0.0
    return float(np.mean(gaps))


def to_int(x, default):
    try:
        return int(x)
    except Exception:
        return default


def extract_qet_from_query(qtext: str, qit_hint: datetime | None = None) -> Tuple[datetime | None, str | None]:
    # Parse time expressions from query; return (QET, granularity)
    import unicodedata
    if not qtext:
        return None, None
    text = unicodedata.normalize("NFKC", qtext)
    candidates = []

    def add_candidate(pos, gran, y, m=1, d=1, hh=0, mi=0):
        try:
            dt = datetime(to_int(y, 1970), to_int(m, 1), to_int(d, 1), to_int(hh, 0), to_int(mi, 0), 0)
            candidates.append((pos, gran, dt))
        except Exception:
            pass

    SUFFIX = r'(?:\s*(?:이후|부터|까지|전))?'

    # hour-level
    for m in re.finditer(rf'(?P<y>\d{{4}})[.\-/]\s*(?P<mo>\d{{1,2}})[.\-/]\s*(?P<d>\d{{1,2}})\s+(?P<h>\d{{1,2}}):(?P<mi>\d{{2}}){SUFFIX}', text):
        add_candidate(m.start(), "hour", m.group("y"), m.group("mo"), m.group("d"), m.group("h"), m.group("mi"))
    for m in re.finditer(rf'(?P<y>\d{{4}})\s*년\s*(?P<mo>\d{{1,2}})\s*월\s*(?P<d>\d{{1,2}})\s*일\s*(?P<h>\d{{1,2}})\s*시(?:\s*(?P<mi>\d{{1,2}})\s*분)?{SUFFIX}', text):
        add_candidate(m.start(), "hour", m.group("y"), m.group("mo"), m.group("d"), m.group("h"), m.group("mi") or 0)

    # day-level
    for m in re.finditer(rf'(?P<y>\d{{4}})[.\-/]\s*(?P<mo>\d{{1,2}})[.\-/]\s*(?P<d>\d{{1,2}})\b{SUFFIX}', text):
        add_candidate(m.start(), "day", m.group("y"), m.group("mo"), m.group("d"))
    for m in re.finditer(rf'(?P<y>\d{{4}})\s*년\s*(?P<mo>\d{{1,2}})\s*월\s*(?P<d>\d{{1,2}})\s*일{SUFFIX}', text):
        add_candidate(m.start(), "day", m.group("y"), m.group("mo"), m.group("d"))

    # month-level
    for m in re.finditer(rf'(?P<y>\d{{4}})[.\-/]\s*(?P<mo>\d{{1,2}})\b{SUFFIX}', text):
        add_candidate(m.start(), "month", m.group("y"), m.group("mo"))
    for m in re.finditer(rf'(?P<y>\d{{4}})\s*년\s*(?P<mo>\d{{1,2}})\s*월{SUFFIX}', text):
        add_candidate(m.start(), "month", m.group("y"), m.group("mo"))

    # year-level
    for m in re.finditer(rf'(?P<y>(?:19|20)\d{{2}})\s*년?{SUFFIX}', text):
        add_candidate(m.start(), "year", m.group("y"))

    # month/day without year; backfill from QIT hint
    for m in re.finditer(rf'(?P<mo>\d{{1,2}})\s*월\s*(?P<d>\d{{1,2}})\s*일{SUFFIX}', text):
        if qit_hint is not None:
            add_candidate(m.start(), "day", qit_hint.year, m.group("mo"), m.group("d"))

    if not candidates:
        return None, None

    gran_order = {"hour": 0, "day": 1, "month": 2, "year": 3}
    candidates.sort(key=lambda x: (gran_order[x[1]], x[0]))
    _, gran, dt = candidates[0]
    return dt, gran