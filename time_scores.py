import numpy as np
from time_utils import unit_diff
from config import MISS_MARGIN

# ── Time distance & affinity helpers ───────────────────────────────────────────

def build_delta_penalty(delta_raw: np.ndarray, cand_idx: np.ndarray, missing_margin: float = MISS_MARGIN) -> np.ndarray:
    x = delta_raw.copy()
    n = len(x)
    pen = np.zeros(n, dtype=float)

    finite_mask = np.isfinite(x)
    if finite_mask.any():
        order = np.argsort(x[finite_mask])
        ranks = np.empty(np.sum(finite_mask), dtype=int)
        ranks[order] = np.arange(np.sum(finite_mask))
        if ranks.size > 1:
            pen[finite_mask] = ranks / (ranks.size - 1)
        else:
            pen[finite_mask] = 0.0
    else:
        pen[:] = 1.0 + missing_margin
        return pen

    pen[~finite_mask] = 1.0 + missing_margin
    tiny = 1e-6 * ((cand_idx % 997) / 997.0)
    pen = np.clip(pen + tiny, 0.0, 1.0 + missing_margin)
    return pen


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    y = x.astype(float).copy()
    finite = np.isfinite(y)
    if not finite.any():
        return y
    lo, hi = np.nanmin(y[finite]), np.nanmax(y[finite])
    if hi > lo:
        y[finite] = (y[finite] - lo) / (hi - lo)
    else:
        y[finite] = 0.0
    return y


def time_affinity_from_delta(delta_norm: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    a = np.zeros_like(delta_norm, dtype=float)
    finite = np.isfinite(delta_norm)
    if finite.any():
        z = np.clip(delta_norm[finite], 0.0, 1.0)
        a[finite] = np.exp(-gamma * z)
    a[~finite] = 0.5  # Unknown → neutral
    return a


def compute_single_delta(qit, qet, dit_arr, det_arr, cand_idx, gran):
    out = np.full(len(cand_idx), np.nan, dtype=float)
    qet_star = qet if qet is not None else qit

    for j, di in enumerate(cand_idx):
        dit = dit_arr[di]
        det = det_arr[di]

        # causality: DIT must be before QIT
        if (qit is not None) and (dit is not None) and (dit >= qit):
            out[j] = np.nan
            continue

        det_star = det if det is not None else dit

        acc = 0.0
        used_main = False

        # main
        d_main = unit_diff(qet_star, det_star, gran) if (qet_star is not None and det_star is not None) else None
        if d_main is not None and np.isfinite(d_main):
            acc += 1.0 * d_main
            used_main = True

        # aux1: |QIT - DIT|
        if qit is not None and dit is not None:
            d = unit_diff(qit, dit, gran)
            if d is not None and np.isfinite(d):
                acc += 0.1 * d

        # aux2: |QIT - DET|
        if qit is not None and det is not None:
            d = unit_diff(qit, det, gran)
            if d is not None and np.isfinite(d):
                acc += 0.1 * d

        # aux3: |QET - DIT|
        if qet is not None and dit is not None:
            d = unit_diff(qet, dit, gran)
            if d is not None and np.isfinite(d):
                acc += 0.1 * d

        out[j] = acc if used_main else np.nan

    return out


def per_candidate_margin_weights(b_norm: np.ndarray, power: float = 1.0) -> np.ndarray:
    bmin, bmax = float(np.min(b_norm)), float(np.max(b_norm))
    denom = max(1e-6, bmax - bmin)
    margin = (bmax - b_norm) / denom
    w = 1.0 - margin
    if power != 1.0:
        w = np.power(np.clip(w, 0.0, 1.0), power)
    return np.clip(w, 0.0, 1.0)