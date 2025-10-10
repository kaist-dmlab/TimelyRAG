import numpy as np
from config import ALPHA_GRID, TIME_STD_MAX, TEXT_STD_MAX, GAP_MAX

# ── Alpha selection & gating ───────────────────────────────────────────────────

def _round_to_grid(x: float, grid: list[float]) -> float:
    return float(min(grid, key=lambda g: abs(g - x)))


def _normalize_std(x: float, max_ref: float) -> float:
    if max_ref <= 1e-9:
        return 0.0
    return float(np.clip(x / max_ref, 0.0, 1.0))


def choose_alpha_from_signals(
    qet_exists: float,
    gran_id: float,
    gap_avg: float,
    valid_ratio_terms: dict,
    base_scores_norm: np.ndarray,
    delta_pen: np.ndarray,
    mix_finite_ratio: float,
    alpha_grid: list[float] | None = None,
):
    if alpha_grid is None:
        alpha_grid = ALPHA_GRID

    # Lower text variance → rely more on time
    text_std = float(np.std(base_scores_norm))
    text_std_n = _normalize_std(text_std, TEXT_STD_MAX)

    # Higher delta penalty variance → better time separability
    with np.errstate(invalid="ignore"):
        dp_std = float(np.nanstd(delta_pen))
    dp_std_n = _normalize_std(dp_std, TIME_STD_MAX)

    # Term availability
    valid_vals = [valid_ratio_terms.get(k, 0.0) for k in ["QET_DET", "QIT_DIT", "QIT_DET", "QET_DIT"]]
    valid_mean = float(np.mean(valid_vals)) if valid_vals else 0.0

    # Gap normalization based on granularity
    gap_max = GAP_MAX.get({0: "hour", 1: "day", 2: "month", 3: "year"}[int(gran_id)], GAP_MAX["month"])
    gap_n = float(np.clip(gap_avg / max(1e-6, gap_max), 0.0, 1.0))

    # Time-importance score
    s_pos = (
        0.32 * gap_n +
        0.18 * qet_exists +
        0.08 * (gran_id / 3.0) +
        0.22 * dp_std_n +
        0.18 * valid_mean +
        0.02 * mix_finite_ratio
    )
    s_neg = 0.20 * text_std_n
    time_importance = float(np.clip(s_pos - s_neg, 0.0, 1.0))

    # α = (1 - time_importance)
    alpha_continuous = 1.0 - time_importance
    alpha = _round_to_grid(alpha_continuous, alpha_grid)

    dbg = {
        "text_std": text_std, "text_std_n": text_std_n,
        "delta_pen_std": dp_std, "delta_pen_std_n": dp_std_n,
        "valid_mean": valid_mean,
        "gap_avg": gap_avg, "gap_n": gap_n,
        "time_importance": time_importance,
        "alpha_continuous": alpha_continuous,
    }
    return alpha, dbg


def compute_time_gate(signals: dict, valid_ratio_terms: dict, delta_pen: np.ndarray) -> float:
    qet_exists = float(signals.get("qet_exists", 0.0))
    gran_id    = float(signals.get("gran_id", 1.0)) / 3.0
    valid_vals = [valid_ratio_terms.get(k, 0.0) for k in ["QET_DET", "QIT_DIT", "QIT_DET", "QET_DIT"]]
    valid_mean = float(np.mean(valid_vals)) if valid_vals else 0.0

    with np.errstate(invalid="ignore"):
        dp_std = float(np.nanstd(delta_pen))
    dp_std_n = np.clip(dp_std / TIME_STD_MAX, 0.0, 1.0)

    s = (
        0.35 * qet_exists +
        0.10 * gran_id +
        0.35 * valid_mean +
        0.20 * dp_std_n
    )

    if qet_exists < 0.5 and valid_mean < 0.4:
        return 0.0
    if dp_std_n < 0.10:
        s *= 0.25

    return float(np.clip(s, 0.0, 1.0))