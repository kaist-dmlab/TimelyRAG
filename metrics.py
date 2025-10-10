import numpy as np

# ── IR metrics ─────────────────────────────────────────────────────────────────

def compute_dcg(relevances, k: int) -> float:
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    return np.sum((2 ** relevances - 1) / np.log2(np.arange(2, relevances.size + 2)))


def compute_ndcg(relevances, k: int) -> float:
    dcg = compute_dcg(relevances, k)
    ideal_dcg = compute_dcg(sorted(relevances, reverse=True), k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_average_precision(relevances, k: int) -> float:
    relevances = np.asarray(relevances)[:k]
    score = 0.0
    num_hits = 0
    for i, rel in enumerate(relevances):
        if rel:
            num_hits += 1
            score += num_hits / (i + 1)
    denom = max(1, np.sum(relevances[:k]))
    return score / denom


def evaluate_ranking_ids(topk_ids, pos_ids, k: int):
    pos_set = set(pos_ids)
    relevances = [1 if gid in pos_set else 0 for gid in topk_ids[:k]]
    hit = int(any(relevances))
    recall = sum(relevances) / max(1, len(pos_set))
    precision = sum(relevances) / k
    ndcg = compute_ndcg(relevances, k)
    ap = compute_average_precision(relevances, k)
    return hit, recall, precision, ndcg, ap


def empty_metrics():
    return {"hit": [], "recall": [], "precision": [], "ndcg": [], "map": []}


def add_metrics(metrics, vals):
    for key, val in zip(metrics.keys(), vals):
        metrics[key].append(val)


def summarize_metrics(metrics):
    return {m: float(np.mean(vals)) if len(vals) else 0.0 for m, vals in metrics.items()}


def print_summary(K: int, title: str, summary: dict):
    print(f"\n[{title}]")
    print(f"Recall@{K}:   {summary['recall']:.4f}")
    print(f"Hit@{K}:      {summary['hit']:.4f}")
    print(f"Precision@{K}:{summary['precision']:.4f}")
    print(f"NDCG@{K}:     {summary['ndcg']:.4f}")
    print(f"MAP@{K}:      {summary['map']:.4f}")