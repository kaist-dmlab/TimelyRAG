import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel

from config import K, STAGE1_TOPN, ALPHA_GRID
from metrics import empty_metrics, add_metrics, evaluate_ranking_ids
from time_utils import infer_gran_from_query_text, infer_gran_from_docs, combine_granularity, stabilize_gran
from time_utils import compute_gap_avg
from time_scores import compute_single_delta, minmax_normalize, time_affinity_from_delta, build_delta_penalty, per_candidate_margin_weights
from alpha import choose_alpha_from_signals, compute_time_gate


class RetrieverBundle:
    """Holds BM25 and BGE-M3 encoders/embeddings for reuse."""
    def __init__(self, docs: list[str], device: str = "cuda"):
        self.docs = docs
        self.tokenized_docs = [d.split() for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)
        self.doc_embeddings = self.model.encode(
            docs,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]

    def bm25_scores(self, query: str):
        return self.bm25.get_scores(query.split())

    def dense_scores(self, query: str):
        q_emb = self.model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"][0]
        return cosine_similarity([q_emb], self.doc_embeddings)[0]


def rerank_with_alpha_grid(b_norm, time_aff, gate, w_margin, cand_idx, all_gids, pos_ids, alpha_grid):
    best_alpha = None
    best_ndcg = -1.0
    best_gids = None

    for a in alpha_grid:
        w = (1.0 - a) * gate * w_margin
        valid_aff = np.isfinite(time_aff) & (time_aff > 0.0)
        w = np.where(valid_aff, w, 0.0)
        final = (1.0 - w) * b_norm + w * time_aff
        reranked = cand_idx[np.argsort(final)[::-1]][:K]
        reranked_gids = [all_gids[i] for i in reranked]
        if pos_ids:
            _, _, _, ndcg, _ = evaluate_ranking_ids(reranked_gids, pos_ids, K)
        else:
            ndcg = 0.0
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_alpha = a
            best_gids = reranked_gids

    return best_alpha, float(best_ndcg), best_gids


def evaluate_2stage(
    method_name: str,
    base_scores_fn,
    all_queries,
    all_gids,
    det_arr,
    dit_arr,
    alpha_grid=ALPHA_GRID,
):
    m_auto = empty_metrics()
    m_best = empty_metrics()
    alpha_hist_auto = {a: 0 for a in alpha_grid}
    alpha_hist_best = {a: 0 for a in alpha_grid}

    for q_idx, qobj in tqdm(list(enumerate(all_queries)), total=len(all_queries)):
        q_text = qobj["query"]
        pos_ids = qobj["positive_ids"]

        base_scores = base_scores_fn(q_text)
        cand_idx = np.argsort(base_scores)[::-1][:min(STAGE1_TOPN, len(base_scores))]
        base_topk_idx_all = np.argsort(base_scores)[::-1][:K]
        base_topk_gids_all = [all_gids[i] for i in base_topk_idx_all]

        # Granularity
        qg = infer_gran_from_query_text(q_text)
        dg = infer_gran_from_docs(dit_arr, det_arr, cand_idx, threshold=0.7)
        gran = stabilize_gran(combine_granularity(qg, dg), qg, dg)

        # Query times
        qit = qobj.get("qit")  # optional if caller populated
        qet = qobj.get("qet")
        if qit is None or qet is None:
            # if missing, pull from annotation arrays if provided by caller
            pass

        # Time distances and affinity
        delta = compute_single_delta(qit, qet, dit_arr, det_arr, cand_idx, gran)
        delta_norm = minmax_normalize(delta)
        time_aff = time_affinity_from_delta(delta_norm, gamma=2.0)
        delta_pen = build_delta_penalty(delta_norm, cand_idx)

        # GapAvg
        gap_avg = compute_gap_avg(cand_idx, det_arr, dit_arr, gran)

        # Normalize base scores
        b = base_scores[cand_idx].astype(float)
        b = (b - np.min(b)) / (np.ptp(b) + 1e-9)

        # Availability & signals
        def compute_valid_ratios_for_terms(qit, qet, dit_arr, det_arr, cand_idx):
            n = len(cand_idx)
            def ratio(mask):
                return float(np.sum(mask)) / max(1, n)

            has_qit = qit is not None
            has_qet = qet is not None

            m_main, m_qit_dit, m_qit_det, m_qet_dit = [], [], [], []
            for di in cand_idx:
                dit = dit_arr[di]
                det = det_arr[di]
                det_star = det if det is not None else dit
                qet_star = qet if qet is not None else qit

                m_main.append((qet_star is not None) and (det_star is not None))
                m_qit_dit.append(has_qit and (dit is not None))
                m_qit_det.append(has_qit and (det is not None))
                m_qet_dit.append(has_qet and (dit is not None))

            return {
                "QET_DET": ratio(np.array(m_main, dtype=bool)),
                "QIT_DIT": ratio(np.array(m_qit_dit, dtype=bool)),
                "QIT_DET": ratio(np.array(m_qit_det, dtype=bool)),
                "QET_DIT": ratio(np.array(m_qet_dit, dtype=bool)),
            }

        valid_ratio_terms = compute_valid_ratios_for_terms(qit, qet, dit_arr, det_arr, cand_idx)
        signals = {
            "qet_exists": float(qet is not None),
            "gran_id": float({"year": 0, "month": 1, "day": 2, "hour": 3}.get(gran, 1)),
            "gap_avg": float(gap_avg),
        }

        # Gate & per-candidate weights
        gate = compute_time_gate(signals, valid_ratio_terms, delta_pen)
        w_margin = per_candidate_margin_weights(b, power=1.0)
        valid_aff = np.isfinite(time_aff) & (time_aff > 0.0)

        # Auto-α
        alpha_auto, alpha_dbg = choose_alpha_from_signals(
            qet_exists=signals["qet_exists"],
            gran_id=signals["gran_id"],
            gap_avg=signals["gap_avg"],
            valid_ratio_terms=valid_ratio_terms,
            base_scores_norm=b,
            delta_pen=delta_pen,
            mix_finite_ratio=float(np.mean(np.isfinite(delta_norm))),
            alpha_grid=alpha_grid,
        )
        alpha_hist_auto[alpha_auto] += 1

        w_auto = (1.0 - alpha_auto) * gate * w_margin
        w_auto = np.where(valid_aff, w_auto, 0.0)
        final_auto = (1.0 - w_auto) * b + w_auto * time_aff
        reranked_auto = cand_idx[np.argsort(final_auto)[::-1]][:K]
        reranked_auto_gids = [all_gids[i] for i in reranked_auto]
        if pos_ids:
            add_metrics(m_auto, evaluate_ranking_ids(reranked_auto_gids, pos_ids, K))

        # Best-α (grid search on NDCG@K)
        best_alpha, best_ndcg, best_gids = rerank_with_alpha_grid(
            b_norm=b, time_aff=time_aff, gate=gate, w_margin=w_margin,
            cand_idx=cand_idx, all_gids=all_gids, pos_ids=pos_ids, alpha_grid=alpha_grid,
        )
        if best_alpha is None:
            best_alpha = 0.0
            best_gids = reranked_auto_gids
        alpha_hist_best[best_alpha] += 1
        if pos_ids:
            add_metrics(m_best, evaluate_ranking_ids(best_gids, pos_ids, K))

    return m_auto, m_best, alpha_hist_auto, alpha_hist_best
