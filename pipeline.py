import os
import numpy as np
from tqdm import tqdm

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
os.environ['MKL_THREADING_LAYER'] = os.environ.get('MKL_THREADING_LAYER', 'GNU')

from config import EXP_DIR, LOG_PATH, DATASET_PATHS, K
from io_utils import TeeLogger, dataset_tag, append_jsonl, load_jsonl, fmt_dt, to_jsonable_list
from metrics import empty_metrics, summarize_metrics, print_summary
from TimelyQABench_datasets import prepare_dataset_collections
from retrieval import RetrieverBundle, evaluate_2stage


def run_pipeline(dataset_paths = None):
    if dataset_paths is None:
        dataset_paths = DATASET_PATHS

    # start tee logging
    tee = TeeLogger(LOG_PATH)
    import sys
    sys.stdout = tee

    print("Loading BGE-M3 model...")

    all_results = {}
    macro_accumulator = {}

    def _accumulate(method_label: str, summary: dict):
        if method_label not in macro_accumulator:
            macro_accumulator[method_label] = empty_metrics()
        for m in ["hit", "recall", "precision", "ndcg", "map"]:
            macro_accumulator[method_label][m].append(float(summary.get(m, 0.0)))

    for dataset_path in dataset_paths:
        tag = dataset_tag(dataset_path)
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_path}")

        items = load_jsonl(dataset_path)
        print(f"  probe -> items:{len(items)}")

        (
            all_docs,
            all_gids,
            gid2idx,
            all_queries,
            qet_arr, qit_arr,
            det_arr, dit_arr,
        ) = prepare_dataset_collections(items)

        print(f"Total queries: {len(all_queries)}, Total docs: {len(all_docs)}")
        if len(all_docs) == 0 or len(all_queries) == 0:
            zero = {m: 0.0 for m in ["hit", "recall", "precision", "ndcg", "map"]}
            all_results[dataset_path] = {
                "BM25": zero,
                "BGE": zero,
                "BM25+TimeRerank(auto)": zero,
                "BM25+TimeRerank(best)": zero,
                "Dense+TimeRerank(auto)": zero,
                "Dense+TimeRerank(best)": zero,
            }
            continue

        # attach QIT/QET to query dicts for downstream use
        for i, q in enumerate(all_queries):
            q["qit"] = qit_arr[i]
            q["qet"] = qet_arr[i]

        # Build retrievers/embeddings once
        bundle = RetrieverBundle(all_docs, device="cuda")

        # ── Baselines ───────────────────────────────────────────────────────────
        print("\n=== BM25 (baseline) ===")
        m_bm25 = empty_metrics()
        for q in tqdm(all_queries):
            q_text = q["query"]
            pos_ids = q["positive_ids"]
            scores = bundle.bm25_scores(q_text)
            topk_idx = np.argsort(scores)[::-1][:K]
            topk_ids = [all_gids[i] for i in topk_idx]
            if pos_ids:
                from .metrics import evaluate_ranking_ids, add_metrics
                add_metrics(m_bm25, evaluate_ranking_ids(topk_ids, pos_ids, K))
        summary_bm25 = summarize_metrics(m_bm25)
        print_summary(K, "BM25 (baseline)", summary_bm25)

        print("\n=== BGE-M3 Dense (baseline) ===")
        m_bge = empty_metrics()
        for q in tqdm(all_queries):
            q_text = q["query"]
            pos_ids = q["positive_ids"]
            sim = bundle.dense_scores(q_text)
            topk_idx = np.argsort(sim)[::-1][:K]
            topk_ids = [all_gids[i] for i in topk_idx]
            if pos_ids:
                from .metrics import evaluate_ranking_ids, add_metrics
                add_metrics(m_bge, evaluate_ranking_ids(topk_ids, pos_ids, K))
        summary_bge = summarize_metrics(m_bge)
        print_summary(K, "BGE-M3 Dense (baseline)", summary_bge)

        # ── 2-Stage: Text Top-N → Time Rerank ──────────────────────────────────
        print("\n=== BM25 + TimeRerank (auto-α vs best-α) ===")
        bm25_auto, bm25_best, alpha_hist_auto, alpha_hist_best = evaluate_2stage(
            method_name="bm25_timererank",
            base_scores_fn=bundle.bm25_scores,
            all_queries=all_queries,
            all_gids=all_gids,
            det_arr=det_arr,
            dit_arr=dit_arr,
        )
        summ_auto = summarize_metrics(bm25_auto)
        print_summary(K, "BM25 + TimeRerank (auto-α)", summ_auto)
        summ_best = summarize_metrics(bm25_best)
        print_summary(K, "BM25 + TimeRerank (best-α)", summ_best)

        # Persist alpha histograms
        total_auto = sum(alpha_hist_auto.values())
        mean_auto = (sum(a * c for a, c in alpha_hist_auto.items()) / total_auto) if total_auto > 0 else 0.0
        append_jsonl(os.path.join(EXP_DIR, f"{tag}__alpha_distribution_top{K}.jsonl"), {
            "dataset": tag,
            "phase": "bm25_timererank_auto",
            "alpha_hist": {str(a): int(c) for a, c in alpha_hist_auto.items()},
            "mean_alpha": mean_auto,
            "ts": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        total_best = sum(alpha_hist_best.values())
        mean_best = (sum(a * c for a, c in alpha_hist_best.items()) / total_best) if total_best > 0 else 0.0
        append_jsonl(os.path.join(EXP_DIR, f"{tag}__alpha_distribution_top{K}.jsonl"), {
            "dataset": tag,
            "phase": "bm25_timererank_best",
            "alpha_hist": {str(a): int(c) for a, c in alpha_hist_best.items()},
            "mean_alpha": mean_best,
            "ts": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        print("\n=== Dense + TimeRerank (auto-α vs best-α) ===")
        dense_auto, dense_best, d_alpha_hist_auto, d_alpha_hist_best = evaluate_2stage(
            method_name="dense_timererank",
            base_scores_fn=bundle.dense_scores,
            all_queries=all_queries,
            all_gids=all_gids,
            det_arr=det_arr,
            dit_arr=dit_arr,
        )
        d_summ_auto = summarize_metrics(dense_auto)
        print_summary(K, "Dense + TimeRerank (auto-α)", d_summ_auto)
        d_summ_best = summarize_metrics(dense_best)
        print_summary(K, "Dense + TimeRerank (best-α)", d_summ_best)

        # Persist alpha histograms
        total_auto = sum(d_alpha_hist_auto.values())
        mean_auto = (sum(a * c for a, c in d_alpha_hist_auto.items()) / total_auto) if total_auto > 0 else 0.0
        append_jsonl(os.path.join(EXP_DIR, f"{tag}__alpha_distribution_top{K}.jsonl"), {
            "dataset": tag,
            "phase": "dense_timererank_auto",
            "alpha_hist": {str(a): int(c) for a, c in d_alpha_hist_auto.items()},
            "mean_alpha": mean_auto,
            "ts": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        total_best = sum(d_alpha_hist_best.values())
        mean_best = (sum(a * c for a, c in d_alpha_hist_best.items()) / total_best) if total_best > 0 else 0.0
        append_jsonl(os.path.join(EXP_DIR, f"{tag}__alpha_distribution_top{K}.jsonl"), {
            "dataset": tag,
            "phase": "dense_timererank_best",
            "alpha_hist": {str(a): int(c) for a, c in d_alpha_hist_best.items()},
            "mean_alpha": mean_best,
            "ts": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Aggregate per-dataset summaries
        all_results[dataset_path] = {
            "BM25": summary_bm25,
            "BGE": summary_bge,
            "BM25+TimeRerank(auto)": summ_auto,
            "BM25+TimeRerank(best)": summ_best,
            "Dense+TimeRerank(auto)": d_summ_auto,
            "Dense+TimeRerank(best)": d_summ_best,
        }

        for method, summary in all_results[dataset_path].items():
            _accumulate(method, summary)

    print("\n" + "=" * 80)
    print("Per-dataset results (summary dict):")
    for path, res in all_results.items():
        print(f"\n{path}")
        for method, summ in res.items():
            print(f"  {method}: {summ}")

    print("\n" + "=" * 80)
    print("Macro averages across datasets:")
    for method_label, metric_lists in macro_accumulator.items():
        macro = {m: float(np.mean(vals)) if len(vals) else 0.0 for m, vals in metric_lists.items()}
        from .metrics import print_summary as _ps
        _ps(K, f"Macro Avg - {method_label}", macro)

    print(f"\n✅ Log saved at {LOG_PATH}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="TimelyRAG Retrieval + TimeRerank Runner")
    parser.add_argument("--exp-dir", type=str, default=None, help="Override experiment output directory.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU ID for CUDA_VISIBLE_DEVICES.")
    args = parser.parse_args()

    if args.exp_dir:
        os.environ["TR_EXPERIMENT_DIR"] = args.exp_dir
        print(f"[INFO] TR_EXPERIMENT_DIR set to: {args.exp_dir}")

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"[INFO] CUDA_VISIBLE_DEVICES set to: {args.gpu}")

    os.environ["MKL_THREADING_LAYER"] = os.environ.get("MKL_THREADING_LAYER", "GNU")

    run_pipeline()