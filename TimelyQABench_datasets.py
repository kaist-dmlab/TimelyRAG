
from typing import Tuple, List, Dict
import numpy as np
from io_utils import load_jsonl
from time_utils import to_dt, extract_qet_from_query


def prepare_dataset_collections(items) -> Tuple[list, list, dict, list, list, list, np.ndarray, np.ndarray]:
    def pick_key(d, candidates):
        for k in candidates:
            if k in d and d[k] is not None:
                return k
        return None

    all_docs, all_queries = [], []
    all_gids = []
    auto_gid = 0

    # Collect global document texts + IDs
    for item in items:
        docs_key = pick_key(item, ["docs_ko", "docs"])
        docs = item.get(docs_key, []) if docs_key else []
        for d in docs:
            all_docs.append(d.get("text", ""))
            gid = d.get("global_doc_id")
            if gid is None:
                gid = f"auto_{auto_gid}"
                auto_gid += 1
            all_gids.append(gid)

    gid2idx = {gid: i for i, gid in enumerate(all_gids)}

    # Queries + positive global IDs
    for item in items:
        qkey = pick_key(item, ["query_ko", "query", "question_ko", "question"])
        if not qkey:
            continue
        q_text = item.get(qkey)
        if not q_text:
            continue

        pos_ids = []
        gkey = pick_key(item, ["gold_docs_final_global", "gold_global_ids", "gold_docs_global"])
        if gkey:
            for gi in item.get(gkey, []):
                if gi in gid2idx:
                    pos_ids.append(gi)
                    continue
                if str(gi) in gid2idx:
                    pos_ids.append(str(gi))
                    continue
                try:
                    gidx = int(gi)
                    if 0 <= gidx < len(all_gids):
                        pos_ids.append(all_gids[gidx])
                        continue
                except Exception:
                    pass

        if not pos_ids:
            lkey = pick_key(item, ["gold_docs_final", "gold_docs", "positive_doc_indices", "positives"])
            if lkey:
                docs_key = pick_key(item, ["docs_ko", "docs"])
                docs = item.get(docs_key, []) if docs_key else []
                for li in item.get(lkey, []):
                    try:
                        lidx = int(li)
                        if 0 <= lidx < len(docs):
                            local_doc = docs[lidx]
                            gid = local_doc.get("global_doc_id")
                            if gid is None:
                                try:
                                    gidx = all_docs.index(local_doc.get("text", ""))
                                    gid = all_gids[gidx]
                                except ValueError:
                                    continue
                            pos_ids.append(gid)
                    except Exception:
                        continue

        all_queries.append({
            "query": q_text,
            "positive_ids": sorted(set(pos_ids)),
            "raw_item": item,
        })

    qet_arr, qit_arr = [], []
    for q in all_queries:
        item = q["raw_item"]
        qit_dt = to_dt(item.get("query_insert_time", ""))
        qet_dt = to_dt(item.get("query_event_time", ""))
        if qet_dt is None:
            qet_dt, _ = extract_qet_from_query(q["query"], qit_hint=qit_dt)
        qit_arr.append(qit_dt)
        qet_arr.append(qet_dt)

    det_arr, dit_arr = [], []
    for item in items:
        docs_key = pick_key(item, ["docs_ko", "docs"])
        docs = item.get(docs_key, []) if docs_key else []
        for d in docs:
            det_arr.append(to_dt(d.get("event_time", "")))
            dit_arr.append(to_dt(d.get("doc_insert_time", "")))

    return (
        all_docs, all_gids, gid2idx,
        all_queries, qet_arr, qit_arr,
        np.array(det_arr, dtype=object), np.array(dit_arr, dtype=object)
    )