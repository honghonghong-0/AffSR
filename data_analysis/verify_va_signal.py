"""
Verification D: Does VA space carry a recommendation signal?

Method:
  Use the mean of each user's sequence VA as r_u,
  then score items by distance to their VA (pure VA-based recommendation).

  score(u, v) = -‖mean(VA_seq_u) - VA(v)‖₂

  Measure R@10, N@10, R@20, N@20 on Valid/Test set.

Reference points:
  - Random baseline: R@10 ≈ 10/15024 = 0.000666
  - SASRec baseline test: R@10 = 0.0555
  - VA-only must clearly outperform random to support the AffSR assumption
"""

import math
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def va_only_eval(data_dir: Path, split: str = "valid"):
    # ── Data loading ───────────────────────────────────────────────
    with open(data_dir / "item_va.json") as f:
        item_va_raw = json.load(f)
    item_va = {}
    for k, v in item_va_raw.items():
        if not k.isdigit():
            continue  # skip __meta__ etc.
        val = v["va"] if isinstance(v, dict) and "va" in v else v
        item_va[int(k)] = np.array(val, dtype=np.float32)
    max_item_idx = max(item_va.keys())

    # all_item_va matrix
    all_va = np.zeros((max_item_idx + 1, 2), dtype=np.float32)
    for k, v in item_va.items():
        all_va[k] = v
    num_items = max_item_idx + 1
    print(f"[Data] Items: {num_items:,}")

    # train sequences (exclude valid/test targets)
    with open(data_dir / "splits" / "train.pkl", "rb") as f:
        sequences = pickle.load(f)

    # split
    with open(data_dir / f"splits/{split}.pkl", "rb") as f:
        split_data = pickle.load(f)
    print(f"[Data] {split} samples: {len(split_data):,} users")

    # ── Evaluation ────────────────────────────────────────────────
    recalls = {10: [], 20: [], 100: []}
    ndcgs = {10: [], 20: [], 100: []}
    target_ranks = []
    skipped = 0

    for user_idx, target_tuple in tqdm(split_data.items(), desc=f"{split}"):
        if user_idx not in sequences:
            skipped += 1
            continue

        seq = sequences[user_idx]
        target_idx = target_tuple[0]

        # mean VA of sequence
        seq_vas = np.array([all_va[s[0]] for s in seq if 0 < s[0] <= max_item_idx])
        if len(seq_vas) == 0:
            skipped += 1
            continue
        r_u = seq_vas.mean(axis=0)  # (2,)

        # distance to all items
        dists = np.linalg.norm(all_va - r_u[None, :], axis=1)  # (N,)
        scores = -dists

        # mask seen items
        seen = set(s[0] for s in seq)
        seen.add(0)  # padding
        for s in seen:
            if 0 <= s <= max_item_idx:
                scores[s] = -np.inf

        # ranking
        sorted_idx = np.argsort(-scores)
        try:
            rank = int(np.where(sorted_idx == target_idx)[0][0])
        except IndexError:
            skipped += 1
            continue
        target_ranks.append(rank)

        for k in [10, 20, 100]:
            hit = 1 if rank < k else 0
            recalls[k].append(hit)
            if hit:
                ndcgs[k].append(1.0 / math.log2(rank + 2))
            else:
                ndcgs[k].append(0.0)

    # ── Results ───────────────────────────────────────────────────
    print(f"\n[{split} results] (n={len(target_ranks):,}, skipped={skipped:,})")
    print(f"  Target rank mean   : {np.mean(target_ranks):.0f}")
    print(f"  Target rank median : {np.median(target_ranks):.0f}")
    print(f"  Target rank p10/p90: {np.percentile(target_ranks, 10):.0f} / {np.percentile(target_ranks, 90):.0f}")
    print(f"\n  Recall@10  : {np.mean(recalls[10]):.4f}  (random={10/num_items:.4f})")
    print(f"  NDCG@10    : {np.mean(ndcgs[10]):.4f}")
    print(f"  Recall@20  : {np.mean(recalls[20]):.4f}  (random={20/num_items:.4f})")
    print(f"  NDCG@20    : {np.mean(ndcgs[20]):.4f}")
    print(f"  Recall@100 : {np.mean(recalls[100]):.4f}  (random={100/num_items:.4f})")
    print(f"  NDCG@100   : {np.mean(ndcgs[100]):.4f}")

    # Improvement over random
    rand_r10 = 10 / num_items
    actual_r10 = np.mean(recalls[10])
    print(f"\n  R@10 vs random : {actual_r10/rand_r10:.2f}×")

    return {
        "R@10": np.mean(recalls[10]),
        "N@10": np.mean(ndcgs[10]),
        "R@20": np.mean(recalls[20]),
        "N@20": np.mean(ndcgs[20]),
        "rank_median": np.median(target_ranks),
        "num_items": num_items,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/movies_tv_2021_2023")
    parser.add_argument("--split", default="valid", choices=["valid", "test"])
    args = parser.parse_args()

    print(f"=== Verification D: VA-only recommendation (score = -‖mean(VA_seq) - VA(item)‖) ===\n")
    va_only_eval(Path(args.data_dir), args.split)