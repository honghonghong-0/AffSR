"""
compute_idm.py
==============
Category-based IDM (Interest Drift Measurement) computation

IDURL paper formula:
  IDM = 1 - |cat(target) ∩ ∪cat(seq)| / |cat(target)|

  - repeat item: if target already appears in seq, IDM=0
  - empty category: if target category is empty, skip user

Output:
  {user_idx: [idm_1, idm_2, ...]} (chronological order)
  - Stores all IDM values per user as a list
  - Designed for leave-one-out split during training
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_category_idm(
    data_dir: str,
    output_path: str = None,
    exclude_cats: set = None,
):
    """
    Category-based IDM computation (IDURL style).

    Args:
        data_dir      : Preprocessed data directory (requires item_cats.json, sequences.pkl)
        output_path   : IDM save path (None -> data_dir/idm.pkl)
        exclude_cats  : Category ID set to exclude (default: {0, 1} — padding, dataset name)

    Returns:
        idm_dict : {user_idx: [idm_1, idm_2, ...]}
    """
    data_dir = Path(data_dir)
    if output_path is None:
        output_path = data_dir / "idm.pkl"
    else:
        output_path = Path(output_path)

    if exclude_cats is None:
        exclude_cats = {0, 1}  # padding, dataset name categories

    # ── Load data ─────────────────────────────────────────────────
    print(f"[IDM] Loading: {data_dir}")

    # item_cats.json: {item_idx: [cat_id_1, cat_id_2, ...]}
    with open(data_dir / "item_cats.json") as f:
        item_cats_raw = json.load(f)

    # Convert to int keys
    item_cats = {}
    for k, v in item_cats_raw.items():
        try:
            item_idx = int(k)
            cats = set(v) - exclude_cats if isinstance(v, list) else set()
            item_cats[item_idx] = cats
        except (ValueError, TypeError):
            continue

    # sequences.pkl: {user_idx: [(item_idx, timestamp, v, a), ...]}
    with open(data_dir / "sequences.pkl", "rb") as f:
        sequences = pickle.load(f)

    print(f"[IDM] Items: {len(item_cats):,} | Users: {len(sequences):,}")

    # ── Compute IDM ───────────────────────────────────────────────
    print("[IDM] Computing...")
    idm_dict = {}
    n_skip = 0

    for user_idx, seq in tqdm(sequences.items(), desc="IDM", mininterval=1):
        seq = list(seq)  # generator → list
        if len(seq) < 2:
            continue

        idm_list = []
        for i in range(1, len(seq)):
            # target: seq[i]
            # seq_before: seq[0:i]
            target_item = seq[i][0]
            seq_before_items = [s[0] for s in seq[:i]]

            # target categories
            target_cats = item_cats.get(target_item, set())
            if len(target_cats) == 0:
                # Skip if target has no categories
                n_skip += 1
                continue

            # Check for repeat item
            if target_item in seq_before_items:
                idm_list.append(0.0)
                continue

            # Union of seq categories
            seq_cats = set()
            for item in seq_before_items:
                seq_cats.update(item_cats.get(item, set()))

            # IDM = 1 - |intersection| / |target|
            intersection = target_cats & seq_cats
            idm = 1.0 - len(intersection) / len(target_cats)
            idm_list.append(float(idm))

        if len(idm_list) > 0:
            idm_dict[user_idx] = idm_list

    print(f"[IDM] Valid users: {len(idm_dict):,} | Skipped: {n_skip:,}")

    # ── Statistics ─────────────────────────────────────────────────────
    all_idm = []
    for idm_list in idm_dict.values():
        all_idm.extend(idm_list)

    if len(all_idm) > 0:
        print(f"[IDM] Statistics:")
        print(f"      mean: {np.mean(all_idm):.4f}")
        print(f"      std:  {np.std(all_idm):.4f}")
        print(f"      min:  {np.min(all_idm):.4f}")
        print(f"      max:  {np.max(all_idm):.4f}")
        print(f"      IDM=0: {sum(1 for x in all_idm if x < 0.01)}/{len(all_idm)} "
              f"({sum(1 for x in all_idm if x < 0.01)/len(all_idm)*100:.1f}%)")
        print(f"      IDM=1: {sum(1 for x in all_idm if x > 0.99)}/{len(all_idm)} "
              f"({sum(1 for x in all_idm if x > 0.99)/len(all_idm)*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────
    with open(output_path, "wb") as f:
        pickle.dump(idm_dict, f)

    print(f"[IDM] Saved: {output_path}")
    return idm_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/movies_tv_2021_2023",
        help="Preprocessed data directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="IDM save path (None -> data_dir/idm.pkl)",
    )
    args = parser.parse_args()

    compute_category_idm(args.data_dir, args.output_path)

