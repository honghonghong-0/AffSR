"""
convert_idm.py
==============
Convert idm.pkl format:
  old: {user_idx: [idm_1, idm_2, ...]}
  new: {(user_idx, target_idx): idm_float}

To avoid index misalignment caused by the skip logic in compute_idm.py,
we recompute from sequences.pkl + item_cats.json rather than enumerating the list.
The resulting values are identical to compute_idm.py.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def convert(data_dir: Path, exclude_cats: set | None = None):
    if exclude_cats is None:
        exclude_cats = {0, 1}

    with open(data_dir / "sequences.pkl", "rb") as f:
        sequences = pickle.load(f)
    with open(data_dir / "item_cats.json") as f:
        item_cats_raw = json.load(f)

    item_cats: dict[int, set] = {}
    for k, v in item_cats_raw.items():
        try:
            idx = int(k)
            cats = set(v) - exclude_cats if isinstance(v, list) else set()
            item_cats[idx] = cats
        except (ValueError, TypeError):
            continue

    new_idm: dict[tuple[int, int], float] = {}
    n_empty_max_drift = 0

    for user_idx, seq in sequences.items():
        seq = list(seq)
        if len(seq) < 2:
            continue

        for i in range(1, len(seq)):
            target_item = seq[i][0]
            seq_before = [s[0] for s in seq[:i]]

            target_cats = item_cats.get(target_item, set())
            # IDURL convention: unknown target category → maximum drift (completely new interest)
            if len(target_cats) == 0:
                new_idm[(user_idx, target_item)] = 1.0
                n_empty_max_drift += 1
                continue

            if target_item in seq_before:
                new_idm[(user_idx, target_item)] = 0.0
                continue

            seq_cats = set()
            for it in seq_before:
                seq_cats.update(item_cats.get(it, set()))

            inter = target_cats & seq_cats
            idm = 1.0 - len(inter) / len(target_cats)
            new_idm[(user_idx, target_item)] = float(idm)

    with open(data_dir / "idm.pkl", "wb") as f:
        pickle.dump(new_idm, f)

    vals = list(new_idm.values())
    print(f"[convert_idm] Saved: {data_dir / 'idm.pkl'}")
    print(f"  (user, item) pairs : {len(new_idm):,}")
    print(f"  empty cat → IDM=1  : {n_empty_max_drift:,}")
    print(f"  IDM mean/std       : {np.mean(vals):.4f} / {np.std(vals):.4f}")
    print(f"  IDM min/max        : {min(vals):.4f} / {max(vals):.4f}")
    print(f"  IDM=0 ratio        : {sum(1 for v in vals if v < 0.01) / len(vals) * 100:.1f}%")
    print(f"  IDM=1 ratio        : {sum(1 for v in vals if v > 0.99) / len(vals) * 100:.1f}%")
    print()
    print(f"[sample 10 entries]")
    for k, v in list(new_idm.items())[:10]:
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str,
        default="data/processed/movies_tv_2021_2023",
    )
    args = parser.parse_args()
    convert(Path(args.data_dir))