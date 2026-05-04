"""
convert_idm.py
==============
idm.pkl 포맷 변환:
  기존: {user_idx: [idm_1, idm_2, ...]}
  신규: {(user_idx, target_idx): idm_float}

compute_idm.py의 skip 로직으로 인한 인덱스 어긋남을 피하기 위해,
list를 enumerate하지 않고 sequences.pkl + item_cats.json에서 재계산한다.
결과 값은 compute_idm.py와 동일.
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
            # IDURL 규약: 타겟 카테고리 불명 → 최대 drift (완전히 새로운 관심)
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
    print(f"[convert_idm] 저장: {data_dir / 'idm.pkl'}")
    print(f"  (user, item) 쌍 수: {len(new_idm):,}")
    print(f"  빈 카테고리→IDM=1  : {n_empty_max_drift:,}")
    print(f"  IDM 평균/std       : {np.mean(vals):.4f} / {np.std(vals):.4f}")
    print(f"  IDM min/max        : {min(vals):.4f} / {max(vals):.4f}")
    print(f"  IDM=0 비율         : {sum(1 for v in vals if v < 0.01) / len(vals) * 100:.1f}%")
    print(f"  IDM=1 비율         : {sum(1 for v in vals if v > 0.99) / len(vals) * 100:.1f}%")
    print()
    print(f"[sample 10개]")
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