"""
preprocessing/convert_to_recbole.py
=====================================
CDS / Movies 데이터를 RecBole(IDURL) 포맷으로 변환

소스: sequences.pkl (VA 필터링 완료된 데이터)
     → AffSR 학습에 사용된 것과 동일한 유저/아이템 집합 보장

출력:
  references/IDURL-main/IDURL-main/dataset/
  ├── cds/
  │   ├── cds.inter    (user_id:token  item_id:token  rating:float  timestamp:float)
  │   └── cds.item     (item_id:token  categories:token_seq)
  └── movies/
      ├── movies.inter
      └── movies.item

사용법:
  python preprocessing/convert_to_recbole.py
"""

import json
import pickle
from pathlib import Path

import pandas as pd

IDURL_DATASET_DIR = Path("references/IDURL-main/IDURL-main/dataset")


def build_recbole_data(processed_dir: str, out_dir: Path, name: str):
    """
    sequences.pkl 기반으로 RecBole .inter / .item 생성.
    sequences.pkl: {user_idx: [(item_idx, timestamp, v, a, has_va), ...]}
    """
    processed_dir = Path(processed_dir)

    # ── 역방향 맵 구성 ──────────────────────────────────────────────
    with open(processed_dir / "user_map.json") as f:
        user_map = json.load(f)          # user_id_str → idx
    idx2user = {v: k for k, v in user_map.items()}

    with open(processed_dir / "item_map.json") as f:
        item_map = json.load(f)          # asin → idx
    idx2item = {v: k for k, v in item_map.items()}

    with open(processed_dir / "item_cats.json") as f:
        item_cats = json.load(f)         # str(idx) → [cat, ...]

    # ── sequences.pkl 로드 ──────────────────────────────────────────
    with open(processed_dir / "sequences.pkl", "rb") as f:
        sequences = pickle.load(f)       # {user_idx: [(item_idx, ts, v, a, ...), ...]}

    # ── .inter 행 생성 ───────────────────────────────────────────────
    rows = []
    for user_idx, seq in sequences.items():
        user_str = idx2user.get(user_idx)
        if user_str is None:
            continue
        for entry in seq:
            item_idx  = entry[0]
            timestamp = entry[1]
            asin = idx2item.get(item_idx)
            if asin is None:
                continue
            rows.append({
                "user_id:token":    user_str,
                "item_id:token":    asin,
                "rating:float":     1.0,
                "timestamp:float":  float(timestamp),
            })

    inter_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    inter_path = out_dir / f"{name}.inter"
    inter_df.to_csv(inter_path, sep="\t", index=False)
    print(f"  .inter → {inter_path}  ({len(inter_df):,}행, {inter_df['user_id:token'].nunique():,}명)")

    # ── .item 생성 ────────────────────────────────────────────────────
    item_rows = []
    for idx, asin in idx2item.items():
        cats = item_cats.get(str(idx), [])
        item_rows.append({
            "item_id:token":       asin,
            "categories:token_seq": ", ".join(cats) if cats else "",
        })
    item_df = pd.DataFrame(item_rows)
    item_path = out_dir / f"{name}.item"
    item_df.to_csv(item_path, sep="\t", index=False)
    no_cat = (item_df["categories:token_seq"] == "").sum()
    print(f"  .item  → {item_path}  ({len(item_df):,}개 아이템, 카테고리 없음: {no_cat}개)")


# ─────────────────────────────────────────────────────────────────────────────
# CDS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("CDS 변환  (VA-filtered sequences.pkl 기반)")
print("=" * 55)
build_recbole_data(
    processed_dir="data/processed/cds_v10",
    out_dir=IDURL_DATASET_DIR / "cds",
    name="cds",
)

# ─────────────────────────────────────────────────────────────────────────────
# Movies
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Movies 변환  (VA-filtered sequences.pkl 기반)")
print("=" * 55)
build_recbole_data(
    processed_dir="data/processed/movies_v10",
    out_dir=IDURL_DATASET_DIR / "movies",
    name="movies",
)

print("\nDone ✓")