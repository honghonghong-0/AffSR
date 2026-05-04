"""
preprocess.py
=============
Movies_and_TV preprocessing Phase 1 — K-core filtering + metadata join → processed.csv

Usage examples:
  # Print statistics for K=5,10,20,50 only and exit
  python preprocessing/preprocess.py \
      --review_path data/raw/Movies_and_TV.jsonl \
      --meta_path   data/raw/meta_Movies_and_TV.jsonl \
      --output_dir  data/processed/movies_tv \
      --stats_only

  # Run preprocessing with K=20 → save processed.csv
  python preprocessing/preprocess.py \
      --review_path data/raw/Movies_and_TV.jsonl \
      --meta_path   data/raw/meta_Movies_and_TV.jsonl \
      --output_dir  data/processed/movies_tv \
      --K 20
"""

import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

EXCLUDE_CATS = {
    "Movies & TV", "Prime Video",
    "Featured Categories", "Genre for Featured Categories", "Amazon Video",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def load_reviews(review_path: str) -> pd.DataFrame:
    print("=" * 60)
    print("Step 1. Load review data")
    print("=" * 60)
    records, skip = [], 0
    with open(review_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reviews", mininterval=5):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skip += 1

    df = pd.DataFrame(records)
    print(f"Load complete: {len(df):,} records (parse failures: {skip:,})")

    print("\nStep 2. Remove missing values / empty text")
    before = len(df)
    df = df.dropna(subset=["user_id", "parent_asin", "text", "timestamp", "rating"])
    print(f"  Missing values removed: {before:,} -> {len(df):,} (-{before - len(df):,})")
    before = len(df)
    df = df[df["text"].astype(str).str.strip().str.len() >= 10]
    print(f"  Empty/short text removed: {before:,} -> {len(df):,} (-{before - len(df):,})")
    return df


def kcore_filter(df: pd.DataFrame, K: int, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    it = 0
    while True:
        it += 1
        before = len(df)
        uc = df["user_id"].value_counts()
        ic = df["parent_asin"].value_counts()
        df = df[
            df["user_id"].isin(uc[uc >= K].index) &
            df["parent_asin"].isin(ic[ic >= K].index)
        ]
        after = len(df)
        if verbose:
            print(f"  iter {it}: {before:,} → {after:,} (-{before - after:,})")
        if before == after:
            break
    return df.reset_index(drop=True)


def kcore_stats(df: pd.DataFrame, K_values: list) -> pd.DataFrame:
    rows = []
    for K in K_values:
        dk = kcore_filter(df, K, verbose=False)
        n_inter = len(dk)
        n_users = dk["user_id"].nunique()
        n_items = dk["parent_asin"].nunique()
        avg_seq = (n_inter / n_users) if n_users > 0 else 0.0
        sparsity = (1 - n_inter / (n_users * n_items)) * 100 if n_users * n_items > 0 else 100.0
        rows.append({
            "K": K,
            "Interactions": n_inter,
            "Users": n_users,
            "Items": n_items,
            "Avg Seq Len": round(avg_seq, 2),
            "Sparsity(%)": round(sparsity, 4),
        })
    result = pd.DataFrame(rows).set_index("K")
    return result


def load_meta(meta_path: str) -> dict:
    print("\nStep 3. Load metadata")
    asin2cats = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Meta", mininterval=5):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = obj.get("parent_asin", "")
            raw  = obj.get("categories") or []
            cats = [c.strip() for c in raw
                    if c.strip() and c.strip() not in EXCLUDE_CATS]
            asin2cats[asin] = cats
    print(f"Metadata load complete: {len(asin2cats):,} items")
    return asin2cats


def print_final_stats(df: pd.DataFrame):
    n_inter  = len(df)
    n_users  = df["user_id"].nunique()
    n_items  = df["parent_asin"].nunique()
    sparsity = (1 - n_inter / (n_users * n_items)) * 100
    seq_len  = df.groupby("user_id").size()

    print("\n" + "=" * 60)
    print("Final Statistics (for paper Table)")
    print("=" * 60)
    print(f"  Interactions      : {n_inter:,}")
    print(f"  Users             : {n_users:,}")
    print(f"  Items             : {n_items:,}")
    print(f"  Sparsity          : {sparsity:.4f}%")
    print(f"  Avg Seq Length    : {seq_len.mean():.2f}")
    print(f"  Median Seq Length : {seq_len.median():.1f}")
    print(f"  Max Seq Length    : {seq_len.max()}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_path", default="data/raw/Movies_and_TV.jsonl")
    parser.add_argument("--meta_path",   default="data/raw/meta_Movies_and_TV.jsonl")
    parser.add_argument("--output_dir",  default="data/processed/movies_tv")
    parser.add_argument("--K",           type=int, default=5,
                        help="K-core filtering K value (default 5)")
    parser.add_argument("--stats_only",  action="store_true",
                        help="Print statistics for K=5,10,20,50 only and exit")
    args = parser.parse_args()

    df_raw = load_reviews(args.review_path)

    # ── K-core statistics ────────────────────────────────────────────────────
    K_list = [5, 10, 20, 50]
    print("\n" + "=" * 60)
    print(f"K-core Statistics (K = {K_list})")
    print("=" * 60)
    stats = kcore_stats(df_raw, K_list)

    fmt = stats.copy()
    for col in ["Interactions", "Users", "Items"]:
        fmt[col] = fmt[col].map("{:,}".format)
    print(fmt.to_string())
    print("=" * 60)

    if args.stats_only:
        return

    # ── K-core filtering ─────────────────────────────────────────────────────
    print(f"\nStep 3. K={args.K} filtering")
    df = kcore_filter(df_raw, args.K, verbose=True)
    print(f"  Result: {len(df):,} records | "
          f"{df['user_id'].nunique():,} users | "
          f"{df['parent_asin'].nunique():,} items")

    # ── Metadata join ────────────────────────────────────────────────────────
    asin2cats = load_meta(args.meta_path)
    df["categories"] = df["parent_asin"].map(asin2cats).apply(
        lambda x: x if isinstance(x, list) else []
    )
    n_with = df["categories"].apply(lambda x: len(x) > 0).sum()
    print(f"  Items with categories: {n_with:,} ({n_with/len(df)*100:.1f}%)")

    # ── Sort by time ─────────────────────────────────────────────────────────
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    cols = ["user_id", "parent_asin", "timestamp", "rating", "text", "categories"]
    df_out = df[cols].copy()
    # Serialize categories list to string for CSV storage
    df_out["categories"] = df_out["categories"].apply(lambda x: "||".join(x))

    out_path = f"{args.output_dir}/processed.csv"
    df_out.to_csv(out_path, ilse)
    print(f"\nStep 5. Save complete -> {out_path}")

    print_final_stats(df)
    print("\nDone ✓")


if __name__ == "__main__":
    main()