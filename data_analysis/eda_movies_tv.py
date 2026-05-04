from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick EDA for Movies_and_TV JSONL")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "data/raw/Movies_and_TV.jsonl",
        help="Path to Movies_and_TV.jsonl",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200000,
        help="How many rows to read for quick EDA",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Chunk size for streaming read",
    )
    return parser.parse_args()


def first_existing(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    seen = 0
    sampled_chunks: list[pd.DataFrame] = []

    for chunk in pd.read_json(args.input, lines=True, chunksize=args.chunksize):
        sampled_chunks.append(chunk)
        seen += len(chunk)
        if seen >= args.sample_size:
            break

    if not sampled_chunks:
        print("No rows read. Check input file.")
        return

    df = pd.concat(sampled_chunks, ignore_index=True)
    if len(df) > args.sample_size:
        df = df.iloc[: args.sample_size].copy()

    print("=" * 80)
    print(f"Loaded rows (sample): {len(df):,}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("=" * 80)

    print("\n[Head]")
    print(df.head(3).to_string(index=False))

    print("\n[Missing ratio top 10]")
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    print(miss.head(10).round(2).astype(str) + "%")

    user_col = first_existing(df.columns.tolist(), ["reviewerID", "user_id", "user", "uid"])
    item_col = first_existing(df.columns.tolist(), ["asin", "item_id", "item", "iid"])
    rating_col = first_existing(df.columns.tolist(), ["overall", "rating", "stars", "score"])
    time_col = first_existing(
        df.columns.tolist(), ["unixReviewTime", "reviewTime", "timestamp", "time"]
    )

    print("\n[Core stats]")
    if user_col:
        print(f"unique users ({user_col}): {df[user_col].nunique():,}")
    else:
        print("unique users: column not found")

    if item_col:
        print(f"unique items ({item_col}): {df[item_col].nunique():,}")
    else:
        print("unique items: column not found")

    if rating_col:
        print(f"\n[rating distribution: {rating_col}]")
        print(df[rating_col].value_counts(dropna=False).sort_index())
    else:
        print("rating distribution: column not found")

    if time_col == "unixReviewTime":
        years = pd.to_datetime(df[time_col], unit="s", errors="coerce").dt.year
        print("\n[year distribution from unixReviewTime]")
        print(years.value_counts(dropna=True).sort_index().tail(15))
    elif time_col == "reviewTime":
        years = pd.to_datetime(df[time_col], errors="coerce").dt.year
        print("\n[year distribution from reviewTime]")
        print(years.value_counts(dropna=True).sort_index().tail(15))
    elif time_col:
        years = pd.to_datetime(df[time_col], errors="coerce").dt.year
        print(f"\n[year distribution from {time_col}]")
        print(years.value_counts(dropna=True).sort_index().tail(15))
    else:
        print("year distribution: time column not found")


if __name__ == "__main__":
    main()