"""
Movies & TV Amazon review dataset EDA.

What this script reports:
- Total number of raw lines in the JSONL file
- Number of parsed/valid interactions
- Yearly interaction counts
- K-core (default K=5) filtering retention stats

Outputs are saved under --output_dir:
- summary_movies_tv_k{K}.json
- yearly_counts_movies_tv_k{K}.csv
- kcore_iterations_k{K}.csv
- yearly_counts_movies_tv_k{K}.png (optional)

Example:
  python data_analysis/EDA/eda_movies_tv_overview.py \
      --review_path data/raw/Movies_and_TV.jsonl \
      --output_dir data_analysis/EDA/results \
      --k 5
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review_path",
        default="data/raw/Movies_and_TV.jsonl",
        help="Path to Amazon Movies_and_TV review JSONL",
    )
    parser.add_argument(
        "--output_dir",
        default="data_analysis/EDA/results",
        help="Directory where EDA outputs will be saved",
    )
    parser.add_argument("--k", type=int, default=5, help="K value for k-core filtering")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap for quick debug runs (uses first N valid interactions)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save yearly count plot as PNG",
    )
    return parser.parse_args()


def _to_year(ts_series: pd.Series) -> pd.Series:
    """Convert unix timestamp series to year, handling seconds/milliseconds."""
    ts = pd.to_numeric(ts_series, errors="coerce")
    median_val = ts.dropna().median() if ts.notna().any() else 0

    # Amazon review dumps usually use milliseconds.
    unit = "ms" if median_val > 1e11 else "s"
    dt = pd.to_datetime(ts, unit=unit, errors="coerce", utc=True)
    return dt.dt.year


def load_reviews(review_path: Path, max_rows=None):
    records = []
    raw_lines = 0
    parsed_lines = 0

    with review_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw_lines += 1
            try:
                obj = json.loads(line)
                parsed_lines += 1
            except json.JSONDecodeError:
                continue

            uid = obj.get("user_id")
            item = obj.get("parent_asin")
            ts = obj.get("timestamp")
            if not uid or not item or ts is None:
                continue

            records.append({"user_id": uid, "parent_asin": item, "timestamp": ts})
            if max_rows is not None and len(records) >= max_rows:
                break

    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        df["timestamp"] = df["timestamp"].astype("int64")

    return df, raw_lines, parsed_lines


def kcore_filter(df: pd.DataFrame, k: int):
    """Iterative user-item k-core filtering until convergence."""
    cur = df.copy()
    logs = []
    itr = 0

    while True:
        itr += 1
        before_n = len(cur)
        before_users = cur["user_id"].nunique()
        before_items = cur["parent_asin"].nunique()

        user_cnt = cur["user_id"].value_counts()
        item_cnt = cur["parent_asin"].value_counts()
        valid_users = user_cnt[user_cnt >= k].index
        valid_items = item_cnt[item_cnt >= k].index

        nxt = cur[cur["user_id"].isin(valid_users) & cur["parent_asin"].isin(valid_items)]

        after_n = len(nxt)
        after_users = nxt["user_id"].nunique()
        after_items = nxt["parent_asin"].nunique()

        logs.append(
            {
                "iteration": itr,
                "reviews_before": int(before_n),
                "reviews_after": int(after_n),
                "users_before": int(before_users),
                "users_after": int(after_users),
                "items_before": int(before_items),
                "items_after": int(after_items),
            }
        )

        if after_n == before_n:
            return nxt, pd.DataFrame(logs)

        cur = nxt


def build_summary(
    df_all: pd.DataFrame, df_k: pd.DataFrame, raw_lines: int, parsed_lines: int, k: int
) -> dict[str, object]:
    summary: dict[str, object] = {
        "dataset": "Amazon Movies_and_TV",
        "k": int(k),
        "raw_jsonl_lines": int(raw_lines),
        "parsed_json_lines": int(parsed_lines),
        "valid_interactions_before_kcore": int(len(df_all)),
        "users_before_kcore": int(df_all["user_id"].nunique()),
        "items_before_kcore": int(df_all["parent_asin"].nunique()),
        "valid_interactions_after_kcore": int(len(df_k)),
        "users_after_kcore": int(df_k["user_id"].nunique()),
        "items_after_kcore": int(df_k["parent_asin"].nunique()),
    }

    before = max(len(df_all), 1)
    after = len(df_k)
    summary["interaction_retention_pct"] = round(after * 100.0 / before, 2)

    return summary


def main():
    args = parse_args()
    review_path = Path(args.review_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] {review_path}")
    df_all, raw_lines, parsed_lines = load_reviews(review_path, max_rows=args.max_rows)

    if df_all.empty:
        raise RuntimeError("No valid interactions were found. Check input path/format.")

    df_all["year"] = _to_year(df_all["timestamp"])

    print(f"[Load] raw lines={raw_lines:,}, parsed={parsed_lines:,}, valid interactions={len(df_all):,}")
    print(f"[Load] users={df_all['user_id'].nunique():,}, items={df_all['parent_asin'].nunique():,}")

    print(f"[K-core] Filtering with K={args.k}")
    df_k, kcore_logs = kcore_filter(df_all[["user_id", "parent_asin", "timestamp", "year"]], k=args.k)

    # Yearly counts before/after k-core
    yearly_before = df_all.groupby("year").size().rename("count_before_kcore").to_frame()
    yearly_after = df_k.groupby("year").size().rename("count_after_kcore").to_frame()
    yearly = yearly_before.join(yearly_after, how="outer").fillna(0).astype(int).reset_index()
    yearly = yearly.sort_values("year")

    summary = build_summary(df_all, df_k, raw_lines, parsed_lines, args.k)

    summary_path = out_dir / f"summary_movies_tv_k{args.k}.json"
    yearly_path = out_dir / f"yearly_counts_movies_tv_k{args.k}.csv"
    iter_path = out_dir / f"kcore_iterations_k{args.k}.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    yearly.to_csv(yearly_path, index=False)
    kcore_logs.to_csv(iter_path, index=False)

    print(f"[Save] {summary_path}")
    print(f"[Save] {yearly_path}")
    print(f"[Save] {iter_path}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(yearly["year"], yearly["count_before_kcore"], marker="o", label="Before K-core")
        ax.plot(yearly["year"], yearly["count_after_kcore"], marker="o", label=f"After K={args.k} core")
        ax.set_xlabel("Year")
        ax.set_ylabel("Review Count")
        ax.set_title("Movies_and_TV Yearly Review Counts")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plot_path = out_dir / f"yearly_counts_movies_tv_k{args.k}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[Save] {plot_path}")

    print("\n[Summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

