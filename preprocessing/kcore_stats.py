"""
K-core filtering statistics checker.
Usage: python preprocessing/kcore_stats.py --interactions_path data/processed/movies_tv/interactions.csv
"""
import argparse
import pandas as pd


def kcore_filter(df, K):
    df = df.copy()
    while True:
        user_cnt = df["user_id"].value_counts()
        item_cnt = df["parent_asin"].value_counts()
        valid_users = user_cnt[user_cnt >= K].index
        valid_items = item_cnt[item_cnt >= K].index
        before = len(df)
        df = df[df["user_id"].isin(valid_users) & df["parent_asin"].isin(valid_items)]
        if len(df) == before:
            break
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions_path",
                        default="data/processed/movies_tv/interactions.csv")
    parser.add_argument("--K_values", nargs="+", type=int,
                        default=[5, 10, 20, 50])
    args = parser.parse_args()

    print(f"Loading: {args.interactions_path}")
    df_raw = pd.read_csv(args.interactions_path,
                         usecols=["user_id", "parent_asin"])
    print(f"Raw: {len(df_raw):,} interactions | "
          f"{df_raw['user_id'].nunique():,} users | "
          f"{df_raw['parent_asin'].nunique():,} items\n")

    rows = []
    for K in args.K_values:
        df_k = kcore_filter(df_raw, K)
        rows.append({
            "K": K,
            "interactions": len(df_k),
            "users": df_k["user_id"].nunique(),
            "items": df_k["parent_asin"].nunique(),
        })
        print(f"K={K:>2}  done")

    result = pd.DataFrame(rows).set_index("K")
    result["interactions"] = result["interactions"].map("{:,}".format)
    result["users"] = result["users"].map("{:,}".format)
    result["items"] = result["items"].map("{:,}".format)

    print("\n" + "=" * 50)
    print(result.to_string())
    print("=" * 50)


if __name__ == "__main__":
    main()
