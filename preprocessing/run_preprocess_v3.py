"""
run_preprocess_v3.py
====================
Movies_and_TV preprocessing Phase 2 — GoEmotions inference + sequence construction.

Run after preprocess.py has produced processed.csv.

Processing order:
  [internal] Index mapping: user_id/parent_asin -> integer indices
  Step 3. GoEmotions inference (slow — background recommended)
  Step 4. Per-item VA average (e_aff)
  Step 5. Sequence construction and leave-one-out split

Examples:
  # Full run (Step 3 -> 5)
  python preprocessing/run_preprocess_v3.py \
      --processed_path data/processed/movies_tv/processed.csv \
      --output_dir     data/processed/movies_tv \
      --device cuda --gpu_id 1

  # Ablation with different top_k (no GoEmotions re-inference)
  python preprocessing/run_preprocess_v3.py \
      --processed_path data/processed/movies_tv/processed.csv \
      --output_dir     data/processed/movies_tv \
      --device cuda --gpu_id 1 \
      --top_k 3

  # Resume from Step 4 (after GoEmotions completes)
  python preprocessing/run_preprocess_v3.py \
      --processed_path data/processed/movies_tv/processed.csv \
      --output_dir     data/processed/movies_tv \
      --start_step 4

  # Recompute VA only (different top_k, no re-inference)
  python preprocessing/run_preprocess_v3.py \
      --processed_path data/processed/movies_tv/processed.csv \
      --output_dir     data/processed/movies_tv \
      --recompute_va --top_k 3 \
      --output_path data/processed/movies_tv/review_va_topk3.csv

Output files:
  data/processed/movies_tv/
  ├── user_map.json           # user_id -> integer index
  ├── item_map.json           # parent_asin -> integer index
  ├── item_cats.json          # item_idx -> list of categories
  ├── item_va.json            # item_idx -> {va: [v, a], dist28: [...]}
  ├── review_va.csv           # (user_idx, item_idx, timestamp, valence, arousal,
  │                           #  top_k_used, admiration, ..., neutral)
  ├── sequences.pkl
  └── splits/
      ├── train.pkl
      ├── valid.pkl
      └── test.pkl
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# GoEmotions configuration
# ─────────────────────────────────────────────────────────────────────────────
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]
NEUTRAL_IDX = GOEMOTIONS_LABELS.index("neutral")

GOEMOTIONS_VA = {
    "admiration":    ( 0.82,  0.41),  "amusement":     ( 0.76,  0.60),
    "anger":         (-0.43,  0.67),  "annoyance":     (-0.55,  0.35),
    "approval":      ( 0.69,  0.10),  "caring":        ( 0.73,  0.05),
    "confusion":     (-0.15,  0.30),  "curiosity":     ( 0.22,  0.55),
    "desire":        ( 0.55,  0.59),  "disappointment":(-0.63, -0.30),
    "disapproval":   (-0.62,  0.20),  "disgust":       (-0.60,  0.35),
    "embarrassment": (-0.45,  0.15),  "excitement":    ( 0.80,  0.78),
    "fear":          (-0.55,  0.70),  "gratitude":     ( 0.88, -0.45),
    "grief":         (-0.75, -0.55),  "joy":           ( 0.90,  0.65),
    "love":          ( 0.88,  0.40),  "nervousness":   (-0.35,  0.55),
    "optimism":      ( 0.72,  0.30),  "pride":         ( 0.77,  0.45),
    "realization":   ( 0.10, -0.10),  "relief":        ( 0.68, -0.50),
    "remorse":       (-0.58, -0.35),  "sadness":       (-0.70, -0.45),
    "surprise":      ( 0.15,  0.60),
}
VA_MATRIX = np.array(
    [GOEMOTIONS_VA.get(lb, (0.0, 0.0)) for lb in GOEMOTIONS_LABELS],
    dtype=np.float32,
)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def dist28_to_va(dist28: np.ndarray, top_k: int = None) -> tuple:
    single = dist28.ndim == 1
    if single:
        dist28 = dist28[np.newaxis, :]

    mtx = dist28.copy().astype(np.float32)
    mtx[:, NEUTRAL_IDX] = 0.0

    if top_k is not None:
        for i in range(len(mtx)):
            row = mtx[i]
            top_idx = np.argsort(row)[::-1][:top_k]
            mask = np.zeros(28, dtype=np.float32)
            mask[top_idx] = 1.0
            mtx[i] = row * mask

    s = mtx.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    normed = mtx / s

    V = normed @ VA_MATRIX[:, 0]
    A = normed @ VA_MATRIX[:, 1]

    if single:
        return float(V[0]), float(A[0])
    return np.stack([V, A], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Index mapping (internal step)
# ─────────────────────────────────────────────────────────────────────────────
def build_index_maps(df: pd.DataFrame, output_dir: str):
    """
    Assign integer user_idx / item_idx from processed.csv.
    Saves user_map.json, item_map.json, item_cats.json.
    Loads from cache if already exists.
    """
    out = Path(output_dir)
    map_path  = out / "user_map.json"
    imap_path = out / "item_map.json"
    cats_path = out / "item_cats.json"

    # parse categories column (pipe-separated string -> list)
    if df["categories"].dtype == object:
        df["categories"] = df["categories"].apply(
            lambda x: x.split("||") if isinstance(x, str) and x else []
        )

    if map_path.exists() and imap_path.exists() and cats_path.exists():
        print("[IndexMap] Cache found -> loading")
        with open(map_path)  as f: user_map = json.load(f)
        with open(imap_path) as f: item_map = json.load(f)
        df["user_idx"] = df["user_id"].map(user_map)
        df["item_idx"] = df["parent_asin"].map(item_map)
        return df, user_map, item_map

    users    = sorted(df["user_id"].unique())
    items    = sorted(df["parent_asin"].unique())
    user_map = {u: i + 1 for i, u in enumerate(users)}
    item_map = {v: i + 1 for i, v in enumerate(items)}

    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["parent_asin"].map(item_map)

    item_cats = {}
    for asin, grp in df.groupby("parent_asin"):
        cats = grp["categories"].iloc[0]
        if isinstance(cats, str):
            cats = cats.split("||") if cats else []
        item_cats[str(item_map[asin])] = cats

    with open(map_path,  "w") as f: json.dump(user_map, f)
    with open(imap_path, "w") as f: json.dump(item_map, f)
    with open(cats_path, "w") as f: json.dump(item_cats, f)
    print(f"[IndexMap] Done: {len(user_map):,} users | {len(item_map):,} items")
    return df, user_map, item_map


# ─────────────────────────────────────────────────────────────────────────────
# VA recomputation for top-k ablation
# ─────────────────────────────────────────────────────────────────────────────
def recompute_va_from_raw(review_va_path: str, top_k: int, output_path: str):
    print(f"[recompute_va] Loading: {review_va_path}")
    df = pd.read_csv(review_va_path)

    if not all(lb in df.columns for lb in GOEMOTIONS_LABELS):
        raise ValueError(
            "review_va.csv is missing 28-dim columns. "
            "Re-run Step 3."
        )

    dist28      = df[GOEMOTIONS_LABELS].values.astype(np.float32)
    top_k_actual = None if top_k == 0 else top_k
    va           = dist28_to_va(dist28, top_k=top_k_actual)

    df["valence"]    = va[:, 0]
    df["arousal"]    = va[:, 1]
    df["top_k_used"] = top_k

    df.to_csv(output_path, index=False)
    print(f"[recompute_va] Done (top_k={top_k}): {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3. GoEmotions inference
# ─────────────────────────────────────────────────────────────────────────────
def step3_goemotions(df, output_dir, batch_size=128, device="cpu", top_k=5):
    out_path  = Path(output_dir) / "review_va.csv"
    chunk_dir = Path(output_dir) / "va_chunks"
    chunk_dir.mkdir(exist_ok=True)

    if out_path.exists():
        existing          = pd.read_csv(out_path, nrows=1)
        cached_top_k      = int(existing["top_k_used"].iloc[0]) if "top_k_used" in existing.columns else -1
        current_top_k_val = top_k if top_k is not None else 0

        if cached_top_k == current_top_k_val:
            print(f"[Step3] Cache found (top_k={cached_top_k}) -> loading: {out_path}")
            return pd.read_csv(out_path)
        elif all(lb in pd.read_csv(out_path, nrows=0).columns for lb in GOEMOTIONS_LABELS):
            print(f"[Step3] top_k mismatch (cache={cached_top_k}, requested={current_top_k_val})")
            print("         28-dim raw found -> recomputing VA without GoEmotions re-inference")
            recompute_va_from_raw(str(out_path), current_top_k_val, str(out_path))
            return pd.read_csv(out_path)
        else:
            print(f"[Step3] top_k mismatch (cache={cached_top_k}, requested={current_top_k_val})")
            print("         28-dim raw not found -> GoEmotions re-inference required")

    from transformers import pipeline

    print(f"[Step3] Starting GoEmotions inference: {len(df):,} reviews")
    print(f"        device={device}, batch_size={batch_size}, top_k={top_k}")

    clf = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None, truncation=True, max_length=128,
        device=0 if device == "cuda" else -1,
    )

    label2idx = {lb: i for i, lb in enumerate(GOEMOTIONS_LABELS)}
    CHUNK_SIZE = 100_000
    texts   = df["text"].tolist()
    indices = df.index.tolist()
    results = {}

    done_chunks = set()
    for cp in chunk_dir.glob("chunk_*.pkl"):
        with open(cp, "rb") as f:
            chunk_results = pickle.load(f)
        results.update(chunk_results)
        done_chunks.add(int(cp.stem.split("_")[1]))
    if done_chunks:
        print(f"[Step3] Already processed chunks: {sorted(done_chunks)} ({len(results):,} loaded)")

    chunk_buf = {}

    for start in tqdm(range(0, len(texts), batch_size), desc="GoEmotions"):
        batch_texts = texts[start: start + batch_size]
        batch_idxs  = indices[start: start + batch_size]

        if all(idx in results for idx in batch_idxs):
            continue

        raw_mtx = np.zeros((len(batch_texts), 28), dtype=np.float32)
        for bi, preds in enumerate(clf(batch_texts)):
            for p in preds:
                idx = label2idx.get(p["label"])
                if idx is not None:
                    raw_mtx[bi, idx] = p["score"]

        for i, df_idx in enumerate(batch_idxs):
            d = raw_mtx[i].tolist()
            results[df_idx]   = (0.0, 0.0, d)
            chunk_buf[df_idx] = (0.0, 0.0, d)

        if len(chunk_buf) >= CHUNK_SIZE:
            cid = len(list(chunk_dir.glob("chunk_*.pkl")))
            with open(chunk_dir / f"chunk_{cid}.pkl", "wb") as f:
                pickle.dump(chunk_buf, f)
            chunk_buf = {}
            print(f"[Step3] chunk_{cid} saved ({start + batch_size:,}/{len(texts):,})")

    if chunk_buf:
        cid = len(list(chunk_dir.glob("chunk_*.pkl")))
        with open(chunk_dir / f"chunk_{cid}.pkl", "wb") as f:
            pickle.dump(chunk_buf, f)

    print(f"[Step3] Final VA computation (top_k={top_k})...")
    dist28_array = np.array([results[i][2] for i in df.index], dtype=np.float32)
    va_final     = dist28_to_va(dist28_array, top_k=top_k)

    df_out = df[["user_idx", "item_idx", "timestamp"]].copy()
    df_out["valence"]    = va_final[:, 0]
    df_out["arousal"]    = va_final[:, 1]
    df_out["top_k_used"] = top_k if top_k is not None else 0

    for j, lb in enumerate(GOEMOTIONS_LABELS):
        df_out[lb] = dist28_array[:, j]

    df_out.to_csv(out_path, index=False)
    print(f"[Step3] Done: {out_path}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Step 4. Per-item VA average (e_aff)
# ─────────────────────────────────────────────────────────────────────────────
def step4_item_va(review_va_df, output_dir, top_k_val: int = 0):
    out_path = Path(output_dir) / "item_va.json"
    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        cached_topk = cached.get("__meta__", {}).get("top_k_used", -1)
        if cached_topk == top_k_val:
            print(f"[Step4] Cache found (top_k={cached_topk}) -> loading: {out_path}")
            return {k: v for k, v in cached.items() if k != "__meta__"}
        print(f"[Step4] top_k mismatch (cache={cached_topk}, requested={top_k_val}) -> recomputing")

    print("[Step4] Computing per-item VA + 28-dim average...")
    has_dist28 = all(lb in review_va_df.columns for lb in GOEMOTIONS_LABELS)
    if not has_dist28:
        print("[Step4] Warning: review_va.csv missing 28-dim columns. Saving VA only.")

    item_va = {}
    for item_idx, grp in tqdm(review_va_df.groupby("item_idx"), desc="Item VA"):
        entry = {
            "va": [
                round(float(grp["valence"].mean()), 6),
                round(float(grp["arousal"].mean()), 6),
            ]
        }
        if has_dist28:
            dist28_mean = grp[GOEMOTIONS_LABELS].mean(axis=0).tolist()
            entry["dist28"] = [round(float(x), 6) for x in dist28_mean]
        item_va[str(item_idx)] = entry

    item_va["__meta__"] = {"top_k_used": top_k_val}
    with open(out_path, "w") as f:
        json.dump(item_va, f)

    print(f"[Step4] Saved {len(item_va) - 1:,} items: {out_path}")
    if has_dist28:
        print("         Includes: va (2-dim) + dist28 (28-dim average)")
    return {k: v for k, v in item_va.items() if k != "__meta__"}


# ─────────────────────────────────────────────────────────────────────────────
# Step 5. Sequence construction + Leave-one-out split
# ─────────────────────────────────────────────────────────────────────────────
def step5_sequences(df, review_va_df, output_dir, top_k_val: int = 0):
    seq_path      = Path(output_dir) / "sequences.pkl"
    split_dir     = Path(output_dir) / "splits"
    split_dir.mkdir(exist_ok=True)

    train_path    = split_dir / "train.pkl"
    valid_path    = split_dir / "valid.pkl"
    test_path     = split_dir / "test.pkl"
    topk_sentinel = split_dir / ".topk"

    if all(p.exists() for p in [train_path, valid_path, test_path]):
        cached_topk = int(topk_sentinel.read_text().strip()) if topk_sentinel.exists() else -1
        if cached_topk == top_k_val:
            print(f"[Step5] Cache found (top_k={cached_topk}) -> skipping")
            return
        print(f"[Step5] top_k mismatch (cache={cached_topk}, requested={top_k_val}) -> recomputing")

    print("[Step5] Building sequences...")
    df_merged = df[["user_idx", "item_idx", "timestamp"]].merge(
        review_va_df[["user_idx", "item_idx", "timestamp", "valence", "arousal"]],
        on=["user_idx", "item_idx", "timestamp"], how="left"
    ).sort_values(["user_idx", "timestamp"])

    sequences  = {}
    train_data = {}
    valid_data = {}
    test_data  = {}

    for uid, grp in tqdm(df_merged.groupby("user_idx"), desc="Sequences"):
        grp = grp.sort_values("timestamp")
        seq = list(zip(
            grp["item_idx"].tolist(),
            grp["timestamp"].tolist(),
            grp["valence"].fillna(0.0).tolist(),
            grp["arousal"].fillna(0.0).tolist(),
        ))
        if len(seq) < 3:
            continue

        sequences[uid]  = seq
        test_item       = seq[-1]
        valid_item      = seq[-2]
        test_data[uid]  = (test_item[0],  test_item[2],  test_item[3])
        valid_data[uid] = (valid_item[0], valid_item[2], valid_item[3])
        train_data[uid] = [(s[0], s[2], s[3]) for s in seq[:-2]]

    print(f"[Step5] Users: {len(sequences):,}")
    print(f"        train: {len(train_data):,} | valid: {len(valid_data):,} | test: {len(test_data):,}")

    with open(seq_path,   "wb") as f: pickle.dump(sequences,  f)
    with open(train_path, "wb") as f: pickle.dump(train_data, f)
    with open(valid_path, "wb") as f: pickle.dump(valid_data, f)
    with open(test_path,  "wb") as f: pickle.dump(test_data,  f)
    topk_sentinel.write_text(str(top_k_val))
    print(f"[Step5] Saved: {split_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_path", required=True,
                        help="Path to processed.csv generated by preprocess.py")
    parser.add_argument("--output_dir",     default="data/processed/movies_tv")
    parser.add_argument("--batch_size",     type=int, default=128)
    parser.add_argument("--device",         default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpu_id",         type=int, default=1)
    parser.add_argument("--start_step",     type=int, default=3,
                        help="Start step (3-5). Use >3 to resume after GoEmotions completes.")
    parser.add_argument("--top_k",          type=int, default=5,
                        help="Number of top emotions for VA conversion (default 5). 0=all, valid: 0 or 2-27.")
    parser.add_argument("--recompute_va",   action="store_true",
                        help="Recompute VA from 28-dim raw without GoEmotions re-inference")
    parser.add_argument("--review_va_path", default=None,
                        help="Input review_va.csv path for --recompute_va")
    parser.add_argument("--output_path",    default=None,
                        help="Output path for --recompute_va (overwrites input if not set)")
    args = parser.parse_args()

    if args.top_k < 0 or args.top_k == 1 or args.top_k > 27:
        parser.error(
            f"--top_k={args.top_k} is invalid. "
            "Use 0 (no masking) or an integer in [2, 27]."
        )
    top_k = None if args.top_k == 0 else args.top_k

    if args.recompute_va:
        src = args.review_va_path or str(Path(args.output_dir) / "review_va.csv")
        dst = args.output_path or src
        recompute_va_from_raw(src, args.top_k, dst)
        return

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load processed.csv and build index maps
    print(f"[Load] Loading processed.csv: {args.processed_path}")
    df = pd.read_csv(args.processed_path)
    df, user_map, item_map = build_index_maps(df, args.output_dir)

    # Step 3. GoEmotions
    if args.start_step <= 3:
        review_va_df = step3_goemotions(
            df, args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            top_k=top_k,
        )
    else:
        print("[Step3] Skipped -> loading review_va.csv")
        review_va_df = pd.read_csv(Path(args.output_dir) / "review_va.csv")

    # Step 4. Per-item VA
    if args.start_step <= 4:
        step4_item_va(review_va_df, args.output_dir, top_k_val=args.top_k)

    # Step 5. Sequence construction
    if args.start_step <= 5:
        step5_sequences(df, review_va_df, args.output_dir, top_k_val=args.top_k)

    print("\nPreprocessing complete.")
    print(f"   Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
