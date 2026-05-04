"""
preprocess_cds.py
=================
CDs_and_Vinyl preprocessing pipeline — for AffSR model
(Same logic as preprocess_movies2023.py, starting from raw jsonl for all years)

Processing order:
  Step 0. raw jsonl -> K-core filtering -> interactions_k5.csv
  Step 1. meta join (add categories column) -> interactions.csv
  Step 2. index mapping -> user_map.json, item_map.json, item_cats.json
  Step 3. GoEmotions inference -> review_va.csv (incremental chunk saving)
  Step 4. Per-item VA average computation -> item_va.json
  Step 5. Sequence construction + leave-one-out split -> sequences.pkl + splits/

Usage:
  # Run from Step 0
  python preprocessing/preprocess_cds.py \
      --device cuda --gpu_id 1

  # Resume from a specific step
  python preprocessing/preprocess_cds.py \
      --device cuda --gpu_id 1 --start_step 3

  # top-k ablation (recompute VA without re-running GoEmotions)
  python preprocessing/preprocess_cds.py \
      --recompute_va --top_k 3

Output files:
  data/processed/cds/
  ├── interactions_k5.csv  # K-core filtering result
  ├── interactions.csv     # interactions with metadata (includes categories)
  ├── user_map.json        # user_id -> integer index
  ├── item_map.json        # parent_asin -> integer index
  ├── item_cats.json       # item_idx -> categories list
  ├── item_va.json         # item_idx -> {va: [v, a], dist28: [...]}
  ├── review_va.csv        # (user_idx, item_idx, timestamp, valence, arousal,
  │                        #  top_k_used, admiration, ..., neutral)
  ├── sequences.pkl        # {user_idx: [(item_idx, timestamp, v, a, has_va), ...]}
  └── splits/
      ├── train.pkl        # {user_idx: [(item_idx, v, a, has_va), ...]}
      ├── valid.pkl        # {user_idx: (item_idx, v, a, has_va)}
      └── test.pkl         # {user_idx: (item_idx, v, a, has_va)}
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
# Constants
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
    "surprise":      ( 0.15,  0.60),  "neutral":       ( 0.00,  0.00),
}
VA_MATRIX = np.array(
    [GOEMOTIONS_VA.get(lb, (0.0, 0.0)) for lb in GOEMOTIONS_LABELS],
    dtype=np.float32,
)

EXCLUDE_CATS = {"CDs & Vinyl", "Featured Categories"}


# ─────────────────────────────────────────────────────────────────────────────
# Utility: convert 28-dim distribution to (valence, arousal)
# ─────────────────────────────────────────────────────────────────────────────
def dist28_to_va(dist28: np.ndarray, top_k: int = None) -> np.ndarray:
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

    va = np.stack([normed @ VA_MATRIX[:, 0], normed @ VA_MATRIX[:, 1]], axis=1)
    return va[0] if single else va


# ─────────────────────────────────────────────────────────────────────────────
# Step 0. raw jsonl -> K-core filtering
# ─────────────────────────────────────────────────────────────────────────────
def kcore_filter(df: pd.DataFrame, K: int) -> pd.DataFrame:
    df = df.copy()
    while True:
        before = len(df)
        uc = df["user_id"].value_counts()
        ic = df["parent_asin"].value_counts()
        df = df[
            df["user_id"].isin(uc[uc >= K].index) &
            df["parent_asin"].isin(ic[ic >= K].index)
        ]
        if len(df) == before:
            break
    return df.reset_index(drop=True)


def step0_kcore(review_path, output_dir, K=5):
    out_path = Path(output_dir) / f"interactions_k{K}.csv"
    if out_path.exists():
        print(f"[Step0] Cache found -> loading: {out_path}")
        return pd.read_csv(out_path)

    print(f"[Step0] Loading raw jsonl: {review_path}")
    records = []
    with open(review_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reviews", mininterval=5):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)
    print(f"[Step0] Load complete: {len(df):,} records")

    before = len(df)
    df = df.dropna(subset=["user_id", "parent_asin", "text", "timestamp", "rating"])
    df = df[df["text"].astype(str).str.strip().str.len() >= 10].copy()
    print(f"[Step0] Missing/empty text removed: {before:,} -> {len(df):,}")

    print(f"[Step0] {K}-core filtering...")
    df = kcore_filter(df, K)
    print(f"[Step0] {K}-core result: {len(df):,} records | "
          f"{df['user_id'].nunique():,} users | "
          f"{df['parent_asin'].nunique():,} items")

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"[Step0] Saved: {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 1. meta join (add categories column)
# ─────────────────────────────────────────────────────────────────────────────
def step1_meta_join(df, meta_path, output_dir):
    out_path = Path(output_dir) / "interactions.csv"
    if out_path.exists():
        print(f"[Step1] Cache found -> loading: {out_path}")
        df = pd.read_csv(out_path)
        df["categories"] = df["categories"].apply(
            lambda x: x.split("||") if isinstance(x, str) and x else []
        )
        return df

    print("[Step1] Loading metadata...")
    asin2cats = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Meta", mininterval=5):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            asin = obj.get("parent_asin", "")
            raw  = obj.get("categories") or []
            cats = [c.strip() for c in raw
                    if c.strip() and c.strip() not in EXCLUDE_CATS]
            if asin:
                asin2cats[asin] = cats

    print(f"[Step1] Metadata load complete: {len(asin2cats):,} items")

    df["categories"] = df["parent_asin"].map(asin2cats).apply(
        lambda x: x if isinstance(x, list) else []
    )

    cat_match = df["categories"].apply(len) > 0
    print(f"[Step1] Items with categories: {cat_match.mean()*100:.1f}%")
    print(f"[Step1] Users: {df['user_id'].nunique():,} | "
          f"Items: {df['parent_asin'].nunique():,} | "
          f"Interactions: {len(df):,}")

    df_save = df.copy()
    df_save["categories"] = df_save["categories"].apply(lambda x: "||".join(x))
    df_save.to_csv(out_path, index=False)
    print(f"[Step1] Saved: {out_path}")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2. Index mapping
# ─────────────────────────────────────────────────────────────────────────────
def step2_mapping(df, output_dir):
    map_path  = Path(output_dir) / "user_map.json"
    imap_path = Path(output_dir) / "item_map.json"
    cats_path = Path(output_dir) / "item_cats.json"

    if all(p.exists() for p in [map_path, imap_path, cats_path]):
        print(f"[Step2] Cache found -> loading")
        with open(map_path)  as f: user_map = json.load(f)
        with open(imap_path) as f: item_map = json.load(f)
        df["user_idx"] = df["user_id"].map(user_map)
        df["item_idx"] = df["parent_asin"].map(item_map)
        return df, user_map, item_map

    users    = sorted(df["user_id"].unique())
    items    = sorted(df["parent_asin"].unique())
    user_map = {u: i+1 for i, u in enumerate(users)}
    item_map = {v: i+1 for i, v in enumerate(items)}

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

    print(f"[Step2] Users: {len(user_map):,} | Items: {len(item_map):,}")
    print(f"[Step2] Saved: {Path(output_dir)}")
    return df, user_map, item_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 3. GoEmotions inference (incremental chunk saving, resumable)
# ─────────────────────────────────────────────────────────────────────────────
def step3_goemotions(df, output_dir, batch_size=128, device="cpu", top_k=5):
    out_path  = Path(output_dir) / "review_va.csv"
    chunk_dir = Path(output_dir) / "va_chunks"
    chunk_dir.mkdir(exist_ok=True)

    if out_path.exists():
        existing = pd.read_csv(out_path, nrows=1)
        cached_top_k = int(existing["top_k_used"].iloc[0]) if "top_k_used" in existing.columns else -1
        current_top_k_val = top_k if top_k is not None else 0

        if cached_top_k == current_top_k_val:
            print(f"[Step3] Cache found (top_k={cached_top_k}) -> loading: {out_path}")
            return pd.read_csv(out_path)

        has_dist28 = all(lb in existing.columns for lb in GOEMOTIONS_LABELS)
        if has_dist28:
            print(f"[Step3] top_k mismatch -> recomputing VA from 28-dim raw (no re-inference)")
            recompute_va_from_raw(str(out_path), current_top_k_val, str(out_path))
            return pd.read_csv(out_path)

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

    CHUNK_SIZE = 50_000
    texts   = df["text"].tolist()
    indices = df.index.tolist()

    results = {}
    done_chunks = set()
    for cp in sorted(chunk_dir.glob("chunk_*.pkl")):
        with open(cp, "rb") as f:
            chunk_results = pickle.load(f)
        results.update(chunk_results)
        done_chunks.add(int(cp.stem.split("_")[1]))
    if done_chunks:
        print(f"[Step3] Loaded existing chunks: {sorted(done_chunks)} ({len(results):,} records)")

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
            chunk_buf[df_idx] = (0.0, 0.0, raw_mtx[i].tolist())
            results[df_idx]   = chunk_buf[df_idx]

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

    print(f"[Step3] Computing final VA (top_k={top_k})...")
    dist28_array = np.array(
        [results[i][2] for i in df.index], dtype=np.float32
    )
    va_final = dist28_to_va(dist28_array, top_k=top_k)

    df_out = df[["user_idx", "item_idx", "timestamp"]].copy()
    df_out["valence"]    = va_final[:, 0]
    df_out["arousal"]    = va_final[:, 1]
    df_out["top_k_used"] = top_k if top_k is not None else 0
    for j, lb in enumerate(GOEMOTIONS_LABELS):
        df_out[lb] = dist28_array[:, j]

    df_out.to_csv(out_path, index=False)
    print(f"[Step3] Complete: {out_path}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Step 4. Per-item VA average
# ─────────────────────────────────────────────────────────────────────────────
def step4_item_va(review_va_df, output_dir, top_k_val=0):
    out_path = Path(output_dir) / "item_va.json"

    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        cached_topk = cached.get("__meta__", {}).get("top_k_used", -1)
        if cached_topk == top_k_val:
            print(f"[Step4] Cache found (top_k={cached_topk}) -> skipping")
            return
        print(f"[Step4] top_k mismatch -> recomputing")

    print("[Step4] Computing per-item VA + 28-dim average...")
    has_dist28 = all(lb in review_va_df.columns for lb in GOEMOTIONS_LABELS)

    item_va = {}
    for item_idx, grp in tqdm(review_va_df.groupby("item_idx"), desc="Item VA"):
        entry = {
            "va": [
                round(float(grp["valence"].mean()), 6),
                round(float(grp["arousal"].mean()), 6),
            ]
        }
        if has_dist28:
            entry["dist28"] = [
                round(float(x), 6) for x in grp[GOEMOTIONS_LABELS].mean(axis=0).tolist()
            ]
        item_va[str(item_idx)] = entry

    item_va["__meta__"] = {"top_k_used": top_k_val}
    with open(out_path, "w") as f:
        json.dump(item_va, f)

    print(f"[Step4] Saved {len(item_va) - 1:,} items: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5. Sequence construction + leave-one-out split
# ─────────────────────────────────────────────────────────────────────────────
def step5_sequences(df, review_va_df, output_dir, top_k_val=0):
    seq_path  = Path(output_dir) / "sequences.pkl"
    split_dir = Path(output_dir) / "splits"
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
        print(f"[Step5] top_k mismatch -> recomputing")

    print("[Step5] Building sequences...")
    df_merged = df[["user_idx", "item_idx", "timestamp"]].merge(
        review_va_df[["user_idx", "item_idx", "timestamp", "valence", "arousal"]],
        on=["user_idx", "item_idx", "timestamp"], how="left"
    ).sort_values(["user_idx", "timestamp"])

    df_merged["has_va"] = ~df_merged["valence"].isna()

    sequences  = {}
    train_data = {}
    valid_data = {}
    test_data  = {}
    skipped    = 0

    for uid, grp in tqdm(df_merged.groupby("user_idx"), desc="Sequences"):
        grp = grp.sort_values(["timestamp", "item_idx"])
        seq = list(zip(
            grp["item_idx"].tolist(),
            grp["timestamp"].tolist(),
            grp["valence"].fillna(0.0).tolist(),
            grp["arousal"].fillna(0.0).tolist(),
            grp["has_va"].tolist(),
        ))

        if len(seq) < 3:
            skipped += 1
            continue

        sequences[uid] = seq

        test_item  = seq[-1]
        valid_item = seq[-2]
        train_seq  = seq[:-2]

        test_data[uid]  = (test_item[0],  test_item[2],  test_item[3],  test_item[4])
        valid_data[uid] = (valid_item[0], valid_item[2], valid_item[3], valid_item[4])
        train_data[uid] = [(s[0], s[2], s[3], s[4]) for s in train_seq]

    print(f"[Step5] Users: {len(sequences):,} (skipped: {skipped:,})")
    print(f"        train: {len(train_data):,} | valid: {len(valid_data):,} | test: {len(test_data):,}")

    with open(seq_path,   "wb") as f: pickle.dump(sequences,  f)
    with open(train_path, "wb") as f: pickle.dump(train_data, f)
    with open(valid_path, "wb") as f: pickle.dump(valid_data, f)
    with open(test_path,  "wb") as f: pickle.dump(test_data,  f)
    topk_sentinel.write_text(str(top_k_val))
    print(f"[Step5] Save complete: {split_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# top-k ablation: recompute VA without re-running GoEmotions
# ─────────────────────────────────────────────────────────────────────────────
def recompute_va_from_raw(review_va_path, top_k, output_path):
    print(f"[recompute_va] Loading: {review_va_path}")
    df = pd.read_csv(review_va_path)

    if not all(lb in df.columns for lb in GOEMOTIONS_LABELS):
        raise ValueError("review_va.csv does not have 28-dim columns. Please re-run Step3.")

    dist28       = df[GOEMOTIONS_LABELS].values.astype(np.float32)
    top_k_actual = None if top_k == 0 else top_k
    va           = dist28_to_va(dist28, top_k=top_k_actual)

    df["valence"]    = va[:, 0]
    df["arousal"]    = va[:, 1]
    df["top_k_used"] = top_k

    df.to_csv(output_path, index=False)
    print(f"[recompute_va] Complete (top_k={top_k}): {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_path", default="data/raw/CDs_and_Vinyl.jsonl")
    parser.add_argument("--meta_path",   default="data/raw/meta_CDs_and_Vinyl.jsonl")
    parser.add_argument("--output_dir",  default="data/processed/cds")
    parser.add_argument("--K",           type=int, default=5,
                        help="K-core filtering K value (default 5)")
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpu_id",      type=int, default=1)
    parser.add_argument("--start_step",  type=int, default=0,
                        help="Step to start from (0~5), for resuming mid-way")
    parser.add_argument("--top_k",       type=int, default=5,
                        help="Number of top emotions for VA conversion (0=use all, 2~27)")
    parser.add_argument("--recompute_va", action="store_true",
                        help="Recompute VA from 28-dim raw without re-running GoEmotions (top-k ablation)")
    parser.add_argument("--review_va_path", default=None,
                        help="Input path for --recompute_va (default: output_dir/review_va.csv)")
    parser.add_argument("--output_path",    default=None,
                        help="Output path for --recompute_va (default: overwrite input file)")
    args = parser.parse_args()

    if args.top_k < 0 or args.top_k == 1 or args.top_k > 27:
        parser.error(f"--top_k={args.top_k}: enter 0 (use all) or an integer between 2~27.")

    top_k     = None if args.top_k == 0 else args.top_k
    top_k_val = args.top_k

    if args.recompute_va:
        src = args.review_va_path or str(Path(args.output_dir) / "review_va.csv")
        dst = args.output_path or src
        recompute_va_from_raw(src, top_k_val, dst)
        return

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 0: K-core filtering
    if args.start_step <= 0:
        df = step0_kcore(args.review_path, args.output_dir, K=args.K)
    else:
        print(f"[Step0] Skipping -> loading interactions_k{args.K}.csv")
        df = pd.read_csv(Path(args.output_dir) / f"interactions_k{args.K}.csv")

    # Step 1: meta join
    if args.start_step <= 1:
        df = step1_meta_join(df, args.meta_path, args.output_dir)
    else:
        print("[Step1] Skipping -> loading interactions.csv")
        df = pd.read_csv(Path(args.output_dir) / "interactions.csv")
        df["categories"] = df["categories"].apply(
            lambda x: x.split("||") if isinstance(x, str) and x else []
        )

    # Step 2: index mapping
    if args.start_step <= 2:
        df, user_map, item_map = step2_mapping(df, args.output_dir)
    else:
        print("[Step2] Skipping -> loading mappings")
        with open(Path(args.output_dir) / "user_map.json") as f:
            user_map = json.load(f)
        with open(Path(args.output_dir) / "item_map.json") as f:
            item_map = json.load(f)
        df["user_idx"] = df["user_id"].map(user_map)
        df["item_idx"] = df["parent_asin"].map(item_map)

    # Step 3: GoEmotions inference
    if args.start_step <= 3:
        review_va_df = step3_goemotions(
            df, args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            top_k=top_k,
        )
    else:
        print("[Step3] Skipping -> loading review_va.csv")
        review_va_df = pd.read_csv(Path(args.output_dir) / "review_va.csv")

    # Step 4: per-item VA average
    if args.start_step <= 4:
        step4_item_va(review_va_df, args.output_dir, top_k_val=top_k_val)

    # Step 5: sequence construction
    if args.start_step <= 5:
        step5_sequences(df, review_va_df, args.output_dir, top_k_val=top_k_val)

    print("\nPreprocessing complete!")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
