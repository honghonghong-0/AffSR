"""
build_sequences_cds.py
======================
Build training data for CDs_and_Vinyl model.

Inputs:
  - data/processed/CDs_and_Vinyl_processed.csv     (k-core filtered)
  - data_analysis/results/emotion_results_cds.csv  (VA + 28-dim emotion probabilities)

Outputs:
  data/processed/cds/
  ├── user_map.json       # user_id -> integer index
  ├── item_map.json       # parent_asin -> integer index
  ├── item_cats.json      # item_idx -> list of categories
  ├── item_va.json        # item_idx -> {va: [v, a], dist28: [...]}
  ├── sequences.pkl       # {user_idx: [(item_idx, timestamp, valence, arousal, dist28), ...]}
  └── splits/
      ├── train.pkl       # {user_idx: [(item_idx, valence, arousal), ...]}
      ├── valid.pkl       # {user_idx: (item_idx, valence, arousal)}
      └── test.pkl        # {user_idx: (item_idx, valence, arousal)}

Usage:
  python preprocessing/build_sequences_cds.py
"""

import ast
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_CSV  = "data/processed/CDs_and_Vinyl_processed.csv"
EMOTION_CSV    = "data_analysis/results/emotion_results_cds.csv"
OUTPUT_DIR     = Path("data/processed/cds_v10")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 1. Load and join data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("Step 1. Load and join data")
print("=" * 55)

df_inter = pd.read_csv(PROCESSED_CSV)
df_emo   = pd.read_csv(EMOTION_CSV)

print(f"  interactions : {len(df_inter):,}")
print(f"  emotion      : {len(df_emo):,}")

# Join VA + emotion probabilities (by user_id + parent_asin + timestamp)
merge_cols = ["user_id", "parent_asin", "timestamp", "valence", "arousal"] + GOEMOTIONS_LABELS
df = df_inter.merge(
    df_emo[merge_cols],
    on=["user_id", "parent_asin", "timestamp"],
    how="left"
)

# Rows without VA (not in emotion_results) -> fill with 0
df["valence"] = df["valence"].fillna(0.0)
df["arousal"] = df["arousal"].fillna(0.0)
for lb in GOEMOTIONS_LABELS:
    df[lb] = df[lb].fillna(0.0)

print(f"  Join complete: {len(df):,}")
print(f"  Rows without VA: {df['valence'].eq(0.0).sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2. Index mapping
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 2. Index mapping")
print("=" * 55)

users = sorted(df["user_id"].unique())
items = sorted(df["parent_asin"].unique())
user_map = {u: i+1 for i, u in enumerate(users)}
item_map = {v: i+1 for i, v in enumerate(items)}

df["user_idx"] = df["user_id"].map(user_map)
df["item_idx"] = df["parent_asin"].map(item_map)

def parse_cats(x):
    try:
        cats = ast.literal_eval(x) if isinstance(x, str) else (x or [])
    except Exception:
        cats = []
    return cats if isinstance(cats, list) else []

df["categories"] = df["categories"].apply(parse_cats)

# item_cats: item_idx -> list of categories
item_cats = {}
for asin, grp in df.groupby("parent_asin"):
    cats = grp["categories"].iloc[0]
    item_cats[str(item_map[asin])] = cats

with open(OUTPUT_DIR / "user_map.json", "w") as f:
    json.dump(user_map, f)
with open(OUTPUT_DIR / "item_map.json", "w") as f:
    json.dump(item_map, f)
with open(OUTPUT_DIR / "item_cats.json", "w") as f:
    json.dump(item_cats, f)

print(f"  Users: {len(user_map):,}")
print(f"  Items: {len(item_map):,}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3. Per-item VA average (e_aff)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 3. Per-item VA average (e_aff)")
print("=" * 55)

item_va = {}
for asin, grp in tqdm(df.groupby("parent_asin"), desc="Item VA"):
    idx = str(item_map[asin])
    va  = [round(float(grp["valence"].mean()), 6),
           round(float(grp["arousal"].mean()), 6)]
    dist28 = [round(float(grp[lb].mean()), 6) for lb in GOEMOTIONS_LABELS]
    item_va[idx] = {"va": va, "dist28": dist28}

with open(OUTPUT_DIR / "item_va.json", "w") as f:
    json.dump(item_va, f)

print(f"  Saved {len(item_va):,} items -> {OUTPUT_DIR / 'item_va.json'}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4. Sequence construction + Leave-one-out split
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 4. Sequence construction + Leave-one-out split")
print("=" * 55)

df_sorted = df.sort_values(["user_idx", "timestamp"])

sequences  = {}
train_data = {}
valid_data = {}
test_data  = {}

skipped = 0
for uid, grp in tqdm(df_sorted.groupby("user_idx"), desc="Sequences"):
    grp = grp.sort_values("timestamp")
    dist28_rows = grp[GOEMOTIONS_LABELS].fillna(0.0).values.tolist()
    seq = [
        (item_i, ts, v, a, np.array(d28, dtype=np.float32))
        for item_i, ts, v, a, d28 in zip(
            grp["item_idx"].tolist(),
            grp["timestamp"].tolist(),
            grp["valence"].tolist(),
            grp["arousal"].tolist(),
            dist28_rows,
        )
    ]

    if len(seq) < 3:
        skipped += 1
        continue

    sequences[uid] = seq

    test_item  = seq[-1]
    valid_item = seq[-2]
    train_seq  = seq[:-2]

    test_data[uid]  = (test_item[0],  test_item[2],  test_item[3])
    valid_data[uid] = (valid_item[0], valid_item[2], valid_item[3])
    train_data[uid] = [(s[0], s[2], s[3], s[4]) for s in train_seq]

print(f"  Sequences built: {len(sequences):,} users (skipped: {skipped:,})")
print(f"  train: {len(train_data):,} | valid: {len(valid_data):,} | test: {len(test_data):,}")

split_dir = OUTPUT_DIR / "splits"
split_dir.mkdir(exist_ok=True)

with open(OUTPUT_DIR / "sequences.pkl", "wb") as f:
    pickle.dump(sequences, f)
with open(split_dir / "train.pkl", "wb") as f:
    pickle.dump(train_data, f)
with open(split_dir / "valid.pkl", "wb") as f:
    pickle.dump(valid_data, f)
with open(split_dir / "test.pkl", "wb") as f:
    pickle.dump(test_data, f)

print(f"\nSaved -> {OUTPUT_DIR}")
print("Done.")
