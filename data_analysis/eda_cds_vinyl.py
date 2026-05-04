"""
EDA for CDs_and_Vinyl.jsonl
Amazon Review Dataset (JSONL format)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict, Counter
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
DATA_PATH = "/home/seohyeon/.00_project/Recsys_02_AffDrift/data/raw/CDs_and_Vinyl.jsonl"
OUTPUT_DIR = Path("/home/seohyeon/.00_project/Recsys_02_AffDrift/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data...")
print("=" * 60)

records = []
skip_count = 0
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skip_count += 1

print(f"Skipped (broken lines): {skip_count:,}")

df = pd.DataFrame(records)
print(f"Total records loaded: {len(df):,}\n")
print("Columns:", df.columns.tolist())
print()

# ── 컬럼명 자동 감지 (Amazon 리뷰 데이터 버전별 대응) ─────────────────────────
# user_id
user_col = next((c for c in ["user_id", "reviewerID", "reviewer_id"] if c in df.columns), None)
# item_id
item_col = next((c for c in ["parent_asin", "asin", "item_id"] if c in df.columns), None)
# review text
text_col = next((c for c in ["text", "reviewText", "review_text"] if c in df.columns), None)
# timestamp
time_col = next((c for c in ["timestamp", "unixReviewTime", "unix_time"] if c in df.columns), None)

print(f"user_col  : {user_col}")
print(f"item_col  : {item_col}")
print(f"text_col  : {text_col}")
print(f"time_col  : {time_col}")
print()

# ── 1. 유저 / 아이템 / 상호작용 수 ────────────────────────────────────────────
print("=" * 60)
print("1. Basic Statistics")
print("=" * 60)

n_interactions = len(df)
n_users  = df[user_col].nunique() if user_col else "N/A"
n_items  = df[item_col].nunique() if item_col else "N/A"

print(f"  # Interactions : {n_interactions:,}")
print(f"  # Users        : {n_users:,}" if isinstance(n_users, int) else f"  # Users        : {n_users}")
print(f"  # Items        : {n_items:,}" if isinstance(n_items, int) else f"  # Items        : {n_items}")

if isinstance(n_users, int) and isinstance(n_items, int):
    sparsity = 1 - n_interactions / (n_users * n_items)
    print(f"  Sparsity       : {sparsity:.6f} ({sparsity*100:.4f}%)")
print()

# ── 2. 시퀀스 길이 분포 ────────────────────────────────────────────────────────
print("=" * 60)
print("2. Sequence Length Distribution (interactions per user)")
print("=" * 60)

if user_col:
    seq_lengths = df.groupby(user_col).size()
    print(seq_lengths.describe().to_string())
    print()
    print(f"  Users with >= 5  interactions: {(seq_lengths >= 5).sum():,} ({(seq_lengths >= 5).mean()*100:.1f}%)")
    print(f"  Users with >= 10 interactions: {(seq_lengths >= 10).sum():,} ({(seq_lengths >= 10).mean()*100:.1f}%)")
    print(f"  Users with >= 20 interactions: {(seq_lengths >= 20).sum():,} ({(seq_lengths >= 20).mean()*100:.1f}%)")
    print()

# ── 3. 리뷰 텍스트 존재 비율 ──────────────────────────────────────────────────
print("=" * 60)
print("3. Review Text Existence")
print("=" * 60)

if text_col:
    has_text = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    n_with_text = has_text.sum()
    ratio = n_with_text / n_interactions
    print(f"  Reviews with text    : {n_with_text:,} / {n_interactions:,} ({ratio*100:.2f}%)")
    print(f"  Reviews without text : {n_interactions - n_with_text:,} ({(1-ratio)*100:.2f}%)")
else:
    print("  ⚠ No review text column found.")
print()

# ── 4. 리뷰당 평균 길이 ───────────────────────────────────────────────────────
print("=" * 60)
print("4. Review Text Length (characters)")
print("=" * 60)

if text_col:
    text_series = df[text_col].dropna().astype(str)
    text_series = text_series[text_series.str.strip() != ""]
    char_lengths = text_series.str.len()
    word_lengths = text_series.str.split().str.len()

    print("  [Character level]")
    print(f"    Mean   : {char_lengths.mean():.1f}")
    print(f"    Median : {char_lengths.median():.1f}")
    print(f"    Std    : {char_lengths.std():.1f}")
    print(f"    Min    : {char_lengths.min()}")
    print(f"    Max    : {char_lengths.max()}")
    print()
    print("  [Word level]")
    print(f"    Mean   : {word_lengths.mean():.1f}")
    print(f"    Median : {word_lengths.median():.1f}")
    print(f"    Std    : {word_lengths.std():.1f}")
    print(f"    Min    : {word_lengths.min()}")
    print(f"    Max    : {word_lengths.max()}")
else:
    print("  ⚠ No review text column found.")
print()

# ── 5. 시각화 ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("5. Saving plots...")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("CDs and Vinyl - EDA", fontsize=14, fontweight="bold")

# (a) 시퀀스 길이 분포
if user_col:
    ax = axes[0]
    clipped = seq_lengths.clip(upper=50)
    ax.hist(clipped, bins=50, edgecolor="white", color="steelblue")
    ax.set_title("Sequence Length Distribution\n(clipped at 50)")
    ax.set_xlabel("# interactions per user")
    ax.set_ylabel("# users")
    ax.axvline(seq_lengths.mean(), color="red", linestyle="--", label=f"mean={seq_lengths.mean():.1f}")
    ax.legend()

# (b) 리뷰 텍스트 존재 비율 파이차트
if text_col:
    ax = axes[1]
    ax.pie(
        [n_with_text, n_interactions - n_with_text],
        labels=["Has text", "No text"],
        autopct="%1.1f%%",
        colors=["steelblue", "lightgray"],
        startangle=90,
    )
    ax.set_title("Review Text Existence")

# (c) 리뷰 단어 수 분포
if text_col:
    ax = axes[2]
    clipped_words = word_lengths.clip(upper=300)
    ax.hist(clipped_words, bins=60, edgecolor="white", color="darkorange")
    ax.set_title("Review Word Count Distribution\n(clipped at 300)")
    ax.set_xlabel("# words per review")
    ax.set_ylabel("# reviews")
    ax.axvline(word_lengths.mean(), color="red", linestyle="--", label=f"mean={word_lengths.mean():.1f}")
    ax.legend()

plt.tight_layout()
save_path = OUTPUT_DIR / "eda_cds_vinyl.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved → {save_path}")
print()
print("Done ✓")