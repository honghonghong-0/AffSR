"""
CDs and Vinyl — EDA comparing before and after filtering
Paper (IDURL) criterion: remove users/items with fewer than 5 interactions
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# ── Font setup ─────────────────────────────────────────────────────────────────
def set_font():
    candidates = ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'AppleGothic']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            print(f'Font set: {font}')
            return
    paths = ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
             '/usr/share/fonts/nanum/NanumGothic.ttf']
    for p in paths:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            plt.rcParams['font.family'] = prop.get_name()
            print(f'Font set (path): {p}')
            return
    print('⚠ No CJK font found — falling back to default')

set_font()
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH  = str(Path(__file__).parent.parent / 'data/raw/CDs_and_Vinyl.jsonl')
OUTPUT_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data loading ────────────────────────────────────────────────────────────────
print('Loading data...')
records = []
skip_count = 0
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skip_count += 1

df = pd.DataFrame(records)
user_col = next((c for c in ['user_id', 'reviewerID'] if c in df.columns), None)
item_col = next((c for c in ['parent_asin', 'asin']   if c in df.columns), None)
print(f'Parse failures (skipped): {skip_count:,}')
print(f'Loaded: {len(df):,} rows | user_col={user_col}, item_col={item_col}\n')

# ── Statistics helper ──────────────────────────────────────────────────────────
def get_stats(df, user_col, item_col, label=''):
    n_inter = len(df)
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    sparsity = 1 - n_inter / (n_users * n_items)
    seq_len = df.groupby(user_col).size()
    print(f'[{label}]')
    print(f'  Interactions      : {n_inter:,}')
    print(f'  Users             : {n_users:,}')
    print(f'  Items             : {n_items:,}')
    print(f'  Sparsity          : {sparsity:.6f} ({sparsity*100:.4f}%)')
    print(f'  Seq length mean   : {seq_len.mean():.2f}')
    print(f'  Seq length median : {seq_len.median():.1f}')
    print(f'  Seq length max    : {seq_len.max()}')
    print()
    return seq_len

def make_summary(df, user_col, item_col):
    n_inter = len(df)
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    sparsity = 1 - n_inter / (n_users * n_items)
    seq_len = df.groupby(user_col).size()
    return {
        'Interactions':      f'{n_inter:,}',
        'Users':             f'{n_users:,}',
        'Items':             f'{n_items:,}',
        'Sparsity':          f'{sparsity*100:.4f}%',
        'Seq mean':          f'{seq_len.mean():.2f}',
        'Seq median':        f'{seq_len.median():.1f}',
        'Seq max':           f'{seq_len.max()}',
    }

# ── Statistics before filtering ───────────────────────────────────────────────
seq_before = get_stats(df, user_col, item_col, label='Before filtering')

# ── 5-core filtering ───────────────────────────────────────────────────────────
def filter_kcore(df, user_col, item_col, k=5):
    iteration = 0
    while True:
        iteration += 1
        before = len(df)
        user_counts = df[user_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= k].index)]
        item_counts = df[item_col].value_counts()
        df = df[df[item_col].isin(item_counts[item_counts >= k].index)]
        after = len(df)
        print(f'  iteration {iteration}: {before:,} → {after:,} rows')
        if (df[user_col].value_counts().min() >= k and
            df[item_col].value_counts().min() >= k):
            break
    return df.reset_index(drop=True)

print('5-core filtering...')
df_filtered = filter_kcore(df, user_col, item_col, k=5)
print()

seq_after = get_stats(df_filtered, user_col, item_col, label='After filtering (5-core)')

# ── Comparison table ───────────────────────────────────────────────────────────
print('=' * 55)
print('Before vs After Filtering')
print('=' * 55)
summary = pd.DataFrame({
    'Before': make_summary(df, user_col, item_col),
    'After':  make_summary(df_filtered, user_col, item_col),
})
print(summary.to_string())
print()

# ── Visualization ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sequence Length Distribution: Before vs After 5-core Filtering',
             fontsize=14, fontweight='bold')

ax = axes[0]
ax.hist(seq_before.clip(upper=50), bins=50, color='steelblue', edgecolor='white')
ax.axvline(seq_before.mean(), color='red', linestyle='--',
           label=f'mean {seq_before.mean():.1f}')
ax.set_title('Before filtering')
ax.set_xlabel('# interactions')
ax.set_ylabel('# users')
ax.legend()

ax = axes[1]
ax.hist(seq_after.clip(upper=50), bins=50, color='darkorange', edgecolor='white')
ax.axvline(seq_after.mean(), color='red', linestyle='--',
           label=f'mean {seq_after.mean():.1f}')
ax.set_title('After filtering (5-core)')
ax.set_xlabel('# interactions')
ax.set_ylabel('# users')
ax.legend()

plt.tight_layout()
save_path = OUTPUT_DIR / 'eda_filtering.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved → {save_path}')
print('Done ✓')