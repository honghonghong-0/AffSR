"""
CDs_and_Vinyl preprocessing pipeline
1. Remove missing values / empty text (keep duplicates to preserve sequence info)
2. 5-core filtering
3. Metadata join (add categories column)
4. Sort by timestamp
5. Save (data/processed/)
"""

import json
import pandas as pd
from pathlib import Path

# ── Path configuration ─────────────────────────────────────────────────────────
RAW_DIR       = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_PATH = RAW_DIR / 'CDs_and_Vinyl.jsonl'
META_PATH   = RAW_DIR / 'meta_CDs_and_Vinyl.jsonl'

user_col = 'user_id'
item_col = 'parent_asin'

# ── Step 1: Load review data ───────────────────────────────────────────────────
print('=' * 55)
print('Step 1. Load review data')
print('=' * 55)
records = []
skip_count = 0
with open(REVIEW_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skip_count += 1

df = pd.DataFrame(records)
print(f'Load complete: {len(df):,} records (parse failures: {skip_count:,})')

# ── Step 2: Remove missing values / empty text / duplicates ────────────────────
print('\n' + '=' * 55)
print('Step 2. Remove missing values / empty text')
print('=' * 55)

before = len(df)

# Remove missing required columns
df = df.dropna(subset=[user_col, item_col, 'text', 'timestamp', 'rating'])
print(f'Missing values removed: {before:,} -> {len(df):,} ({before - len(df):,} removed)')

# Remove empty text
before = len(df)
df = df[df['text'].astype(str).str.strip() != '']
print(f'Empty text removed: {before:,} -> {len(df):,} ({before - len(df):,} removed)')

# Keep duplicates to preserve temporal sequence information
print(f'Duplicates retained (user-item duplicates ~0.45% -> preserving sequence info)')

# ── Step 3: 5-core filtering ───────────────────────────────────────────────────
print('\n' + '=' * 55)
print('Step 3. 5-core filtering (remove users/items with fewer than 5 interactions)')
print('=' * 55)

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
        print(f'  iteration {iteration}: {before:,} -> {after:,} records')
        if (df[user_col].value_counts().min() >= k and
            df[item_col].value_counts().min() >= k):
            break
    return df.reset_index(drop=True)

df = filter_kcore(df, user_col, item_col, k=5)
print(f'Filtering complete: {len(df):,} records | {df[user_col].nunique():,} users | {df[item_col].nunique():,} items')

# ── Step 4: Load and join metadata ────────────────────────────────────────────
print('\n' + '=' * 55)
print('Step 4. Metadata join (add categories column)')
print('=' * 55)

asin2cats = {}
with open(META_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            cats = r.get('categories') or []
            # Skip first "CDs & Vinyl" entry and store the rest
            asin2cats[r['parent_asin']] = cats[1:] if len(cats) > 1 else []
        except json.JSONDecodeError:
            continue

print(f'Metadata load complete: {len(asin2cats):,} items')

df['categories'] = df[item_col].map(asin2cats)

# Check category mapping results
n_with_cats = df['categories'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
n_no_cats   = len(df) - n_with_cats
print(f'Items with categories: {n_with_cats:,} ({n_with_cats/len(df)*100:.1f}%)')
print(f'Items without categories: {n_no_cats:,} ({n_no_cats/len(df)*100:.1f}%)')

# Fill missing categories with empty list
df['categories'] = df['categories'].apply(lambda x: x if isinstance(x, list) else [])

# ── Step 5: Sort by timestamp ──────────────────────────────────────────────────
print('\n' + '=' * 55)
print('Step 5. Sort by timestamp')
print('=' * 55)

df = df.sort_values([user_col, 'timestamp']).reset_index(drop=True)
print('Sort complete')

# ── Step 6: Select required columns and save ──────────────────────────────────
print('\n' + '=' * 55)
print('Step 6. Save')
print('=' * 55)

cols = [user_col, item_col, 'timestamp', 'rating', 'text', 'categories']
df_final = df[cols].copy()

save_path = PROCESSED_DIR / 'CDs_and_Vinyl_processed.csv'
df_final.to_csv(save_path, index=False)
print(f'Save complete -> {save_path}')

# ── Final statistics ───────────────────────────────────────────────────────────
print('\n' + '=' * 55)
print('Final Statistics')
print('=' * 55)
n_inter = len(df_final)
n_users = df_final[user_col].nunique()
n_items = df_final[item_col].nunique()
sparsity = 1 - n_inter / (n_users * n_items)
seq_len = df_final.groupby(user_col).size()

print(f'  Interactions    : {n_inter:,}')
print(f'  Users           : {n_users:,}')
print(f'  Items           : {n_items:,}')
print(f'  Sparsity        : {sparsity*100:.4f}%')
print(f'  Avg Seq Length  : {seq_len.mean():.2f}')
print(f'  Med Seq Length  : {seq_len.median():.1f}')
print(f'  Max Seq Length  : {seq_len.max()}')
print('\nDone ✓')
