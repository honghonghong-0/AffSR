"""
CDs and Vinyl — 필터링 전후 비교 EDA
논문(IDURL) 기준: 5회 미만 유저/아이템 제거
"""
 
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
 
# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = ['NanumGothic', 'NanumBarunGothic', 'Malgun Gothic', 'AppleGothic']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            print(f'폰트 설정: {font}')
            return
    paths = ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
             '/usr/share/fonts/nanum/NanumGothic.ttf']
    for p in paths:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            plt.rcParams['font.family'] = prop.get_name()
            print(f'폰트 설정 (경로): {p}')
            return
    print('⚠ 한글 폰트 없음 — 영문으로 대체됩니다')
 
set_korean_font()
plt.rcParams['axes.unicode_minus'] = False
 
DATA_PATH  = str(Path(__file__).parent.parent / 'data/raw/CDs_and_Vinyl.jsonl')
OUTPUT_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print('데이터 로딩 중...')
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
print(f'파싱 실패(스킵): {skip_count:,}건')
print(f'로드 완료: {len(df):,}건 | user_col={user_col}, item_col={item_col}\n')
 
# ── 통계 출력 함수 ─────────────────────────────────────────────────────────────
def get_stats(df, user_col, item_col, label=''):
    n_inter = len(df)
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    sparsity = 1 - n_inter / (n_users * n_items)
    seq_len = df.groupby(user_col).size()
    print(f'[{label}]')
    print(f'  상호작용 수     : {n_inter:,}')
    print(f'  유저 수         : {n_users:,}')
    print(f'  아이템 수       : {n_items:,}')
    print(f'  희소성          : {sparsity:.6f} ({sparsity*100:.4f}%)')
    print(f'  시퀀스 길이 평균: {seq_len.mean():.2f}')
    print(f'  시퀀스 길이 중앙: {seq_len.median():.1f}')
    print(f'  시퀀스 길이 최대: {seq_len.max()}')
    print()
    return seq_len
 
def make_summary(df, user_col, item_col):
    n_inter = len(df)
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    sparsity = 1 - n_inter / (n_users * n_items)
    seq_len = df.groupby(user_col).size()
    return {
        '상호작용 수': f'{n_inter:,}',
        '유저 수': f'{n_users:,}',
        '아이템 수': f'{n_items:,}',
        '희소성': f'{sparsity*100:.4f}%',
        '시퀀스 길이 평균': f'{seq_len.mean():.2f}',
        '시퀀스 길이 중앙값': f'{seq_len.median():.1f}',
        '시퀀스 길이 최대': f'{seq_len.max()}',
    }
 
# ── 필터링 전 통계 ─────────────────────────────────────────────────────────────
seq_before = get_stats(df, user_col, item_col, label='필터링 전')
 
# ── 5-core 필터링 ──────────────────────────────────────────────────────────────
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
        print(f'  iteration {iteration}: {before:,} → {after:,}건')
        if (df[user_col].value_counts().min() >= k and
            df[item_col].value_counts().min() >= k):
            break
    return df.reset_index(drop=True)
 
print('5-core 필터링 중...')
df_filtered = filter_kcore(df, user_col, item_col, k=5)
print()
 
seq_after = get_stats(df_filtered, user_col, item_col, label='필터링 후 (5-core)')
 
# ── 비교표 ─────────────────────────────────────────────────────────────────────
print('=' * 55)
print('필터링 전후 비교')
print('=' * 55)
summary = pd.DataFrame({
    '필터링 전': make_summary(df, user_col, item_col),
    '필터링 후': make_summary(df_filtered, user_col, item_col),
})
print(summary.to_string())
print()
 
# ── 시각화 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('필터링 전후 시퀀스 길이 분포 비교', fontsize=14, fontweight='bold')
 
ax = axes[0]
ax.hist(seq_before.clip(upper=50), bins=50, color='steelblue', edgecolor='white')
ax.axvline(seq_before.mean(), color='red', linestyle='--', label=f'평균 {seq_before.mean():.1f}')
ax.set_title('필터링 전')
ax.set_xlabel('상호작용 수')
ax.set_ylabel('유저 수')
ax.legend()
 
ax = axes[1]
ax.hist(seq_after.clip(upper=50), bins=50, color='darkorange', edgecolor='white')
ax.axvline(seq_after.mean(), color='red', linestyle='--', label=f'평균 {seq_after.mean():.1f}')
ax.set_title('필터링 후 (5-core)')
ax.set_xlabel('상호작용 수')
ax.set_ylabel('유저 수')
ax.legend()
 
plt.tight_layout()
save_path = OUTPUT_DIR / 'eda_filtering.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'그래프 저장 → {save_path}')
print('Done ✓')
 