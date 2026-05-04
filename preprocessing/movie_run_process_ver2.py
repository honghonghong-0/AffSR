"""
run_preprocess.py
=================
Movies_and_TV 전처리 파이프라인 — AffSR 모델용

주요 변경사항 (v2):
  [핵심] GoEmotions 28차원 raw 분포를 review_va.csv에 함께 저장
         → ADM 수식 변경, top-k ablation 시 GPU 재추론 불필요
  [핵심] top-k masking을 --top_k 파라미터로 분리 (기본값 5)
         → k=3/5/10 ablation을 재추론 없이 실험 가능
  [수정] results dict 구조 변경: (valence, arousal) → (valence, arousal, dist_28)
  [추가] item_va.json에 28차원 평균도 함께 저장 (e_aff 후보)

실행 순서:
  Step 1. K-core 필터링
  Step 2. Leave-one-out split
  Step 3. GoEmotions 추론 (시간 오래 걸림 — 백그라운드 권장)
  Step 4. 아이템별 VA 평균 계산 (e_aff)
  Step 5. 시퀀스 구성 및 저장

사용법:
  # 전체 실행
  python preprocessing/run_preprocess.py \
      --review_path data/raw/Movies_and_TV.jsonl \
      --meta_path   data/raw/meta_Movies_and_TV.jsonl \
      --output_dir  data/processed/movies_tv \
      --device cuda --gpu_id 1

  # top-k 변경 (기본값 5, ablation용)
  python preprocessing/run_preprocess.py \
      --review_path data/raw/Movies_and_TV.jsonl \
      --meta_path   data/raw/meta_Movies_and_TV.jsonl \
      --output_dir  data/processed/movies_tv \
      --device cuda --gpu_id 1 \
      --top_k 3    # 3, 5, 10 등

  # Step 지정 실행 (이어서 할 때)
  python preprocessing/run_preprocess.py \
      --review_path data/raw/Movies_and_TV.jsonl \
      --meta_path   data/raw/meta_Movies_and_TV.jsonl \
      --output_dir  data/processed/movies_tv \
      --device cuda --gpu_id 1 \
      --start_step 3   # 1~5 중 선택

출력 파일:
  data/processed/movies_tv/
  ├── interactions.csv        # K-core 후 전체 인터랙션
  ├── user_map.json           # user_id → 정수 인덱스
  ├── item_map.json           # parent_asin → 정수 인덱스
  ├── item_cats.json          # item_idx → categories 리스트
  ├── item_va.json            # item_idx → {va: [v, a], dist28: [...]} (e_aff)
  ├── review_va.csv           # (user_idx, item_idx, timestamp, valence, arousal,
  │                           #  top_k_used, admiration, amusement, ..., neutral)
  │                           #  ↑ 28차원 raw 분포 포함 (재추론 불필요)
  ├── sequences.pkl           # {user_idx: [(item_idx, timestamp, valence, arousal), ...]}
  └── splits/
      ├── train.pkl           # {user_idx: [(item_idx, valence, arousal), ...]} 시간순
      ├── valid.pkl           # {user_idx: (item_idx, valence, arousal)}  직전 1개
      └── test.pkl            # {user_idx: (item_idx, valence, arousal)}  마지막 1개

top-k ablation 방법 (재추론 없이):
  review_va.csv에 28차원 raw와 top_k_used가 저장되어 있으므로,
  아래 스크립트로 k만 바꿔서 VA 값을 재계산 가능:

  python preprocessing/recompute_va.py \
      --review_va_path data/processed/movies_tv/review_va.csv \
      --top_k 3 \
      --output_path data/processed/movies_tv/review_va_topk3.csv
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# GoEmotions 설정
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
# neutral은 VA 좌표 없음 → (0.0, 0.0) fallback
VA_MATRIX = np.array(
    [GOEMOTIONS_VA.get(lb, (0.0, 0.0)) for lb in GOEMOTIONS_LABELS],
    dtype=np.float32,
)

EXCLUDE_CATS = {
    "Movies & TV", "Prime Video",
    "Featured Categories", "Genre for Featured Categories", "Amazon Video",
}


# ─────────────────────────────────────────────────────────────────────────────
# 유틸: raw 28차원 분포 → (valence, arousal) 변환
# top_k=None이면 masking 없이 전체 사용
# ─────────────────────────────────────────────────────────────────────────────
def dist28_to_va(dist28: np.ndarray, top_k: int = None) -> tuple:
    """
    28차원 감정 분포 → (valence, arousal) 변환

    Args:
        dist28: shape (28,) or (N, 28), neutral 제외 전 raw 분포
        top_k:  상위 k개 감정만 사용. None이면 masking 없음.

    Returns:
        (valence, arousal) or (N, 2)
    """
    single = dist28.ndim == 1
    if single:
        dist28 = dist28[np.newaxis, :]  # (1, 28)

    mtx = dist28.copy().astype(np.float32)

    # neutral 제거
    mtx[:, NEUTRAL_IDX] = 0.0

    # top-k masking (파라미터화)
    if top_k is not None:
        for i in range(len(mtx)):
            row = mtx[i]
            top_idx = np.argsort(row)[::-1][:top_k]
            mask = np.zeros(28, dtype=np.float32)
            mask[top_idx] = 1.0
            mtx[i] = row * mask

    # 정규화
    s = mtx.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    normed = mtx / s

    # VA 계산
    V = normed @ VA_MATRIX[:, 0]  # (N,)
    A = normed @ VA_MATRIX[:, 1]  # (N,)

    if single:
        return float(V[0]), float(A[0])
    return np.stack([V, A], axis=1)  # (N, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1. 데이터 로딩 + K-core 필터링
# ─────────────────────────────────────────────────────────────────────────────
def step1_kcore(review_path, meta_path, output_dir, K=5):
    out_path = Path(output_dir) / "interactions.csv"
    if out_path.exists():
        print(f"[Step1] 캐시 발견 → 로드: {out_path}")
        return pd.read_csv(out_path)

    # 1-1. 메타데이터 로딩
    print("[Step1] 메타데이터 로딩...")
    asin2cats = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Meta", mininterval=5):
            try:
                obj = json.loads(line)
            except:
                continue
            asin = obj.get("parent_asin", "")
            raw  = obj.get("categories") or []
            cats = [c.strip() for c in raw
                    if c.strip() and c.strip() not in EXCLUDE_CATS]
            asin2cats[asin] = cats

    # 1-2. 리뷰 로딩
    print("[Step1] 리뷰 로딩...")
    records = []
    with open(review_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reviews", mininterval=5):
            try:
                obj = json.loads(line)
            except:
                continue
            text = (obj.get("text") or "").strip()
            uid  = obj.get("user_id", "")
            asin = obj.get("parent_asin", "")
            if not uid or not asin or len(text) < 10:
                continue
            records.append({
                "user_id":     uid,
                "parent_asin": asin,
                "timestamp":   obj.get("timestamp", 0),
                "rating":      float(obj.get("rating", 3.0)),
                "text":        text,
                "categories":  asin2cats.get(asin, []),
            })

    df = pd.DataFrame(records)
    print(f"[Step1] 원본: {len(df):,} 리뷰 | "
          f"{df['user_id'].nunique():,} 유저 | "
          f"{df['parent_asin'].nunique():,} 아이템")

    # 1-3. K-core 필터링 (수렴까지 반복)
    print(f"[Step1] K-core 필터링 (K={K})...")
    while True:
        user_cnt = df["user_id"].value_counts()
        item_cnt = df["parent_asin"].value_counts()
        valid_users = user_cnt[user_cnt >= K].index
        valid_items = item_cnt[item_cnt >= K].index
        before = len(df)
        df = df[df["user_id"].isin(valid_users) & df["parent_asin"].isin(valid_items)]
        after = len(df)
        print(f"         {before:,} → {after:,} (-{before-after:,})")
        if before == after:
            break

    print(f"[Step1] K-core 완료: {len(df):,} 리뷰 | "
          f"{df['user_id'].nunique():,} 유저 | "
          f"{df['parent_asin'].nunique():,} 아이템")

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    df["categories_str"] = df["categories"].apply(lambda x: "||".join(x))
    df.drop(columns=["categories"]).rename(
        columns={"categories_str": "categories"}
    ).to_csv(out_path, index=False)
    print(f"[Step1] 저장: {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2. 인덱스 매핑 + Leave-one-out split
# ─────────────────────────────────────────────────────────────────────────────
def step2_split(df, output_dir):
    map_path  = Path(output_dir) / "user_map.json"
    imap_path = Path(output_dir) / "item_map.json"
    cats_path = Path(output_dir) / "item_cats.json"

    if df["categories"].dtype == object and df["categories"].str.contains("||", regex=False).any():
        df["categories"] = df["categories"].apply(
            lambda x: x.split("||") if isinstance(x, str) and x else []
        )

    users = sorted(df["user_id"].unique())
    items = sorted(df["parent_asin"].unique())
    user_map = {u: i+1 for i, u in enumerate(users)}
    item_map = {v: i+1 for i, v in enumerate(items)}

    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["parent_asin"].map(item_map)

    item_cats = {}
    for asin, group in df.groupby("parent_asin"):
        cats = group["categories"].iloc[0]
        if isinstance(cats, str):
            cats = cats.split("||") if cats else []
        item_cats[str(item_map[asin])] = cats

    with open(map_path,  "w") as f: json.dump(user_map, f)
    with open(imap_path, "w") as f: json.dump(item_map, f)
    with open(cats_path, "w") as f: json.dump(item_cats, f)
    print(f"[Step2] 매핑 저장: {map_path.parent}")
    print(f"        유저 {len(user_map):,}명 | 아이템 {len(item_map):,}개")

    return df, user_map, item_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 3. GoEmotions 추론
# ─────────────────────────────────────────────────────────────────────────────
def step3_goemotions(df, output_dir, batch_size=128, device="cpu", top_k=5):
    """
    GoEmotions 추론 + 28차원 raw 분포 저장

    변경사항:
    - results[idx] = (valence, arousal, dist_28)  ← dist_28 추가
    - review_va.csv에 28개 감정 레이블 컬럼 추가
    - top_k를 파라미터로 받아 top_k_used 컬럼에 기록
    - 28차원이 저장되므로 ADM 수식 변경 시 이 step 재실행 불필요
    """
    out_path   = Path(output_dir) / "review_va.csv"
    chunk_dir  = Path(output_dir) / "va_chunks"
    chunk_dir.mkdir(exist_ok=True)

    if out_path.exists():
        print(f"[Step3] 캐시 발견 → 로드: {out_path}")
        return pd.read_csv(out_path)

    from transformers import pipeline

    print(f"[Step3] GoEmotions 추론 시작: {len(df):,}개 리뷰")
    print(f"        device={device}, batch_size={batch_size}, top_k={top_k}")

    clf = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None, truncation=True, max_length=128,
        device=0 if device == "cuda" else -1,
    )

    label2idx = {lb: i for i, lb in enumerate(GOEMOTIONS_LABELS)}

    CHUNK_SIZE = 100_000
    texts      = df["text"].tolist()
    indices    = df.index.tolist()

    # results[idx] = (valence, arousal, dist_28_array)
    results = {}

    # 이미 처리된 chunk 확인
    done_chunks = set()
    for cp in chunk_dir.glob("chunk_*.pkl"):
        with open(cp, "rb") as f:
            chunk_results = pickle.load(f)
        results.update(chunk_results)
        done_chunks.add(int(cp.stem.split("_")[1]))
    if done_chunks:
        print(f"[Step3] 이미 처리된 chunk: {sorted(done_chunks)}")

    chunk_buf = {}  # idx → (v, a, dist28)

    for start in tqdm(range(0, len(texts), batch_size), desc="GoEmotions"):
        batch_texts = texts[start: start + batch_size]
        batch_idxs  = indices[start: start + batch_size]

        if all(idx in results for idx in batch_idxs):
            continue

        # raw 28차원 분포 계산 (neutral 포함, masking 없음)
        raw_mtx = np.zeros((len(batch_texts), 28), dtype=np.float32)
        for bi, preds in enumerate(clf(batch_texts)):
            for p in preds:
                idx = label2idx.get(p["label"])
                if idx is not None:
                    raw_mtx[bi, idx] = p["score"]

        # VA 변환 (top_k 적용)
        va_pairs = dist28_to_va(raw_mtx, top_k=top_k)  # (N, 2)

        for i, df_idx in enumerate(batch_idxs):
            v = float(va_pairs[i, 0])
            a = float(va_pairs[i, 1])
            d = raw_mtx[i].tolist()  # 28차원 raw (neutral 포함, masking 전)
            results[df_idx]   = (v, a, d)
            chunk_buf[df_idx] = (v, a, d)

        # CHUNK_SIZE마다 중간 저장 (pkl로 저장 — 28차원 배열 효율적)
        if len(chunk_buf) >= CHUNK_SIZE:
            cid = len(list(chunk_dir.glob("chunk_*.pkl")))
            with open(chunk_dir / f"chunk_{cid}.pkl", "wb") as f:
                pickle.dump(chunk_buf, f)
            chunk_buf = {}
            print(f"[Step3] chunk_{cid} 저장 ({start+batch_size:,}/{len(texts):,})")

    if chunk_buf:
        cid = len(list(chunk_dir.glob("chunk_*.pkl")))
        with open(chunk_dir / f"chunk_{cid}.pkl", "wb") as f:
            pickle.dump(chunk_buf, f)

    # 전체 합치기 → review_va.csv
    print("[Step3] review_va.csv 생성 중...")
    df_out = df[["user_idx", "item_idx", "timestamp"]].copy()
    df_out["valence"]    = [results[i][0] for i in df.index]
    df_out["arousal"]    = [results[i][1] for i in df.index]
    df_out["top_k_used"] = top_k  # ablation 추적용

    # 28차원 raw 분포 컬럼 추가 (레이블명 그대로)
    dist28_array = np.array([results[i][2] for i in df.index], dtype=np.float32)
    for j, lb in enumerate(GOEMOTIONS_LABELS):
        df_out[lb] = dist28_array[:, j]

    df_out.to_csv(out_path, index=False)
    print(f"[Step3] 완료: {out_path}")
    print(f"         컬럼: user_idx, item_idx, timestamp, valence, arousal, "
          f"top_k_used, {', '.join(GOEMOTIONS_LABELS)}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Step 4. 아이템별 VA 평균 (e_aff)
# ─────────────────────────────────────────────────────────────────────────────
def step4_item_va(review_va_df, output_dir):
    """
    변경사항:
    - item_va.json에 VA 2차원 + 28차원 평균 분포 모두 저장
      {item_idx: {"va": [v, a], "dist28": [...]}}
    - dist28 평균을 저장해두면 MoE gating 방식 변경 시 재추론 불필요
    """
    out_path = Path(output_dir) / "item_va.json"
    if out_path.exists():
        print(f"[Step4] 캐시 발견 → 로드: {out_path}")
        with open(out_path) as f:
            return json.load(f)

    print("[Step4] 아이템별 VA + 28차원 평균 계산...")

    # 28차원 컬럼 유무 확인
    has_dist28 = all(lb in review_va_df.columns for lb in GOEMOTIONS_LABELS)
    if not has_dist28:
        print("[Step4] 경고: review_va.csv에 28차원 컬럼 없음. VA만 저장.")

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

    with open(out_path, "w") as f:
        json.dump(item_va, f)

    print(f"[Step4] 아이템 {len(item_va):,}개 저장: {out_path}")
    if has_dist28:
        print("         포함 정보: va (2차원) + dist28 (28차원 평균)")
    return item_va


# ─────────────────────────────────────────────────────────────────────────────
# Step 5. 시퀀스 구성 + Leave-one-out split 저장
# ─────────────────────────────────────────────────────────────────────────────
def step5_sequences(df, review_va_df, output_dir):
    seq_path   = Path(output_dir) / "sequences.pkl"
    split_dir  = Path(output_dir) / "splits"
    split_dir.mkdir(exist_ok=True)

    train_path = split_dir / "train.pkl"
    valid_path = split_dir / "valid.pkl"
    test_path  = split_dir / "test.pkl"

    if all(p.exists() for p in [train_path, valid_path, test_path]):
        print(f"[Step5] 캐시 발견 → 스킵")
        return

    print("[Step5] 시퀀스 구성 중...")

    df_merged = df[["user_idx", "item_idx", "timestamp"]].merge(
        review_va_df[["user_idx", "item_idx", "timestamp", "valence", "arousal"]],
        on=["user_idx", "item_idx", "timestamp"], how="left"
    )
    df_merged = df_merged.sort_values(["user_idx", "timestamp"])

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

        sequences[uid] = seq

        test_item  = seq[-1]
        valid_item = seq[-2]
        train_seq  = seq[:-2]

        test_data[uid]  = (test_item[0],  test_item[2],  test_item[3])
        valid_data[uid] = (valid_item[0], valid_item[2], valid_item[3])
        train_data[uid] = [(s[0], s[2], s[3]) for s in train_seq]

    print(f"[Step5] 유저: {len(sequences):,}명")
    print(f"        train: {len(train_data):,} | valid: {len(valid_data):,} | test: {len(test_data):,}")

    with open(seq_path,   "wb") as f: pickle.dump(sequences,  f)
    with open(train_path, "wb") as f: pickle.dump(train_data, f)
    with open(valid_path, "wb") as f: pickle.dump(valid_data, f)
    with open(test_path,  "wb") as f: pickle.dump(test_data,  f)
    print(f"[Step5] 저장 완료: {split_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_path", default="data/raw/Movies_and_TV.jsonl")
    parser.add_argument("--meta_path",   default="data/raw/meta_Movies_and_TV.jsonl")
    parser.add_argument("--output_dir",  default="data/processed/movies_tv")
    parser.add_argument("--K",           type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpu_id",      type=int, default=1)
    parser.add_argument("--start_step",  type=int, default=1,
                        help="1~5 중 시작할 step")
    parser.add_argument("--top_k",       type=int, default=5,
                        help="VA 변환 시 사용할 상위 감정 수 (기본값 5). "
                             "None(0)으로 설정하면 masking 없이 전체 사용.")
    args = parser.parse_args()

    # top_k=0 → masking 없음
    top_k = None if args.top_k == 0 else args.top_k

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.start_step <= 1:
        df = step1_kcore(
            args.review_path, args.meta_path,
            args.output_dir, K=args.K
        )
    else:
        print("[Step1] 스킵 → interactions.csv 로드")
        df = pd.read_csv(Path(args.output_dir) / "interactions.csv")
        df["categories"] = df["categories"].apply(
            lambda x: x.split("||") if isinstance(x, str) and x else []
        )

    if args.start_step <= 2:
        df, user_map, item_map = step2_split(df, args.output_dir)
    else:
        print("[Step2] 스킵 → 매핑 로드")
        with open(Path(args.output_dir) / "user_map.json") as f:
            user_map = json.load(f)
        with open(Path(args.output_dir) / "item_map.json") as f:
            item_map = json.load(f)
        df["user_idx"] = df["user_id"].map(user_map)
        df["item_idx"] = df["parent_asin"].map(item_map)

    if args.start_step <= 3:
        review_va_df = step3_goemotions(
            df, args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            top_k=top_k,
        )
    else:
        print("[Step3] 스킵 → review_va.csv 로드")
        review_va_df = pd.read_csv(Path(args.output_dir) / "review_va.csv")

    if args.start_step <= 4:
        step4_item_va(review_va_df, args.output_dir)

    if args.start_step <= 5:
        step5_sequences(df, review_va_df, args.output_dir)

    print("\n✅ 전처리 완료!")
    print(f"   출력 디렉토리: {args.output_dir}")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# 별도 스크립트: top-k ablation용 VA 재계산 (재추론 불필요)
# 사용: python run_preprocess.py --recompute_va --top_k 3 ...
#       또는 별도 파일로 분리해서 사용
# ─────────────────────────────────────────────────────────────────────────────
def recompute_va_from_raw(review_va_path: str, top_k: int, output_path: str):
    """
    저장된 review_va.csv의 28차원 raw로부터 다른 top_k로 VA 재계산.
    GoEmotions 재추론 없이 ablation 가능.

    Args:
        review_va_path: 28차원 컬럼이 포함된 review_va.csv 경로
        top_k:          새로운 top_k 값 (0이면 masking 없음)
        output_path:    결과 저장 경로
    """
    print(f"[recompute_va] 로드: {review_va_path}")
    df = pd.read_csv(review_va_path)

    if not all(lb in df.columns for lb in GOEMOTIONS_LABELS):
        raise ValueError(
            "review_va.csv에 28차원 컬럼이 없습니다. "
            "Step3를 v2 코드로 재실행하세요."
        )

    dist28 = df[GOEMOTIONS_LABELS].values.astype(np.float32)  # (N, 28)
    top_k_actual = None if top_k == 0 else top_k
    va = dist28_to_va(dist28, top_k=top_k_actual)  # (N, 2)

    df["valence"]    = va[:, 0]
    df["arousal"]    = va[:, 1]
    df["top_k_used"] = top_k

    df.to_csv(output_path, index=False)
    print(f"[recompute_va] 완료 (top_k={top_k}): {output_path}")