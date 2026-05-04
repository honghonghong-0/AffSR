"""
preprocess_movies2023.py
========================
Movies_and_TV_2021_2023 전처리 파이프라인 — AffSR 모델용

전제: 5-core 필터링은 이미 완료됨
  (data/processed/movies_tv_2021_2023/interactions_2021_2023_k5.csv)
  → 19,958 유저 | 15,027 아이템 | 190,871 상호작용

실행 순서:
  Step 1. meta 조인 (categories 컬럼 추가) → interactions.csv
  Step 2. 인덱스 매핑 → user_map.json, item_map.json, item_cats.json
  Step 3. GoEmotions 추론 → review_va.csv (chunk 단위 중간 저장)
  Step 4. 아이템별 VA 평균 계산 → item_va.json
  Step 5. 시퀀스 구성 + leave-one-out split → sequences.pkl + splits/

사용법:
  # 전체 실행 (Step 1부터)
  python preprocessing/preprocess_movies2023.py \
      --device cuda --gpu_id 1

  # Step 지정 (이어서 할 때)
  python preprocessing/preprocess_movies2023.py \
      --device cuda --gpu_id 1 --start_step 3

  # top-k ablation (GoEmotions 재추론 없이 VA 재계산)
  python preprocessing/preprocess_movies2023.py \
      --recompute_va --top_k 3

출력 파일:
  data/processed/movies_tv_2021_2023/
  ├── interactions.csv    # meta 조인된 상호작용 (categories 포함)
  ├── user_map.json       # user_id → 정수 인덱스
  ├── item_map.json       # parent_asin → 정수 인덱스
  ├── item_cats.json      # item_idx → categories 리스트
  ├── item_va.json        # item_idx → {va: [v, a], dist28: [...]}
  ├── review_va.csv       # (user_idx, item_idx, timestamp, valence, arousal,
  │                       #  top_k_used, admiration, ..., neutral)
  ├── sequences.pkl       # {user_idx: [(item_idx, timestamp, v, a, has_va), ...]}
  └── splits/
      ├── train.pkl       # {user_idx: [(item_idx, v, a, has_va), ...]}
      ├── valid.pkl       # {user_idx: (item_idx, v, a, has_va)}
      └── test.pkl        # {user_idx: (item_idx, v, a, has_va)}
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
# 상수
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

# VA 좌표: 각 GoEmotions 감정 레이블을 대표하는 단어들의
# NRC VAD Lexicon 평균값으로 유도 (Mohammad, 2018)
# ref: https://saifmohammad.com/WebPages/nrc-vad.html
# neutral은 VA 공간에서 원점(0,0)으로 정의 (기여 없음)
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

EXCLUDE_CATS = {
    "Movies & TV", "Prime Video",
    "Featured Categories", "Genre for Featured Categories", "Amazon Video",
}

# ─────────────────────────────────────────────────────────────────────────────
# 유틸: 28차원 분포 → (valence, arousal) 변환
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
# Step 1. meta 조인 (categories 컬럼 추가)
# ─────────────────────────────────────────────────────────────────────────────
def step1_meta_join(input_csv, meta_path, output_dir):
    out_path = Path(output_dir) / "interactions.csv"
    if out_path.exists():
        print(f"[Step1] 캐시 발견 → 로드: {out_path}")
        df = pd.read_csv(out_path)
        df["categories"] = df["categories"].apply(
            lambda x: x.split("||") if isinstance(x, str) and x else []
        )
        return df

    print("[Step1] 메타데이터 로딩...")
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

    print(f"[Step1] 메타 로딩 완료: {len(asin2cats):,}개 아이템")

    print(f"[Step1] interactions 로드: {input_csv}")
    df = pd.read_csv(input_csv)

    # text 빈 행 제거
    df = df[df["text"].astype(str).str.strip().str.len() >= 10].copy()

    df["categories"] = df["parent_asin"].map(asin2cats).apply(
        lambda x: x if isinstance(x, list) else []
    )

    cat_match = df["categories"].apply(len) > 0
    print(f"[Step1] 카테고리 있는 아이템 비율: {cat_match.mean()*100:.1f}%")
    print(f"[Step1] 유저: {df['user_id'].nunique():,} | "
          f"아이템: {df['parent_asin'].nunique():,} | "
          f"상호작용: {len(df):,}")

    df_save = df.copy()
    df_save["categories"] = df_save["categories"].apply(lambda x: "||".join(x))
    df_save.to_csv(out_path, index=False)
    print(f"[Step1] 저장: {out_path}")

    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2. 인덱스 매핑
# ─────────────────────────────────────────────────────────────────────────────
def step2_mapping(df, output_dir):
    map_path  = Path(output_dir) / "user_map.json"
    imap_path = Path(output_dir) / "item_map.json"
    cats_path = Path(output_dir) / "item_cats.json"

    if all(p.exists() for p in [map_path, imap_path, cats_path]):
        print(f"[Step2] 캐시 발견 → 로드")
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

    print(f"[Step2] 유저 {len(user_map):,}명 | 아이템 {len(item_map):,}개")
    print(f"[Step2] 저장: {Path(output_dir)}")
    return df, user_map, item_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 3. GoEmotions 추론 (chunk 단위 중간 저장, 재시작 가능)
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
            print(f"[Step3] 캐시 발견 (top_k={cached_top_k}) → 로드: {out_path}")
            return pd.read_csv(out_path)

        has_dist28 = all(lb in existing.columns for lb in GOEMOTIONS_LABELS)
        if has_dist28:
            print(f"[Step3] top_k 불일치 → 28차원 raw로 VA 재계산 (재추론 없음)")
            recompute_va_from_raw(str(out_path), current_top_k_val, str(out_path))
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

    CHUNK_SIZE = 50_000
    texts   = df["text"].tolist()
    indices = df.index.tolist()

    # 이미 처리된 chunk 로드
    results = {}
    done_chunks = set()
    for cp in sorted(chunk_dir.glob("chunk_*.pkl")):
        with open(cp, "rb") as f:
            chunk_results = pickle.load(f)
        results.update(chunk_results)
        done_chunks.add(int(cp.stem.split("_")[1]))
    if done_chunks:
        print(f"[Step3] 기존 chunk 로드: {sorted(done_chunks)} ({len(results):,}건)")

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
            print(f"[Step3] chunk_{cid} 저장 ({start + batch_size:,}/{len(texts):,})")

    if chunk_buf:
        cid = len(list(chunk_dir.glob("chunk_*.pkl")))
        with open(chunk_dir / f"chunk_{cid}.pkl", "wb") as f:
            pickle.dump(chunk_buf, f)

    # 최종 VA 계산 (현재 top_k 기준으로 재계산)
    print(f"[Step3] VA 최종 계산 (top_k={top_k})...")
    dist28_array = np.array(
        [results[i][2] for i in df.index], dtype=np.float32
    )
    va_final = dist28_to_va(dist28_array, top_k=top_k)  # (N, 2)

    df_out = df[["user_idx", "item_idx", "timestamp"]].copy()
    df_out["valence"]    = va_final[:, 0]
    df_out["arousal"]    = va_final[:, 1]
    df_out["top_k_used"] = top_k if top_k is not None else 0
    for j, lb in enumerate(GOEMOTIONS_LABELS):
        df_out[lb] = dist28_array[:, j]

    df_out.to_csv(out_path, index=False)
    print(f"[Step3] 완료: {out_path}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Step 4. 아이템별 VA 평균 (e_aff)
# ─────────────────────────────────────────────────────────────────────────────
def step4_item_va(review_va_df, output_dir, top_k_val=0):
    out_path = Path(output_dir) / "item_va.json"

    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        cached_topk = cached.get("__meta__", {}).get("top_k_used", -1)
        if cached_topk == top_k_val:
            print(f"[Step4] 캐시 발견 (top_k={cached_topk}) → 스킵")
            return
        print(f"[Step4] top_k 불일치 → 재계산")

    print("[Step4] 아이템별 VA + 28차원 평균 계산...")
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

    print(f"[Step4] 아이템 {len(item_va) - 1:,}개 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5. 시퀀스 구성 + leave-one-out split
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
            print(f"[Step5] 캐시 발견 (top_k={cached_topk}) → 스킵")
            return
        print(f"[Step5] top_k 불일치 → 재계산")

    print("[Step5] 시퀀스 구성 중...")
    df_merged = df[["user_idx", "item_idx", "timestamp"]].merge(
        review_va_df[["user_idx", "item_idx", "timestamp", "valence", "arousal"] + GOEMOTIONS_LABELS],
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
        dist28_rows = grp[GOEMOTIONS_LABELS].fillna(0.0).values.tolist()
        seq = [
            (item_i, ts, v, a, has_va, np.array(d28, dtype=np.float32))
            for item_i, ts, v, a, has_va, d28 in zip(
                grp["item_idx"].tolist(),
                grp["timestamp"].tolist(),
                grp["valence"].fillna(0.0).tolist(),
                grp["arousal"].fillna(0.0).tolist(),
                grp["has_va"].tolist(),
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

        test_data[uid]  = (test_item[0],  test_item[2],  test_item[3],  test_item[4])
        valid_data[uid] = (valid_item[0], valid_item[2], valid_item[3], valid_item[4])
        train_data[uid] = [(s[0], s[2], s[3], s[4], s[5]) for s in train_seq]

    print(f"[Step5] 유저: {len(sequences):,}명 (스킵: {skipped:,}명)")
    print(f"        train: {len(train_data):,} | valid: {len(valid_data):,} | test: {len(test_data):,}")

    with open(seq_path,   "wb") as f: pickle.dump(sequences,  f)
    with open(train_path, "wb") as f: pickle.dump(train_data, f)
    with open(valid_path, "wb") as f: pickle.dump(valid_data, f)
    with open(test_path,  "wb") as f: pickle.dump(test_data,  f)
    topk_sentinel.write_text(str(top_k_val))
    print(f"[Step5] 저장 완료: {split_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# top-k ablation: GoEmotions 재추론 없이 VA 재계산
# ─────────────────────────────────────────────────────────────────────────────
def recompute_va_from_raw(review_va_path, top_k, output_path):
    print(f"[recompute_va] 로드: {review_va_path}")
    df = pd.read_csv(review_va_path)

    if not all(lb in df.columns for lb in GOEMOTIONS_LABELS):
        raise ValueError("review_va.csv에 28차원 컬럼이 없습니다. Step3를 재실행하세요.")

    dist28   = df[GOEMOTIONS_LABELS].values.astype(np.float32)
    top_k_actual = None if top_k == 0 else top_k
    va       = dist28_to_va(dist28, top_k=top_k_actual)

    df["valence"]    = va[:, 0]
    df["arousal"]    = va[:, 1]
    df["top_k_used"] = top_k

    df.to_csv(output_path, index=False)
    print(f"[recompute_va] 완료 (top_k={top_k}): {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",   default="data/processed/movies_tv_2021_2023/interactions_2021_2023_k5.csv")
    parser.add_argument("--meta_path",   default="data/raw/meta_Movies_and_TV.jsonl")
    parser.add_argument("--output_dir",  default="data/processed/movies_tv_2021_2023")
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpu_id",      type=int, default=1)
    parser.add_argument("--start_step",  type=int, default=1,
                        help="1~5 중 시작할 step (중간 재시작용)")
    parser.add_argument("--top_k",       type=int, default=5,
                        help="VA 변환 시 상위 감정 수 (0=전체 사용, 2~27)")
    parser.add_argument("--recompute_va", action="store_true",
                        help="GoEmotions 재추론 없이 28차원 raw로 VA 재계산 (top-k ablation)")
    parser.add_argument("--review_va_path", default=None,
                        help="--recompute_va 시 입력 경로 (기본: output_dir/review_va.csv)")
    parser.add_argument("--output_path",    default=None,
                        help="--recompute_va 시 출력 경로 (기본: 입력 파일 덮어쓰기)")
    args = parser.parse_args()

    if args.top_k < 0 or args.top_k == 1 or args.top_k > 27:
        parser.error(f"--top_k={args.top_k}: 0(전체) 또는 2~27 사이 정수를 입력하세요.")

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

    # Step 1: meta 조인
    if args.start_step <= 1:
        df = step1_meta_join(args.input_csv, args.meta_path, args.output_dir)
    else:
        print("[Step1] 스킵 → interactions.csv 로드")
        df = pd.read_csv(Path(args.output_dir) / "interactions.csv")
        df["categories"] = df["categories"].apply(
            lambda x: x.split("||") if isinstance(x, str) and x else []
        )

    # Step 2: 인덱스 매핑
    if args.start_step <= 2:
        df, user_map, item_map = step2_mapping(df, args.output_dir)
    else:
        print("[Step2] 스킵 → 매핑 로드")
        with open(Path(args.output_dir) / "user_map.json") as f:
            user_map = json.load(f)
        with open(Path(args.output_dir) / "item_map.json") as f:
            item_map = json.load(f)
        df["user_idx"] = df["user_id"].map(user_map)
        df["item_idx"] = df["parent_asin"].map(item_map)

    # Step 3: GoEmotions 추론
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

    # Step 4: 아이템별 VA 평균
    if args.start_step <= 4:
        step4_item_va(review_va_df, args.output_dir, top_k_val=top_k_val)

    # Step 5: 시퀀스 구성
    if args.start_step <= 5:
        step5_sequences(df, review_va_df, args.output_dir, top_k_val=top_k_val)

    print("\n전처리 완료!")
    print(f"  출력 디렉토리: {args.output_dir}")


if __name__ == "__main__":
    main()