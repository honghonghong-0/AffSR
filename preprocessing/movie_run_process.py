"""
run_preprocess.py
=================
Movies_and_TV 전처리 파이프라인 — AffSR 모델용

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
  ├── item_va.json            # item_idx → [valence_mean, arousal_mean] (e_aff)
  ├── review_va.csv           # (user_idx, item_idx, timestamp, valence, arousal)
  ├── sequences.pkl           # {user_idx: [(item_idx, timestamp, valence, arousal), ...]}
  └── splits/
      ├── train.pkl           # {user_idx: [(item_idx, valence, arousal), ...]} 시간순
      ├── valid.pkl           # {user_idx: (item_idx, valence, arousal)}  직전 1개
      └── test.pkl            # {user_idx: (item_idx, valence, arousal)}  마지막 1개
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
# GoEmotions 설정 (eda_idm_adm_v4.py 와 동일)
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

EXCLUDE_CATS = {
    "Movies & TV", "Prime Video",
    "Featured Categories", "Genre for Featured Categories", "Amazon Video",
}


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

    # 1-4. 시간순 정렬
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # 1-5. 저장
    os.makedirs(output_dir, exist_ok=True)
    # categories는 리스트라 CSV 저장 시 문자열로 변환
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

    # categories 컬럼 복원 (CSV 로드 시 문자열)
    if df["categories"].dtype == object and df["categories"].str.contains("||", regex=False).any():
        df["categories"] = df["categories"].apply(
            lambda x: x.split("||") if isinstance(x, str) and x else []
        )

    # 인덱스 매핑
    users = sorted(df["user_id"].unique())
    items = sorted(df["parent_asin"].unique())
    user_map = {u: i+1 for i, u in enumerate(users)}   # 1-indexed (0=padding)
    item_map = {v: i+1 for i, v in enumerate(items)}

    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["parent_asin"].map(item_map)

    # 아이템별 categories 저장
    item_cats = {}
    for asin, group in df.groupby("parent_asin"):
        cats = group["categories"].iloc[0]
        if isinstance(cats, str):
            cats = cats.split("||") if cats else []
        item_cats[str(item_map[asin])] = cats

    # 저장
    with open(map_path,  "w") as f: json.dump(user_map, f)
    with open(imap_path, "w") as f: json.dump(item_map, f)
    with open(cats_path, "w") as f: json.dump(item_cats, f)
    print(f"[Step2] 매핑 저장: {map_path.parent}")
    print(f"        유저 {len(user_map):,}명 | 아이템 {len(item_map):,}개")

    return df, user_map, item_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 3. GoEmotions 추론
# ─────────────────────────────────────────────────────────────────────────────
def step3_goemotions(df, output_dir, batch_size=128, device="cpu"):
    """
    7.5M 리뷰 전체 추론 — GPU L40S 기준 5~8시간 예상
    중간에 끊겨도 괜찮도록 chunk 단위로 저장
    """
    out_path   = Path(output_dir) / "review_va.csv"
    chunk_dir  = Path(output_dir) / "va_chunks"
    chunk_dir.mkdir(exist_ok=True)

    if out_path.exists():
        print(f"[Step3] 캐시 발견 → 로드: {out_path}")
        return pd.read_csv(out_path)

    from transformers import pipeline

    print(f"[Step3] GoEmotions 추론 시작: {len(df):,}개 리뷰")
    print(f"        device={device}, batch_size={batch_size}")

    clf = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None, truncation=True, max_length=128,
        device=0 if device == "cuda" else -1,
    )

    label2idx = {lb: i for i, lb in enumerate(GOEMOTIONS_LABELS)}
    va_mat    = np.array(
        [GOEMOTIONS_VA.get(lb, (0.0, 0.0)) for lb in GOEMOTIONS_LABELS],
        dtype=np.float32
    )

    CHUNK_SIZE = 100_000  # 10만 건마다 중간 저장
    texts      = df["text"].tolist()
    indices    = df.index.tolist()
    results    = {}  # idx → (valence, arousal)

    # 이미 처리된 chunk 확인
    done_chunks = set()
    for cp in chunk_dir.glob("chunk_*.csv"):
        chunk_df = pd.read_csv(cp)
        for _, row in chunk_df.iterrows():
            results[int(row["df_idx"])] = (row["valence"], row["arousal"])
        done_chunks.add(int(cp.stem.split("_")[1]))
    if done_chunks:
        print(f"[Step3] 이미 처리된 chunk: {sorted(done_chunks)}")

    # 추론
    chunk_id   = 0
    chunk_buf  = []

    for start in tqdm(range(0, len(texts), batch_size), desc="GoEmotions"):
        batch_texts = texts[start: start + batch_size]
        batch_idxs  = indices[start: start + batch_size]

        # 이미 처리된 인덱스 스킵
        if all(idx in results for idx in batch_idxs):
            continue

        mtx = np.zeros((len(batch_texts), 28), dtype=np.float32)
        for bi, preds in enumerate(clf(batch_texts)):
            for p in preds:
                idx = label2idx.get(p["label"])
                if idx is not None:
                    mtx[bi, idx] = p["score"]

        # neutral 제외 + top-5
        mtx[:, NEUTRAL_IDX] = 0.0
        for i in range(len(mtx)):
            row = mtx[i].copy()
            top_idx = np.argsort(row)[::-1][:5]
            mask = np.zeros(28); mask[top_idx] = 1.0
            mtx[i] = row * mask

        s = mtx.sum(axis=1, keepdims=True)
        s = np.where(s == 0, 1, s)
        n = mtx / s
        V = (n @ va_mat[:, 0]).tolist()
        A = (n @ va_mat[:, 1]).tolist()

        for df_idx, v, a in zip(batch_idxs, V, A):
            results[df_idx] = (v, a)
            chunk_buf.append({"df_idx": df_idx, "valence": v, "arousal": a})

        # CHUNK_SIZE마다 중간 저장
        if len(chunk_buf) >= CHUNK_SIZE:
            cid = len(list(chunk_dir.glob("chunk_*.csv")))
            pd.DataFrame(chunk_buf).to_csv(
                chunk_dir / f"chunk_{cid}.csv", index=False)
            chunk_buf = []
            print(f"[Step3] chunk_{cid} 저장 ({start+batch_size:,}/{len(texts):,})")

    # 나머지 저장
    if chunk_buf:
        cid = len(list(chunk_dir.glob("chunk_*.csv")))
        pd.DataFrame(chunk_buf).to_csv(
            chunk_dir / f"chunk_{cid}.csv", index=False)

    # 전체 합치기
    valences = [results[i][0] for i in df.index]
    arousals = [results[i][1] for i in df.index]
    df_out = df[["user_idx", "item_idx", "timestamp"]].copy()
    df_out["valence"] = valences
    df_out["arousal"] = arousals
    df_out.to_csv(out_path, index=False)
    print(f"[Step3] 완료: {out_path}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Step 4. 아이템별 VA 평균 (e_aff)
# ─────────────────────────────────────────────────────────────────────────────
def step4_item_va(review_va_df, output_dir):
    out_path = Path(output_dir) / "item_va.json"
    if out_path.exists():
        print(f"[Step4] 캐시 발견 → 로드: {out_path}")
        with open(out_path) as f:
            return json.load(f)

    print("[Step4] 아이템별 VA 평균 계산...")
    item_va = {}
    for item_idx, grp in review_va_df.groupby("item_idx"):
        item_va[str(item_idx)] = [
            round(float(grp["valence"].mean()), 6),
            round(float(grp["arousal"].mean()), 6),
        ]

    with open(out_path, "w") as f:
        json.dump(item_va, f)
    print(f"[Step4] 아이템 {len(item_va):,}개 VA 저장: {out_path}")
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

    # review_va_df와 merge
    df_merged = df[["user_idx", "item_idx", "timestamp"]].merge(
        review_va_df[["user_idx", "item_idx", "timestamp", "valence", "arousal"]],
        on=["user_idx", "item_idx", "timestamp"], how="left"
    )
    df_merged = df_merged.sort_values(["user_idx", "timestamp"])

    sequences = {}   # {user_idx: [(item_idx, timestamp, v, a), ...]}
    train_data = {}  # {user_idx: [(item_idx, v, a), ...]}
    valid_data = {}  # {user_idx: (item_idx, v, a)}
    test_data  = {}  # {user_idx: (item_idx, v, a)}

    for uid, grp in tqdm(df_merged.groupby("user_idx"), desc="Sequences"):
        grp = grp.sort_values("timestamp")
        seq = list(zip(
            grp["item_idx"].tolist(),
            grp["timestamp"].tolist(),
            grp["valence"].fillna(0.0).tolist(),
            grp["arousal"].fillna(0.0).tolist(),
        ))

        if len(seq) < 3:
            # train만 가능한 경우 스킵 (test, valid 못 만듦)
            continue

        sequences[uid] = seq

        # Leave-one-out split
        # test  = 마지막 1개
        # valid = 직전 1개
        # train = 나머지
        test_item  = seq[-1]
        valid_item = seq[-2]
        train_seq  = seq[:-2]

        test_data[uid]  = (test_item[0],  test_item[2],  test_item[3])
        valid_data[uid] = (valid_item[0], valid_item[2], valid_item[3])
        train_data[uid] = [(s[0], s[2], s[3]) for s in train_seq]

    print(f"[Step5] 유저: {len(sequences):,}명")
    print(f"        train 유저: {len(train_data):,} | "
          f"valid: {len(valid_data):,} | test: {len(test_data):,}")

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
    parser.add_argument("--gpu_id",      type=int, default=1,
                        help="사용할 GPU 인덱스 (기본값: 1)")
    parser.add_argument("--start_step",  type=int, default=1,
                        help="1~5 중 시작할 step (이어서 실행할 때)")
    args = parser.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1
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

    # Step 2
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

    # Step 3 (가장 오래 걸림)
    if args.start_step <= 3:
        review_va_df = step3_goemotions(
            df, args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        print("[Step3] 스킵 → review_va.csv 로드")
        review_va_df = pd.read_csv(Path(args.output_dir) / "review_va.csv")

    # Step 4
    if args.start_step <= 4:
        step4_item_va(review_va_df, args.output_dir)

    # Step 5
    if args.start_step <= 5:
        step5_sequences(df, review_va_df, args.output_dir)

    print("\n✅ 전처리 완료!")
    print(f"   출력 디렉토리: {args.output_dir}")


if __name__ == "__main__":
    main()