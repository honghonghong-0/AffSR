"""
analysis/ad_idm_correlation.py
================================
CDS v10 데이터셋에서 AD(Affective Drift)와 IDM(Interest Drift Measurement) 상관관계 분석

AD 계산: AffDrift 논문과 동일 방식
  h_long  = Σ wₜ · dist28_t   (EMA, λ=0.5 고정)
  va_long = h_long @ VA_MATRIX
  AD      = ‖aₙ - va_long‖₂

IDM 로드: data/processed/cds_v10/idm.pkl
  포맷 1: {user_idx: [idm_1, idm_2, ...]}          ← compute_idm.py 출력
  포맷 2: {(user_idx, target_item_idx): float}      ← base_dataset.py 기대 포맷

사용법:
  python data_analysis/EDA/ad_idm_correlation_cds.py \
      --data_dir data/processed/cds_v10 \
      --output_dir outputs/data_analysis/results/ad_idm_cds_cor
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm


# ── GoEmotions 28 → VA 고정 변환 행렬 (affdrift.py와 동일) ──────────────────
_GOEMOTIONS_VA = [
    ( 0.82,  0.41),  # admiration
    ( 0.76,  0.60),  # amusement
    (-0.43,  0.67),  # anger
    (-0.55,  0.35),  # annoyance
    ( 0.69,  0.10),  # approval
    ( 0.73,  0.05),  # caring
    (-0.15,  0.30),  # confusion
    ( 0.22,  0.55),  # curiosity
    ( 0.55,  0.59),  # desire
    (-0.63, -0.30),  # disappointment
    (-0.62,  0.20),  # disapproval
    (-0.60,  0.35),  # disgust
    (-0.45,  0.15),  # embarrassment
    ( 0.80,  0.78),  # excitement
    (-0.55,  0.70),  # fear
    ( 0.88, -0.45),  # gratitude
    (-0.75, -0.55),  # grief
    ( 0.90,  0.65),  # joy
    ( 0.88,  0.40),  # love
    (-0.35,  0.55),  # nervousness
    ( 0.72,  0.30),  # optimism
    ( 0.77,  0.45),  # pride
    ( 0.10, -0.10),  # realization
    ( 0.68, -0.50),  # relief
    (-0.58, -0.35),  # remorse
    (-0.70, -0.45),  # sadness
    ( 0.15,  0.60),  # surprise
    ( 0.00,  0.00),  # neutral
]
VA_MATRIX = np.array(_GOEMOTIONS_VA, dtype=np.float32)  # (28, 2)
NEUTRAL_IDX = 27


def compute_ad_for_sequence(seq, lambda_=0.5):
    """
    단일 유저 시퀀스에서 각 타임스텝의 AD 계산.

    Args:
        seq     : [(item_idx, v, a, dist28), ...]  (train.pkl 포맷, CDS)
        lambda_ : EMA 감쇠율

    Returns:
        ad_list : [float, ...]  (t=1,...,len(seq)-1)
        items   : [item_idx, ...]  (타겟 아이템)
    """
    ad_list = []
    items = []

    for t in range(1, len(seq)):
        past = seq[:t]

        # a_n: 마지막 리뷰 VA
        a_n = np.array([past[-1][1], past[-1][2]], dtype=np.float32)

        # dist28 시퀀스
        dist28s = []
        for item_tuple in past:
            if len(item_tuple) >= 4:
                d28 = np.asarray(item_tuple[3], dtype=np.float32)
            else:
                d28 = np.zeros(28, dtype=np.float32)
            dist28s.append(d28)
        dist28s = np.stack(dist28s, axis=0)  # (t, 28)

        L = len(dist28s)

        # EMA 가중치
        t_idx = np.arange(L, dtype=np.float32)
        w = np.exp(-lambda_ * (L - 1 - t_idx))  # (L,)
        w = w / (w.sum() + 1e-8)

        # neutral 제거 후 정규화
        d28 = dist28s.copy()
        d28[:, NEUTRAL_IDX] = 0.0
        d28_sum = d28.sum(axis=-1, keepdims=True).clip(min=1e-8)
        d28 = d28 / d28_sum

        # 장기 감성
        h_long = (w[:, None] * d28).sum(axis=0)  # (28,)
        va_long = h_long @ VA_MATRIX              # (2,)

        ad = float(np.linalg.norm(a_n - va_long))
        ad_list.append(ad)
        items.append(seq[t][0])

    return ad_list, items


def load_idm(idm_path):
    """IDM 로드 (두 가지 포맷 모두 지원)."""
    with open(idm_path, "rb") as f:
        idm_raw = pickle.load(f)

    sample_key = next(iter(idm_raw))
    if isinstance(sample_key, tuple):
        return idm_raw, "tuple"
    else:
        return idm_raw, "list"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="data/processed/cds_v10")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis/ad_idm_cds")
    parser.add_argument("--lambda_",    type=float, default=0.5, help="EMA 감쇠율")
    parser.add_argument("--max_users",  type=int, default=None, help="분석할 최대 유저 수 (None=전체)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print("[1] 데이터 로드...")
    with open(data_dir / "splits" / "train.pkl", "rb") as f:
        train_seqs = pickle.load(f)

    idm_raw, idm_fmt = load_idm(data_dir / "idm.pkl")
    print(f"    IDM 포맷: {idm_fmt}")
    print(f"    유저 수: {len(train_seqs):,}")

    # ── AD 계산 ──────────────────────────────────────────────────────────────
    print("[2] AD 계산 중...")
    ad_values  = []
    idm_values = []

    user_ids = list(train_seqs.keys())
    if args.max_users:
        user_ids = user_ids[:args.max_users]

    for uid in tqdm(user_ids, desc="AD 계산"):
        seq = train_seqs[uid]
        if len(seq) < 2:
            continue

        ad_list, target_items = compute_ad_for_sequence(seq, lambda_=args.lambda_)

        for i, (ad, item) in enumerate(zip(ad_list, target_items)):
            if idm_fmt == "tuple":
                idm_val = idm_raw.get((uid, item), None)
            else:
                user_idm = idm_raw.get(uid, [])
                idm_val = user_idm[i] if i < len(user_idm) else None

            if idm_val is None:
                continue

            ad_values.append(float(ad))
            idm_values.append(float(idm_val))

    ad_arr  = np.array(ad_values)
    idm_arr = np.array(idm_values)
    print(f"    유효 샘플: {len(ad_arr):,}")

    # ── 상관관계 분석 ────────────────────────────────────────────────────────
    print("[3] 상관관계 분석...")
    pearson_r,  pearson_p  = stats.pearsonr(ad_arr, idm_arr)
    spearman_r, spearman_p = stats.spearmanr(ad_arr, idm_arr)

    print(f"\n{'='*50}")
    print(f"  AD-IDM 상관관계 (n={len(ad_arr):,})")
    print(f"{'='*50}")
    print(f"  Pearson  r = {pearson_r:+.4f}  (p={pearson_p:.2e})")
    print(f"  Spearman ρ = {spearman_r:+.4f}  (p={spearman_p:.2e})")
    print(f"\n  AD  — mean={ad_arr.mean():.4f}, std={ad_arr.std():.4f}, "
          f"min={ad_arr.min():.4f}, max={ad_arr.max():.4f}")
    print(f"  IDM — mean={idm_arr.mean():.4f}, std={idm_arr.std():.4f}, "
          f"min={idm_arr.min():.4f}, max={idm_arr.max():.4f}")
    print(f"{'='*50}\n")

    # ── 시각화 ──────────────────────────────────────────────────────────────
    print("[4] 시각화...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"AD–IDM Correlation Analysis (CDS v10, n={len(ad_arr):,})", fontsize=14)

    # 1. 산점도
    ax = axes[0, 0]
    ax.scatter(ad_arr, idm_arr, alpha=0.05, s=5, color="steelblue")
    z = np.polyfit(ad_arr, idm_arr, 1)
    x_line = np.linspace(ad_arr.min(), ad_arr.max(), 200)
    ax.plot(x_line, np.poly1d(z)(x_line), "r-", linewidth=1.5, label=f"trend (r={pearson_r:+.3f})")
    ax.set_xlabel("AD (Affective Drift)")
    ax.set_ylabel("IDM (Interest Drift Measurement)")
    ax.set_title("Scatter Plot")
    ax.legend()

    # 2. AD 분포
    ax = axes[0, 1]
    ax.hist(ad_arr, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(ad_arr.mean(), color="red", linestyle="--", label=f"mean={ad_arr.mean():.3f}")
    ax.set_xlabel("AD")
    ax.set_ylabel("Count")
    ax.set_title("AD Distribution")
    ax.legend()

    # 3. IDM 분포
    ax = axes[1, 0]
    ax.hist(idm_arr, bins=30, color="coral", edgecolor="white", linewidth=0.3)
    ax.axvline(idm_arr.mean(), color="red", linestyle="--", label=f"mean={idm_arr.mean():.3f}")
    ax.set_xlabel("IDM")
    ax.set_ylabel("Count")
    ax.set_title("IDM Distribution")
    ax.legend()

    # 4. IDM 구간별 AD 분포 (박스플롯)
    ax = axes[1, 1]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    groups = [ad_arr[(idm_arr >= bins[i]) & (idm_arr < bins[i+1])] for i in range(len(bins)-1)]
    ax.boxplot(groups, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_xlabel("IDM range")
    ax.set_ylabel("AD")
    ax.set_title(f"AD by IDM Bin\n(Pearson r={pearson_r:+.4f}, Spearman ρ={spearman_r:+.4f})")

    plt.tight_layout()
    fig_path = output_dir / "ad_idm_correlation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    저장: {fig_path}")

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    result_path = output_dir / "ad_idm_stats.txt"
    with open(result_path, "w") as f:
        f.write(f"AD-IDM Correlation Analysis (CDS v10)\n")
        f.write(f"{'='*50}\n")
        f.write(f"n_samples   : {len(ad_arr):,}\n")
        f.write(f"Pearson  r  : {pearson_r:+.4f}  (p={pearson_p:.2e})\n")
        f.write(f"Spearman rho: {spearman_r:+.4f}  (p={spearman_p:.2e})\n")
        f.write(f"\nAD  — mean={ad_arr.mean():.4f}, std={ad_arr.std():.4f}\n")
        f.write(f"IDM — mean={idm_arr.mean():.4f}, std={idm_arr.std():.4f}\n")
    print(f"    저장: {result_path}")
    print("\n완료!")


if __name__ == "__main__":
    main()
