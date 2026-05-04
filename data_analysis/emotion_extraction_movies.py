"""
eda_idm_adm_v4.py
=================
Category-based IDM vs ADM correlation analysis — v4

Changes from v3:
  [Required] Added EXCLUDE_CATS filter — removes dataset-name categories such as "Movies & TV", "Prime Video"
             → Prevents artificial skewing of IDM distribution (reflects IDURL sequential_dataset.py:157)
  [Optional] Repeat item handling — if the target has already appeared in the past sequence, set IDM=0
             (reflects IDURL line:151; minor effect in EDA but added for consistency)
  [Note]     IDQ quantization is not applied in EDA — continuous values are more appropriate here
             When used as a model feature, separate quantization is required (noted in comments)

File location: preprocessing/eda_idm_adm_v4.py
Usage:
  python preprocessing/eda_idm_adm_v4.py \
      --review_path  data/raw/Movies_and_TV.jsonl \
      --meta_path    data/raw/meta_Movies_and_TV.jsonl \
      --emotion_dir  data_analysis/results \
      --sample_users 200 \
      --min_seq_len  5 \
      --output_dir   data_analysis/results \
      --device cuda --gpu_id 1
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# [Key fix] Reflects IDURL sequential_dataset.py:157
# Dataset-name and top-level categories are excluded from IDM computation
# → Including these categories creates intersection for almost all item pairs, artificially pushing IDM toward 0
EXCLUDE_CATS = {
    "Movies & TV",
    "Prime Video",
    "Featured Categories",
    "Genre for Featured Categories",
    "Amazon Video",
}

# GoEmotions
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

QUADRANT_INFO = {
    "Q1 (High V, High A)": {"color": "#E8A838"},
    "Q2 (Low V, High A)":  {"color": "#E85252"},
    "Q3 (High V, Low A)":  {"color": "#52B788"},
    "Q4 (Low V, Low A)":   {"color": "#6B8CBA"},
}

def get_quadrant(v, a):
    if   v >= 0 and a >= 0: return "Q1 (High V, High A)"
    elif v <  0 and a >= 0: return "Q2 (Low V, High A)"
    elif v >= 0 and a <  0: return "Q3 (High V, Low A)"
    else:                   return "Q4 (Low V, Low A)"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metadata loading
# ─────────────────────────────────────────────────────────────────────────────
def load_item_categories(meta_path: str) -> dict:
    """parent_asin → frozenset(categories) — stored after removing EXCLUDE_CATS"""
    print(f"[Meta] Loading: {meta_path}")
    asin2cats = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading meta", mininterval=5):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = obj.get("parent_asin", "")
            raw  = obj.get("categories") or []
            # Apply EXCLUDE_CATS filter
            cats = frozenset(
                c.strip() for c in raw
                if c.strip() and c.strip() not in EXCLUDE_CATS
            )
            asin2cats[asin] = cats

    n_with = sum(1 for v in asin2cats.values() if len(v) > 0)
    print(f"[Meta] {len(asin2cats):,} items | "
          f"with valid categories (excluding {'/'.join(list(EXCLUDE_CATS)[:2])}...): "
          f"{n_with:,} ({n_with/len(asin2cats)*100:.1f}%)")
    return asin2cats


# ─────────────────────────────────────────────────────────────────────────────
# 2. Review loading (user-centric)
# ─────────────────────────────────────────────────────────────────────────────
def load_user_sequences(review_path, asin2cats, sample_users=200,
                        min_seq_len=5, seed=42):
    print(f"[Load] Scanning reviews: {review_path}")
    rng = random.Random(seed)
    user_reviews = defaultdict(list)

    with open(review_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning", mininterval=5):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (obj.get("text") or "").strip()
            if len(text) < 20:
                continue
            uid  = obj.get("user_id", "")
            asin = obj.get("parent_asin", "")
            if not uid or not asin:
                continue
            # Include even if categories are missing (handled during IDM computation)
            user_reviews[uid].append({
                "user_id":     uid,
                "parent_asin": asin,
                "timestamp":   obj.get("timestamp", 0),
                "rating":      float(obj.get("rating", 3.0)),
                "text":        text,
                "categories":  asin2cats.get(asin, frozenset()),
            })

    eligible = [u for u, r in user_reviews.items() if len(r) >= min_seq_len]
    print(f"[Load] Total users: {len(user_reviews):,} | "
          f"seq≥{min_seq_len}: {len(eligible):,}")

    n = min(sample_users, len(eligible))
    sampled = rng.sample(eligible, n)
    records = [r for uid in sampled for r in user_reviews[uid]]

    df = pd.DataFrame(records).sort_values(
        ["user_id", "timestamp"]).reset_index(drop=True)
    print(f"[Load] Sampled {n} users | total reviews: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. VA acquisition (cache-first)
# ─────────────────────────────────────────────────────────────────────────────
def get_va_values(df, emotion_dir, batch_size=32, device="cpu"):
    cache = Path(emotion_dir) / "va_results.csv"
    if cache.exists():
        print(f"[VA] Loading from cache: {cache}")
        va_df  = pd.read_csv(cache)
        merged = df.merge(
            va_df[["user_id", "timestamp", "valence", "arousal"]],
            on=["user_id", "timestamp"], how="left"
        )
        missing = merged["valence"].isna()
        if missing.sum() > 0:
            print(f"[VA] {missing.sum()} cache misses → running inference")
            v_new, a_new = _infer_goemotions(
                df[missing]["text"].tolist(), batch_size, device)
            merged.loc[missing, "valence"] = v_new
            merged.loc[missing, "arousal"] = a_new
        return merged["valence"].values, merged["arousal"].values

    print("[VA] No cache found → running full inference")
    v, a = _infer_goemotions(df["text"].tolist(), batch_size, device)
    save = df[["user_id", "timestamp"]].copy()
    save["valence"], save["arousal"] = v, a
    os.makedirs(emotion_dir, exist_ok=True)
    save.to_csv(cache, index=False)
    print(f"[VA] Saved: {cache}")
    return v, a


def _infer_goemotions(texts, batch_size, device, top_k=5):
    from transformers import pipeline
    print(f"[GoEmotions] Running inference on {len(texts):,} texts...")
    clf = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None, truncation=True, max_length=128,
        device=0 if device == "cuda" else -1,
    )
    label2idx = {lb: i for i, lb in enumerate(GOEMOTIONS_LABELS)}
    va_mat = np.array(
        [GOEMOTIONS_VA.get(lb, (0.0, 0.0)) for lb in GOEMOTIONS_LABELS],
        dtype=np.float32)

    V, A = [], []
    for s in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch = texts[s: s + batch_size]
        mtx = np.zeros((len(batch), 28), dtype=np.float32)
        for bi, preds in enumerate(clf(batch)):
            for p in preds:
                idx = label2idx.get(p["label"])
                if idx is not None:
                    mtx[bi, idx] = p["score"]
        mtx[:, NEUTRAL_IDX] = 0.0          # exclude neutral
        if top_k < 27:                      # top-K masking
            for i in range(len(mtx)):
                row = mtx[i].copy()
                top_idx = np.argsort(row)[::-1][:top_k]
                mask = np.zeros(28); mask[top_idx] = 1.0
                mtx[i] = row * mask
        s_ = mtx.sum(axis=1, keepdims=True)
        s_ = np.where(s_ == 0, 1, s_)
        n  = mtx / s_
        V.extend((n @ va_mat[:, 0]).tolist())
        A.extend((n @ va_mat[:, 1]).tolist())
    return np.array(V), np.array(A)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Category-based IDM (IDURL formula + applied fixes)
# ─────────────────────────────────────────────────────────────────────────────
def compute_category_idm(df: pd.DataFrame):
    """
    IDURL formula:
      IDM = 1 - |cats(target) ∩ ∪cats(seq)| / |cats(target)|

    Applied fixes:
      - EXCLUDE_CATS: already removed in load_item_categories()
      - repeat item: if target already appears in seq, set IDM=0 (IDURL line:151)
      - if cats(target) is empty, skip that user

    NOTE: IDQ quantization (discretization into 1~4) is not applied here — continuous values are preferred for EDA.
          Use the quantize_idm() function below when preparing model features.
    """
    idm_list, uid_list = [], []

    for uid, grp in df.groupby("user_id"):
        grp    = grp.sort_values("timestamp")
        asins  = grp["parent_asin"].tolist()
        cats   = list(grp["categories"])   # list of frozenset

        if len(asins) < 2:
            continue

        target_asin = asins[-1]
        target_cats = cats[-1]
        seq_asins   = asins[:-1]
        seq_cats    = set().union(*cats[:-1]) if len(cats) > 1 else set()

        # skip if target has no categories
        if len(target_cats) == 0:
            continue

        # repeat item handling (IDURL line:151)
        if target_asin in seq_asins:
            idm_list.append(0.0)
            uid_list.append(uid)
            continue

        intersection = target_cats & seq_cats
        idm = 1.0 - len(intersection) / len(target_cats)
        idm_list.append(idm)
        uid_list.append(uid)

    return np.array(idm_list), uid_list


def quantize_idm(idm_arr: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """
    IDURL IDQ quantization (for model features; not used in EDA):
      same_ratio = 1 - IDM
      1.0        → degree 1  (completely identical)
      0.5~1.0    → degree 2
      0.0~0.5    → degree 3
      0.0        → degree 4  (completely new category)
    """
    same = 1.0 - idm_arr
    deg  = np.where(same == 1.0, 1,
           np.where(same >= 0.5,  2,
           np.where(same >  0.0,  3, 4)))
    return deg


# ─────────────────────────────────────────────────────────────────────────────
# 5. ADM computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_adm(df, valence, arousal, valid_uids):
    df = df.copy()
    df["valence"] = valence
    df["arousal"] = arousal
    adm_list = []
    for uid in valid_uids:
        grp   = df[df["user_id"] == uid].sort_values("timestamp")
        v_seq = grp["valence"].values
        a_seq = grp["arousal"].values
        if len(v_seq) < 2:
            adm_list.append(np.nan); continue
        adm = 0.5 * abs(v_seq[-1] - v_seq[:-1].mean()) + \
              0.5 * abs(a_seq[-1] - a_seq[:-1].mean())
        adm_list.append(adm)
    return np.array(adm_list)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization
# ─────────────────────────────────────────────────────────────────────────────
def plot_idm_dist(idm, out_path):
    vals = idm[np.isfinite(idm)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=np.linspace(0, 1, 21), color="#6B8CBA",
            edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color="#E85252", lw=1.5, ls="--",
               label=f"Mean = {vals.mean():.3f}")
    # show quantize boundary lines
    for thr, lbl in [(0.0, "deg1"), (0.5, "deg2/3"), (1.0, "deg4")]:
        ax.axvline(thr, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("IDM  (category-based, EXCLUDE_CATS applied)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("IDM Distribution  (top-level categories excluded)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] IDM dist → {out_path}")
    # print distribution summary
    print(f"       IDM=0 (repeat or full overlap): {(vals==0).mean()*100:.1f}%")
    print(f"       IDM=1 (completely new cats):     {(vals==1).mean()*100:.1f}%")
    print(f"       IDM mean={vals.mean():.4f}  std={vals.std():.4f}")


def plot_idm_adm_corr(idm, adm, out_path):
    mask = np.isfinite(idm) & np.isfinite(adm)
    x, y = idm[mask], adm[mask]
    n    = len(x)

    if n < 10:
        print(f"[⚠️] Insufficient valid samples: {n}.")
        return None, None

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.45, s=18, color="#6B8CBA", linewidths=0)
    m, b = np.polyfit(x, y, 1)
    xl   = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, m * xl + b, color="#E85252", lw=1.5, ls="--")

    ax.set_xlabel("IDM  (category-based behavioral drift)", fontsize=10)
    ax.set_ylabel("ADM  (GoEmotions VA-based affective drift)", fontsize=10)
    ax.set_title(
        f"IDM vs ADM  —  n={n} users\n"
        f"Pearson r={pr:.3f} (p={pp:.3f})   Spearman ρ={sr:.3f} (p={sp:.3f})",
        fontsize=10,
    )

    if abs(pr) < 0.3:
        msg   = (f"✅  r = {pr:.3f}  <  0.3\n"
                 f"→ Affective drift is independent\n"
                 f"   of behavioral drift\n"
                 f"→ Motivation confirmed ✓")
        color = "green"
    else:
        msg   = (f"⚠️   r = {pr:.3f}  ≥  0.3\n"
                 f"→ Correlation exists.\n"
                 f"   Consider increasing sample_users\n"
                 f"   or using Beauty/Sports data")
        color = "darkorange"

    ax.text(0.03, 0.97, msg, transform=ax.transAxes, fontsize=8.5, va="top",
            color=color, bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                   alpha=0.8, ec=color))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] IDM vs ADM → {out_path}")
    print(f"       Pearson r={pr:.4f}  Spearman ρ={sr:.4f}  n={n}")
    return pr, sr


def plot_idm0_adm_high(idm, adm, out_path, idm_thr=0.1, adm_thr=0.1):
    """Key motivation figure for the paper: proportion of IDM≈0 cases where ADM>0"""
    idm_zero = idm <= idm_thr
    n_zero   = idm_zero.sum()
    n_high   = (adm[idm_zero] > adm_thr).sum()
    ratio    = n_high / n_zero * 100 if n_zero > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(adm[idm_zero], bins=20, color="#6B8CBA",
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(adm_thr, color="#E85252", lw=1.5, ls="--",
                    label=f"ADM threshold={adm_thr}")
    axes[0].set_xlabel("ADM (Affective Drift)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title(
        f"ADM distribution when IDM ≤ {idm_thr}\n(n={n_zero} users)", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].pie(
        [n_high, n_zero - n_high],
        labels=[f"ADM > {adm_thr}\n({ratio:.1f}%)",
                f"ADM ≤ {adm_thr}\n({100-ratio:.1f}%)"],
        colors=["#E85252", "#D3D3D3"],
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9},
    )
    axes[1].set_title(
        f"Among IDM≈0 users:\nhow many show affective drift?", fontsize=10)

    fig.suptitle(
        "Behavioral Drift ≈ 0  BUT  Affective Drift > 0\n"
        "→ Emotion captures signals invisible to behavior  (Motivation Evidence)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] IDM≈0 & ADM>0 → {out_path}")
    print(f"       IDM≤{idm_thr}: {n_zero} users, ADM>{adm_thr}: {n_high} ({ratio:.1f}%)")
    return ratio


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_path",  default="data/raw/Movies_and_TV.jsonl")
    parser.add_argument("--meta_path",    default="data/raw/meta_Movies_and_TV.jsonl")
    parser.add_argument("--emotion_dir",  default="data_analysis/results")
    parser.add_argument("--sample_users", type=int, default=200)
    parser.add_argument("--min_seq_len",  type=int, default=5)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--device",       default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpu_id",       type=int, default=1)
    parser.add_argument("--output_dir",   default="data_analysis/results")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    os.makedirs(args.output_dir, exist_ok=True)
    p = lambda name: os.path.join(args.output_dir, name)

    # 1. Load metadata (with EXCLUDE_CATS applied)
    asin2cats = load_item_categories(args.meta_path)

    # 2. Load reviews
    df = load_user_sequences(
        args.review_path, asin2cats,
        sample_users=args.sample_users,
        min_seq_len=args.min_seq_len,
        seed=args.seed,
    )

    # 3. VA values
    valence, arousal = get_va_values(
        df, args.emotion_dir, args.batch_size, args.device)

    # 4. IDM (category-based + EXCLUDE_CATS + repeat item)
    print("\n[IDM] Computing...")
    idm_vals, valid_uids = compute_category_idm(df)
    print(f"[IDM] Valid users: {len(idm_vals)}")
    plot_idm_dist(idm_vals, p("idm_dist_v4.png"))

    # 5. ADM computation
    adm_vals = compute_adm(df, valence, arousal, valid_uids)
    print(f"[ADM] mean={np.nanmean(adm_vals):.4f}  std={np.nanstd(adm_vals):.4f}")

    # 6. Correlation analysis + motivation figure
    pr, sr = plot_idm_adm_corr(idm_vals, adm_vals, p("idm_adm_corr_v4.png"))
    ratio  = plot_idm0_adm_high(idm_vals, adm_vals, p("idm0_adm_high_v4.png"))

    # 7. Summary
    summary = {
        "version":              "v4",
        "idm_type":             "category-based (EXCLUDE_CATS + repeat_item)",
        "exclude_cats":         sorted(EXCLUDE_CATS),
        "n_users":              len(valid_uids),
        "n_reviews":            len(df),
        "idm_mean":             round(float(np.nanmean(idm_vals)), 4),
        "idm_std":              round(float(np.nanstd(idm_vals)),  4),
        "idm_zero_ratio_%":     round(float((idm_vals == 0).mean() * 100), 1),
        "idm_one_ratio_%":      round(float((idm_vals == 1).mean() * 100), 1),
        "adm_mean":             round(float(np.nanmean(adm_vals)), 4),
        "adm_std":              round(float(np.nanstd(adm_vals)),  4),
        "pearson_r":            round(float(pr), 4) if pr is not None else None,
        "spearman_r":           round(float(sr), 4) if sr is not None else None,
        "motivation_confirmed": bool(abs(pr) < 0.3) if pr is not None else None,
        "idm0_adm_high_%":      round(ratio, 1),
    }
    out_json = p("summary_v4.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()