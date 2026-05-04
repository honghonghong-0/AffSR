"""
emotion_extraction_cds.py
=========================
Emotion extraction and IDM/ADM analysis on CDs_and_Vinyl preprocessed data.

Pipeline:
  1. Load preprocessed CSV (data/processed/CDs_and_Vinyl_processed.csv)
  2. Sample users with seq_len >= min_seq_len
  3. GoEmotions -> VA mapping + store 28-dim emotion probabilities
  4. Compute IDM (category-based, IDURL formula)
  5. Compute ADM (VA-based affective drift)
  6. Correlation analysis + visualization
  7. Save CSV results

Usage:
  python preprocessing/emotion_extraction_cds.py \
      --processed_path data/processed/CDs_and_Vinyl_processed.csv \
      --sample_users 200 \
      --min_seq_len 5 \
      --batch_size 32 \
      --device cuda \
      --gpu_id 3 \
      --output_dir data_analysis/results
"""

import argparse
import ast
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EXCLUDE_CATS = {
    "CDs & Vinyl",
    "Digital Music",
}

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

def emotion_quadrant(label):
    v, a = GOEMOTIONS_VA.get(label, (0, 0))
    return get_quadrant(v, a)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load preprocessed CSV and sample users
# ─────────────────────────────────────────────────────────────────────────────
def load_processed_data(processed_path, sample_users=200, min_seq_len=5, seed=42):
    print(f"[Load] Loading preprocessed data: {processed_path}")
    df = pd.read_csv(processed_path)

    def parse_cats(x):
        try:
            cats = ast.literal_eval(x) if isinstance(x, str) else (x or [])
        except Exception:
            cats = []
        return frozenset(
            c.strip() for c in cats
            if c.strip() and c.strip() not in EXCLUDE_CATS
        )

    df["categories"] = df["categories"].apply(parse_cats)
    df = df[df["text"].astype(str).str.strip().str.len() >= 20].copy()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    seq_counts = df.groupby("user_id").size()
    eligible = seq_counts[seq_counts >= min_seq_len].index.tolist()
    print(f"[Load] Total users: {df['user_id'].nunique():,} | seq>={min_seq_len}: {len(eligible):,}")

    rng = random.Random(seed)
    n = min(sample_users, len(eligible))
    sampled = rng.sample(eligible, n)
    df = df[df["user_id"].isin(sampled)].reset_index(drop=True)
    print(f"[Load] Sampled {n} users | total reviews: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. GoEmotions inference -> VA + 28-dim emotion probabilities
# ─────────────────────────────────────────────────────────────────────────────
def get_va_values(df, emotion_dir, batch_size=32, device="cpu"):
    cache = Path(emotion_dir) / "va_results_cds.csv"
    if cache.exists():
        print(f"[VA] Loading cache: {cache}")
        va_df = pd.read_csv(cache)
        has_probs = all(lb in va_df.columns for lb in GOEMOTIONS_LABELS)
        if not has_probs:
            print("[VA] Cache missing emotion probabilities -> re-running inference")
            os.remove(cache)
            return get_va_values(df, emotion_dir, batch_size, device)

        merged = df.merge(
            va_df[["user_id", "timestamp", "valence", "arousal"] + GOEMOTIONS_LABELS],
            on=["user_id", "timestamp"], how="left"
        )
        missing = merged["valence"].isna()
        if missing.sum() > 0:
            print(f"[VA] {missing.sum()} unmatched rows -> running inference")
            v_new, a_new, probs_new = _infer_goemotions(
                df[missing]["text"].tolist(), batch_size, device)
            merged.loc[missing, "valence"] = v_new
            merged.loc[missing, "arousal"] = a_new
            for i, lb in enumerate(GOEMOTIONS_LABELS):
                merged.loc[missing, lb] = probs_new[:, i]
        emotion_probs = merged[GOEMOTIONS_LABELS].values
        return merged["valence"].values, merged["arousal"].values, emotion_probs

    print("[VA] No cache found -> running full inference")
    v, a, probs = _infer_goemotions(df["text"].tolist(), batch_size, device)
    save = df[["user_id", "timestamp"]].copy()
    save["valence"], save["arousal"] = v, a
    for i, lb in enumerate(GOEMOTIONS_LABELS):
        save[lb] = probs[:, i]
    os.makedirs(emotion_dir, exist_ok=True)
    save.to_csv(cache, index=False)
    print(f"[VA] Saved: {cache}")
    return v, a, probs


def _infer_goemotions(texts, batch_size, device, top_k=5):
    from transformers import pipeline
    print(f"[GoEmotions] Inferring {len(texts):,} texts...")
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

    V, A, ALL_PROBS = [], [], []
    for s in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch = texts[s: s + batch_size]
        mtx = np.zeros((len(batch), 28), dtype=np.float32)
        for bi, preds in enumerate(clf(batch)):
            for p in preds:
                idx = label2idx.get(p["label"])
                if idx is not None:
                    mtx[bi, idx] = p["score"]
        mtx[:, NEUTRAL_IDX] = 0.0
        if top_k < 27:
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
        ALL_PROBS.append(n)
    return np.array(V), np.array(A), np.vstack(ALL_PROBS)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute IDM
# ─────────────────────────────────────────────────────────────────────────────
def compute_category_idm(df: pd.DataFrame):
    idm_list, uid_list = [], []
    for uid, grp in df.groupby("user_id"):
        grp   = grp.sort_values("timestamp")
        asins = grp["parent_asin"].tolist()
        cats  = list(grp["categories"])
        if len(asins) < 2:
            continue
        target_asin = asins[-1]
        target_cats = cats[-1]
        seq_asins   = asins[:-1]
        seq_cats    = set().union(*cats[:-1]) if len(cats) > 1 else set()
        if len(target_cats) == 0:
            continue
        if target_asin in seq_asins:
            idm_list.append(0.0)
            uid_list.append(uid)
            continue
        intersection = target_cats & seq_cats
        idm = 1.0 - len(intersection) / len(target_cats)
        idm_list.append(idm)
        uid_list.append(uid)
    return np.array(idm_list), uid_list


# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute ADM
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
            adm_list.append(np.nan)
            continue
        adm = 0.5 * abs(v_seq[-1] - v_seq[:-1].mean()) + \
              0.5 * abs(a_seq[-1] - a_seq[:-1].mean())
        adm_list.append(adm)
    return np.array(adm_list)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualization
# ─────────────────────────────────────────────────────────────────────────────
def plot_va_scatter(df, valence, arousal, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    quadrants = [get_quadrant(v, a) for v, a in zip(valence, arousal)]
    for q, info in QUADRANT_INFO.items():
        mask = [qd == q for qd in quadrants]
        ax.scatter(
            np.array(valence)[mask], np.array(arousal)[mask],
            alpha=0.3, s=10, color=info["color"],
            label=f"{q} ({sum(mask)})"
        )
    ax.set_xlabel("Valence", fontsize=10)
    ax.set_ylabel("Arousal", fontsize=10)
    ax.set_title("VA Distribution — CDs & Vinyl", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] VA scatter -> {out_path}")


def plot_va_quadrant_bar(valence, arousal, out_path):
    quadrants = [get_quadrant(v, a) for v, a in zip(valence, arousal)]
    order  = list(QUADRANT_INFO.keys())
    total  = len(quadrants)
    values = [quadrants.count(q) / total * 100 for q in order]
    colors = [QUADRANT_INFO[q]["color"] for q in order]
    labels = ["Q1\nHigh V\nHigh A", "Q2\nLow V\nHigh A",
              "Q3\nHigh V\nLow A",  "Q4\nLow V\nLow A"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.axhline(25, color="gray", lw=1.2, ls="--", label="Uniform (25%)")
    ax.set_ylabel("Proportion (%)", fontsize=10)
    ax.set_title("VA Quadrant Distribution (neutral excluded)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] VA quadrant bar -> {out_path}")


def plot_top10_emotions(emotion_probs, out_path, top_k=10):
    mean_probs = emotion_probs.mean(axis=0)
    sorted_idx = np.argsort(mean_probs)[::-1][:top_k]
    labels = [GOEMOTIONS_LABELS[i] for i in sorted_idx]
    values = [mean_probs[i] for i in sorted_idx]
    colors = [QUADRANT_INFO[emotion_quadrant(lb)]["color"]
              if lb != "neutral" else "#AAAAAA" for lb in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Mean Probability", fontsize=10)
    ax.set_title(f"Top-{top_k} GoEmotions Labels in Sample", fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=info["color"], label=q)
                       for q, info in QUADRANT_INFO.items()]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Top-10 emotions -> {out_path}")


def plot_idm_dist(idm, out_path):
    vals = idm[np.isfinite(idm)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=np.linspace(0, 1, 21), color="#6B8CBA",
            edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color="#E85252", lw=1.5, ls="--",
               label=f"Mean = {vals.mean():.3f}")
    for thr in [0.0, 0.5, 1.0]:
        ax.axvline(thr, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("IDM (category-based, EXCLUDE_CATS applied)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("IDM Distribution — CDs & Vinyl", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] IDM dist -> {out_path}")
    print(f"       IDM=0: {(vals==0).mean()*100:.1f}%")
    print(f"       IDM=1: {(vals==1).mean()*100:.1f}%")
    print(f"       IDM mean={vals.mean():.4f}  std={vals.std():.4f}")


def plot_idm_adm_corr(idm, adm, out_path):
    mask = np.isfinite(idm) & np.isfinite(adm)
    x, y = idm[mask], adm[mask]
    n    = len(x)
    if n < 10:
        print(f"[Warning] Only {n} valid samples — insufficient.")
        return None, None
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.45, s=18, color="#6B8CBA", linewidths=0)
    m, b = np.polyfit(x, y, 1)
    xl   = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, m * xl + b, color="#E85252", lw=1.5, ls="--")
    ax.set_xlabel("IDM (category-based behavioral drift)", fontsize=10)
    ax.set_ylabel("ADM (GoEmotions VA-based affective drift)", fontsize=10)
    ax.set_title(
        f"IDM vs ADM — n={n} users\n"
        f"Pearson r={pr:.3f} (p={pp:.3f})   Spearman r={sr:.3f} (p={sp:.3f})",
        fontsize=10,
    )
    if abs(pr) < 0.3:
        msg   = (f"r = {pr:.3f} < 0.3\n"
                 f"-> Affective drift is independent\n"
                 f"   of behavioral drift\n"
                 f"-> Motivation confirmed")
        color = "green"
    else:
        msg   = (f"r = {pr:.3f} >= 0.3\n"
                 f"-> Correlation exists.")
        color = "darkorange"
    ax.text(0.03, 0.97, msg, transform=ax.transAxes, fontsize=8.5, va="top",
            color=color, bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                   alpha=0.8, ec=color))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] IDM vs ADM -> {out_path}")
    print(f"       Pearson r={pr:.4f}  Spearman r={sr:.4f}  n={n}")
    return pr, sr


def plot_idm0_adm_high(idm, adm, out_path, idm_thr=0.1, adm_thr=0.1):
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
        f"ADM distribution when IDM <= {idm_thr}\n(n={n_zero} users)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].pie(
        [n_high, n_zero - n_high],
        labels=[f"ADM > {adm_thr}\n({ratio:.1f}%)",
                f"ADM <= {adm_thr}\n({100-ratio:.1f}%)"],
        colors=["#E85252", "#D3D3D3"],
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9},
    )
    axes[1].set_title(
        f"Among IDM~0 users:\nhow many show affective drift?", fontsize=10)
    fig.suptitle(
        "Behavioral Drift ~ 0  BUT  Affective Drift > 0\n"
        "-> Emotion captures signals invisible to behavior  (Motivation Evidence)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] IDM~0 & ADM>0 -> {out_path}")
    print(f"       IDM<={idm_thr}: {n_zero} users with ADM>{adm_thr}: {n_high} ({ratio:.1f}%)")
    return ratio


# ─────────────────────────────────────────────────────────────────────────────
# 6. Save results
# ─────────────────────────────────────────────────────────────────────────────
def save_results(df, valence, arousal, emotion_probs, valid_uids, idm_vals, adm_vals, out_path):
    df = df.copy()
    df["valence"] = valence
    df["arousal"] = arousal
    df["quadrant"] = [get_quadrant(v, a) for v, a in zip(valence, arousal)]
    for i, lb in enumerate(GOEMOTIONS_LABELS):
        df[lb] = emotion_probs[:, i]
    uid2idm = dict(zip(valid_uids, idm_vals))
    uid2adm = dict(zip(valid_uids, adm_vals))
    df["idm"] = df["user_id"].map(uid2idm)
    df["adm"] = df["user_id"].map(uid2adm)
    df.to_csv(out_path, index=False)
    print(f"[Save] Results CSV -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_path", default="data/processed/CDs_and_Vinyl_processed.csv")
    parser.add_argument("--sample_users",   type=int, default=200)
    parser.add_argument("--min_seq_len",    type=int, default=5)
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--device",         default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpu_id",         type=int, default=3)
    parser.add_argument("--output_dir",     default="data_analysis/results")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    os.makedirs(args.output_dir, exist_ok=True)
    p = lambda name: os.path.join(args.output_dir, name)

    # 1. Load data
    df = load_processed_data(
        args.processed_path,
        sample_users=args.sample_users,
        min_seq_len=args.min_seq_len,
        seed=args.seed,
    )

    # 2. VA + emotion probability inference
    valence, arousal, emotion_probs = get_va_values(
        df, args.output_dir, args.batch_size, args.device)

    # 3. Visualization
    plot_va_scatter(df, valence, arousal, p("va_scatter_cds.png"))
    plot_va_quadrant_bar(valence, arousal, p("va_quadrant_bar_cds.png"))
    plot_top10_emotions(emotion_probs, p("top10_emotions_cds.png"))

    # 4. IDM
    print("\n[IDM] Computing...")
    idm_vals, valid_uids = compute_category_idm(df)
    print(f"[IDM] Valid users: {len(idm_vals)}")
    plot_idm_dist(idm_vals, p("idm_dist_cds.png"))

    # 5. ADM
    adm_vals = compute_adm(df, valence, arousal, valid_uids)
    print(f"[ADM] mean={np.nanmean(adm_vals):.4f}  std={np.nanstd(adm_vals):.4f}")

    # 6. Correlation analysis
    pr, sr = plot_idm_adm_corr(idm_vals, adm_vals, p("idm_adm_corr_cds.png"))
    ratio  = plot_idm0_adm_high(idm_vals, adm_vals, p("idm0_adm_high_cds.png"))

    # 7. Save results
    save_results(df, valence, arousal, emotion_probs, valid_uids,
                 idm_vals, adm_vals, p("emotion_results_cds.csv"))

    # 8. Summary JSON
    summary = {
        "version":              "cds_v2",
        "dataset":              "CDs_and_Vinyl",
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
    out_json = p("summary_cds.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
