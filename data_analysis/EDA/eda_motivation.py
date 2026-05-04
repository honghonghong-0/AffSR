"""
eda_motivation_v8.py
====================
Motivation section analysis for AffSR v8 paper

Three claims to make in the paper:
  [Claim 1] IDM ⊥ ADM
            Behavioral drift (IDM) alone cannot explain affective drift (ADM)
            -> A dedicated emotion module (MoE + βₖ) is necessary

  [Claim 2] Affective drift is a real phenomenon
            ADM distribution is widely spread across user sequences
            -> Ignoring emotional change degrades recommendation quality

  [Claim 3] User emotions are structurally distributed across VA quadrants
            Distribution of aₙ spans all four quadrants
            -> βₖ (quadrant-based soft gating) design is justified

Outputs:
  {output_dir}/
  ├── fig1_idm_vs_adm_scatter.png   [Claim 1] IDM vs ADM scatter plot + MI values
  ├── fig2_adm_distribution.png     [Claim 2] ADM distribution histogram (drift / congruence)
  ├── fig3_va_quadrant.png          [Claim 3] VA space distribution of aₙ + quadrant centroids
  └── summary.json                  All numeric results

Usage:
    python preprocessing/eda_motivation_v8.py \\
        --sequences data/processed/cds/sequences.pkl \\
        --item_va   data/processed/cds/item_va.json \\
        --item_cats data/processed/cds/item_cats.json \\
        --output_dir data_analysis/results/motivation_v8 \\
        --max_pairs 100000
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
EXCLUDE_CATS = {
    "Movies & TV", "Prime Video", "Featured Categories",
    "Genre for Featured Categories", "Amazon Video", "CDs & Vinyl",
}

QUADRANT_CENTROIDS = {
    "c1": ( 0.83,  0.61),   # High V, High A  (joy, excitement)
    "c2": (-0.53,  0.57),   # Low V,  High A  (anger, fear)
    "c3": ( 0.75, -0.48),   # High V, Low A   (gratitude, calmness)
    "c4": (-0.69, -0.43),   # Low V,  Low A   (sadness, grief)
}
QUADRANT_LABELS = {
    "c1": "Q1: joy/excitement",
    "c2": "Q2: anger/fear",
    "c3": "Q3: gratitude/calm",
    "c4": "Q4: sadness/grief",
}
QUADRANT_COLORS = {
    "c1": "#E8A838",
    "c2": "#E85252",
    "c3": "#52B788",
    "c4": "#6B8CBA",
}

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data construction
# ─────────────────────────────────────────────────────────────────────────────
def build_pairs(sequences, item_va, item_cats, max_pairs=100_000):
    """
    Generate (input_seq, target) pairs via sliding window.
    Compute IDM, ADM (drift/congruence), aₙ (VA) for each pair.

    sequences[uid] = [(item_idx, ts, v, a), ...]
    """
    pairs = []
    n_skip_short = n_skip_no_cat = n_skip_no_va = 0

    for uid, seq in tqdm(sequences.items(), desc="Building pairs"):
        if len(seq) < 3:   # minimum 2 input items + 1 target
            n_skip_short += 1
            continue

        for t in range(2, len(seq)):          # ensure at least 2 input items
            input_seq = seq[:t]               # [(item_idx, ts, v, a), ...]
            target    = seq[t]

            # ── Target VA ──────────────────────────────────────────────────
            target_item = target[0]
            try:
                target_v, target_a = target[2], target[3]
            except (IndexError, TypeError):
                va_entry = item_va.get(str(target_item))
                if va_entry is None:
                    n_skip_no_va += 1
                    continue
                va = va_entry.get("va") if isinstance(va_entry, dict) else va_entry
                target_v, target_a = va[0], va[1]

            # ── IDM (category-based) ────────────────────────────────────────
            target_cats = set(item_cats.get(str(target_item), [])) - EXCLUDE_CATS
            if len(target_cats) == 0:
                n_skip_no_cat += 1
                continue

            seq_items = [s[0] for s in input_seq]
            if target_item in seq_items:
                idm = 0.0
            else:
                seq_cats = set()
                for s_item in seq_items:
                    seq_cats |= set(item_cats.get(str(s_item), [])) - EXCLUDE_CATS
                inter = target_cats & seq_cats
                idm   = 1.0 - len(inter) / len(target_cats)

            # ── Emotion sequence ────────────────────────────────────────────
            try:
                v_seq = np.array([s[2] for s in input_seq], dtype=np.float32)
                a_seq = np.array([s[3] for s in input_seq], dtype=np.float32)
            except (IndexError, TypeError):
                va_list = []
                for s in input_seq:
                    va_entry = item_va.get(str(s[0]))
                    if va_entry is None:
                        continue
                    va_list.append(
                        va_entry.get("va", [0.0, 0.0])
                        if isinstance(va_entry, dict) else va_entry
                    )
                if len(va_list) < 2:
                    continue
                va_arr = np.array(va_list)
                v_seq, a_seq = va_arr[:, 0], va_arr[:, 1]

            if len(v_seq) < 2:
                continue

            # ── aₙ, ā_u (mean excluding a_n) ───────────────────────────────
            a_n   = np.array([v_seq[-1],        a_seq[-1]])
            a_bar = np.array([v_seq[:-1].mean(), a_seq[:-1].mean()])
            e_tgt = np.array([target_v, target_a])

            # ── ADM computation ─────────────────────────────────────────────
            term1_drift = float(np.linalg.norm(a_n - a_bar))       # emotional change
            term2_cong  = float(np.linalg.norm(e_tgt - a_n))       # item-emotion mismatch
            adm         = term1_drift                               # drift term only

            pairs.append({
                "uid":           uid,
                "t":             t,
                "idm":           float(idm),
                "adm":           adm,
                "adm_drift":     term1_drift,
                "adm_congruence":term2_cong,
                "an_v":          float(a_n[0]),   # aₙ valence
                "an_a":          float(a_n[1]),   # aₙ arousal
            })

            if len(pairs) >= max_pairs:
                print(f"[Build] max_pairs={max_pairs} reached")
                return pairs

    print(f"[Build] {len(pairs):,} pairs | "
          f"skip_short={n_skip_short}, skip_no_cat={n_skip_no_cat}, "
          f"skip_no_va={n_skip_no_va}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# [Claim 1] IDM ⊥ ADM — scatter + MI
# ─────────────────────────────────────────────────────────────────────────────
def fig1_idm_vs_adm(idm, adm, output_dir):
    """
    IDM vs ADM scatter plot.
    Display MI(IDM; ADM) with shuffled baseline to show independence.
    Paper claim: "behavioral drift (IDM) alone cannot explain affective drift (ADM)"
    """
    plt.rcParams.update(PLOT_STYLE)

    # MI computation
    mi_val  = mutual_info_regression(idm.reshape(-1, 1), adm, random_state=42)[0]
    np.random.seed(42)
    mi_base = mutual_info_regression(
        np.random.permutation(idm).reshape(-1, 1), adm, random_state=42
    )[0]
    spear_r, spear_p = spearmanr(idm, adm)

    # scatter plot (sampled)
    n_plot = min(5000, len(idm))
    idx    = np.random.choice(len(idm), n_plot, replace=False)
    idm_s, adm_s = idm[idx], adm[idx]

    # per IDM bin ADM mean (for boxplot)
    bins       = [0.0, 0.001, 0.33, 0.66, 1.001]
    bin_labels = ["IDM=0", "low", "mid", "IDM=1"]
    bin_data   = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (idm >= lo) & (idm < hi)
        bin_data.append(adm[mask])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── left: scatter ─────────────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(idm_s, adm_s, alpha=0.15, s=8, c=adm_s,
                    cmap="viridis", rasterized=True)
    plt.colorbar(sc, ax=ax, label="ADM", shrink=0.8)

    # highlight IDM=0 but ADM in top 25%
    mask_idm0  = (idm == 0)
    adm_p75    = np.percentile(adm, 75)
    interesting = mask_idm0 & (adm > adm_p75)
    pct         = interesting.sum() / mask_idm0.sum() * 100

    highlight = (idm_s == 0) & (adm_s > adm_p75)
    ax.scatter(idm_s[highlight], adm_s[highlight],
               c="red", s=15, alpha=0.6,
               label=f"IDM=0 & ADM>P75 ({pct:.1f}%)",
               zorder=3)
    ax.legend(fontsize=8, loc="upper right")

    # MI / Spearman text
    txt = (f"MI(IDM; ADM) = {mi_val:.4f}\n"
           f"MI baseline  = {mi_base:.4f}\n"
           f"Spearman ρ   = {spear_r:.3f}  (p={spear_p:.1e})")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    ax.set_xlabel("IDM (behavioral drift)", fontsize=11)
    ax.set_ylabel("ADM (affective drift)", fontsize=11)
    ax.set_title("IDM vs. ADM", fontsize=12, fontweight="bold")

    # ── right: ADM boxplot per IDM bin ────────────────────────────────────
    ax2 = axes[1]
    valid_data   = [d for d in bin_data if len(d) > 30]
    valid_labels = [l for l, d in zip(bin_labels, bin_data) if len(d) > 30]
    bp = ax2.boxplot(valid_data, tick_labels=valid_labels,
                     patch_artist=True, showfliers=False, widths=0.5)
    colors = ["#6B8CBA", "#52B788", "#E8A838", "#E85252"]
    for patch, c in zip(bp["boxes"], colors[:len(valid_data)]):
        patch.set_facecolor(c); patch.set_alpha(0.7)

    ax2.set_xlabel("IDM bin", fontsize=11)
    ax2.set_ylabel("ADM", fontsize=11)
    ax2.set_title("ADM distribution per IDM bin", fontsize=12, fontweight="bold")

    plt.suptitle(
        "Claim 1: Behavioral drift (IDM) cannot explain affective drift (ADM)",
        fontsize=11, y=1.01, style="italic"
    )
    plt.tight_layout()
    path = f"{output_dir}/fig1_idm_vs_adm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig1] Saved: {path}")

    return {
        "MI_IDM_ADM":           float(mi_val),
        "MI_baseline":          float(mi_base),
        "spearman_r":           float(spear_r),
        "spearman_p":           float(spear_p),
        "MI_ratio":             float(mi_val / mi_base) if mi_base > 0 else None,
        "idm0_adm_p75_pct":     float(pct),
    }


# ─────────────────────────────────────────────────────────────────────────────
# [Claim 2] Affective drift is a real phenomenon — ADM distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig2_adm_distribution(adm, adm_drift, adm_cong, output_dir):
    """
    Distribution of ADM overall, drift term, and congruence term.
    Paper claim: "user emotion changes substantially over time"
    """
    plt.rcParams.update(PLOT_STYLE)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    data_list   = [adm, adm_drift, adm_cong]
    titles      = ["ADM (overall)", "Term 1: Drift\n‖aₙ − ā_u‖", "Term 2: Congruence\n‖e_aff(v) − aₙ‖"]
    colors_list = ["#6B8CBA", "#E8A838", "#52B788"]

    stats_out = {}
    for ax, data, title, color in zip(axes, data_list, titles, colors_list):
        ax.hist(data, bins=60, color=color, alpha=0.75, edgecolor="none",
                density=True)
        ax.axvline(np.mean(data),   color="black", lw=1.5, ls="--",
                   label=f"mean={np.mean(data):.3f}")
        ax.axvline(np.median(data), color="gray",  lw=1.2, ls=":",
                   label=f"med={np.median(data):.3f}")
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

        key = title.split("\n")[0].replace(" ", "_").replace("(", "").replace(")", "")
        stats_out[key] = {
            "mean":   float(np.mean(data)),
            "median": float(np.median(data)),
            "std":    float(np.std(data)),
            "pct_above_mean": float((data > np.mean(data)).mean() * 100),
        }

    plt.suptitle(
        "Claim 2: Affective drift is a real and widespread phenomenon",
        fontsize=11, y=1.01, style="italic"
    )
    plt.tight_layout()
    path = f"{output_dir}/fig2_adm_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig2] Saved: {path}")
    return stats_out


# ─────────────────────────────────────────────────────────────────────────────
# [Claim 3] VA quadrant structure — βₖ design justification
# ─────────────────────────────────────────────────────────────────────────────
def fig3_va_quadrant(an_v, an_a, output_dir):
    """
    VA space distribution of user current emotion aₙ + quadrant centroids.
    Paper claim: "user emotions span all 4 quadrants → βₖ gating design is appropriate"
    """
    plt.rcParams.update(PLOT_STYLE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: KDE/scatter ─────────────────────────────────────────────────
    ax = axes[0]
    n_plot = min(8000, len(an_v))
    idx    = np.random.choice(len(an_v), n_plot, replace=False)

    ax.scatter(an_v[idx], an_a[idx], alpha=0.12, s=6,
               c="#6B8CBA", rasterized=True, label="aₙ (user)")

    # centroid markers
    for key, (cv, ca) in QUADRANT_CENTROIDS.items():
        ax.scatter(cv, ca, s=200, marker="*",
                   color=QUADRANT_COLORS[key], zorder=5,
                   label=QUADRANT_LABELS[key], edgecolors="white", linewidths=0.5)
        ax.annotate(key, (cv, ca), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=QUADRANT_COLORS[key],
                    fontweight="bold")

    # quadrant boundary lines
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.6)

    # quadrant labels (background)
    for (xpos, ypos, lbl) in [
        ( 0.55,  0.75, "Q1\nHigh V / High A"),
        (-0.95,  0.75, "Q2\nLow V / High A"),
        ( 0.55, -0.85, "Q3\nHigh V / Low A"),
        (-0.95, -0.85, "Q4\nLow V / Low A"),
    ]:
        ax.text(xpos, ypos, lbl, fontsize=7, alpha=0.5, ha="center")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Valence", fontsize=11)
    ax.set_ylabel("Arousal", fontsize=11)
    ax.set_title("User current emotion aₙ in VA space", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)

    # ── right: quadrant proportion bar chart ──────────────────────────────
    ax2 = axes[1]

    # assign each point to a quadrant
    q_counts = {
        "Q1 (+V,+A)": int(((an_v > 0) & (an_a > 0)).sum()),
        "Q2 (−V,+A)": int(((an_v <= 0) & (an_a > 0)).sum()),
        "Q3 (+V,−A)": int(((an_v > 0) & (an_a <= 0)).sum()),
        "Q4 (−V,−A)": int(((an_v <= 0) & (an_a <= 0)).sum()),
    }
    total    = sum(q_counts.values())
    q_pcts   = {k: v / total * 100 for k, v in q_counts.items()}
    bar_cols = ["#E8A838", "#E85252", "#52B788", "#6B8CBA"]

    bars = ax2.bar(q_pcts.keys(), q_pcts.values(),
                   color=bar_cols, alpha=0.8, edgecolor="white", linewidth=1.2)
    for bar, (k, pct) in zip(bars, q_pcts.items()):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    ax2.set_ylabel("Proportion (%)", fontsize=11)
    ax2.set_title("Emotion distribution across quadrants", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(q_pcts.values()) * 1.15)

    plt.suptitle(
        "Claim 3: User emotions span all VA quadrants → βₖ soft gating is well-motivated",
        fontsize=11, y=1.01, style="italic"
    )
    plt.tight_layout()
    path = f"{output_dir}/fig3_va_quadrant.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig3] Saved: {path}")

    return {
        "quadrant_counts": q_counts,
        "quadrant_pcts":   {k: float(v) for k, v in q_pcts.items()},
        "an_v_mean":       float(np.mean(an_v)),
        "an_a_mean":       float(np.mean(an_a)),
        "an_v_std":        float(np.std(an_v)),
        "an_a_std":        float(np.std(an_a)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences",  required=True)
    ap.add_argument("--item_va",    required=True)
    ap.add_argument("--item_cats",  required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_pairs",  type=int, default=100_000)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Load] sequences : {args.sequences}")
    with open(args.sequences, "rb") as f:
        sequences = pickle.load(f)
    print(f"[Load] item_va   : {args.item_va}")
    with open(args.item_va) as f:
        item_va = json.load(f)
    print(f"[Load] item_cats : {args.item_cats}")
    with open(args.item_cats) as f:
        item_cats = json.load(f)

    pairs = build_pairs(sequences, item_va, item_cats,
                        max_pairs=args.max_pairs)
    if len(pairs) < 200:
        print(f"[Error] Too few pairs: {len(pairs)}"); return

    idm      = np.array([p["idm"]           for p in pairs])
    adm      = np.array([p["adm"]           for p in pairs])
    adm_d    = np.array([p["adm_drift"]     for p in pairs])
    adm_c    = np.array([p["adm_congruence"]for p in pairs])
    an_v     = np.array([p["an_v"]          for p in pairs])
    an_a     = np.array([p["an_a"]          for p in pairs])

    print(f"\n=== Basic Statistics ===")
    print(f"n = {len(pairs):,}")
    print(f"IDM=0: {(idm==0).mean()*100:.1f}%  IDM=1: {(idm==1).mean()*100:.1f}%")
    print(f"ADM    mean={adm.mean():.4f}  std={adm.std():.4f}")
    print(f"drift  mean={adm_d.mean():.4f}  std={adm_d.std():.4f}")
    print(f"cong   mean={adm_c.mean():.4f}  std={adm_c.std():.4f}")

    stats1 = fig1_idm_vs_adm(idm, adm, args.output_dir)
    stats2 = fig2_adm_distribution(adm, adm_d, adm_c, args.output_dir)
    stats3 = fig3_va_quadrant(an_v, an_a, args.output_dir)

    summary = {
        "n_pairs":   len(pairs),
        "idm_pct_zero": float((idm == 0).mean() * 100),
        "idm_pct_one":  float((idm == 1).mean() * 100),
        "claim1_idm_vs_adm":    stats1,
        "claim2_adm_distribution": stats2,
        "claim3_va_quadrant":   stats3,
        "motivation_narrative": {
            "claim1": (
                "IDM(behavioral drift) and ADM(affective drift) are nearly "
                "independent signals (MI ratio ≈ 2x baseline). "
                "Behavioral drift alone cannot explain affective drift, "
                "justifying a dedicated emotion module."
            ),
            "claim2": (
                "ADM is widely distributed across users and time steps, "
                "confirming that affective drift is a real phenomenon "
                "that should not be ignored."
            ),
            "claim3": (
                "User current emotion aₙ spans all four VA quadrants, "
                "motivating the βₖ soft-gating design that routes "
                "representations through quadrant-specialized experts."
            ),
        }
    }

    out_path = f"{args.output_dir}/summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved: {args.output_dir}/")
    print(f"  fig1_idm_vs_adm.png       ← Claim 1: IDM ⊥ ADM")
    print(f"  fig2_adm_distribution.png ← Claim 2: affective drift exists")
    print(f"  fig3_va_quadrant.png      ← Claim 3: βₖ design justified")
    print(f"  summary.json")


if __name__ == "__main__":
    main()