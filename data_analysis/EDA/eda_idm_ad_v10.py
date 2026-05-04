"""
data_analysis/EDA/eda_idm_ad_v10.py
=====================================
IDM vs AD correlation analysis based on v10

AD definition (v10):
  va_long = EMA(dist28_seq) @ VA_MATRIX   (using learned lambda)
  AD      = ||a_n - va_long||_2

IDM definition:
  idm.pkl: {(user_idx, target_item_idx): float}

Method:
  - Call AffDrift.forward() from the trained best.pt checkpoint
  - Extract AD and IDM for all users in the test split
  - Pearson r, Spearman r, KS test, Kruskal-Wallis, Mutual Information

Usage:
  python data_analysis/EDA/eda_idm_ad_v10.py \
      --ckpt outputs/v10_final/affsr_full_movies/best.pt \
      --data_dir data/processed/movies_tv_2021_2023 \
      --output_dir data_analysis/results/idm_ad_v10_movies

  python data_analysis/EDA/eda_idm_ad_v10.py \
      --ckpt outputs/v10_final/affsr_full_cds/best.pt \
      --data_dir data/processed/cds \
      --output_dir data_analysis/results/idm_ad_v10_cds
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, ks_2samp, kruskal
from sklearn.feature_selection import mutual_info_regression
from torch.utils.data import DataLoader

from datasets.base_dataset import AffSRDataset
from models.modules.affsr import AffSR


# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
    "axes.edgecolor":   "#AAAAAA",
    "axes.linewidth":   0.8,
})


def load_model(ckpt_path: str, data_dir: str, device: torch.device):
    ds = AffSRDataset(data_dir, split="train", max_seq_len=50, full_ce=True)
    model = AffSR(num_items=ds.num_items, d_model=64, n_heads=2, n_layers=2,
                  max_seq_len=50, K=4, dropout=0.0)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    lam = torch.nn.functional.softplus(model.affdrift.lambda_raw).item()
    print(f"  Learned EMA lambda = {lam:.4f}")
    return model


def extract_ad_idm(model, data_dir: str, device: torch.device):
    """Extract AD and IDM from the entire test split."""
    with open(Path(data_dir) / "idm.pkl", "rb") as f:
        idm_map = pickle.load(f)   # {(user_idx, item_idx): float}

    ds = AffSRDataset(data_dir, split="test", max_seq_len=50, full_ce=True)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    ad_list, idm_list = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            ad, _ = model.affdrift(
                batch["a_n"], batch["dist28_seq"], batch["seq_mask"]
            )
            ad_np      = ad.cpu().numpy()
            user_idxs  = batch["user_idx"].cpu().numpy()
            target_idxs = batch["target"].cpu().numpy()

            for i in range(len(ad_np)):
                key = (int(user_idxs[i]), int(target_idxs[i]))
                idm_val = idm_map.get(key, None)
                if idm_val is not None:
                    ad_list.append(float(ad_np[i]))
                    idm_list.append(float(idm_val))

    ad  = np.array(ad_list)
    idm = np.array(idm_list)
    print(f"  Extracted: {len(ad):,} pairs  (IDM match succeeded)")
    print(f"  AD:  mean={ad.mean():.4f}  std={ad.std():.4f}  "
          f"min={ad.min():.4f}  max={ad.max():.4f}")
    print(f"  IDM: mean={idm.mean():.4f}  std={idm.std():.4f}  "
          f"0={(idm==0).mean()*100:.1f}%  1={(idm==1).mean()*100:.1f}%")
    return ad, idm


def analyze(ad: np.ndarray, idm: np.ndarray, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Basic correlation ──────────────────────────────────────────────────
    pr, pp = pearsonr(idm, ad)
    sr, sp = spearmanr(idm, ad)
    print(f"\n=== Correlation Coefficients ===")
    print(f"  Pearson  r={pr:.4f}  p={pp:.2e}")
    print(f"  Spearman r={sr:.4f}  p={sp:.2e}")

    # ── 2. AD distribution per IDM bin (boxplot) ───────────────────────────────
    bins       = [(0.0, 0.001), (0.001, 0.5), (0.5, 0.999), (0.999, 1.001)]
    bin_labels = ["IDM=0\n(repeat)", "0<IDM<0.5\n(partial)",
                  "0.5≤IDM<1\n(mostly new)", "IDM=1\n(full drift)"]
    bin_data, bin_stats = [], []
    for (lo, hi), lbl in zip(bins, bin_labels):
        mask = (idm >= lo) & (idm < hi)
        bin_data.append(ad[mask])
        n = int(mask.sum())
        bin_stats.append({
            "bin": lbl.replace("\n", " "),
            "n": n, "pct": float(mask.mean() * 100),
            "ad_mean": float(ad[mask].mean()) if n > 0 else None,
            "ad_std":  float(ad[mask].std())  if n > 0 else None,
        })

    print(f"\n=== AD statistics by IDM bin ===")
    for s in bin_stats:
        if s["n"] > 0:
            print(f"  {s['bin']:30} n={s['n']:6,} ({s['pct']:5.1f}%)  "
                  f"AD={s['ad_mean']:.4f}±{s['ad_std']:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    valid_data   = [d for d in bin_data   if len(d) > 0]
    valid_labels = [l for l, d in zip(bin_labels, bin_data) if len(d) > 0]
    bp = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="crimson", lw=2))
    colors = ["#6B8CBA", "#52B788", "#E8A838", "#E85252"]
    for patch, color in zip(bp["boxes"], colors[:len(valid_data)]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel("AD  (||a_n - va_long||_2)", fontsize=11)
    ax.set_title(f"AD distribution across IDM bins\n"
                 f"Pearson r={pr:.3f}  Spearman r={sr:.3f}", fontsize=11)
    plt.tight_layout()
    bp_path = str(Path(output_dir) / "idm_bins_ad_boxplot.png")
    plt.savefig(bp_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. KS test (adjacent bins) ────────────────────────────────────────
    print(f"\n=== KS test (adjacent bins) ===")
    ks_results = []
    for i in range(len(valid_data) - 1):
        if len(valid_data[i]) > 30 and len(valid_data[i+1]) > 30:
            stat, pval = ks_2samp(valid_data[i], valid_data[i+1])
            sig = ("***" if pval < 0.001 else "**" if pval < 0.01
                   else "*" if pval < 0.05 else "n.s.")
            a = valid_labels[i].replace("\n", " ")
            b = valid_labels[i+1].replace("\n", " ")
            print(f"  {a[:24]:24} vs {b[:24]:24}: KS={stat:.3f}  p={pval:.2e}  {sig}")
            ks_results.append({"compare": f"{a} vs {b}",
                               "ks_stat": float(stat), "p_value": float(pval)})

    # ── 4. Kruskal-Wallis ─────────────────────────────────────────────
    kw_stat = kw_p = None
    kw_data = [d for d in valid_data if len(d) > 30]
    if len(kw_data) >= 2:
        kw_stat, kw_p = kruskal(*kw_data)
        print(f"\n=== Kruskal-Wallis ===")
        print(f"  H={kw_stat:.3f}  p={kw_p:.2e}  "
              f"{'reject (distributions differ)' if kw_p < 0.05 else 'fail to reject'}")

    # ── 5. Mutual Information ─────────────────────────────────────────
    mi = mutual_info_regression(idm.reshape(-1, 1), ad, random_state=42)[0]
    rng = np.random.default_rng(42)
    mi_base = mutual_info_regression(
        rng.permutation(idm).reshape(-1, 1), ad, random_state=42)[0]
    print(f"\n=== Mutual Information ===")
    print(f"  I(IDM; AD)        = {mi:.4f}")
    print(f"  I(shuffled; AD)   = {mi_base:.4f}  <- random baseline")
    print(f"  ratio             = {mi / max(mi_base, 1e-6):.2f}x")

    # ── 6. Scatter ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_plot = min(5000, len(idm))
    idx = np.random.default_rng(0).choice(len(idm), n_plot, replace=False)
    axes[0].scatter(idm[idx], ad[idx], alpha=0.2, s=8, color="#4393C3")
    axes[0].set_xlabel("IDM", fontsize=11)
    axes[0].set_ylabel("AD  (||a_n - va_long||_2)", fontsize=11)
    axes[0].set_title(f"IDM vs AD  (n={n_plot:,} subsampled)\n"
                      f"r={pr:.3f}  rho={sr:.3f}", fontsize=10)
    axes[1].hist(ad, bins=30, color="#D6604D", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("AD", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(f"AD distribution  (n={len(ad):,})", fontsize=10)
    plt.tight_layout()
    sc_path = str(Path(output_dir) / "idm_ad_scatter.png")
    plt.savefig(sc_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 7. Summary JSON ───────────────────────────────────────────────
    summary = {
        "n_pairs": len(ad),
        "ad_dist":  {"mean": float(ad.mean()),  "std": float(ad.std()),
                     "min":  float(ad.min()),    "max": float(ad.max())},
        "idm_dist": {"mean": float(idm.mean()), "std": float(idm.std()),
                     "pct_zero": float((idm == 0).mean() * 100),
                     "pct_one":  float((idm == 1).mean() * 100)},
        "correlation": {
            "pearson_r":  float(pr), "pearson_p":  float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
        },
        "bin_stats": bin_stats,
        "ks_results": ks_results,
        "kruskal_wallis": {
            "stat":    float(kw_stat) if kw_stat is not None else None,
            "p_value": float(kw_p)    if kw_p    is not None else None,
        },
        "mutual_info": {
            "I_IDM_AD":      float(mi),
            "I_shuffled_AD": float(mi_base),
            "ratio":         float(mi / max(mi_base, 1e-6)),
        },
        "interpretation": {
            "ks_significant": any(r["p_value"] < 0.05 for r in ks_results),
            "kw_significant": bool(kw_p < 0.05) if kw_p is not None else None,
            "mi_significant": float(mi / max(mi_base, 1e-6)) > 1.5,
        },
    }
    summary_path = str(Path(output_dir) / "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] {summary_path}")
    print(f"[Saved] {bp_path}")
    print(f"[Saved] {sc_path}")

    # ── Final verdict ─────────────────────────────────────────────────────
    si = summary["interpretation"]
    print(f"\n{'='*55}")
    print("Final Verdict  (IDM ⊥ AD hypothesis)")
    print(f"{'='*55}")
    if not si["mi_significant"] and not si["kw_significant"]:
        print("O  IDM and AD are statistically independent")
        print("   -> motivation strongly supported: both are needed")
    elif si["mi_significant"] and si["kw_significant"]:
        print("X  IDM and AD are NOT independent (both significant)")
        print("   -> motivation weak: one side needs re-examination")
    else:
        print("~  Mixed results")
        print(f"   KW sig={si['kw_significant']}, MI sig={si['mi_significant']}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True,
                    help="path to best.pt checkpoint (affsr_full_movies or affsr_full_cds)")
    ap.add_argument("--data_dir",   required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\n[1] Loading model: {args.ckpt}")
    model = load_model(args.ckpt, args.data_dir, device)

    print(f"\n[2] Extracting AD / IDM...")
    ad, idm = extract_ad_idm(model, args.data_dir, device)

    print(f"\n[3] Correlation analysis...")
    analyze(ad, idm, args.output_dir)


if __name__ == "__main__":
    main()