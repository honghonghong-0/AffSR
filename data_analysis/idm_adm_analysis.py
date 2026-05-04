"""
idm_adm_analysis.py
===================
IDM/ADM distribution visualization and similar-user empty-set rate analysis
on the full dataset.

Inputs:
  data_analysis/results/emotion_results_cds.csv

Outputs:
  data_analysis/results/
  ├── idm_dist_full.png       # IDM distribution
  ├── adm_dist_full.png       # ADM distribution
  ├── idm_adm_joint.png       # IDM vs ADM scatter plot
  └── similar_user_stats.json # empty-set rate per epsilon

Usage:
  python preprocessing/idm_adm_analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_CSV = "data_analysis/results/emotion_results_cds.csv"
OUTPUT_DIR  = Path("data_analysis/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data — extract per-user IDM/ADM
# ─────────────────────────────────────────────────────────────────────────────
print("[Load] Loading data...")
df = pd.read_csv(EMOTION_CSV)

# IDM and ADM have one value per user -> take first occurrence per user
user_df = df.groupby("user_id")[["idm", "adm"]].first().reset_index()
user_df = user_df.dropna(subset=["idm", "adm"])

print(f"  Total reviews: {len(df):,}")
print(f"  Valid users: {len(user_df):,}")
print(f"  IDM mean={user_df['idm'].mean():.4f}  std={user_df['idm'].std():.4f}")
print(f"  ADM mean={user_df['adm'].mean():.4f}  std={user_df['adm'].std():.4f}")

idm = user_df["idm"].values
adm = user_df["adm"].values

# ─────────────────────────────────────────────────────────────────────────────
# 2. IDM distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(idm, bins=np.linspace(0, 1, 21), color="#6B8CBA",
        edgecolor="white", alpha=0.85)
ax.axvline(idm.mean(), color="#E85252", lw=1.5, ls="--",
           label=f"Mean = {idm.mean():.3f}")
ax.axvline(np.median(idm), color="#52B788", lw=1.5, ls="--",
           label=f"Median = {np.median(idm):.3f}")
for thr in [0.0, 0.5, 1.0]:
    ax.axvline(thr, color="gray", lw=0.8, ls=":")
ax.set_xlabel("IDM (Interest Drift Magnitude)", fontsize=11)
ax.set_ylabel("# Users", fontsize=11)
ax.set_title(f"IDM Distribution — CDs & Vinyl (n={len(idm):,})", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "idm_dist_full.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Plot] IDM dist -> {OUTPUT_DIR / 'idm_dist_full.png'}")
print(f"       IDM=0: {(idm==0).mean()*100:.1f}%")
print(f"       IDM=1: {(idm==1).mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ADM distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(adm, bins=40, color="#E8A838", edgecolor="white", alpha=0.85)
ax.axvline(adm.mean(), color="#E85252", lw=1.5, ls="--",
           label=f"Mean = {adm.mean():.3f}")
ax.axvline(np.median(adm), color="#52B788", lw=1.5, ls="--",
           label=f"Median = {np.median(adm):.3f}")
ax.set_xlabel("ADM (Affective Drift Magnitude)", fontsize=11)
ax.set_ylabel("# Users", fontsize=11)
ax.set_title(f"ADM Distribution — CDs & Vinyl (n={len(adm):,})", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "adm_dist_full.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Plot] ADM dist -> {OUTPUT_DIR / 'adm_dist_full.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. IDM vs ADM scatter
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(idm, adm, alpha=0.2, s=8, color="#6B8CBA", linewidths=0)
ax.set_xlabel("IDM (behavioral drift)", fontsize=11)
ax.set_ylabel("ADM (affective drift)", fontsize=11)
ax.set_title(f"IDM vs ADM — n={len(idm):,} users", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "idm_adm_joint.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[Plot] IDM vs ADM joint -> {OUTPUT_DIR / 'idm_adm_joint.png'}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Similar-user empty-set rate per epsilon
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Similar User] Computing empty-set rate per epsilon...")

epsilons = [0.1, 0.2, 0.3]
stats = {}

for eps in epsilons:
    total = len(user_df)
    idm_arr = idm[:, np.newaxis]   # (N, 1)
    adm_arr = adm[:, np.newaxis]   # (N, 1)

    idm_diff = np.abs(idm_arr - idm_arr.T)  # (N, N)
    adm_diff = np.abs(adm_arr - adm_arr.T)  # (N, N)

    # exclude self (diagonal)
    np.fill_diagonal(idm_diff, np.inf)

    similar_mask = (idm_diff <= eps) & (adm_diff <= eps)  # (N, N)
    has_similar  = similar_mask.any(axis=1)                # (N,)
    empty_count  = (~has_similar).sum()

    empty_ratio = empty_count / total * 100
    stats[str(eps)] = {
        "epsilon": eps,
        "total_users": int(total),
        "empty_count": int(empty_count),
        "empty_ratio_%": round(float(empty_ratio), 2),
        "has_similar_%": round(float(100 - empty_ratio), 2),
    }
    print(f"  eps={eps}: empty {empty_count:,}/{total:,} "
          f"({empty_ratio:.2f}%) | has similar: {100-empty_ratio:.2f}%")

out_json = OUTPUT_DIR / "similar_user_stats.json"
with open(out_json, "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"\n[Save] -> {out_json}")
print("\nDone.")
