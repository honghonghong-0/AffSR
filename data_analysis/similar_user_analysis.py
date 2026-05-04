"""
similar_user_analysis.py
========================
Analyze distribution of similar-user counts per user.
Condition (eps=0.1): same target item + |IDM diff| < 0.1 + |ADM diff| < 0.1

Usage:
  python preprocessing/similar_user_analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

EMOTION_CSV = "data_analysis/results/emotion_results_cds.csv"
OUTPUT_DIR  = Path("data_analysis/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("[Load] Loading data...")
df = pd.read_csv(EMOTION_CSV)

# Extract per-user last item (target) + IDM/ADM
user_df = (
    df.sort_values("timestamp")
    .groupby("user_id")
    .last()
    .reset_index()[["user_id", "parent_asin", "idm", "adm"]]
    .rename(columns={"parent_asin": "target_item"})
    .dropna(subset=["idm", "adm"])
)

print(f"  Valid users: {len(user_df):,}")
print(f"  Unique target items: {user_df['target_item'].nunique():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute similar-user count per user
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Similar User] Computing with eps={EPS}...")

similar_counts = []

for target, grp in tqdm(user_df.groupby("target_item"), desc="Target items"):
    grp = grp.reset_index(drop=True)
    n = len(grp)

    if n == 1:
        # only one user with this target item -> no similar users
        similar_counts.append({"user_id": grp["user_id"].iloc[0], "similar_count": 0})
        continue

    idm_arr = grp["idm"].values[:, np.newaxis]  # (n, 1)
    adm_arr = grp["adm"].values[:, np.newaxis]  # (n, 1)

    idm_diff = np.abs(idm_arr - idm_arr.T)  # (n, n)
    adm_diff = np.abs(adm_arr - adm_arr.T)  # (n, n)

    # exclude self
    np.fill_diagonal(idm_diff, np.inf)

    mask = (idm_diff <= EPS) & (adm_diff <= EPS)  # (n, n)
    counts = mask.sum(axis=1)                       # (n,)

    for i, uid in enumerate(grp["user_id"]):
        similar_counts.append({"user_id": uid, "similar_count": int(counts[i])})

result_df = pd.DataFrame(similar_counts)
counts = result_df["similar_count"].values

# ─────────────────────────────────────────────────────────────────────────────
# 3. Statistics
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Stats] Similar-user counts (epsilon={EPS})")
print(f"  Mean   : {counts.mean():.2f}")
print(f"  Median : {np.median(counts):.1f}")
print(f"  Min    : {counts.min()}")
print(f"  Max    : {counts.max()}")
print(f"  Empty  : {(counts == 0).sum():,} users ({(counts == 0).mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Histogram
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Similar User Count Distribution (eps={EPS}, same target item)", fontsize=12)

ax = axes[0]
ax.hist(counts, bins=50, color="#6B8CBA", edgecolor="white", alpha=0.85)
ax.axvline(counts.mean(), color="#E85252", lw=1.5, ls="--",
           label=f"Mean = {counts.mean():.1f}")
ax.axvline(np.median(counts), color="#52B788", lw=1.5, ls="--",
           label=f"Median = {np.median(counts):.1f}")
ax.set_xlabel("# Similar Users", fontsize=11)
ax.set_ylabel("# Users", fontsize=11)
ax.set_title("Full Distribution", fontsize=11)
ax.legend(fontsize=9)

# clipped view (0-50)
ax = axes[1]
clipped = np.clip(counts, 0, 50)
ax.hist(clipped, bins=50, color="#E8A838", edgecolor="white", alpha=0.85)
ax.axvline(counts.mean(), color="#E85252", lw=1.5, ls="--",
           label=f"Mean = {counts.mean():.1f}")
ax.set_xlabel("# Similar Users (clipped at 50)", fontsize=11)
ax.set_ylabel("# Users", fontsize=11)
ax.set_title("Clipped at 50 (detail view)", fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout()
save_path = OUTPUT_DIR / "similar_user_dist.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Plot] -> {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save summary
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    "epsilon": EPS,
    "condition": "same target item + IDM diff < eps + ADM diff < eps",
    "total_users": int(len(counts)),
    "mean": round(float(counts.mean()), 2),
    "median": round(float(np.median(counts)), 1),
    "min": int(counts.min()),
    "max": int(counts.max()),
    "empty_count": int((counts == 0).sum()),
    "empty_ratio_%": round(float((counts == 0).mean() * 100), 2),
}
out_json = OUTPUT_DIR / "similar_user_full_stats.json"
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[Save] -> {out_json}")
print("\nDone.")
