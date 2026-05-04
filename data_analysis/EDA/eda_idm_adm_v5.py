"""
eda_idm_adm_v5.py
=================
IDM-ADM 독립성 가설 직접 검증

v4 대비 핵심 변경:
  - 유저당 1개 → 모든 sliding window 쌍 (수만 개)
  - Pearson r → KS test (분포 비교) + Mutual Information (비선형 관계)
  - bin별 ADM 분포 boxplot

판정:
  H₀ (motivation): IDM ⊥ ADM  → 각 IDM bin의 ADM 분포 동일
  H₁ (의존):       IDM ↔ ADM  → bin별 ADM 분포 다름

사용법:
  python data_analysis/EDA/eda_idm_adm_v5.py \
      --sequences data/processed/cds/sequences.pkl \
      --item_va   data/processed/cds/item_va.json \
      --item_cats data/processed/cds/item_cats.json \
      --output_dir data_analysis/results/cds_v5
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, kruskal
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm


EXCLUDE_CATS = {
    "Movies & TV", "Prime Video", "Featured Categories",
    "Genre for Featured Categories", "Amazon Video", "CDs & Vinyl",
}


def compute_pairs(sequences, item_va, item_cats, min_seq_len=3, max_pairs=100000):
    """모든 sliding window 쌍 생성."""
    pairs = []
    n_skip_short = 0
    n_skip_no_cat = 0

    for uid, seq in tqdm(sequences.items(), desc="Building pairs"):
        if len(seq) < min_seq_len:
            n_skip_short += 1
            continue

        for t in range(1, len(seq)):
            input_seq = seq[:t]
            target = seq[t]

            target_item = target[0]

            target_cats_raw = item_cats.get(str(target_item), [])
            if not isinstance(target_cats_raw, list):
                target_cats_raw = []
            target_cats = set(target_cats_raw) - EXCLUDE_CATS
            if len(target_cats) == 0:
                n_skip_no_cat += 1
                continue

            # IDM
            seq_items = [s[0] for s in input_seq]
            if target_item in seq_items:
                idm = 0.0
            else:
                seq_cats = set()
                for it in seq_items:
                    it_cats = item_cats.get(str(it), [])
                    if isinstance(it_cats, list):
                        seq_cats |= (set(it_cats) - EXCLUDE_CATS)
                intersection = target_cats & seq_cats
                idm = 1.0 - len(intersection) / len(target_cats)

            # ADM (VA 기반) — sequences 포맷: (item_idx, ts, v, a, ...)
            try:
                v_seq = np.array([s[2] for s in input_seq])
                a_seq = np.array([s[3] for s in input_seq])
                target_v = target[2]
                target_a = target[3]
            except (IndexError, TypeError):
                # fallback: item_va에서 조회
                va_list = []
                for s in input_seq:
                    va_entry = item_va.get(str(s[0]))
                    if va_entry is None:
                        continue
                    if isinstance(va_entry, dict):
                        va_list.append(va_entry.get("va", [0.0, 0.0]))
                    else:
                        va_list.append(va_entry)
                if len(va_list) == 0:
                    continue
                va_arr = np.array(va_list)
                v_seq = va_arr[:, 0]
                a_seq = va_arr[:, 1]
                t_entry = item_va.get(str(target_item))
                if t_entry is None:
                    continue
                t_va = t_entry.get("va") if isinstance(t_entry, dict) else t_entry
                target_v, target_a = t_va[0], t_va[1]

            a_n = np.array([v_seq[-1], a_seq[-1]])
            a_bar = np.array([v_seq.mean(), a_seq.mean()])
            e_target = np.array([target_v, target_a])

            term1_drift = np.linalg.norm(a_n - a_bar)
            term2_cong = np.linalg.norm(e_target - a_n)
            adm = 0.5 * term1_drift + 0.5 * term2_cong

            pairs.append({
                "idm": idm,
                "adm": adm,
                "adm_drift": term1_drift,
                "adm_cong": term2_cong,
                "is_repeat": target_item in seq_items,
            })

            if len(pairs) >= max_pairs:
                print(f"[Build] max_pairs={max_pairs} 도달")
                return pairs

    print(f"[Build] {len(pairs):,} 쌍  (short skip={n_skip_short}, no-cat skip={n_skip_no_cat})")
    return pairs


def analyze(pairs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    idm = np.array([p["idm"] for p in pairs])
    adm = np.array([p["adm"] for p in pairs])
    adm_drift = np.array([p["adm_drift"] for p in pairs])
    adm_cong = np.array([p["adm_cong"] for p in pairs])

    print(f"\n=== 기본 통계 ===")
    print(f"n = {len(pairs):,}")
    print(f"IDM: mean={idm.mean():.3f}  std={idm.std():.3f}")
    print(f"     =0: {(idm == 0).mean()*100:.1f}%   "
          f"=1: {(idm == 1).mean()*100:.1f}%   "
          f"mid: {((idm > 0) & (idm < 1)).mean()*100:.1f}%")
    print(f"ADM: mean={adm.mean():.3f}  std={adm.std():.3f}")

    # ── 1. IDM bin별 ADM 분포 ──
    bins = [(0.0, 0.001), (0.001, 0.5), (0.5, 0.999), (0.999, 1.001)]
    bin_labels = ["IDM=0\n(repeat/overlap)",
                  "0<IDM<0.5\n(partial)",
                  "0.5≤IDM<1\n(mostly new)",
                  "IDM=1\n(complete drift)"]

    bin_data = []
    bin_stats = []
    for (lo, hi), lbl in zip(bins, bin_labels):
        mask = (idm >= lo) & (idm < hi)
        bin_data.append(adm[mask])
        bin_stats.append({
            "bin": lbl,
            "n": int(mask.sum()),
            "pct": float(mask.mean() * 100),
            "adm_mean": float(adm[mask].mean()) if mask.sum() > 0 else None,
            "adm_std": float(adm[mask].std()) if mask.sum() > 0 else None,
        })

    print(f"\n=== Bin별 통계 ===")
    for s in bin_stats:
        if s["n"] > 0:
            lbl = s["bin"].replace("\n", " ")
            print(f"  {lbl:30} n={s['n']:7,} ({s['pct']:5.1f}%)  "
                  f"ADM={s['adm_mean']:.3f}±{s['adm_std']:.3f}")

    # boxplot
    fig, ax = plt.subplots(figsize=(9, 5))
    valid_data = [d for d in bin_data if len(d) > 0]
    valid_labels = [l for l, d in zip(bin_labels, bin_data) if len(d) > 0]
    bp = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="red", lw=2))
    colors = ["#6B8CBA", "#52B788", "#E8A838", "#E85252"]
    for patch, color in zip(bp["boxes"], colors[:len(valid_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("ADM (affective drift)")
    ax.set_title("ADM distribution across IDM bins\nH0: 독립이면 4 박스 동일")
    plt.xticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/idm_bins_adm_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. KS test (인접 bin) ──
    print(f"\n=== KS test (인접 bin) ===")
    ks_results = []
    for i in range(len(valid_data) - 1):
        if len(valid_data[i]) > 30 and len(valid_data[i + 1]) > 30:
            stat, pval = ks_2samp(valid_data[i], valid_data[i + 1])
            sig = ("***" if pval < 0.001 else "**" if pval < 0.01
                   else "*" if pval < 0.05 else "n.s.")
            a = valid_labels[i].replace("\n", " ")
            b = valid_labels[i + 1].replace("\n", " ")
            print(f"  {a[:22]:22} vs {b[:22]:22}: KS={stat:.3f}  p={pval:.2e}  {sig}")
            ks_results.append({"compare": f"{a} vs {b}",
                               "ks_stat": float(stat), "p_value": float(pval)})

    # ── 3. Kruskal-Wallis ──
    kw_stat, kw_p = None, None
    kw_data = [d for d in valid_data if len(d) > 30]
    if len(kw_data) >= 2:
        kw_stat, kw_p = kruskal(*kw_data)
        print(f"\n=== Kruskal-Wallis ===")
        print(f"  H0: 모든 bin의 ADM 분포 동일")
        print(f"  H={kw_stat:.3f}  p={kw_p:.2e}  "
              f"{'reject (분포 다름)' if kw_p < 0.05 else 'fail to reject (분포 비슷)'}")

    # ── 4. Mutual Information ──
    print(f"\n=== Mutual Information ===")
    mi = mutual_info_regression(idm.reshape(-1, 1), adm, random_state=42)[0]
    rng = np.random.default_rng(42)
    idm_shuf = rng.permutation(idm)
    mi_base = mutual_info_regression(idm_shuf.reshape(-1, 1), adm, random_state=42)[0]
    print(f"  I(IDM; ADM)      = {mi:.4f}")
    print(f"  I(shuffled; ADM) = {mi_base:.4f}  ← random baseline")
    print(f"  ratio            = {mi/max(mi_base, 1e-6):.2f}x")

    # ── 5. ADM 두 항 분리 ──
    print(f"\n=== ADM 두 항 분리 ===")
    mi_drift = mutual_info_regression(idm.reshape(-1, 1), adm_drift, random_state=42)[0]
    mi_cong = mutual_info_regression(idm.reshape(-1, 1), adm_cong, random_state=42)[0]
    print(f"  drift term  : mean={adm_drift.mean():.3f}  I(IDM;drift) ={mi_drift:.4f}")
    print(f"  congruence  : mean={adm_cong.mean():.3f}  I(IDM;cong)  ={mi_cong:.4f}")

    # ── 6. Scatter + hist ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_plot = min(5000, len(idm))
    idx = np.random.choice(len(idm), n_plot, replace=False)
    axes[0].scatter(idm[idx], adm[idx], alpha=0.2, s=8, color="#6B8CBA")
    axes[0].set_xlabel("IDM")
    axes[0].set_ylabel("ADM")
    axes[0].set_title(f"Pair-level scatter (n={n_plot:,} subsampled)")
    axes[1].hist(idm, bins=20, color="#52B788", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("IDM")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"IDM distribution (n={len(idm):,})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/idm_adm_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 7. Summary ──
    summary = {
        "n_pairs": len(pairs),
        "idm_dist": {"mean": float(idm.mean()), "std": float(idm.std()),
                     "pct_zero": float((idm == 0).mean() * 100),
                     "pct_one": float((idm == 1).mean() * 100),
                     "pct_middle": float(((idm > 0) & (idm < 1)).mean() * 100)},
        "adm_dist": {"mean": float(adm.mean()), "std": float(adm.std())},
        "bin_stats": bin_stats,
        "ks_results": ks_results,
        "kruskal_wallis": {"stat": float(kw_stat) if kw_stat is not None else None,
                           "p_value": float(kw_p) if kw_p is not None else None},
        "mutual_info": {"I_IDM_ADM": float(mi),
                        "I_shuffled_ADM": float(mi_base),
                        "ratio": float(mi / max(mi_base, 1e-6)),
                        "I_IDM_drift": float(mi_drift),
                        "I_IDM_cong": float(mi_cong)},
        "interpretation": {
            "ks_significant": any(r["p_value"] < 0.05 for r in ks_results),
            "kw_significant": bool(kw_p < 0.05) if kw_p is not None else None,
            "mi_significant": float(mi / max(mi_base, 1e-6)) > 1.5,
        },
    }
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {output_dir}/summary.json")
    print(f"[Saved] {output_dir}/idm_bins_adm_boxplot.png")
    print(f"[Saved] {output_dir}/idm_adm_overview.png")

    # 최종 판정
    print(f"\n{'='*60}")
    print(f"최종 판정")
    print(f"{'='*60}")
    si = summary["interpretation"]
    if si["mi_significant"] and si["kw_significant"]:
        print(f"X IDM과 ADM이 독립이 아님 (둘 다 유의)")
        print(f"   -> motivation 'IDM ⊥ ADM' 약함 -> 둘 중 하나 생략 가능")
    elif not si["mi_significant"] and (si["kw_significant"] is False):
        print(f"O IDM과 ADM은 통계적으로 독립")
        print(f"   -> motivation 강하게 지지됨 -> 둘 다 필요")
    else:
        print(f"~ 결과 mixed — 해석 필요")
        print(f"   KS/KW sig: {si['kw_significant']}, MI sig: {si['mi_significant']}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--item_va", required=True)
    ap.add_argument("--item_cats", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_seq_len", type=int, default=3)
    ap.add_argument("--max_pairs", type=int, default=100000)
    args = ap.parse_args()

    print(f"[Load] sequences: {args.sequences}")
    with open(args.sequences, "rb") as f:
        sequences = pickle.load(f)
    print(f"        {len(sequences):,} users")

    print(f"[Load] item_va: {args.item_va}")
    with open(args.item_va) as f:
        item_va = json.load(f)

    print(f"[Load] item_cats: {args.item_cats}")
    with open(args.item_cats) as f:
        item_cats = json.load(f)

    pairs = compute_pairs(sequences, item_va, item_cats,
                          min_seq_len=args.min_seq_len,
                          max_pairs=args.max_pairs)

    if len(pairs) < 100:
        print(f"[Error] 쌍이 너무 적음: {len(pairs)}")
        return

    analyze(pairs, args.output_dir)


if __name__ == "__main__":
    main()