"""
eda_idm_adm_v5.py
=================
ADM 분해 검증 (drift vs congruence) — sliding window 단위

v4와의 차이:
  - 유저당 1개 → 모든 sliding window 쌍
  - Pearson r → KS test + Mutual Information
  - ADM의 두 항을 분리해서 IDM과의 관계 따로 분석 ← 핵심

사용법:
    python preprocessing/eda_idm_adm_v5.py \
        --sequences data/processed/cds/sequences.pkl \
        --item_va   data/processed/cds/item_va.json \
        --item_cats data/processed/cds/item_cats.json \
        --output_dir data_analysis/results/cds_v5 \
        --max_pairs 100000

기대 결과:
  - I(IDM; drift) > I(IDM; congruence) → drift는 IDM과 중복
  - 즉 v9는 congruence term만 써야 함 (정보 효율)
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
    "Genre for Featured Categories", "Amazon Video",
    "CDs & Vinyl",
}


def compute_pairs(sequences, item_va, item_cats, min_seq_len=3, max_pairs=100000):
    pairs = []
    n_skip_short = n_skip_no_cat = 0

    for uid, seq in tqdm(sequences.items(), desc="Building pairs"):
        if len(seq) < min_seq_len:
            n_skip_short += 1
            continue

        for t in range(1, len(seq)):
            input_seq = seq[:t]
            target = seq[t]

            target_item = target[0]
            try:
                target_v, target_a = target[2], target[3]
            except (IndexError, TypeError):
                # fallback: item_va에서 조회
                t_entry = item_va.get(str(target_item))
                if t_entry is None:
                    continue
                t_va = t_entry.get("va") if isinstance(t_entry, dict) else t_entry
                target_v, target_a = t_va[0], t_va[1]

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
                idm = 1.0 - len(inter) / len(target_cats)

            try:
                v_seq = np.array([s[2] for s in input_seq])
                a_seq = np.array([s[3] for s in input_seq])
            except (IndexError, TypeError):
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

            a_n = np.array([v_seq[-1], a_seq[-1]])

            if len(v_seq) < 2:
                continue

            a_bar = np.array([v_seq[:-1].mean(), a_seq[:-1].mean()])
            e_target = np.array([target_v, target_a])

            term1_drift = np.linalg.norm(a_n - a_bar)
            term2_cong = np.linalg.norm(e_target - a_n)
            adm = 0.5 * term1_drift + 0.5 * term2_cong

            pairs.append({
                "uid": uid, "t": t, "idm": idm, "adm": adm,
                "adm_drift": term1_drift, "adm_congruence": term2_cong,
            })

            if len(pairs) >= max_pairs:
                print(f"[Build] max_pairs={max_pairs} 도달")
                return pairs

    print(f"[Build] {len(pairs):,} pairs / skip_short={n_skip_short}, skip_no_cat={n_skip_no_cat}")
    return pairs


def analyze(pairs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    idm = np.array([p["idm"] for p in pairs])
    adm = np.array([p["adm"] for p in pairs])
    adm_drift = np.array([p["adm_drift"] for p in pairs])
    adm_cong = np.array([p["adm_congruence"] for p in pairs])

    print(f"\n=== 기본 통계 ===")
    print(f"n = {len(pairs):,}")
    print(f"IDM=0: {(idm==0).mean()*100:.1f}%, IDM=1: {(idm==1).mean()*100:.1f}%")

    bins = [(0.0, 0.001), (0.001, 0.5), (0.5, 0.999), (0.999, 1.001)]
    bin_labels = ["IDM=0", "0<IDM<0.5", "0.5<=IDM<1", "IDM=1"]

    bin_data = []
    bin_stats = []
    for (lo, hi), lbl in zip(bins, bin_labels):
        mask = (idm >= lo) & (idm < hi)
        bin_data.append(adm[mask])
        bin_stats.append({
            "bin": lbl, "n": int(mask.sum()),
            "pct": float(mask.mean() * 100),
            "adm_mean": float(adm[mask].mean()) if mask.sum() > 0 else None,
        })
    print(f"\n=== Bin 통계 ===")
    for s in bin_stats:
        print(f"  {s['bin']:12} n={s['n']:6,} ({s['pct']:5.1f}%)  ADM={s['adm_mean']}")

    # Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_data = [d for d in bin_data if len(d) > 0]
    valid_labels = [l for l, d in zip(bin_labels, bin_data) if len(d) > 0]
    bp = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True, showfliers=False)
    colors = ["#6B8CBA", "#52B788", "#E8A838", "#E85252"]
    for patch, c in zip(bp["boxes"], colors[:len(valid_data)]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("ADM"); ax.set_title("ADM across IDM bins")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/idm_bins_adm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # KS test
    print(f"\n=== KS test ===")
    ks_results = []
    for i in range(len(valid_data) - 1):
        if len(valid_data[i]) > 30 and len(valid_data[i+1]) > 30:
            stat, pval = ks_2samp(valid_data[i], valid_data[i+1])
            print(f"  {valid_labels[i]} vs {valid_labels[i+1]}: KS={stat:.3f} p={pval:.2e}")
            ks_results.append({"compare": f"{valid_labels[i]} vs {valid_labels[i+1]}",
                               "ks": float(stat), "p": float(pval)})

    # Kruskal-Wallis
    kw_data = [d for d in valid_data if len(d) > 30]
    kw_stat = kw_p = None
    if len(kw_data) >= 2:
        kw_stat, kw_p = kruskal(*kw_data)
        print(f"\n=== Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.2e} ===")

    # MI
    print(f"\n=== Mutual Information (핵심!) ===")
    mi_full = mutual_info_regression(idm.reshape(-1, 1), adm, random_state=42)[0]
    np.random.seed(42)
    idm_shuf = np.random.permutation(idm)
    mi_baseline = mutual_info_regression(idm_shuf.reshape(-1, 1), adm, random_state=42)[0]

    mi_drift = mutual_info_regression(idm.reshape(-1, 1), adm_drift, random_state=42)[0]
    mi_cong = mutual_info_regression(idm.reshape(-1, 1), adm_cong, random_state=42)[0]

    print(f"  I(IDM; ADM)         = {mi_full:.4f}  (baseline={mi_baseline:.4f})")
    print(f"  I(IDM; drift)       = {mi_drift:.4f}  <- higher = more dependent")
    print(f"  I(IDM; congruence)  = {mi_cong:.4f}   <- lower  = more independent")
    if mi_cong > 0:
        print(f"  ratio (drift/cong)  = {mi_drift/mi_cong:.2f}x")

    # Save
    summary = {
        "n_pairs": len(pairs),
        "idm_pct_zero": float((idm == 0).mean() * 100),
        "idm_pct_one": float((idm == 1).mean() * 100),
        "bin_stats": bin_stats,
        "ks_results": ks_results,
        "kruskal_wallis": {"H": float(kw_stat) if kw_stat else None,
                           "p": float(kw_p) if kw_p else None},
        "mutual_info": {
            "I_IDM_ADM_full": float(mi_full),
            "I_IDM_ADM_baseline_shuffled": float(mi_baseline),
            "I_IDM_drift_term": float(mi_drift),
            "I_IDM_congruence_term": float(mi_cong),
            "drift_over_cong_ratio": float(mi_drift / mi_cong) if mi_cong > 0 else None,
        },
        "interpretation": {
            "drift_more_dependent_than_cong": bool(mi_drift > mi_cong),
            "cong_is_independent_signal": bool(mi_cong < mi_drift * 0.5),
        }
    }
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {output_dir}/summary.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--item_va", required=True)
    ap.add_argument("--item_cats", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_pairs", type=int, default=100000)
    args = ap.parse_args()

    print(f"[Load] sequences: {args.sequences}")
    with open(args.sequences, "rb") as f:
        sequences = pickle.load(f)
    print(f"[Load] item_va: {args.item_va}")
    with open(args.item_va) as f:
        item_va = json.load(f)
    print(f"[Load] item_cats: {args.item_cats}")
    with open(args.item_cats) as f:
        item_cats = json.load(f)

    pairs = compute_pairs(sequences, item_va, item_cats, max_pairs=args.max_pairs)
    if len(pairs) < 100:
        print(f"[Error] 쌍 부족: {len(pairs)}"); return
    analyze(pairs, args.output_dir)


if __name__ == "__main__":
    main()