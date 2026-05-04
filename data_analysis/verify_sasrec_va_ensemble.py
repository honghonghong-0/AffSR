"""
Verification D-2: SASRec + VA simple ensemble

Post-hoc combination of SASRec scores with VA similarity (no training).
Sweep α to determine at which α the model outperforms the baseline.

score_final = sasrec_score + α · va_similarity

Normalization: combine after z-score normalizing both scores to the same scale.

Reference points:
  α = 0.0: SASRec baseline
  α > 0:   VA auxiliary signal added

Interpretation patterns:
  Pattern 1 (VA helps): peak at intermediate α → proceed with AffSR direction
  Pattern 2 (VA hurts): monotone decrease as α↑
  Pattern 3 (no effect): stable regardless of α
"""

import argparse
import json
import pickle
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.base_dataset import AffSRDataset
from models.affsr import AffSR


def evaluate_ensemble(
    sasrec_ckpt_path: Path,
    data_dir: Path,
    split: str = "valid",
    alphas: list = None,
    batch_size: int = 128,
    device: str = "cuda",
):
    if alphas is None:
        alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    # ── Data loading ─────────────────────────────────────────────────
    print(f"[Load] Dataset: {data_dir} / {split}")
    dataset = AffSRDataset(
        str(data_dir), split=split, max_seq_len=50, num_neg=1, seed=42,
    )
    num_items = dataset.num_items

    # all_item_va (N+1, 2)
    all_va = torch.zeros(num_items + 1, 2)
    for idx, va in dataset.item_va.items():
        if idx <= num_items:
            all_va[idx] = torch.from_numpy(va)
    all_va = all_va.to(device)

    # ── Load SASRec model (baseline_only) ────────────────────────────
    print(f"[Load] SASRec checkpoint: {sasrec_ckpt_path}")
    ckpt = torch.load(sasrec_ckpt_path, map_location=device, weights_only=False)

    # restore as baseline_only (checkpoint structure is v8 full — load full then use baseline mode)
    model = AffSR(
        num_items=num_items,
        d_model=64, n_heads=2, n_layers=2, max_seq_len=50,
        K=4, dropout=0.5, tau=1.0,
        baseline_only=True,
    ).to(device)
    # baseline checkpoint was trained with baseline_only=True, so structure matches
    try:
        model.load_state_dict(ckpt["model"], strict=False)
    except Exception as e:
        print(f"[Warning] Loading with strict=False: {e}")
    model.eval()

    # ── Loader ──────────────────────────────────────────────────────
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # ── Collect both scores ─────────────────────────────────────────
    results = {alpha: {"recalls": {10: [], 20: []}, "ndcgs": {10: [], 20: []}}
               for alpha in alphas}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{split} ensemble"):
            item_seq = batch["item_seq"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            a_n = batch["a_n"].to(device)
            a_bar_u = batch["a_bar_u"].to(device)
            idm = batch["idm"].to(device)
            targets = batch["target"].to(device)
            B = item_seq.size(0)
            N = num_items + 1

            # SASRec scores (B, N)
            sasrec_scores = model.predict(
                item_seq, seq_mask, a_n, a_bar_u, idm, all_va,
            )

            # VA similarity scores (B, N)
            # user r_u_va = mean of sequence VA
            va_seq = batch["va_seq"].to(device)  # (B, L, 2)
            mask_float = seq_mask.float().unsqueeze(-1)  # (B, L, 1)
            r_u_va = (va_seq * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)  # (B, 2)
            # distance (B, N)
            dists = torch.norm(
                r_u_va.unsqueeze(1) - all_va.unsqueeze(0), dim=-1,
            )
            va_scores = -dists  # higher score for closer items

            # mask seen items (both scores set to -inf at same positions)
            sasrec_scores[:, 0] = float("-inf")
            va_scores[:, 0] = float("-inf")
            for b in range(B):
                seen = item_seq[b]
                sasrec_scores[b, seen] = float("-inf")
                va_scores[b, seen] = float("-inf")

            # z-score normalization per user (excluding -inf)
            def zscore_per_row(s):
                out = torch.full_like(s, float("-inf"))
                for i in range(s.size(0)):
                    row = s[i]
                    valid = row != float("-inf")
                    if valid.sum() < 2:
                        continue
                    vals = row[valid]
                    mean = vals.mean()
                    std = vals.std().clamp(min=1e-6)
                    out[i, valid] = (vals - mean) / std
                return out

            sasrec_z = zscore_per_row(sasrec_scores)
            va_z = zscore_per_row(va_scores)

            # α sweep
            for alpha in alphas:
                combined = sasrec_z + alpha * va_z  # (B, N)

                for k in [10, 20]:
                    topk = combined.topk(k, dim=-1).indices  # (B, k)
                    hit = (topk == targets.unsqueeze(1)).any(dim=-1).float()
                    results[alpha]["recalls"][k].extend(hit.cpu().tolist())

                    rank_pos = (topk == targets.unsqueeze(1)).nonzero(as_tuple=False)
                    ndcg_vec = torch.zeros(B, device=device)
                    if rank_pos.numel() > 0:
                        b_idx = rank_pos[:, 0]
                        p_idx = rank_pos[:, 1].float()
                        ndcg_vec[b_idx] = 1.0 / torch.log2(p_idx + 2)
                    results[alpha]["ndcgs"][k].extend(ndcg_vec.cpu().tolist())

    # ── Results ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"Verification D-2 Results ({split})")
    print(f"{'='*72}")
    print(f"{'α':>6}  {'R@10':>8}  {'N@10':>8}  {'R@20':>8}  {'N@20':>8}  Δ(R@10 vs α=0)")
    print("-" * 72)

    baseline_r10 = np.mean(results[0.0]["recalls"][10]) if 0.0 in results else None

    for alpha in alphas:
        r10 = np.mean(results[alpha]["recalls"][10])
        n10 = np.mean(results[alpha]["ndcgs"][10])
        r20 = np.mean(results[alpha]["recalls"][20])
        n20 = np.mean(results[alpha]["ndcgs"][20])
        delta = f"{(r10 - baseline_r10)*1e4:+.1f}e-4" if baseline_r10 is not None else "-"
        marker = " ⭐" if baseline_r10 is not None and r10 > baseline_r10 else ""
        print(f"{alpha:>6.2f}  {r10:>8.4f}  {n10:>8.4f}  {r20:>8.4f}  {n20:>8.4f}  {delta}{marker}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="outputs/checkpoints/movies_sasrec/best.pt")
    parser.add_argument("--data_dir", type=str,
                        default="data/processed/movies_tv_2021_2023")
    parser.add_argument("--split", type=str, default="valid")
    args = parser.parse_args()

    evaluate_ensemble(
        sasrec_ckpt_path=Path(args.ckpt),
        data_dir=Path(args.data_dir),
        split=args.split,
    )