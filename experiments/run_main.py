"""
experiments/run_main.py
=======================
AffSR training entry point

Usage:
  # CDs dataset
  CUDA_VISIBLE_DEVICES=1 python experiments/run_main.py \
      --data_dir data/processed/cds \
      --dataset cds

  # Movies dataset
  CUDA_VISIBLE_DEVICES=1 python experiments/run_main.py \
      --data_dir data/processed/movies_tv_2021_2023 \
      --dataset movies

  # ablation: remove ℒ_disen
  CUDA_VISIBLE_DEVICES=1 python experiments/run_main.py \
      --data_dir data/processed/cds \
      --lambda_disen 0.0

  # ablation: remove Cross-attention (use_cross_attn flag in affsr.py)
  CUDA_VISIBLE_DEVICES=1 python experiments/run_main.py \
      --data_dir data/processed/cds \
      --no_cross_attn
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.base_dataset import AffSRDataset
from models.affsr import AffSR
from trainers.base_trainer import AffSRTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_all_item_va(dataset: AffSRDataset, num_items: int) -> torch.Tensor:
    """
    Build full item VA tensor. Shape (num_items+1, 2), index 0 is padding (zero vector).
    Passed to model.predict() during evaluation.
    """
    va = torch.zeros(num_items + 1, 2, dtype=torch.float32)
    for item_idx, va_arr in dataset.item_va.items():
        if item_idx <= num_items:
            va[item_idx] = torch.from_numpy(va_arr)

    # [Sanity check 1] Verify index alignment
    item_va_keys = list(dataset.item_va.keys())
    sample_targets = [s["target_idx"] for s in dataset.samples[:5]]
    print(f"\n[Sanity check 1] Index range verification")
    print(f"  all_item_va shape         : {tuple(va.shape)}  (expected ({num_items+1}, 2))")
    print(f"  num_items                 : {num_items}")
    print(f"  item_va keys range        : [{min(item_va_keys)}, {max(item_va_keys)}]")
    print(f"  item_va keys sample       : {sorted(item_va_keys)[:5]}")
    print(f"  target sample (train[:5]) : {sample_targets}")
    nonzero_count = (va.norm(dim=-1) > 0).sum().item()
    print(f"  va nonzero entries        : {nonzero_count} / {va.size(0)}")
    print()
    return va


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir",    type=str, required=True)
    parser.add_argument("--dataset",     type=str, default="cds")

    # Model
    parser.add_argument("--d_model",     type=int, default=64)
    parser.add_argument("--n_heads",     type=int, default=2)
    parser.add_argument("--n_layers",    type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--K",           type=int, default=4)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--tau",         type=float, default=1.0)

    # Training
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-5)  # lowered from 5e-5 to 1e-5
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_neg",      type=int,   default=50)
    parser.add_argument("--patience",     type=int,   default=10)

    # Loss
    parser.add_argument("--lambda_disen",   type=float, default=0.1)
    parser.add_argument("--contrastive_tau",type=float, default=0.1)
    parser.add_argument("--lambda_inter",   type=float, default=0.5)

    # Misc
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--device",  type=str, default="cuda")
    parser.add_argument("--save_dir",type=str, default="outputs/checkpoints")

    # Ablation (v8)
    parser.add_argument(
        "--baseline_only", action="store_true",
        help="SASRec standalone baseline (skip all affective modules)",
    )
    parser.add_argument(
        "--no_moe", action="store_true",
        help="MoE ablation — e_final = e_id",
    )
    parser.add_argument(
        "--no_long", action="store_true",
        help="Long-term sentiment ablation — β = β(a_n) only",
    )
    parser.add_argument(
        "--no_short", action="store_true",
        help="Short-term sentiment ablation — β = β(va_long) only",
    )
    parser.add_argument(
        "--no_ad", action="store_true",
        help="AD ablation — replace α with a learnable fixed scalar",
    )

    parser.add_argument(
        "--full_ce", action="store_true",
        help="Full-softmax CE loss (instead of sampled CE)",
    )

    # Run name (auto timestamp)
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="If not specified, auto-generated as {dataset}_{YYYYMMDD_HHMMSS}",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────
    print(f"[Loading data] {args.data_dir}")
    train_ds = AffSRDataset(
        args.data_dir, split="train",
        max_seq_len=args.max_seq_len, num_neg=args.num_neg, seed=args.seed,
        full_ce=args.full_ce,
    )
    valid_ds = AffSRDataset(
        args.data_dir, split="valid",
        max_seq_len=args.max_seq_len, num_neg=args.num_neg, seed=args.seed,
    )
    test_ds = AffSRDataset(
        args.data_dir, split="test",
        max_seq_len=args.max_seq_len, num_neg=args.num_neg, seed=args.seed,
    )
    print(train_ds)
    print(f"  valid: {len(valid_ds)} | test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Full item VA ──────────────────────────────────────────────────
    all_item_va = build_all_item_va(train_ds, train_ds.num_items)

    # ── Model (v8) ────────────────────────────────────────────────────
    model = AffSR(
        num_items=train_ds.num_items,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        K=args.K,
        dropout=args.dropout,
        tau=args.tau,
        baseline_only=args.baseline_only,
        no_moe=args.no_moe,
        no_long=args.no_long,
        no_short=args.no_short,
        no_ad=args.no_ad,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.baseline_only:
        print(f"[BASELINE MODE] SASRec + ID embedding only")
    if args.no_moe:
        print(f"[ABLATION] MoE removed — e_final = e_id")
    if args.no_long:
        print(f"[ABLATION] Long-term sentiment removed — β = β(a_n)")
    if args.no_short:
        print(f"[ABLATION] Short-term sentiment removed — β = β(va_long)")
    if args.no_ad:
        print(f"[ABLATION] AD removed — α = learnable fixed scalar")
    print(f"Model parameters: {num_params:,}")

    # ── Config ───────────────────────────────────────────────────────
    config = vars(args)

    # ── Run name (with timestamp) ────────────────────────────────────
    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.dataset}_{ts}"
    else:
        run_name = args.run_name
    print(f"[run_name] {run_name}")

    # ── Trainer ──────────────────────────────────────────────────────
    save_dir = os.path.join(args.save_dir, run_name)
    trainer = AffSRTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        all_item_va=all_item_va,
        config=config,
        device=device,
        save_dir=save_dir,
    )

    # ── Start training ────────────────────────────────────────────────
    print(f"\nStarting training (epochs={args.epochs}, patience={args.patience})")
    test_metrics = trainer.train(args.epochs)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")


if __name__ == "__main__":
    main()
