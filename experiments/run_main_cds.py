"""
experiments/run_main_cds.py
============================
AffSR two-stage training entry point

Usage:
  # Full run (Stage 1 + Stage 2)
  python -m experiments.run_main_cds \
      --d 64 --lambda_ 0.1 --num_neg 1 \
      --gpu_id 1 --output_dir outputs/affsr_cds_d64_lam01

  # Stage 2 only (when Stage 1 ckpt already exists)
  python -m experiments.run_main_cds \
      --d 64 --lambda_ 0.1 --num_neg 1 \
      --gpu_id 1 --output_dir outputs/affsr_cds_d64_lam01 \
      --skip_stage1
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from pathlib import Path

from models.modules.affsr_cds import AffSR
from datasets.cds_dataset_cds import get_dataloaders, load_processed_data
from trainers.affsr_trainer_cds import AffSRTrainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir",    default="data/processed/cds")
    parser.add_argument("--max_len",     type=int, default=50)
    # Model
    parser.add_argument("--d",           type=int, default=64)
    parser.add_argument("--num_layers",  type=int, default=2)
    parser.add_argument("--num_heads",   type=int, default=2)
    parser.add_argument("--dropout",     type=float, default=0.5)
    parser.add_argument("--K",           type=int, default=4)
    # Training (shared)
    parser.add_argument("--num_neg",     type=int, default=1)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    # Stage 1
    parser.add_argument("--s1_epochs",   type=int, default=200)
    parser.add_argument("--s1_patience", type=int, default=10)
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip Stage 1 (when stage1_best.pt already exists)")
    parser.add_argument("--stage1_ckpt", default=None,
                        help="Stage 1 ckpt path (specify when using skip_stage1; defaults to output_dir/stage1_best.pt)")
    # Stage 2
    parser.add_argument("--s2_epochs",   type=int, default=100)
    parser.add_argument("--s2_patience", type=int, default=20)
    parser.add_argument("--lambda_",     type=float, default=0.1)
    # Environment
    parser.add_argument("--gpu_id",      type=int, default=1)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir",  default="outputs/affsr_cds")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(Path(args.output_dir) / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("[Config]")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[Data] Loading...")
    train_loader, valid_loader, test_loader, num_items = get_dataloaders(
        data_dir=args.data_dir,
        max_len=args.max_len,
        num_neg=args.num_neg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    _, _, _, _, item_va, _ = load_processed_data(args.data_dir)
    print(f"  num_items: {num_items:,}")
    print(f"  train/valid/test batches: "
          f"{len(train_loader)}/{len(valid_loader)}/{len(test_loader)}")

    # ── Initialize model ─────────────────────────────────────────────────
    print("\n[Model] Initializing AffSR...")
    model = AffSR(
        num_items=num_items,
        d_model=args.d,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_len=args.max_len,
        dropout=args.dropout,
        K=args.K,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = AffSRTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        item_va=item_va,
        num_items=num_items,
        config={
            "device":     "cuda",
            "lambda":     args.lambda_,
            "output_dir": args.output_dir,
        },
    )

    # ── Stage 1 ──────────────────────────────────────────────────────────
    stage1_ckpt = Path(args.output_dir) / "stage1_best.pt"
    if args.skip_stage1 and stage1_ckpt.exists():
        print(f"\n[Stage 1 Skip] Loading {stage1_ckpt}")
    else:
        stage1_ckpt = trainer.train_stage1(
            num_epochs=args.s1_epochs,
            patience=args.s1_patience,
            lr=args.lr,
        )

    # ── Stage 2 ──────────────────────────────────────────────────────────
    trainer.train_stage2(
        stage1_ckpt=stage1_ckpt,
        num_epochs=args.s2_epochs,
        patience=args.s2_patience,
        lr=args.lr,
    )

    # ── Test ─────────────────────────────────────────────────────────────
    trainer.test()
    print("\nDone!")


if __name__ == "__main__":
    main()
