"""
trainers/base_trainer.py
========================
AffSR Trainer

Usage:
  python experiments/run_main.py --data_dir data/processed/cds --dataset cds
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trainers.losses import AffSRLoss


class AffSRTrainer:
    """
    Args:
        model       : AffSR model
        train_loader: training DataLoader
        valid_loader: validation DataLoader
        test_loader : evaluation DataLoader
        all_item_va : (num_items, 2) full item VA tensor (for evaluation)
        config      : hyperparameter dict
        device      : 'cuda' or 'cpu'
        save_dir    : checkpoint save directory
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        all_item_va: torch.Tensor,
        config: dict,
        device: str = "cuda",
        save_dir: str = "outputs/checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.all_item_va = all_item_va.to(device)
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # Scheduler (optional)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get("lr_decay_step", 20),
            gamma=config.get("lr_decay_gamma", 0.5),
        )

        # Loss
        self.criterion = AffSRLoss(
            lambda_disen=config.get("lambda_disen", 0.1),
            tau=config.get("contrastive_tau", 0.1),
            lambda_inter=config.get("lambda_inter", 0.5),
        )

        self.full_ce = config.get("full_ce", False)

        self.best_ndcg = 0.0
        self.best_epoch = 0
        self.patience = config.get("patience", 10)
        self.patience_counter = 0

    # ─────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────

    def train(self, epochs: int):
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            self.scheduler.step()

            metrics = self.evaluate(self.valid_loader, k_list=[10, 20])
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
                f"R@10={metrics['Recall@10']:.4f} N@10={metrics['NDCG@10']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Early stopping (based on NDCG@10)
            if metrics["NDCG@10"] > self.best_ndcg:
                self.best_ndcg = metrics["NDCG@10"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint("best.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch} (best={self.best_epoch})")
                    break

        # Final test
        print("\n── Test (best checkpoint) ──")
        self._load_checkpoint("best.pt")
        test_metrics = self.evaluate(self.test_loader, k_list=[10, 20])
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        result_path = self.save_dir / "test_results.json"
        with open(result_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nResults saved: {result_path}")
        return test_metrics

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        valid_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.full_ce:
                # ── Full-softmax CE ─────────────────────────────────────
                all_scores = self.model.predict(
                    item_seq=batch["item_seq"],
                    seq_mask=batch["seq_mask"],
                    a_n=batch["a_n"],
                    dist28_seq=batch["dist28_seq"],
                    idm=batch["idm"],
                    all_item_va=self.all_item_va,
                )  # (B, N)
                if torch.isnan(all_scores).any():
                    continue
                l_rec = F.cross_entropy(all_scores, batch["target"])
                loss_dict = {"loss": l_rec, "l_rec": l_rec, "l_disen": torch.tensor(0.0)}
            else:
                # ── Sampled CE (original) ───────────────────────────────
                out = self.model(
                    item_seq=batch["item_seq"],
                    seq_mask=batch["seq_mask"],
                    a_n=batch["a_n"],
                    dist28_seq=batch["dist28_seq"],
                    e_aff_v=batch["e_aff_pos"],
                    idm=batch["idm"],
                    target_id=batch["target"],
                )
                score_pos = out["score"]
                if torch.isnan(score_pos).any():
                    continue

                neg_items = batch["neg_items"]
                score_neg = self.model.score_neg_batch(
                    r_bar_u=out["r_u_tilde"],
                    beta=out["beta"],
                    neg_ids=neg_items,
                )
                if torch.isnan(score_neg).any():
                    continue

                loss_dict = self.criterion(
                    score_pos=score_pos,
                    score_neg=score_neg,
                    r_u_tilde=out["r_u_tilde"],
                    beta=out["beta"],
                )
                if torch.isnan(loss_dict["loss"]):
                    continue

            self.optimizer.zero_grad()
            loss_dict["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss_dict["loss"].item()
            valid_batches += 1

        return total_loss / max(valid_batches, 1)

    # ─────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        k_list: list = [10, 20],
    ) -> dict:
        self.model.eval()

        recalls = {k: [] for k in k_list}
        ndcgs   = {k: [] for k in k_list}

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            B = batch["item_seq"].size(0)

            # Full item scores (B, num_items)
            scores = self.model.predict(
                item_seq=batch["item_seq"],
                seq_mask=batch["seq_mask"],
                a_n=batch["a_n"],
                dist28_seq=batch["dist28_seq"],
                idm=batch["idm"],
                all_item_va=self.all_item_va,
            )

            # Mask seen items (including padding=0)
            scores[:, 0] = float("-inf")
            seen_mask = batch["item_seq"]  # (B, L)
            for b in range(B):
                scores[b, seen_mask[b]] = float("-inf")

            targets = batch["target"]  # (B,)

            for k in k_list:
                topk_idx = scores.topk(k, dim=-1).indices  # (B, k)
                hit = (topk_idx == targets.unsqueeze(1)).any(dim=-1).float()  # (B,)

                # NDCG
                rank = (topk_idx == targets.unsqueeze(1)).nonzero(as_tuple=False)
                ndcg_scores = torch.zeros(B, device=self.device)
                if rank.numel() > 0:
                    batch_idx = rank[:, 0]
                    pos_idx   = rank[:, 1].float()
                    ndcg_scores[batch_idx] = 1.0 / torch.log2(pos_idx + 2)

                recalls[k].append(hit.cpu())
                ndcgs[k].append(ndcg_scores.cpu())

        metrics = {}
        for k in k_list:
            metrics[f"Recall@{k}"] = torch.cat(recalls[k]).mean().item()
            metrics[f"NDCG@{k}"]   = torch.cat(ndcgs[k]).mean().item()

        return metrics

    # ─────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ndcg": self.best_ndcg,
                "best_epoch": self.best_epoch,
                "config": self.config,
            },
            self.save_dir / name,
        )

    def _load_checkpoint(self, name: str):
        ckpt = torch.load(self.save_dir / name, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

