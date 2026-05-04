"""
trainers/affsr_trainer_cds.py
==============================
AffSR two-stage training trainer

Stage 1 — SASRec pre-train:
  - Softmax cross-entropy loss over all items
  - AffSR modules (DRG, IDRD, MoE, cross-attn) are not used
  - Early stopping patience=10, based on NDCG@10
  - Prints Recall@10, NDCG@10, hit_rate every epoch

Stage 2 — AffSR fine-tune:
  - Load Stage 1 best model and freeze SASRec parameters
  - Train only DRG, IDRD, MoE, cross-attn
  - loss = BPR (r̄_u* · e_final*) + λ·L_disen
  - Early stopping patience=20, based on NDCG@10
  - Prints Stage 1 hit rate + Stage 2 metrics every epoch
"""

import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class AffSRTrainer:
    def __init__(self, model, train_loader, valid_loader, test_loader,
                 item_va, num_items, config):
        self.model        = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader  = test_loader
        self.item_va      = item_va
        self.num_items    = num_items
        self.config       = config

        self.device = torch.device(
            config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.lam        = config.get("lambda", 0.1)
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Item VA tensor (index 0 = padding)
        va_list = [[0.0, 0.0]]
        for i in range(1, num_items + 1):
            va_list.append(item_va.get(i, [0.0, 0.0]))
        self.item_va_tensor = torch.tensor(
            va_list, dtype=torch.float32
        ).to(self.device)  # (num_items+1, 2)

    # ─────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────
    def bpr_loss(self, score_pos, score_neg):
        diff = score_pos.unsqueeze(-1) - score_neg   # (B, num_neg)
        return -F.logsigmoid(diff).mean()

    def ce_loss_full(self, r_u, pos_item):
        """Softmax cross-entropy over all items. pos_item is 1-indexed."""
        E = self.model.item_emb.weight      # (num_items+1, d), index 0 = padding
        logits = r_u @ E[1:].T             # (B, num_items)
        targets = pos_item - 1             # 0-indexed
        return F.cross_entropy(logits, targets)

    # ─────────────────────────────────────────────────────────────────────
    # Stage 1 Top-100 Hit Rate
    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_stage1_hitrate(self, loader, top_k=100):
        self.model.eval()
        hits = []
        for batch in loader:
            item_seq = batch["item_seq"].to(self.device)
            pos_item = batch["pos_item"].to(self.device)
            _, all_scores = self.model.forward_stage1(item_seq)
            _, top_indices = all_scores.topk(top_k, dim=-1)
            top_item_ids = top_indices + 1
            for b in range(item_seq.size(0)):
                hits.append(
                    1 if pos_item[b].item() in top_item_ids[b].tolist() else 0
                )
        return float(np.mean(hits)) * 100

    # ─────────────────────────────────────────────────────────────────────
    # Stage 1 evaluation only (ranking based on r_u · e_id)
    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_stage1(self, loader, ks=(10, 20)):
        self.model.eval()
        recalls = {k: [] for k in ks}
        ndcgs   = {k: [] for k in ks}

        for batch in loader:
            item_seq = batch["item_seq"].to(self.device)
            pos_item = batch["pos_item"].to(self.device)
            B = item_seq.size(0)

            _, all_scores = self.model.forward_stage1(item_seq)  # (B, num_items)
            _, top_indices = all_scores.topk(max(ks), dim=-1)    # (B, max_k)
            top_item_ids = top_indices + 1                        # (B, max_k)

            for b in range(B):
                pos = pos_item[b].item()
                ranked = top_item_ids[b].tolist()
                for k in ks:
                    top_k = ranked[:k]
                    hit = 1 if pos in top_k else 0
                    recalls[k].append(hit)
                    ndcg = 1.0 / math.log2(top_k.index(pos) + 2) if pos in top_k else 0.0
                    ndcgs[k].append(ndcg)

        metrics = {}
        for k in ks:
            metrics[f"Recall@{k}"] = float(np.mean(recalls[k]))
            metrics[f"NDCG@{k}"]   = float(np.mean(ndcgs[k]))
        return metrics

    # ─────────────────────────────────────────────────────────────────────
    # Stage 2 evaluation (top-100 → cross-attn re-ranking)
    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_stage2(self, loader, top_k_stage1=100, ks=(10, 20)):
        self.model.eval()
        recalls   = {k: [] for k in ks}
        ndcgs     = {k: [] for k in ks}
        all_betas = []
        all_sims  = []

        for batch in loader:
            item_seq = batch["item_seq"].to(self.device)
            a_n      = batch["a_n"].to(self.device)
            pos_item = batch["pos_item"].to(self.device)
            B = item_seq.size(0)

            r_u, all_scores = self.model.forward_stage1(item_seq)
            _, top_indices  = all_scores.topk(top_k_stage1, dim=-1)
            top_item_ids    = top_indices + 1                          # (B, 100)
            e_aff_cand      = self.item_va_tensor[top_item_ids]        # (B, 100, 2)
            scores_s2       = self.model.forward_stage2(r_u, a_n, e_aff_cand)  # (B, 100)

            beta = self.model.compute_beta(a_n)
            all_betas.append(beta.cpu())
            drift_reprs = self.model.drg(r_u)
            all_sims.append(self.model.compute_collapse_similarity(drift_reprs))

            for b in range(B):
                pos = pos_item[b].item()
                sorted_idx   = scores_s2[b].argsort(descending=True)
                ranked_items = top_item_ids[b][sorted_idx].tolist()
                for k in ks:
                    top_k = ranked_items[:k]
                    hit   = 1 if pos in top_k else 0
                    recalls[k].append(hit)
                    ndcg  = 1.0 / math.log2(top_k.index(pos) + 2) if pos in top_k else 0.0
                    ndcgs[k].append(ndcg)

        metrics = {}
        for k in ks:
            metrics[f"Recall@{k}"] = float(np.mean(recalls[k]))
            metrics[f"NDCG@{k}"]   = float(np.mean(ndcgs[k]))

        all_betas = torch.cat(all_betas, dim=0)
        metrics["beta_mean"]    = all_betas.mean(dim=0).tolist()
        metrics["collapse_sim"] = torch.stack(all_sims).mean(dim=0).tolist()
        return metrics

    # ─────────────────────────────────────────────────────────────────────
    # Stage 1 training — SASRec pre-train
    # ─────────────────────────────────────────────────────────────────────
    def train_stage1(self, num_epochs=200, patience=10, lr=1e-3):
        print("=" * 60)
        print("Stage 1: SASRec Pre-train")
        print(f"  device={self.device} | patience={patience} | lr={lr}")
        print("=" * 60)

        optimizer = torch.optim.Adam(
            self.model.sasrec.parameters(), lr=lr
        )

        best_ndcg     = -1
        patience_cnt  = 0
        history       = []
        save_path     = self.output_dir / "stage1_best.pt"

        for epoch in range(1, num_epochs + 1):
            # ── Train ──
            self.model.train()
            total_loss = 0.0
            for batch in self.train_loader:
                item_seq = batch["item_seq"].to(self.device)
                pos_item = batch["pos_item"].to(self.device)

                r_u  = self.model.encode_user(item_seq)          # (B, d)
                loss = self.ce_loss_full(r_u, pos_item)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.sasrec.parameters(), max_norm=1.0
                )
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)

            # ── Eval ──
            val_metrics = self.evaluate_stage1(self.valid_loader)
            hit_rate    = self.evaluate_stage1_hitrate(self.valid_loader)
            ndcg10      = val_metrics["NDCG@10"]

            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                  f"R@10={val_metrics['Recall@10']:.4f} | "
                  f"N@10={ndcg10:.4f} | "
                  f"R@20={val_metrics['Recall@20']:.4f} | "
                  f"N@20={val_metrics['NDCG@20']:.4f} | "
                  f"Hit@100={hit_rate:.2f}%")

            history.append({"epoch": epoch, "loss": avg_loss,
                            "hit_rate": hit_rate, **val_metrics})

            if ndcg10 > best_ndcg:
                best_ndcg    = ndcg10
                patience_cnt = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"          [Best] NDCG@10={ndcg10:.4f} | Hit@100={hit_rate:.2f}%")
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"[Stage 1 Early Stop] patience={patience} reached")
                    break

        with open(self.output_dir / "stage1_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n[Stage 1 Done] best NDCG@10={best_ndcg:.4f} | saved: {save_path}")
        return save_path

    # ─────────────────────────────────────────────────────────────────────
    # Stage 2 training — AffSR fine-tune
    # ─────────────────────────────────────────────────────────────────────
    def train_stage2(self, stage1_ckpt, num_epochs=100, patience=20, lr=1e-4):
        print("=" * 60)
        print("Stage 2: AffSR Fine-tune")
        print(f"  device={self.device} | patience={patience} | lr={lr} | λ={self.lam}")
        print("=" * 60)

        # Load Stage 1 best model
        self.model.load_state_dict(
            torch.load(stage1_ckpt, map_location=self.device)
        )

        # Freeze SASRec
        for param in self.model.sasrec.parameters():
            param.requires_grad = False
        print("  SASRec parameters frozen")

        # Train only AffSR modules
        trainable = [
            *self.model.drg.parameters(),
            *self.model.moe.parameters(),
            *self.model.cross_attn.parameters(),
        ]
        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Trainable parameters: {n_trainable:,}")

        optimizer    = torch.optim.Adam(trainable, lr=lr)
        best_ndcg    = -1
        patience_cnt = 0
        history      = []
        save_path    = self.output_dir / "stage2_best.pt"

        for epoch in range(1, num_epochs + 1):
            # ── Train ──
            self.model.train()
            # Keep SASRec in eval mode (disables BN/Dropout)
            self.model.sasrec.eval()
            total_loss = 0.0

            for batch in self.train_loader:
                item_seq  = batch["item_seq"].to(self.device)
                a_n       = batch["a_n"].to(self.device)
                e_aff_pos = batch["e_aff_pos"].to(self.device)
                e_aff_neg = batch["e_aff_neg"].to(self.device)
                pos_item  = batch["pos_item"].to(self.device)

                score_pos, score_neg, beta, drift_reprs = self.model.forward_train(
                    item_seq, a_n, e_aff_pos, e_aff_neg
                )

                l_bpr   = self.bpr_loss(score_pos, score_neg)
                l_disen = self.model.compute_disen_loss(drift_reprs, beta, pos_item)
                loss    = l_bpr + self.lam * l_disen

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)

            # ── Eval ──
            val_metrics = self.evaluate_stage2(self.valid_loader)
            hit_rate    = self.evaluate_stage1_hitrate(self.valid_loader)
            ndcg10      = val_metrics["NDCG@10"]

            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                  f"R@10={val_metrics['Recall@10']:.4f} | "
                  f"N@10={ndcg10:.4f} | "
                  f"R@20={val_metrics['Recall@20']:.4f} | "
                  f"N@20={val_metrics['NDCG@20']:.4f} | "
                  f"Hit@100={hit_rate:.2f}%")
            print(f"          beta={[f'{x:.3f}' for x in val_metrics['beta_mean']]}")

            history.append({"epoch": epoch, "loss": avg_loss,
                            "hit_rate": hit_rate, **val_metrics})

            if ndcg10 > best_ndcg:
                best_ndcg    = ndcg10
                patience_cnt = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"          [Best] NDCG@10={ndcg10:.4f}")
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"[Stage 2 Early Stop] patience={patience} reached")
                    break

        # Unfreeze SASRec (needed for testing)
        for param in self.model.sasrec.parameters():
            param.requires_grad = True

        with open(self.output_dir / "stage2_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n[Stage 2 Done] best NDCG@10={best_ndcg:.4f} | saved: {save_path}")
        return save_path

    # ─────────────────────────────────────────────────────────────────────
    # Test
    # ─────────────────────────────────────────────────────────────────────
    def test(self, stage2_ckpt=None):
        ckpt = stage2_ckpt or (self.output_dir / "stage2_best.pt")
        print(f"\n[Test] Loading model: {ckpt}")
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device)
        )

        # Stage 1 hit rate
        hit_rate = self.evaluate_stage1_hitrate(self.test_loader)
        print(f"\n  Stage 1 Top-100 Hit Rate: {hit_rate:.2f}%")

        # Stage 2 metrics
        test_metrics = self.evaluate_stage2(self.test_loader)
        test_metrics["stage1_top100_hit_rate_%"] = round(hit_rate, 2)

        print("\n[Test Results]")
        for k, v in test_metrics.items():
            if k not in ("beta_mean", "collapse_sim"):
                print(f"  {k}: {v:.4f}")
        print(f"  beta_mean (Q1~Q4): {[f'{x:.3f}' for x in test_metrics['beta_mean']]}")
        print(f"  collapse_sim:\n{np.array(test_metrics['collapse_sim'])}")

        out_path = self.output_dir / "test_results.json"
        with open(out_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\n[Save] {out_path}")
        return test_metrics