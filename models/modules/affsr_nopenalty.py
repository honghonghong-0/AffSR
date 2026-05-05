"""
models/modules/affsr_nopenalty.py
=======================
AffSR integrated model (no λ_mc penalty variant)

Overall flow:
  [Encoding]  item_seq → SASRec → rᵤ
  dist28_seq + seq_mask → AffDrift → β_final

  [User repr]   rᵤ used as-is
  [Item repr]   e_id(v) → MoE(β_final) → e_final

  [Matching]    CrossAttn(rᵤ, e_final) → rᵤ*, e_final*
  score = rᵤ* · e_final*

  [Loss] L = L_rec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from models.backbone.sasrec import SASRec
from models.modules.affdrift import AffDrift
from models.modules.emotion_moe import EmotionMoE
from models.modules.cross_attention import BidirectionalCrossAttention


class AffSR(nn.Module):
    """
    AffSR (no penalty variant).

    Args:
        num_items     : total number of items
        d_model       : embedding dimension
        n_heads       : SASRec attention head count
        n_layers      : SASRec layer count
        max_seq_len   : maximum sequence length
        K             : number of emotion quadrants
        dropout       : dropout rate
        tau           : βₖ softmax temperature initial value
        baseline_only : SASRec + dot product only (skips all affective modules)
        no_moe        : ablation removing MoE (e_final = e_id)
        no_long       : ablation removing va_long (β = β(a_n))
        no_short      : ablation removing a_n (β = β(va_long))
        no_ad         : ablation removing AD (α = learnable fixed scalar)
    """

    def __init__(
        self,
        num_items: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 50,
        K: int = 4,
        dropout: float = 0.2,
        tau: float = 1.0,
        baseline_only: bool = False,
        no_moe: bool = False,
        no_long: bool = False,
        no_short: bool = False,
        no_ad: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_items = num_items
        self.K = K
        self.baseline_only = baseline_only
        self.no_moe = no_moe

        # ── Modules ───────────────────────────────────────────────────
        self.sasrec = SASRec(
            num_items=num_items,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.affdrift = AffDrift(K=K, tau=tau, no_long=no_long, no_short=no_short, no_ad=no_ad)

        self.moe = EmotionMoE(d_model=d_model, K=K)

        self.cross_attn = BidirectionalCrossAttention(
            d_model=d_model, dropout=dropout,
        )

        # Item ID embedding — shared with SASRec internal embedding (single embedding space)
        self.item_emb = self.sasrec.item_embedding

        # Term2 penalty: score -= softplus(λ_mc) * ‖e_aff(v) - aₙ‖₂
        self.lambda_mc = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        # item_emb is shared with sasrec.item_embedding — already initialized by SASRec
        self.item_emb.weight.data[0].zero_()  # ensure padding idx=0 is zero

    # ── Common helpers ────────────────────────────────────────────────
    def _compute_beta(self, a_n: torch.Tensor) -> torch.Tensor:
        """Compute β (requires only a_n, independent of e_aff_v)."""
        dists = torch.norm(
            a_n.unsqueeze(1) - self.affdrift.centroids.unsqueeze(0),
            dim=-1,
        )  # (B, K)
        return F.softmax(-dists / self.affdrift.tau, dim=-1)

    def _user_repr(
        self, r_u: torch.Tensor, beta: torch.Tensor,
    ) -> torch.Tensor:
        return r_u

    def _item_repr(
        self, e_id: torch.Tensor, beta: torch.Tensor,
    ) -> torch.Tensor:
        """Generate item representation via MoE. Returns e_id directly if no_moe."""
        if self.no_moe:
            return e_id
        return self.moe(e_id, beta)

    def score_neg_batch(
        self,
        r_bar_u: torch.Tensor,   # (B, d)  — cached user repr from pos forward
        beta: torch.Tensor,      # (B, K)  — cached β from pos forward
        neg_ids: torch.Tensor,   # (B, num_neg)
    ) -> torch.Tensor:           # (B, num_neg)
        """Score all negative items in a single batched operation.
        Avoids recomputing SASRec / AffDrift / UserDriftRepr, saving significant memory.
        """
        B, num_neg = neg_ids.shape
        d = self.d_model

        if self.baseline_only:
            e_neg = self.item_emb(neg_ids)             # (B, num_neg, d)
            return (r_bar_u.unsqueeze(1) * e_neg).sum(-1)  # (B, num_neg)

        # flatten (B, num_neg) → (B*num_neg,)
        neg_flat = neg_ids.reshape(-1)                 # (B*num_neg,)
        e_id_flat = self.item_emb(neg_flat)            # (B*num_neg, d)

        beta_exp = beta.unsqueeze(1).expand(B, num_neg, self.K).reshape(B * num_neg, self.K)
        e_final_flat = self._item_repr(e_id_flat, beta_exp)  # (B*num_neg, d)

        r_bar_u_exp = r_bar_u.unsqueeze(1).expand(B, num_neg, d).reshape(B * num_neg, d)
        r_u_star, e_final_star = self.cross_attn(r_bar_u_exp, e_final_flat)
        scores = (r_u_star * e_final_star).sum(-1)     # (B*num_neg,)
        return scores.reshape(B, num_neg)

    # ── Forward (training) ────────────────────────────────────────────
    def forward(
        self,
        item_seq: torch.Tensor,      # (B, L)
        seq_mask: torch.Tensor,      # (B, L)
        a_n: torch.Tensor,           # (B, 2)
        dist28_seq: torch.Tensor,    # (B, L, 28)
        e_aff_v: torch.Tensor,       # (B, 2)  target item VA (for score penalty)
        idm: torch.Tensor,           # (B,)
        target_id: torch.Tensor,     # (B,)
    ) -> dict:
        """
        Returns dict:
            score     : (B,)
            adm       : (B,)    AD scalar (for analysis)
            beta      : (B, K)  β_final
            r_u_tilde : (B, d)  r̄ᵤ
        """
        r_u = self.sasrec(item_seq, seq_mask)  # (B, d)

        if self.baseline_only:
            e_target = self.item_emb(target_id)
            score = (r_u * e_target).sum(dim=-1)
            B = score.size(0)
            return {
                "score":     score,
                "adm":       torch.zeros_like(score),
                "beta":      torch.zeros(B, self.K, device=score.device),
                "r_u_tilde": r_u,
            }

        adm, beta = self.affdrift(a_n, dist28_seq, seq_mask)

        r_bar_u = self._user_repr(r_u, beta)            # (B, d)
        e_id = self.item_emb(target_id)                 # (B, d)
        e_final = self._item_repr(e_id, beta)           # (B, d)

        r_u_star, e_final_star = self.cross_attn(r_bar_u, e_final)
        score = (r_u_star * e_final_star).sum(dim=-1)

        return {
            "score":     score,
            "adm":       adm,
            "beta":      beta,
            "r_u_tilde": r_bar_u,
        }

    def _chunk_forward(
        self,
        e_id_flat: torch.Tensor,    # (B*C, d)
        beta_exp: torch.Tensor,     # (B*C, K)
        r_bar_u_exp: torch.Tensor,  # (B*C, d)
    ) -> torch.Tensor:              # (B*C,)
        e_final_flat = self._item_repr(e_id_flat, beta_exp)
        r_u_star, e_final_star = self.cross_attn(r_bar_u_exp, e_final_flat)
        return (r_u_star * e_final_star).sum(dim=-1)

    # ── Predict (evaluation and full-softmax CE training) ─────────────
    def predict(
        self,
        item_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        a_n: torch.Tensor,
        dist28_seq: torch.Tensor,    # (B, L, 28)
        idm: torch.Tensor,
        all_item_va: torch.Tensor,   # (num_items+1, 2)  unused, kept for compatibility
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """Evaluation: compute scores over all items (chunked)."""
        B = item_seq.size(0)
        N = all_item_va.size(0)
        d = self.d_model
        device = item_seq.device

        r_u = self.sasrec(item_seq, seq_mask)  # (B, d)

        if self.baseline_only:
            return r_u @ self.item_emb.weight.T  # (B, N)

        # compute β once from dist28_seq + seq_mask
        adm, beta = self.affdrift(a_n, dist28_seq, seq_mask)  # (B,), (B, K)
        r_bar_u = self._user_repr(r_u, beta)            # (B, d)

        use_ckpt = self.training and torch.is_grad_enabled()
        chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            C = end - start

            item_ids_c = torch.arange(start, end, device=device)  # (C,)
            e_id_c = self.item_emb(item_ids_c)                    # (C, d)
            e_id_flat = e_id_c.unsqueeze(0).expand(B, C, d).reshape(B * C, d).contiguous()
            beta_exp = beta.unsqueeze(1).expand(B, C, self.K).reshape(B * C, self.K).contiguous()
            r_bar_u_exp = r_bar_u.unsqueeze(1).expand(B, C, d).reshape(B * C, d).contiguous()

            if use_ckpt:
                scores_c = ckpt.checkpoint(
                    self._chunk_forward, e_id_flat, beta_exp, r_bar_u_exp,
                    use_reentrant=False,
                )
            else:
                scores_c = self._chunk_forward(e_id_flat, beta_exp, r_bar_u_exp)

            chunks.append(scores_c.view(B, C))

        scores = torch.cat(chunks, dim=1)  # (B, N)

        return scores
