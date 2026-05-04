"""
trainers/losses.py
==================
AffSR Loss functions

ℒ = ℒ_rec + λ · ℒ_disen

ℒ_rec  : BPR loss (positive vs negative pair comparison)
          → Converges faster than cross-entropy; standard in sequential rec

ℒ_disen: Contrastive learning based on soft βₖ weights (ablation switch available)
  ℒ_intra: Pull users in the same quadrant closer together
  ℒ_inter: Push users in different quadrants apart
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss.
    Trains the model so that score_pos > score_neg.
    """

    def forward(
        self,
        score_pos: torch.Tensor,  # (B,)
        score_neg: torch.Tensor,  # (B,) or (B, num_neg)
    ) -> torch.Tensor:

        if score_neg.dim() == 2:
            score_pos = score_pos.unsqueeze(1)  # (B, 1)

        diff = score_pos - score_neg
        loss = -F.logsigmoid(diff).mean()
        return loss


class CELoss(nn.Module):
    """
    Sampled-softmax Cross-Entropy.
    The first element in [score_pos, score_neg_1, ..., score_neg_K] is the ground truth.
    """

    def forward(
        self,
        score_pos: torch.Tensor,  # (B,)
        score_neg: torch.Tensor,  # (B,) or (B, num_neg)
    ) -> torch.Tensor:
        if score_neg.dim() == 1:
            score_neg = score_neg.unsqueeze(1)
        logits = torch.cat([score_pos.unsqueeze(1), score_neg], dim=1)  # (B, 1+K)
        labels = torch.zeros(
            logits.size(0), dtype=torch.long, device=logits.device
        )
        return F.cross_entropy(logits, labels)


class DisentanglementLoss(nn.Module):
    """
    Contrastive learning based on soft βₖ weights.

    ℒ_intra: Pull representations of dominant users in the same quadrant closer together.
    ℒ_inter: Push users in different quadrants apart.

    Args:
        tau         : contrastive temperature
        lambda_inter: weight for ℒ_inter
    """

    def __init__(self, tau: float = 0.1, lambda_inter: float = 0.5):
        super().__init__()
        self.tau = tau
        self.lambda_inter = lambda_inter

    def forward(
        self,
        r_u_tilde: torch.Tensor,  # (B, d)
        beta: torch.Tensor,       # (B, K)
    ) -> torch.Tensor:
        B, d = r_u_tilde.shape

        dominant = beta.argmax(dim=-1)  # (B,)

        r_norm = F.normalize(r_u_tilde, dim=-1)  # (B, d)
        sim = r_norm @ r_norm.T / self.tau        # (B, B)

        mask_self = torch.eye(B, dtype=torch.bool, device=r_u_tilde.device)
        sim = sim.masked_fill(mask_self, float("-inf"))

        soft_weight = beta @ beta.T  # (B, B)
        soft_weight = soft_weight.masked_fill(mask_self, 0.0)

        log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)  # (B, B)
        # Diagonal is -inf and soft_weight diagonal is 0, so mask to prevent 0 * (-inf) = NaN
        log_prob = log_prob.masked_fill(mask_self, 0.0)
        l_intra = -(soft_weight * log_prob).sum() / (soft_weight.sum() + 1e-8)

        diff_quad = dominant.unsqueeze(0) != dominant.unsqueeze(1)  # (B, B)
        diff_quad = diff_quad & ~mask_self

        if diff_quad.sum() == 0:
            l_inter = torch.tensor(0.0, device=r_u_tilde.device)
        else:
            same_quad = dominant.unsqueeze(0) == dominant.unsqueeze(1)
            same_quad = same_quad & ~mask_self

            sim_pos = sim.clone()
            sim_pos = sim_pos.masked_fill(~same_quad, float("-inf"))
            best_pos, _ = sim_pos.max(dim=-1)  # (B,)

            sim_neg = sim.clone()
            sim_neg = sim_neg.masked_fill(~diff_quad, float("-inf"))
            denom = torch.logsumexp(
                torch.cat([best_pos.unsqueeze(1), sim_neg], dim=1),
                dim=-1,
            )  # (B,)

            valid = same_quad.any(dim=-1) & diff_quad.any(dim=-1)
            if valid.sum() == 0:
                l_inter = torch.tensor(0.0, device=r_u_tilde.device)
            else:
                l_inter = -(best_pos[valid] - denom[valid]).mean()

        return l_intra + self.lambda_inter * l_inter


class AffSRLoss(nn.Module):
    """
    Full AffSR Loss.

    ℒ = ℒ_rec + λ · ℒ_disen

    Args:
        lambda_disen : weight for ℒ_disen (0 = ablation w/o ℒ_disen)
        tau          : contrastive temperature
        lambda_inter : internal weight for ℒ_inter
    """

    def __init__(
        self,
        lambda_disen: float = 0.1,
        tau: float = 0.1,
        lambda_inter: float = 0.5,
    ):
        super().__init__()
        self.lambda_disen = lambda_disen
        self.rec_loss = CELoss()
        self.disen = DisentanglementLoss(tau=tau, lambda_inter=lambda_inter)

    def forward(
        self,
        score_pos: torch.Tensor,   # (B,)
        score_neg: torch.Tensor,   # (B,) or (B, num_neg)
        r_u_tilde: torch.Tensor,   # (B, d)
        beta: torch.Tensor,        # (B, K)
    ) -> dict:
        l_rec = self.rec_loss(score_pos, score_neg)

        if self.lambda_disen > 0:
            l_disen = self.disen(r_u_tilde, beta)
        else:
            l_disen = torch.tensor(0.0, device=score_pos.device)

        total = l_rec + self.lambda_disen * l_disen

        return {
            "loss":    total,
            "l_rec":   l_rec,
            "l_disen": l_disen,
        }