"""
models/modules/emotion_moe.py
=============================
Affective MoE module

Role:
  Transforms **item ID embeddings** per quadrant based on current emotion state (βₖ).

Formula:
  e_final = Σₖ βₖ · (Eₖ(e_id) + W_ck · cₖ)

  Eₖ ∈ ℝ^{d×d} : quadrant-specific linear transform (initialized as Identity)
  W_ck ∈ ℝ^{d×2}: projects centroid to d-dim bias

Inputs:
  e_id  : (B, d)    item ID embedding
  beta  : (B, K)    gating weights from AffDrift

Output:
  e_final : (B, d)  emotion-conditioned item representation
"""

import torch
import torch.nn as nn

from models.modules.affdrift import QUADRANT_CENTROIDS


class EmotionMoE(nn.Module):
    """
    Affective Mixture-of-Experts (item ID embedding input variant).

    Args:
        d_model : ID embedding dimension (= output dimension)
        K       : number of experts (= number of quadrants, default 4)
    """

    def __init__(self, d_model: int, K: int = 4):
        super().__init__()
        self.d_model = d_model
        self.K = K

        # Eₖ: quadrant-specific linear transform (d → d)
        # Identity initialization — passes ID embedding through unchanged at training start
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(K)
        ])
        self._init_experts()

        # W_ck: centroid → d-dim bias projection (K transforms, each 2 → d)
        self.centroid_proj = nn.ModuleList([
            nn.Linear(2, d_model, bias=False) for _ in range(K)
        ])

        # centroids are fixed (shared with AffDrift); slice/pad when K≠4
        if K <= 4:
            centroids = QUADRANT_CENTROIDS[:K]
        else:
            center = torch.zeros(K - 4, 2, dtype=torch.float32)
            centroids = torch.cat([QUADRANT_CENTROIDS, center], dim=0)
        self.register_buffer("centroids", centroids)  # (K, 2)

    def _init_experts(self):
        """Initialize Eₖ as Identity (d×d diagonal = 1)."""
        for expert in self.experts:
            nn.init.eye_(expert.weight)

    def forward(
        self,
        e_id: torch.Tensor,   # (B, d)
        beta: torch.Tensor,   # (B, K)
    ) -> torch.Tensor:        # (B, d)
        """
        e_final = Σₖ βₖ · (Eₖ(e_id) + W_ck · cₖ)
        """
        B = e_id.size(0)
        e_final = torch.zeros(B, self.d_model, device=e_id.device)

        for k in range(self.K):
            # Eₖ(e_id): (B, d)
            expert_out = self.experts[k](e_id)

            # W_ck · cₖ: centroid is a shared bias across the batch
            centroid_bias = self.centroid_proj[k](
                self.centroids[k].unsqueeze(0)  # (1, 2)
            )  # (1, d)

            gate = beta[:, k].unsqueeze(1)  # (B, 1)

            e_final = e_final + gate * (expert_out + centroid_bias)

        return e_final  # (B, d)