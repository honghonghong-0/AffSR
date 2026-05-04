"""
models/modules/user_drift_repr.py
==================================
User Drift Representation Generator — v8 Main Contribution 1

Role:
  Reinterprets IDURL's DRG (Drift Representation Generation) structure
  for VA quadrants.
  While IDURL generates K representations via IDM discretization (hard routing),
  v8 uses βₖ (soft routing) to generate K representations.

Formulas:
  Fₖ(x) = LN(Dropout(x + Linear_k(x)))
  r̃ᵤᵏ = Fₖ(rᵤ),  k=1..K

  L_disen disentangles the k-th representation during training.
  r̄ᵤ = Σₖ βₖ · r̃ᵤᵏ   (weighted sum applied externally)

Input:
  r_u : (B, d)  SASRec-encoded user representation

Output:
  drift_reprs : list[K] of (B, d)
"""

from typing import List

import torch
import torch.nn as nn


class UserDriftRepr(nn.Module):
    """
    Generates user drift representations using K independent FFNs.

    Args:
        d_model : input/output dimension
        K       : number of quadrants
        dropout : dropout rate
    """

    def __init__(self, d_model: int, K: int = 4, dropout: float = 0.5):
        super().__init__()
        self.K = K
        self.d_model = d_model

        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(K)
        ])
        self.lns = nn.ModuleList([
            nn.LayerNorm(d_model, eps=1e-8) for _ in range(K)
        ])
        self.dropout = nn.Dropout(dropout)

        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, r_u: torch.Tensor) -> List[torch.Tensor]:
        """
        r_u: (B, d)
        returns: list of K tensors [(B, d), ...]
        """
        drift_reprs = []
        for k in range(self.K):
            x_k = self.dropout(r_u + self.linears[k](r_u))  # residual
            x_k = self.lns[k](x_k)
            drift_reprs.append(x_k)
        return drift_reprs