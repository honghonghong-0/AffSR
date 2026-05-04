"""
models/modules/cross_attention.py
==================================
Bidirectional cross-attention module

Role:
  Enables precise user-item matching by having user representation (r̃ᵤ)
  and emotion-conditioned item representation (e_final) attend to each other.

Formulas:
  # User → Item direction
  Qᵤ = W_Q · r̃ᵤ,  Kᵥ = W_K · e_final,  Vᵥ = W_V · e_final
  r̃ᵤ* = r̃ᵤ + softmax(Qᵤ Kᵥᵀ / √d) · Vᵥ

  # Item → User direction
  Qᵥ = W_Q' · e_final,  Kᵤ = W_K' · r̃ᵤ,  Vᵤ = W_V' · r̃ᵤ
  e_final* = e_final + softmax(Qᵥ Kᵤᵀ / √d) · Vᵤ

Inputs:
  r_u_tilde  : (B, d)   α-blended user representation
  e_final    : (B, d)   MoE output item representation

Outputs:
  r_u_star   : (B, d)   refined user representation
  e_final_star: (B, d)  refined item representation
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention.

    Operates at the vector level (not sequence level) since
    attention is applied to a single (user, item) vector pair.

    Args:
        d_model : embedding dimension
        dropout : dropout rate
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(d_model)

        # User → Item direction
        self.W_Q_u = nn.Linear(d_model, d_model, bias=False)
        self.W_K_v = nn.Linear(d_model, d_model, bias=False)
        self.W_V_v = nn.Linear(d_model, d_model, bias=False)

        # Item → User direction
        self.W_Q_v = nn.Linear(d_model, d_model, bias=False)
        self.W_K_u = nn.Linear(d_model, d_model, bias=False)
        self.W_V_u = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.ln_u = nn.LayerNorm(d_model)
        self.ln_v = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        r_u: torch.Tensor,    # (B, d)
        e_final: torch.Tensor,  # (B, d)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            r_u_star    : (B, d)
            e_final_star: (B, d)
        """
        # ── User → Item direction ─────────────────────────────────────
        # attention weight is a scalar since input is a vector pair
        Q_u = self.W_Q_u(r_u)        # (B, d)
        K_v = self.W_K_v(e_final)    # (B, d)
        V_v = self.W_V_v(e_final)    # (B, d)

        # dot product → scalar attention weight → (B, 1)
        attn_u = (Q_u * K_v).sum(dim=-1, keepdim=True) / self.scale  # (B, 1)
        attn_u = torch.sigmoid(attn_u)  # use sigmoid for single pair

        r_u_star = self.ln_u(r_u + self.dropout(attn_u * V_v))

        # ── Item → User direction ─────────────────────────────────────
        Q_v = self.W_Q_v(e_final)    # (B, d)
        K_u = self.W_K_u(r_u)        # (B, d)
        V_u = self.W_V_u(r_u)        # (B, d)

        attn_v = (Q_v * K_u).sum(dim=-1, keepdim=True) / self.scale  # (B, 1)
        attn_v = torch.sigmoid(attn_v)

        e_final_star = self.ln_v(e_final + self.dropout(attn_v * V_u))

        return r_u_star, e_final_star