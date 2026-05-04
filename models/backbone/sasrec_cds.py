"""
models/backbone/sasrec_cds.py
================
SASRec backbone implementation (Kang & McAuley, 2018)
Self-Attentive Sequential Recommendation

AffSR interface:
  encoder = SASRec(num_items, d, num_layers, num_heads, max_len, dropout)
  r_u = encoder(item_seq)   # (B, d)  ← last-timestep representation
  E   = encoder.item_emb    # (num_items+1, d) ← item embedding table
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.d_k)

    def forward(self, x, attn_mask=None):
        """
        x: (B, L, d)
        attn_mask: (B, 1, L, L) or (1, 1, L, L)  — causal mask
        """
        B, L, _ = x.shape

        Q = self.W_Q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        # Q,K,V: (B, H, L, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn   = self.dropout(F.softmax(scores, dim=-1))             # (B, H, L, L)
        out    = torch.matmul(attn, V)                               # (B, H, L, d_k)
        out    = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_O(out)                                         # (B, L, d)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# SASRec Layer (Transformer Block)
# ─────────────────────────────────────────────────────────────────────────────
class SASRecLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.attn    = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn     = FeedForward(d_model, dropout=dropout)
        self.ln1     = nn.LayerNorm(d_model, eps=1e-8)
        self.ln2     = nn.LayerNorm(d_model, eps=1e-8)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Self-Attention + Residual
        x = self.ln1(x + self.dropout(self.attn(x, attn_mask)))
        # FFN + Residual
        x = self.ln2(x + self.dropout(self.ffn(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SASRec
# ─────────────────────────────────────────────────────────────────────────────
class SASRec(nn.Module):
    """
    SASRec backbone

    Args:
        num_items  : number of items (includes padding index 0)
        d_model    : embedding dimension
        num_layers : number of Transformer layers
        num_heads  : number of Multi-Head Attention heads
        max_len    : maximum sequence length
        dropout    : dropout rate

    Forward:
        item_seq: (B, L)  — item index sequence (padding=0)
        returns: r_u (B, d)  — last valid timestep representation
    """

    def __init__(self, num_items, d_model=64, num_layers=2,
                 num_heads=2, max_len=50, dropout=0.5):
        super().__init__()
        self.d_model  = d_model
        self.max_len  = max_len

        # item embedding (index 0 = padding)
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        # position embedding
        self.pos_emb  = nn.Embedding(max_len + 1, d_model)

        self.layers   = nn.ModuleList([
            SASRecLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln       = nn.LayerNorm(d_model, eps=1e-8)
        self.dropout  = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _causal_mask(self, L, device):
        """Upper-triangular mask to prevent attending to future positions."""
        mask = torch.ones(1, 1, L, L, device=device)
        mask = torch.tril(mask)
        return mask  # (1, 1, L, L)

    def forward(self, item_seq):
        """
        item_seq: (B, L)
        returns: (B, d) — last valid timestep representation
        """
        B, L = item_seq.shape
        device = item_seq.device

        # position indices (1-indexed)
        pos = torch.arange(1, L + 1, device=device).unsqueeze(0).expand(B, -1)

        # zero out position embedding for padding positions
        pos = pos * (item_seq != 0).long()

        x = self.dropout(self.item_emb(item_seq) + self.pos_emb(pos))  # (B, L, d)

        attn_mask = self._causal_mask(L, device)

        for layer in self.layers:
            x = layer(x, attn_mask)

        x = self.ln(x)  # (B, L, d)

        # extract last valid timestep
        # find last non-padding position in item_seq
        seq_len = (item_seq != 0).sum(dim=1)           # (B,)
        seq_len = seq_len.clamp(min=1) - 1             # 0-indexed
        idx     = seq_len.view(B, 1, 1).expand(B, 1, self.d_model)
        r_u     = x.gather(1, idx).squeeze(1)          # (B, d)

        return r_u


if __name__ == "__main__":
    # simple test
    model = SASRec(num_items=1000, d_model=64, num_layers=2,
                   num_heads=2, max_len=50, dropout=0.5)
    item_seq = torch.randint(0, 1001, (4, 50))
    item_seq[:, -5:] = 0  # last 5 positions are padding
    r_u = model(item_seq)
    print(f"r_u shape: {r_u.shape}")  # (4, 64)
    print(f"item_emb shape: {model.item_emb.weight.shape}")  # (1001, 64)
    print("SASRec test complete!")
