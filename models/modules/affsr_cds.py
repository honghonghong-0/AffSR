"""
models/modules/affsr_cds.py
===============
AffSR: Affective Sequential Recommendation Model

Components:
  - SASRec backbone (r_u encoding)
  - DRG (Drift Representation Generation): K=4 independent FFNs
  - IDRD (Interest Drift-guided Representation Disentanglement)
  - Affective MoE (VA quadrant-based gating)
  - Two-stage prediction (Stage1: r_u·e_id, Stage2: Cross-Attention re-ranking)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.sasrec_cds import SASRec


# ─────────────────────────────────────────────────────────────────────────────
# VA Quadrant Centroids (fixed, not learned)
# ─────────────────────────────────────────────────────────────────────────────
VA_CENTROIDS = torch.tensor([
    [ 0.83,  0.61],   # Q1: joy, excitement
    [-0.53,  0.57],   # Q2: anger, fear
    [ 0.75, -0.48],   # Q3: gratitude, calmness
    [-0.69, -0.43],   # Q4: sadness, grief
], dtype=torch.float32)   # (4, 2)


# ─────────────────────────────────────────────────────────────────────────────
# DRG: Drift Representation Generator (K independent FFNs)
# ─────────────────────────────────────────────────────────────────────────────
class DRG(nn.Module):
    """
    Generates drift representations using K independent FFNs.
    DRG_k(x) = LN(Dropout(x + W_k·x + b_k))
    """
    def __init__(self, d_model, K=4, dropout=0.5):
        super().__init__()
        self.K = K
        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(K)
        ])
        self.lns     = nn.ModuleList([
            nn.LayerNorm(d_model, eps=1e-8) for _ in range(K)
        ])
        self.dropout = nn.Dropout(dropout)

        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, r_u):
        """
        r_u: (B, d)
        returns: list of K tensors, each (B, d)
        """
        drift_reprs = []
        for k in range(self.K):
            x_k = self.dropout(r_u + self.linears[k](r_u))  # residual
            x_k = self.lns[k](x_k)
            drift_reprs.append(x_k)
        return drift_reprs  # [(B, d)] * K


# ─────────────────────────────────────────────────────────────────────────────
# Affective MoE: VA quadrant-based gating
# ─────────────────────────────────────────────────────────────────────────────
class AffectiveMoE(nn.Module):
    """
    VA quadrant-based gating + affective item representation generation.

    β_k = softmax(-‖a_n - c_k‖₂ / τ)
    e_final = Σ_k β_k · (E_k(e_aff(v)) + W_{c_k}·c_k)
    """
    def __init__(self, d_model, K=4, dropout=0.5):
        super().__init__()
        self.K = K
        self.d_model = d_model

        # temperature (learnable)
        self.tau = nn.Parameter(torch.ones(1))

        # E_k: quadrant-specific linear transform R^{d×2} (e_aff ∈ R^2 → R^d)
        self.E = nn.ModuleList([
            nn.Linear(2, d_model, bias=True) for _ in range(K)
        ])
        # W_{c_k}: project centroid to d-dim R^{d×2}
        self.W_c = nn.ModuleList([
            nn.Linear(2, d_model, bias=False) for _ in range(K)
        ])

        for k in range(K):
            nn.init.xavier_uniform_(self.E[k].weight)
            nn.init.zeros_(self.E[k].bias)
            nn.init.xavier_uniform_(self.W_c[k].weight)

        self.dropout = nn.Dropout(dropout)

    def compute_beta(self, a_n, centroids):
        """
        a_n: (B, 2)  current emotion state
        centroids: (K, 2)
        returns: β (B, K)
        """
        # (B, 1, 2) - (1, K, 2) → (B, K, 2)
        diff = a_n.unsqueeze(1) - centroids.unsqueeze(0)
        dist = diff.norm(dim=-1)               # (B, K)
        beta = F.softmax(-dist / self.tau.abs().clamp(min=1e-6), dim=-1)  # (B, K)
        return beta

    def forward(self, e_aff, a_n, centroids):
        """
        e_aff: (B, 2) or (B, N, 2)  — item VA representation
        a_n:   (B, 2)                — current emotion state
        centroids: (K, 2)            — VA quadrant centroids

        returns:
          e_final: (B, d) or (B, N, d)
          beta:    (B, K)
        """
        beta = self.compute_beta(a_n, centroids)  # (B, K)

        batched = e_aff.dim() == 3  # check if shape is (B, N, 2)
        if not batched:
            e_aff = e_aff.unsqueeze(1)  # (B, 1, 2)

        B, N, _ = e_aff.shape

        e_final = torch.zeros(B, N, self.d_model, device=e_aff.device)
        for k in range(self.K):
            ek   = self.E[k](e_aff)                    # (B, N, d)
            wck  = self.W_c[k](centroids[k])           # (d,)
            wck  = wck.unsqueeze(0).unsqueeze(0)       # (1, 1, d)
            bk   = beta[:, k].view(B, 1, 1)            # (B, 1, 1)
            e_final = e_final + bk * (ek + wck)        # (B, N, d)

        e_final = self.dropout(e_final)

        if not batched:
            e_final = e_final.squeeze(1)  # (B, d)

        return e_final, beta


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Attention (Stage 2 re-ranking)
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttention(nn.Module):
    """
    Bidirectional cross-attention.
    Query: r̄_u → attend to e_final
    Query: e_final → attend to r̄_u
    """
    def __init__(self, d_model):
        super().__init__()
        self.scale = math.sqrt(d_model)

        # User → Item attention
        self.W_Q_u = nn.Linear(d_model, d_model, bias=False)
        self.W_K_v = nn.Linear(d_model, d_model, bias=False)
        self.W_V_v = nn.Linear(d_model, d_model, bias=False)

        # Item → User attention
        self.W_Q_v = nn.Linear(d_model, d_model, bias=False)
        self.W_K_u = nn.Linear(d_model, d_model, bias=False)
        self.W_V_u = nn.Linear(d_model, d_model, bias=False)

        # LayerNorm for residual connections (prevents NaN)
        self.ln_u = nn.LayerNorm(d_model, eps=1e-8)
        self.ln_v = nn.LayerNorm(d_model, eps=1e-8)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, r_u_bar, e_final):
        """
        r_u_bar: (B, d)
        e_final: (B, N, d)  — top-100 candidates

        returns:
          r_u_star:    (B, d)
          e_final_star:(B, N, d)
        """
        B, N, d = e_final.shape

        # User → Item
        Q_u = self.W_Q_u(r_u_bar).unsqueeze(1)   # (B, 1, d)
        K_v = self.W_K_v(e_final)                  # (B, N, d)
        V_v = self.W_V_v(e_final)                  # (B, N, d)
        attn_u = F.softmax(
            torch.bmm(Q_u, K_v.transpose(1, 2)) / self.scale, dim=-1
        )  # (B, 1, N)
        r_u_star = self.ln_u(r_u_bar + torch.bmm(attn_u, V_v).squeeze(1))  # (B, d)

        # Item → User
        Q_v = self.W_Q_v(e_final)                  # (B, N, d)
        K_u = self.W_K_u(r_u_bar).unsqueeze(1)    # (B, 1, d)
        V_u = self.W_V_u(r_u_bar).unsqueeze(1)    # (B, 1, d)
        attn_v = F.softmax(
            torch.bmm(Q_v, K_u.transpose(1, 2)) / self.scale, dim=-1
        )  # (B, N, 1)
        e_final_star = self.ln_v(e_final + torch.bmm(attn_v, V_u))  # (B, N, d)

        return r_u_star, e_final_star


# ─────────────────────────────────────────────────────────────────────────────
# AffSR
# ─────────────────────────────────────────────────────────────────────────────
class AffSR(nn.Module):
    """
    AffSR: Affective Sequential Recommendation

    Args:
        num_items  : number of items
        d_model    : embedding dimension
        num_layers : SASRec layer count
        num_heads  : SASRec attention head count
        max_len    : maximum sequence length
        dropout    : dropout rate
        K          : number of MoE experts (= number of VA quadrants = 4)
    """

    def __init__(self, num_items, d_model=64, num_layers=2,
                 num_heads=2, max_len=50, dropout=0.5, K=4):
        super().__init__()
        self.d_model = d_model
        self.K = K

        # 1. SASRec backbone
        self.sasrec = SASRec(
            num_items=num_items,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout,
        )

        # 2. DRG
        self.drg = DRG(d_model, K=K, dropout=dropout)

        # 3. Affective MoE
        self.moe = AffectiveMoE(d_model, K=K, dropout=dropout)

        # 4. Cross-Attention (Stage 2)
        self.cross_attn = CrossAttention(d_model)

        # Register VA centroids as buffer (not learned)
        self.register_buffer("centroids", VA_CENTROIDS)  # (K, 2)

    # ── Convenience property ───────────────────────────────────────────────
    @property
    def item_emb(self):
        return self.sasrec.item_emb

    # ── Forward ───────────────────────────────────────────────────────────
    def encode_user(self, item_seq):
        """
        item_seq: (B, L)
        returns: r_u (B, d)
        """
        return self.sasrec(item_seq)

    def compute_beta(self, a_n):
        """
        a_n: (B, 2)
        returns: β (B, K)
        """
        return self.moe.compute_beta(a_n, self.centroids)

    def forward_train(self, item_seq, a_n, e_aff_pos, e_aff_neg):
        """
        Forward pass for training.

        Args:
            item_seq   : (B, L)       item sequence
            a_n        : (B, 2)       current emotion state (last review VA)
            e_aff_pos  : (B, 2)       target item VA
            e_aff_neg  : (B, neg, 2)  negative item VAs

        returns:
            score_pos  : (B,)         positive item scores
            score_neg  : (B, neg)     negative item scores
            beta       : (B, K)       gating weights
            drift_reprs: list of K (B, d)  for collapse monitoring
        """
        B = item_seq.size(0)

        # 1. SASRec encoding
        r_u = self.encode_user(item_seq)          # (B, d)

        # 2. DRG
        drift_reprs = self.drg(r_u)               # [(B, d)] * K

        # 3. compute β
        beta = self.compute_beta(a_n)             # (B, K)

        # 4. final user representation
        r_u_bar = sum(beta[:, k].unsqueeze(-1) * drift_reprs[k]
                      for k in range(self.K))     # (B, d)

        # 5. item representation via affective MoE
        # process positive and negative together
        neg = e_aff_neg.size(1)
        # (B, 1+neg, 2)
        e_aff_all = torch.cat([e_aff_pos.unsqueeze(1), e_aff_neg], dim=1)
        e_all, _  = self.moe(e_aff_all, a_n, self.centroids)  # (B, 1+neg, d)

        # 6. Stage 2: Cross-Attention re-ranking
        r_u_star, e_all_star = self.cross_attn(r_u_bar, e_all)  # (B, d), (B, 1+neg, d)

        # scores
        scores = torch.bmm(
            e_all_star,
            r_u_star.unsqueeze(-1)
        ).squeeze(-1)                              # (B, 1+neg)
        scores = torch.clamp(scores, -50, 50)

        score_pos = scores[:, 0]                  # (B,)
        score_neg = scores[:, 1:]                 # (B, neg)

        return score_pos, score_neg, beta, drift_reprs

    def forward_stage1(self, item_seq):
        """
        Inference Stage 1: rough ranking via r_u · e_id(v)

        returns:
            r_u: (B, d)
            all_scores: (B, num_items)  — scores over all items
        """
        r_u = self.encode_user(item_seq)          # (B, d)
        E = self.item_emb.weight[1:]               # (num_items, d)
        all_scores = torch.matmul(r_u, E.T)        # (B, num_items)
        return r_u, all_scores

    def forward_stage2(self, r_u, a_n, e_aff_cand):
        """
        Inference Stage 2: cross-attention re-ranking over top-100 candidates.

        Args:
            r_u        : (B, d)         Stage 1 user representation
            a_n        : (B, 2)         current emotion state
            e_aff_cand : (B, 100, 2)    top-100 candidate item VAs

        returns:
            scores: (B, 100)
        """
        drift_reprs = self.drg(r_u)               # [(B, d)] * K
        beta = self.compute_beta(a_n)             # (B, K)
        r_u_bar = sum(beta[:, k].unsqueeze(-1) * drift_reprs[k]
                      for k in range(self.K))     # (B, d)

        e_final, _ = self.moe(e_aff_cand, a_n, self.centroids)  # (B, 100, d)
        r_u_star, e_final_star = self.cross_attn(r_u_bar, e_final)

        scores = torch.bmm(
            e_final_star,
            r_u_star.unsqueeze(-1)
        ).squeeze(-1)                              # (B, 100)
        scores = torch.clamp(scores, -50, 50)

        return scores

    def compute_disen_loss(self, drift_reprs, beta, pos_item_ids):
        """
        IDRD Disentanglement Loss

        L_disen = -Σ log [ exp(r̃_u^{k*} · e_id(v+)) / Σ_j exp(r̃_u^j · e_id(v+)) ]

        Args:
            drift_reprs  : list of K (B, d)
            beta         : (B, K)
            pos_item_ids : (B,)  positive item indices

        returns: scalar loss
        """
        # dominant quadrant k*
        k_star = beta.argmax(dim=-1)   # (B,)

        # positive item embedding
        e_pos = self.item_emb(pos_item_ids)  # (B, d)

        # dot product of each drift repr with e_pos (normalized for scale stability)
        drift_stack = torch.stack(drift_reprs, dim=1)   # (B, K, d)
        drift_stack_n = F.normalize(drift_stack, dim=-1)
        e_pos_n = F.normalize(e_pos, dim=-1).unsqueeze(-1)
        logits = torch.bmm(drift_stack_n, e_pos_n).squeeze(-1)  # (B, K)

        # cross-entropy to push k* index logit higher
        loss = F.cross_entropy(logits, k_star)
        return loss

    @torch.no_grad()
    def compute_collapse_similarity(self, drift_reprs):
        """
        Measures pairwise cosine similarity between drift representations (for collapse monitoring).

        returns: (K, K) mean similarity matrix
        """
        K = len(drift_reprs)
        # mean representations
        mean_reprs = [dr.mean(dim=0) for dr in drift_reprs]  # [(d,)] * K
        sim_matrix = torch.zeros(K, K)
        for i in range(K):
            for j in range(K):
                sim_matrix[i, j] = F.cosine_similarity(
                    mean_reprs[i].unsqueeze(0),
                    mean_reprs[j].unsqueeze(0)
                ).item()
        return sim_matrix


if __name__ == "__main__":
    B, L, d, num_items = 4, 50, 64, 1000
    model = AffSR(num_items=num_items, d_model=d)

    item_seq   = torch.randint(1, num_items + 1, (B, L))
    a_n        = torch.randn(B, 2)
    e_aff_pos  = torch.randn(B, 2)
    e_aff_neg  = torch.randn(B, 5, 2)
    pos_ids    = torch.randint(1, num_items + 1, (B,))

    score_pos, score_neg, beta, drift_reprs = model.forward_train(
        item_seq, a_n, e_aff_pos, e_aff_neg
    )
    print(f"score_pos: {score_pos.shape}")   # (4,)
    print(f"score_neg: {score_neg.shape}")   # (4, 5)
    print(f"beta:      {beta.shape}")        # (4, 4)

    disen_loss = model.compute_disen_loss(drift_reprs, beta, pos_ids)
    print(f"disen_loss: {disen_loss.item():.4f}")

    sim = model.compute_collapse_similarity(drift_reprs)
    print(f"collapse sim:\n{sim}")

    print("AffSR test complete!")
