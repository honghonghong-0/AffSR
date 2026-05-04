"""
models/modules/affdrift.py
==========================
AffDrift module — v10

Role:
  1. Compute EMA long-term emotion: dist28_seq → h_long (28-dim) → va_long (2-dim)
  2. Compute AD: ‖aₙ - va_long‖₂
  3. Compute α: σ(w·AD + b) — larger AD increases weight on short-term emotion
  4. β_final: (1-α)·βₖ(va_long) + α·βₖ(aₙ)

Formulas:
  h_long  = Σ wₜ · e_t       (wₜ ∝ exp(-λ·(L-1-t)), padding masked)
  va_long = h_long @ VA_MATRIX
  AD      = ‖aₙ - va_long‖₂
  α       = σ(w·AD + b)
  βₖ(x)  = softmax(-‖x - cₖ‖₂ / τ)
  β_final = (1-α)·βₖ(va_long) + α·βₖ(aₙ)

Inputs:
  a_n        : (B, 2)     current emotion (last timestep, from sequences.pkl)
  dist28_seq : (B, L, 28) GoEmotions 28-dim distribution per review
  seq_mask   : (B, L)     padding mask (True = valid)

Outputs:
  adm  : (B,)    AD scalar (for analysis)
  beta : (B, K)  β_final gating weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# VA quadrant centroids based on NRC VAD (fixed values)
QUADRANT_CENTROIDS = torch.tensor([
    [ 0.83,  0.61],   # c1: High V, High A (joy, excitement)
    [-0.53,  0.57],   # c2: Low V, High A  (anger, fear)
    [ 0.75, -0.48],   # c3: High V, Low A  (gratitude, calmness)
    [-0.69, -0.43],   # c4: Low V, Low A   (sadness, grief)
], dtype=torch.float32)  # (K=4, 2)

# Fixed GoEmotions 28-dim → VA transform matrix (based on NRC VAD Lexicon)
# neutral has no VA signal and is mapped to (0, 0)
_GOEMOTIONS_VA = [
    ( 0.82,  0.41),  # admiration
    ( 0.76,  0.60),  # amusement
    (-0.43,  0.67),  # anger
    (-0.55,  0.35),  # annoyance
    ( 0.69,  0.10),  # approval
    ( 0.73,  0.05),  # caring
    (-0.15,  0.30),  # confusion
    ( 0.22,  0.55),  # curiosity
    ( 0.55,  0.59),  # desire
    (-0.63, -0.30),  # disappointment
    (-0.62,  0.20),  # disapproval
    (-0.60,  0.35),  # disgust
    (-0.45,  0.15),  # embarrassment
    ( 0.80,  0.78),  # excitement
    (-0.55,  0.70),  # fear
    ( 0.88, -0.45),  # gratitude
    (-0.75, -0.55),  # grief
    ( 0.90,  0.65),  # joy
    ( 0.88,  0.40),  # love
    (-0.35,  0.55),  # nervousness
    ( 0.72,  0.30),  # optimism
    ( 0.77,  0.45),  # pride
    ( 0.10, -0.10),  # realization
    ( 0.68, -0.50),  # relief
    (-0.58, -0.35),  # remorse
    (-0.70, -0.45),  # sadness
    ( 0.15,  0.60),  # surprise
    ( 0.00,  0.00),  # neutral → mapped to 0
]
VA_MATRIX = torch.tensor(_GOEMOTIONS_VA, dtype=torch.float32)  # (28, 2)
NEUTRAL_IDX = 27  # neutral is the last index


class AffDrift(nn.Module):
    """
    AffDrift v10 module.

    Args:
        K          : number of quadrants (default 4)
        tau        : βₖ softmax temperature initial value (learnable parameter)
        init_lambda: EMA decay initial value (in softplus space, default 0.0 → λ≈0.69)
    """

    def __init__(
        self,
        K: int = 4,
        tau: float = 1.0,
        init_lambda: float = 0.0,
        no_long: bool = False,
        no_short: bool = False,
        no_ad: bool = False,
    ):
        super().__init__()
        self.K = K
        self.no_long = no_long    # ablation: remove va_long → β = β(a_n)
        self.no_short = no_short  # ablation: remove a_n → β = β(va_long)
        self.no_ad = no_ad        # ablation: remove AD → α fixed at 0.5

        # τ: βₖ temperature (stored in log space to ensure positivity)
        self.log_tau = nn.Parameter(torch.tensor(tau).log())

        # λ: EMA decay (stored in softplus space to ensure positivity)
        self.lambda_raw = nn.Parameter(torch.tensor(init_lambda))

        # scalar parameters for computing α
        self.alpha_w = nn.Parameter(torch.tensor(1.0))
        self.alpha_b = nn.Parameter(torch.tensor(0.0))

        # fixed buffers: slice centroids to match K (pad with center point if K=5)
        if K <= 4:
            centroids = QUADRANT_CENTROIDS[:K]
        else:
            center = torch.zeros(K - 4, 2, dtype=torch.float32)
            centroids = torch.cat([QUADRANT_CENTROIDS, center], dim=0)
        self.register_buffer("centroids", centroids)             # (K, 2)
        self.register_buffer("va_matrix", VA_MATRIX)            # (28, 2)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=1e-6)

    @property
    def lambda_(self) -> torch.Tensor:
        return F.softplus(self.lambda_raw)

    def _beta(self, x: torch.Tensor) -> torch.Tensor:
        """βₖ(x): centroid-distance-based softmax. x: (B, 2)"""
        dists = torch.norm(
            x.unsqueeze(1) - self.centroids.unsqueeze(0), dim=-1
        )  # (B, K)
        return F.softmax(-dists / self.tau, dim=-1)  # (B, K)

    def forward(
        self,
        a_n: torch.Tensor,          # (B, 2)
        dist28_seq: torch.Tensor,   # (B, L, 28)
        seq_mask: torch.Tensor,     # (B, L)  True=valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            adm  : (B,)    AD scalar (‖aₙ - va_long‖₂, for analysis)
            beta : (B, K)  β_final gating weights
        """
        B, L, _ = dist28_seq.shape

        # ── EMA weight computation ─────────────────────────────────────
        t = torch.arange(L, device=dist28_seq.device, dtype=torch.float32)
        w = torch.exp(-self.lambda_ * (L - 1 - t))   # (L,)  higher weight for recent steps
        w = w * seq_mask.float()                       # (B, L) mask padding
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)   # (B, L) normalize

        # ── Long-term emotion computation ──────────────────────────────
        # zero out neutral then renormalize
        d28 = dist28_seq.clone()
        d28[:, :, NEUTRAL_IDX] = 0.0
        d28_sum = d28.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        d28 = d28 / d28_sum                            # (B, L, 28) renormalized

        h_long = (w.unsqueeze(-1) * d28).sum(dim=1)   # (B, 28)
        va_long = h_long @ self.va_matrix              # (B, 2)

        # ── AD and α computation ───────────────────────────────────────
        ad = torch.norm(a_n - va_long, dim=-1)         # (B,)
        alpha = torch.sigmoid(self.alpha_w * ad + self.alpha_b).unsqueeze(-1)  # (B, 1)

        # ── β_final: blend long- and short-term ───────────────────────
        beta_long  = self._beta(va_long)               # (B, K)
        beta_short = self._beta(a_n)                   # (B, K)

        if self.no_long:
            beta = beta_short
        elif self.no_short:
            beta = beta_long
        elif self.no_ad:
            # remove AD: replace α with a learnable fixed scalar (removes dynamic adaptation only)
            alpha = torch.sigmoid(self.alpha_b).expand(B, 1)
            beta = (1 - alpha) * beta_long + alpha * beta_short
        else:
            beta = (1 - alpha) * beta_long + alpha * beta_short  # (B, K)

        return ad, beta