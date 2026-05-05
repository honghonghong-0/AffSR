"""
models/backbone/sasrec.py
=========================
SASRec backbone (ported from validated IDURL/RecBole implementation).

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." ICDM 2018.
    RecBole: recbole/model/sequential_recommender/sasrec.py
    RecBole: recbole/model/layers.py (TransformerEncoder family)

AffSR interface (original signature preserved):
    sasrec = SASRec(num_items, d_model, n_heads, n_layers, max_seq_len, dropout)
    r_u = sasrec(item_seq, seq_mask)    # (B, d)
    E   = sasrec.item_emb               # (num_items+1, d)

Key stability measures (prevents NaN/convergence issues):
  - Attention mask added to softmax as (1-mask)*(-10000) (avoids -inf)
  - Post-LN applied on all residual paths
  - Embedding/Linear initialized with normal_(std=0.02) (BERT-style)
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention (ported from RecBole)
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
            )
        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor: torch.Tensor,     # (B, L, d)
        attention_mask: torch.Tensor,   # (B, 1, L, L) — mask added as values
    ) -> torch.Tensor:
        q = self.transpose_for_scores(self.query(input_tensor))
        k = self.transpose_for_scores(self.key(input_tensor))
        v = self.transpose_for_scores(self.value(input_tensor))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        scores = scores + attention_mask       # masked positions become -10000

        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)

        hidden = self.dense(context)
        hidden = self.out_dropout(hidden)
        hidden = self.LayerNorm(hidden + input_tensor)
        return hidden


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward (ported from RecBole)
# ─────────────────────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.act_fn = self._get_act(hidden_act)
        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _get_act(self, act: str):
        table = {
            "gelu": self._gelu,
            "relu": F.relu,
            "swish": lambda x: x * torch.sigmoid(x),
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return table[act]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        h = self.dense_1(input_tensor)
        h = self.act_fn(h)
        h = self.dense_2(h)
        h = self.dropout(h)
        h = self.LayerNorm(h + input_tensor)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Layer / Encoder
# ─────────────────────────────────────────────────────────────────────────────
class TransformerLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.multi_head_attention(hidden_states, attention_mask)
        ff_out = self.feed_forward(attn_out)
        return ff_out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int = 2,
        n_heads: int = 2,
        hidden_size: int = 64,
        inner_size: int = 256,
        hidden_dropout_prob: float = 0.2,
        attn_dropout_prob: float = 0.2,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        template = TransformerLayer(
            n_heads, hidden_size, inner_size,
            hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(template) for _ in range(n_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# SASRec — AffSR interface wrapper
# ─────────────────────────────────────────────────────────────────────────────
class SASRec(nn.Module):
    """
    SASRec backbone (ported from IDURL/RecBole).

    Args:
        num_items   : number of items (0 is the padding index)
        d_model     : embedding dimension
        n_heads     : number of attention heads
        n_layers    : number of Transformer layers
        max_seq_len : maximum sequence length
        dropout     : dropout rate (shared for hidden and attention)
        inner_size  : FFN intermediate dimension (defaults to d_model * 4)
        initializer_range: BERT-style initialization std
        layer_norm_eps   : LayerNorm epsilon

    Forward:
        item_seq : (B, L) integer item IDs (padding=0, left-padded)
        seq_mask : (B, L) bool, True=valid token
        returns  : (B, d) — hidden state at the last valid timestep
    """

    def __init__(
        self,
        num_items: int,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
        inner_size: int | None = None,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.initializer_range = initializer_range

        inner_size = inner_size if inner_size is not None else d_model * 4

        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.trm_encoder = TransformerEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=d_model,
            inner_size=inner_size,
            hidden_dropout_prob=dropout,
            attn_dropout_prob=dropout,
            hidden_act="gelu",
            layer_norm_eps=layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @property
    def item_emb(self) -> nn.Embedding:
        return self.item_embedding

    def _extended_attention_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        """RecBole-style attention mask:
           - sets padding positions to -10000 (≈0 after softmax)
           - simultaneously applies causal (left-to-right) triangular mask
        """
        # padding mask: (B, L) → (B, 1, 1, L)
        attention_mask = (item_seq > 0).long()
        extended = attention_mask.unsqueeze(1).unsqueeze(2)

        L = item_seq.size(-1)
        subsequent = torch.triu(
            torch.ones((1, L, L), device=item_seq.device), diagonal=1
        )
        subsequent = (subsequent == 0).unsqueeze(1).long()  # (1, 1, L, L)

        extended = extended * subsequent
        extended = extended.to(dtype=next(self.parameters()).dtype)
        extended = (1.0 - extended) * -10000.0
        return extended

    def forward(
        self,
        item_seq: torch.Tensor,  # (B, L)
        seq_mask: torch.Tensor,  # (B, L) bool, True=valid
    ) -> torch.Tensor:
        B, L = item_seq.shape

        position_ids = torch.arange(L, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        pos_emb = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        x = item_emb + pos_emb
        x = self.LayerNorm(x)
        x = self.dropout(x)

        attention_mask = self._extended_attention_mask(item_seq)
        x = self.trm_encoder(x, attention_mask)

        # Extract hidden state at the last valid timestep.
        # Directly compute the last True index in seq_mask regardless of padding side.
        # last_idx = (L-1) - argmax of reversed mask
        reversed_mask = torch.flip(seq_mask.long(), dims=[1])
        # if all False, defaults to 0
        last_idx = (L - 1) - reversed_mask.argmax(dim=1)
        last_idx = last_idx.clamp(min=0, max=L - 1)

        r_u = x[torch.arange(B, device=item_seq.device), last_idx]  # (B, d)
        return r_u

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_ids)
