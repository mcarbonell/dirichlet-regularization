"""
Topographic Transformer Architecture Module
============================================
Implements the causal autoregressive Transformer with native 2D cortical lattice
topology, SwiGLU FFNs, and integrated Dirichlet loss accumulation.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology import dirichlet_energy_2d, get_grid_dimensions
from .spectral import dct2d, idct2d, BlockDCTTiler


@dataclass
class TopographicConfig:
    vocab_size: int = 4096
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    ffn_dim: int = 512
    max_seq_len: int = 256
    dropout: float = 0.0
    topo_lambda: float = 0.01
    use_block_dct: bool = True
    block_size: int = 64


class TopographicLinear(nn.Linear):
    """
    nn.Linear subclass that tracks its 2D cortical grid mapping
    and computes its local Dirichlet harmonic energy.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.grid_shape = (out_features, in_features)

    def dirichlet_energy(self) -> torch.Tensor:
        return dirichlet_energy_2d(self.weight, self.grid_shape)


class TopographicAttention(nn.Module):
    def __init__(self, cfg: TopographicConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        assert self.d_head * cfg.n_heads == cfg.d_model, "d_model must be divisible by n_heads"

        self.q_proj = TopographicLinear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = TopographicLinear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = TopographicLinear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = TopographicLinear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, d = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask[:, :, :s, :s] == 0, float("-inf"))
        else:
            causal_mask = torch.triu(torch.full((s, s), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, d)
        return self.out_proj(out)


class TopographicMLP(nn.Module):
    """
    SwiGLU feed-forward network with topographic linear projections.
    """
    def __init__(self, cfg: TopographicConfig):
        super().__init__()
        self.gate_proj = TopographicLinear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up_proj = TopographicLinear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down_proj = TopographicLinear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TopographicBlock(nn.Module):
    def __init__(self, cfg: TopographicConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = TopographicAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = TopographicMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TopographicTransformer(nn.Module):
    def __init__(self, cfg: TopographicConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([TopographicBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # NOTE: lm_head intentionally uses standard nn.Linear, not TopographicLinear.
        # Applying Dirichlet smoothness to the vocabulary projection would constrain
        # the logit geometry, forcing semantically unrelated tokens with adjacent indices
        # to share similar output weights and degrading generation quality.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        nn.init.normal_(self.pos_emb, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, s = idx.shape
        x = self.token_emb(idx) + self.pos_emb[:, :s, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.reshape(-1))
        return logits, loss

    def topographic_loss(self) -> torch.Tensor:
        """
        Computes the average 2D Dirichlet harmonic energy across all
        TopographicLinear projections in the network.
        """
        total_energy = torch.tensor(0.0, device=self.token_emb.weight.device)
        count = 0
        for m in self.modules():
            if isinstance(m, TopographicLinear):
                total_energy = total_energy + m.dirichlet_energy()
                count += 1
        if count == 0:
            return total_energy
        return (total_energy / count) * self.cfg.topo_lambda

    def get_linear_projections(self) -> List[TopographicLinear]:
        return [m for m in self.modules() if isinstance(m, TopographicLinear)]

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 64, temperature: float = 0.8) -> torch.Tensor:
        out = prompt_tokens.clone()
        for _ in range(max_new_tokens):
            idx_cond = out[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
        return out
