"""
Baseline Transformer for modular arithmetic.

Identical task setup to the Mamba model — lets us compare:
  - Does grokking occur in Mamba as it does in Transformers?
  - Do both learn Fourier (clock-algorithm) features?
  - How do SAE features differ between architectures?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj= nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)                                       # each (B, L, H, d_head)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]             # (B, H, L, d_head)

        scale = self.d_head ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale                     # (B, H, L, L)
        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))
        attn  = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn.detach()                       # return weights for analysis


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        attn_out, attn_w = self.attn(self.norm1(x), mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_w


class TransformerForModularArithmetic(nn.Module):
    """
    Causal Transformer for modular arithmetic.

    Input : [a, '+', b, '=']
    Output: logits over {0 .. p-1}  at the last position
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len,    cfg.d_model)

        # Causal mask: upper-triangular
        mask = torch.triu(
            torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", mask)

        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads)
            for _ in range(cfg.n_layers)
        ])

        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.p, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        B, L = x.shape
        pos  = torch.arange(L, device=x.device).unsqueeze(0)

        h    = self.tok_emb(x) + self.pos_emb(pos)
        mask = self.causal_mask[:L, :L]

        hiddens     = {"embed": h.detach().clone()}
        attn_weights= []

        for i, layer in enumerate(self.layers):
            h, attn_w = layer(h, mask)
            hiddens[f"layer_{i}"] = h.detach().clone()
            attn_weights.append(attn_w)

        h = self.norm(h)
        hiddens["final"] = h.detach().clone()

        logits = self.head(h[:, -1, :])

        if return_hidden:
            return logits, hiddens, attn_weights
        return logits

    def get_activations(self, x: torch.Tensor, layer: int = 0, position: int = -1) -> torch.Tensor:
        """Same interface as MambaForModularArithmetic.get_activations."""
        logits, hiddens, _ = self.forward(x, return_hidden=True)
        key = f"layer_{layer}" if layer >= 0 else "final"
        return hiddens[key][:, position, :]

    def get_attention_patterns(self, x: torch.Tensor):
        """Return attention weight tensors for visualisation. Shape: list of (B, H, L, L)."""
        _, _, attn_weights = self.forward(x, return_hidden=True)
        return attn_weights

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
