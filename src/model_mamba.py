"""
Minimal, from-scratch Mamba implementation for mechanistic interpretability.

Architecture (per block):
    x  →  LayerNorm  →  in_proj  →  [x_branch, z_gate]
    x_branch  →  depthwise Conv1d  →  SiLU  →  S6 (selective SSM)
    output = out_proj( SSM_output ⊙ SiLU(z_gate) ) + residual

The S6 (selective state-space model) core:
    Continuous:   h'(t) = A h(t) + B(x) x(t),   y(t) = C(x) h(t)
    Discretized:  h_t   = dA h_{t-1} + dB x_t,  y_t  = C h_t + D x_t
    where dA, dB come from Zero-Order Hold with input-dependent Δ (delta).

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
           Selective State Spaces", 2023.

Changes vs original:
  - MambaBlock.get_x_branch() exposed for interpretability hooks
    (needed by analysis.fourier_hidden_state_analysis)
  - MambaBlock.forward() stores last x_branch as self._last_x_branch
    so SSM hidden states can be extracted post-hoc without re-running
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    S6: the selective state-space model at the heart of Mamba.

    Key idea: B, C, and Δ are functions of the input x, making the
    state-space *selective* (unlike fixed-parameter SSMs like S4).

    Shapes in comments use:
        B = batch, L = seq_len, D = d_inner, N = d_state
    """

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        dt_rank = math.ceil(d_inner / 16)
        self.dt_rank = dt_rank

        # Projects x → [Δ_raw (dt_rank), B (d_state), C (d_state)]
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # Δ projection: low-rank → full d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # A: diagonal structured matrix, log-parameterized for positivity
        # Initialized with HiPPO-style 1..N values
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_init = A_init.unsqueeze(0).expand(d_inner, -1)   # (D, N)
        self.A_log = nn.Parameter(torch.log(A_init))

        # D: skip / residual connection scalar per channel
        self.D = nn.Parameter(torch.ones(d_inner))

        # Good initialization for dt_proj bias
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        B, L, D = x.shape

        # ── Compute selective parameters ──────────────────────────────────
        xp   = self.x_proj(x)                                          # (B, L, dt_rank + 2N)
        dt_r, B_mat, C_mat = xp.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(dt_r))                        # (B, L, D)  Δ > 0
        A     = -torch.exp(self.A_log.float())                         # (D, N)   A < 0

        # ── Discretize (ZOH) ──────────────────────────────────────────────
        # dA[b,l,d,n] = exp(Δ[b,l,d] * A[d,n])
        dA = torch.exp(delta.unsqueeze(-1) * A)                        # (B, L, D, N)
        # dB[b,l,d,n] = Δ[b,l,d] * B[b,l,n]
        dB = delta.unsqueeze(-1) * B_mat.unsqueeze(2)                  # (B, L, D, N)

        # ── Sequential scan ───────────────────────────────────────────────
        # For short sequences (L=4) sequential scan is perfectly fine.
        h = x.new_zeros(B, D, self.d_state)
        ys = []
        for t in range(L):
            h  = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)     # (B, D, N)
            yt = (h * C_mat[:, t].unsqueeze(1)).sum(-1)               # (B, D)
            ys.append(yt)

        y = torch.stack(ys, dim=1)                                     # (B, L, D)
        return y + x * self.D                                          # skip connection

    def get_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return all SSM hidden states for interpretability analysis.

        Returns: h of shape (B, L, D, N)
        """
        B, L, D = x.shape
        xp      = self.x_proj(x)
        dt_r, B_mat, _ = xp.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta   = F.softplus(self.dt_proj(dt_r))
        A       = -torch.exp(self.A_log.float())
        dA      = torch.exp(delta.unsqueeze(-1) * A)
        dB      = delta.unsqueeze(-1) * B_mat.unsqueeze(2)

        h       = x.new_zeros(B, D, self.d_state)
        states  = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            states.append(h.clone())                                   # (B, D, N)

        return torch.stack(states, dim=1)                              # (B, L, D, N)


class MambaBlock(nn.Module):
    """Single Mamba block with pre-norm and residual."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = int(expand * d_model)

        self.norm    = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise conv for local context mixing
        self.conv1d  = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        self.act     = nn.SiLU()
        self.ssm     = SelectiveSSM(self.d_inner, d_state)
        self.out_proj= nn.Linear(self.d_inner, d_model, bias=False)

        # Cached for interpretability hooks (set during forward)
        self._last_x_branch: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x        = self.norm(x)                                        # pre-norm

        xz               = self.in_proj(x)                             # (B, L, 2*D_inner)
        x_branch, z_gate = xz.chunk(2, dim=-1)

        # Causal conv (trim to original seq length)
        x_branch = x_branch.transpose(1, 2)                           # (B, D_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :x.shape[1]]
        x_branch = x_branch.transpose(1, 2)                           # (B, L, D_inner)
        x_branch = self.act(x_branch)

        # Cache for external interpretability access
        self._last_x_branch = x_branch.detach()

        y = self.ssm(x_branch)                                         # (B, L, D_inner)
        y = y * self.act(z_gate)                                       # gated output

        return self.out_proj(y) + residual

    def get_x_branch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Re-compute and return the conv-activated x_branch for a given input.
        This is the tensor fed into the SSM — used by fourier_hidden_state_analysis
        in analysis.py to extract SSM hidden states h_t directly.

        Args:
            x: (B, L, d_model) — token embeddings (pre-block)
        Returns:
            x_branch: (B, L, d_inner)
        """
        x_normed         = self.norm(x)
        xz               = self.in_proj(x_normed)
        x_branch, _      = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :x.shape[1]]
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.act(x_branch)

        return x_branch


class MambaForModularArithmetic(nn.Module):
    """
    Full Mamba model for the modular arithmetic task.

    Input : integer token sequence  [a, '+', b, '=']  (length 4)
    Output: logit vector over {0 .. p-1}  (predicted at '=' position)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len,    cfg.d_model)

        self.layers = nn.ModuleList([
            MambaBlock(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand)
            for _ in range(cfg.n_layers)
        ])

        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.p, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """
        Args:
            x            : (B, L) token indices
            return_hidden: if True, also return dict of layer activations
        Returns:
            logits  : (B, p)  — prediction at the last position
            hiddens : dict    — only if return_hidden=True
        """
        B, L = x.shape
        pos  = torch.arange(L, device=x.device).unsqueeze(0)

        h = self.tok_emb(x) + self.pos_emb(pos)                       # (B, L, D)

        hiddens = {"embed": h.detach().clone()}
        for i, layer in enumerate(self.layers):
            h = layer(h)
            hiddens[f"layer_{i}"] = h.detach().clone()

        h = self.norm(h)
        hiddens["final"] = h.detach().clone()

        logits = self.head(h[:, -1, :])                                # predict at '='

        if return_hidden:
            return logits, hiddens
        return logits

    def get_activations(self, x: torch.Tensor, layer: int = 0, position: int = -1) -> torch.Tensor:
        """
        Extract residual-stream activations at a given layer and sequence position.
        Used for probing and SAE training.

        Args:
            layer   : which layer (0-indexed); use -1 for post-norm final
            position: which token position (-1 = last = '=')
        Returns:
            acts: (B, d_model)
        """
        _, hiddens = self.forward(x, return_hidden=True)
        key = f"layer_{layer}" if layer >= 0 else "final"
        return hiddens[key][:, position, :]

    def get_ssm_states(self, x: torch.Tensor, layer: int = 0) -> torch.Tensor:
        """
        Extract SSM hidden states h_t at every sequence position for a given layer.
        Used by analysis.fourier_hidden_state_analysis.

        Args:
            x    : (B, L) token indices
            layer: which MambaBlock to probe (0-indexed)
        Returns:
            states: (B, L, d_inner, d_state)
        """
        B, L = x.shape
        pos  = torch.arange(L, device=x.device).unsqueeze(0)
        h    = self.tok_emb(x) + self.pos_emb(pos)                    # (B, L, D)

        # Run all layers up to (but not including) target layer
        for i, layer_mod in enumerate(self.layers):
            if i == layer:
                # Get x_branch then extract SSM states directly
                x_branch = layer_mod.get_x_branch(h)
                return layer_mod.ssm.get_states(x_branch)             # (B, L, D_inner, N)
            h = layer_mod(h)

        raise ValueError(f"Layer {layer} out of range (model has {len(self.layers)} layers)")

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
