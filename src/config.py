from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ── Task ──────────────────────────────────────────────────────────────────
    p: int = 97             # Prime modulus (standard: 97 or 113)
    train_frac: float = 0.3 # Grokking requires low data fraction
    seed: int = 42

    # ── Model (shared) ────────────────────────────────────────────────────────
    d_model: int = 128
    n_layers: int = 1       # 1-layer is enough and easier to interpret

    # ── Mamba-specific ────────────────────────────────────────────────────────
    d_state: int = 16       # SSM state dimension N
    d_conv: int = 4         # Local conv kernel width
    expand: int = 2         # Inner dim = expand * d_model

    # ── Transformer-specific ─────────────────────────────────────────────────
    n_heads: int = 4

    # ── Training ──────────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1.0   # High WD is critical for grokking
    batch_size: int = 512
    n_epochs: int = 15_000
    log_every: int = 100

    # ── Sparse Autoencoder ────────────────────────────────────────────────────
    sae_expansion: int = 4      # dict_size = sae_expansion * d_model
    sae_l1_coeff: float = 1e-3  # Sparsity penalty
    sae_lr: float = 2e-4
    sae_epochs: int = 5_000
    sae_batch_size: int = 2048

    # ── I/O ───────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    # ── Derived (read-only) ───────────────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        # 0..p-1: numbers | p: '+' operator | p+1: '=' token
        return self.p + 2

    @property
    def seq_len(self) -> int:
        return 4  # [a, +, b, =]

    @property
    def sae_dict_size(self) -> int:
        return self.sae_expansion * self.d_model

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
