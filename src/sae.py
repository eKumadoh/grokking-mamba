"""
Sparse Autoencoder (SAE) for mechanistic interpretability.

Goal: learn an *overcomplete* dictionary of monosemantic features
from model activations, decomposing the polysemantic residual stream
into interpretable, human-readable directions.

Loss:
    L = ||x - x̂||²  +  λ ||f(x)||₁
        reconstruction      sparsity

Constrained: decoder columns kept at unit norm (so λ has consistent scale).

References:
    Cunningham et al. 2023 — "Sparse Autoencoders Find Highly Interpretable
        Features in Language Models"
    Bricken et al. 2023   — "Towards Monosemanticity" (Anthropic)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_dict: int, l1_coeff: float = 1e-3):
        """
        Args:
            d_in    : input dimensionality (= model d_model)
            d_dict  : dictionary size (overcomplete, e.g. 4× d_in)
            l1_coeff: L1 sparsity penalty weight λ
        """
        super().__init__()
        self.d_in     = d_in
        self.d_dict   = d_dict
        self.l1_coeff = l1_coeff

        # Encoder: d_in → d_dict
        self.W_enc = nn.Parameter(torch.empty(d_in,   d_dict))
        self.b_enc = nn.Parameter(torch.zeros(d_dict))

        # Decoder: d_dict → d_in  (columns kept unit-norm)
        self.W_dec = nn.Parameter(torch.empty(d_dict, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=np.sqrt(5))
        self._normalize_decoder()

    # ── Internals ──────────────────────────────────────────────────────────

    def _normalize_decoder(self):
        """Project decoder rows to unit norm (no-grad)."""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1.0)
            self.W_dec.div_(norms)

    # ── Forward ────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_in) → feature activations (B, d_dict)."""
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """(B, d_dict) → reconstruction (B, d_in)."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        """
        Returns:
            loss       : scalar total loss
            recon_loss : reconstruction MSE
            l1_loss    : L1 sparsity loss
            features   : (B, d_dict) feature activations
            x_hat      : (B, d_in)   reconstruction
        """
        features = self.encode(x)
        x_hat    = self.decode(features)

        recon_loss = ((x - x_hat) ** 2).sum(-1).mean()
        l1_loss    = features.abs().sum(-1).mean()
        loss       = recon_loss + self.l1_coeff * l1_loss

        return loss, recon_loss, l1_loss, features, x_hat

    @property
    def feature_directions(self) -> torch.Tensor:
        """Decoder rows = learned feature directions. Shape: (d_dict, d_in)."""
        return self.W_dec.detach()


# ── Training ───────────────────────────────────────────────────────────────

def train_sae(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    cfg,
    verbose: bool = True,
) -> dict:
    """
    Train a SparseAutoencoder on a fixed matrix of model activations.

    Args:
        sae        : SAE model (moved to cfg.device inside)
        activations: (N, d_model) tensor of collected activations
        cfg        : Config with sae_lr, sae_epochs, sae_batch_size
    Returns:
        history dict with per-epoch losses and sparsity
    """
    device = cfg.device
    sae    = sae.to(device)
    acts   = activations.to(device)

    dataset = TensorDataset(acts)
    loader  = DataLoader(dataset, batch_size=cfg.sae_batch_size, shuffle=True)

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.sae_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.sae_epochs)

    history = {"loss": [], "recon": [], "l1": [], "sparsity": [], "dead_features": []}

    for epoch in tqdm(range(cfg.sae_epochs), desc="SAE training", disable=not verbose):
        ep_loss = ep_recon = ep_l1 = ep_spar = 0.0
        feature_activated = torch.zeros(sae.d_dict, device=device)
        n_batches = 0

        for (batch,) in loader:
            optimizer.zero_grad()
            loss, recon, l1, feats, _ = sae(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            sae._normalize_decoder()

            feature_activated += (feats.detach() > 0).float().sum(0)
            ep_loss  += loss.item()
            ep_recon += recon.item()
            ep_l1    += l1.item()
            ep_spar  += (feats.detach() > 0).float().mean().item()
            n_batches += 1

        scheduler.step()

        dead = (feature_activated == 0).float().mean().item()
        for k, v in zip(
            ["loss", "recon", "l1", "sparsity", "dead_features"],
            [ep_loss, ep_recon, ep_l1, ep_spar, dead],
        ):
            history[k].append(v / n_batches if k != "dead_features" else dead)

        if verbose and epoch % 500 == 0:
            tqdm.write(
                f"SAE {epoch:5d} | loss {history['loss'][-1]:.4f} "
                f"| recon {history['recon'][-1]:.4f} "
                f"| l1 {history['l1'][-1]:.4f} "
                f"| sparsity {history['sparsity'][-1]:.3f} "
                f"| dead {history['dead_features'][-1]:.3f}"
            )

    return history


# ── Activation collection ─────────────────────────────────────────────────

@torch.no_grad()
def collect_activations(
    model,
    dataloader,
    cfg,
    layer: int = 0,
    position: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run model in eval mode, collect activations and labels for every batch.

    Returns:
        activations: (N, d_model) on CPU
        labels     : (N,)         on CPU
    """
    model.eval()
    all_acts, all_labels = [], []

    for x, y in dataloader:
        x = x.to(cfg.device)
        acts = model.get_activations(x, layer=layer, position=position)
        all_acts.append(acts.cpu())
        all_labels.append(y)

    return torch.cat(all_acts), torch.cat(all_labels)
