"""
Analysis and visualisation for the Mamba-vs-Transformer grokking study.

Plots produced:
  1. grokking_curves      — train/test acc over time (the signature grokking plot)
  2. architecture_compare — Mamba vs Transformer side-by-side
  3. fourier_embedding    — SVD + FFT of token embeddings (clock algorithm check)
  4. linear_probe         — decode (a+b) mod p from activations
  5. sae_features         — SAE feature selectivity & Fourier structure
  6. sae_training         — SAE training curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from scipy.fft import fft


# ── Colour palette (consistent across all plots) ─────────────────────────
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
GREY   = "#8C8C8C"


# ─────────────────────────────────────────────────────────────────────────
# 1.  Grokking curves
# ─────────────────────────────────────────────────────────────────────────

def plot_grokking_curves(history: dict, grokking_epoch, title: str = "", save_path=None):
    """
    Two-panel plot: accuracy (left) and loss (right) over training.
    A vertical dashed line marks the grokking epoch if detected.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = history["epoch"]

    # Accuracy
    ax1.plot(epochs, history["train_acc"], label="Train", color=BLUE,   lw=2)
    ax1.plot(epochs, history["test_acc"],  label="Test",  color=ORANGE, lw=2)
    if grokking_epoch is not None:
        ax1.axvline(grokking_epoch, color=GREEN, ls="--", lw=1.5, alpha=0.8,
                    label=f"Grokking (ep {grokking_epoch:,})")
    ax1.set(xlabel="Epoch", ylabel="Accuracy", title=f"{title} — Accuracy", ylim=(-0.05, 1.05))
    ax1.legend(); ax1.grid(alpha=0.3)

    # Loss (log scale)
    ax2.plot(epochs, history["train_loss"], label="Train", color=BLUE,   lw=2)
    ax2.plot(epochs, history["test_loss"],  label="Test",  color=ORANGE, lw=2)
    if grokking_epoch is not None:
        ax2.axvline(grokking_epoch, color=GREEN, ls="--", lw=1.5, alpha=0.8,
                    label=f"Grokking (ep {grokking_epoch:,})")
    ax2.set(xlabel="Epoch", ylabel="Loss (log)", title=f"{title} — Loss", yscale="log")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────
# 2.  Architecture comparison
# ─────────────────────────────────────────────────────────────────────────

def plot_comparison(mamba_history: dict, transformer_history: dict,
                    mamba_grok: int, transformer_grok: int, save_path=None):
    """Side-by-side accuracy curves: Mamba vs Transformer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, hist, grok, col) in zip(
        axes,
        [("Mamba",       mamba_history,       mamba_grok,       BLUE),
         ("Transformer", transformer_history, transformer_grok, ORANGE)],
    ):
        epochs = hist["epoch"]
        ax.plot(epochs, hist["train_acc"], lw=2, ls="--", color=col, alpha=0.6, label="Train")
        ax.plot(epochs, hist["test_acc"],  lw=2, color=col,             label="Test")
        if grok is not None:
            ax.axvline(grok, color=GREEN, ls=":", lw=1.5, label=f"Grokking (ep {grok:,})")
        ax.set(xlabel="Epoch", ylabel="Accuracy", title=name, ylim=(-0.05, 1.05))
        ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Grokking in SSMs vs Transformers", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────
# 3.  Fourier analysis of embeddings (clock algorithm check)
# ─────────────────────────────────────────────────────────────────────────

def fourier_embedding_analysis(model, cfg, title: str = "", save_path=None):
    """
    Check whether the model encodes numbers in a Fourier basis.

    The transformer 'clock algorithm' (Nanda et al. 2023) uses specific
    frequency components.  If Mamba learns the same algorithm, we expect
    similar structure in the top SVD components of the embedding matrix.
    """
    W = model.tok_emb.weight[:cfg.p].detach().cpu().numpy()   # (p, d_model)

    # SVD: dominant embedding directions
    U, S, Vt = np.linalg.svd(W, full_matrices=False)          # U: (p, p)

    n_components = 8
    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(2, n_components, figure=fig, hspace=0.5)

    for i in range(n_components):
        vec   = U[:, i]                                        # (p,)
        freqs = np.abs(fft(vec))

        # Row 0: raw embedding component
        ax0 = fig.add_subplot(gs[0, i])
        ax0.plot(vec, color=BLUE, lw=1.2)
        ax0.set_title(f"SV {i+1}\n(σ={S[i]:.1f})", fontsize=8)
        ax0.set_xticks([]); ax0.grid(alpha=0.3)
        if i == 0:
            ax0.set_ylabel("SV component", fontsize=8)

        # Row 1: FFT magnitude
        ax1 = fig.add_subplot(gs[1, i])
        half = cfg.p // 2
        ax1.bar(range(half), freqs[1:half+1], color=ORANGE, width=1, alpha=0.8)
        ax1.set_title("FFT", fontsize=8)
        ax1.set_xlabel("Freq", fontsize=7)
        ax1.grid(alpha=0.3)
        if i == 0:
            ax1.set_ylabel("Magnitude", fontsize=8)

    fig.suptitle(
        f"{title} — Fourier Structure of Embeddings\n"
        "(Clock algorithm = sharp peaks at one or two frequencies)",
        fontsize=13, fontweight="bold",
    )
    _save_or_show(fig, save_path)
    return U, S


# ─────────────────────────────────────────────────────────────────────────
# 4.  Linear probe
# ─────────────────────────────────────────────────────────────────────────

def linear_probe(activations: torch.Tensor, labels: torch.Tensor, cfg,
                 n_epochs: int = 1000, verbose: bool = True) -> tuple:
    """
    Train a linear classifier on activations to decode (a+b) mod p.

    High accuracy → the information is *linearly accessible* in the
    residual stream (i.e. not entangled with other features).
    """
    device = cfg.device
    acts   = activations.to(device).float()
    labs   = labels.to(device)

    n       = len(acts)
    n_train = int(0.8 * n)
    perm    = torch.randperm(n)
    tr, te  = perm[:n_train], perm[n_train:]

    probe     = nn.Linear(acts.shape[-1], cfg.p).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        criterion(probe(acts[tr]), labs[tr]).backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(acts[te]).argmax(-1)
        acc   = (preds == labs[te]).float().mean().item()

    if verbose:
        print(f"  Linear probe accuracy: {acc:.3f}")

    return acc, probe


# ─────────────────────────────────────────────────────────────────────────
# 5.  SAE feature analysis
# ─────────────────────────────────────────────────────────────────────────

def plot_sae_features(sae, activations: torch.Tensor, labels: torch.Tensor,
                      cfg, n_top: int = 8, save_path=None):
    """
    For each of the top-n most active SAE features, show:
      • Mean activation conditioned on the true label (a+b) mod p
      • FFT of that activation pattern (Fourier structure check)

    If features are monosemantic (fire for a specific residue class or
    for a Fourier component of the clock algorithm), we can identify
    which algorithm the model is using.
    """
    sae.eval()
    with torch.no_grad():
        feats = sae.encode(activations - sae.b_dec)              # (N, d_dict)

    feats_np  = feats.cpu().numpy()
    labels_np = labels.cpu().numpy()

    activation_freq = (feats_np > 0).mean(0)                    # (d_dict,)
    top_feat_idx    = activation_freq.argsort()[-n_top:][::-1]  # highest-freq features

    fig, axes = plt.subplots(2, n_top, figsize=(3 * n_top, 7))

    for col, feat_idx in enumerate(top_feat_idx):
        # Average activation per label value
        mean_by_label = np.array([
            feats_np[labels_np == l, feat_idx].mean()
            if (labels_np == l).any() else 0.0
            for l in range(cfg.p)
        ])

        # ── Top row: raw activation pattern ──────────────────────────────
        ax0 = axes[0, col]
        ax0.bar(range(cfg.p), mean_by_label, color=BLUE, alpha=0.8, width=1)
        ax0.set_title(f"F{feat_idx}\nfreq={activation_freq[feat_idx]:.2f}", fontsize=8)
        ax0.set_xlabel("(a+b) mod p", fontsize=7)
        if col == 0:
            ax0.set_ylabel("Mean activation", fontsize=8)
        ax0.grid(alpha=0.3)

        # ── Bottom row: FFT of activation pattern ─────────────────────────
        ax1 = axes[1, col]
        fft_vals = np.abs(fft(mean_by_label))
        half     = cfg.p // 2
        ax1.bar(range(1, half + 1), fft_vals[1:half + 1], color=ORANGE, alpha=0.8, width=1)
        ax1.set_title("FFT", fontsize=8)
        ax1.set_xlabel("Frequency", fontsize=7)
        if col == 0:
            ax1.set_ylabel("Magnitude", fontsize=8)
        ax1.grid(alpha=0.3)

    fig.suptitle(
        "SAE Feature Selectivity — Activation Pattern and Fourier Spectrum\n"
        "(Clock-algorithm features show sharp FFT peaks)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save_or_show(fig, save_path)

    return feats_np, activation_freq, top_feat_idx


def plot_sae_training(history: dict, save_path=None):
    """Plot SAE training curves: loss, sparsity, dead features."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(len(history["loss"]))

    axes[0].plot(epochs, history["loss"],   color=BLUE,   lw=1.5)
    axes[0].plot(epochs, history["recon"],  color=ORANGE, lw=1.5, ls="--", label="Recon")
    axes[0].set(title="SAE Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(["Total", "Recon"]); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["sparsity"], color=GREEN, lw=1.5)
    axes[1].set(title="Feature Sparsity (fraction active)", xlabel="Epoch", ylabel="Fraction > 0")
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["dead_features"], color=GREY, lw=1.5)
    axes[2].set(title="Dead Features (never activated)", xlabel="Epoch", ylabel="Fraction dead")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────
# 6.  Attention pattern visualisation (Transformer only)
# ─────────────────────────────────────────────────────────────────────────

def plot_attention_patterns(attn_weights, token_labels: list = None,
                             n_examples: int = 4, save_path=None):
    """
    Visualise attention matrices.
    attn_weights: list of (B, H, L, L) per layer
    """
    n_layers = len(attn_weights)
    n_heads  = attn_weights[0].shape[1]
    L        = attn_weights[0].shape[2]
    labels   = token_labels or [str(i) for i in range(L)]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers))
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    for l in range(n_layers):
        for h in range(n_heads):
            avg_attn = attn_weights[l][: min(n_examples, attn_weights[l].shape[0]), h].mean(0)
            ax = axes[l, h]
            im = ax.imshow(avg_attn.cpu().numpy(), vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(L)); ax.set_yticks(range(L))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(f"Layer {l} Head {h}", fontsize=9)
            plt.colorbar(im, ax=ax)

    plt.suptitle("Attention Patterns", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────

def _save_or_show(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.show()
    plt.close(fig)
