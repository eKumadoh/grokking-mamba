"""
run.py — Full experiment pipeline

Phases
------
1. Train Mamba on modular arithmetic, detect grokking
2. Train Transformer baseline, detect grokking
3. Compare grokking curves (do SSMs grok? faster/slower?)
4. Fourier analysis of learned embeddings (clock algorithm?)
5. Linear probing of residual stream
6. Train Sparse Autoencoder, analyse discovered features

Run:
    python run.py
    python run.py --p 97 --n_epochs 20000 --d_model 128
"""

import argparse
import json
import os
import torch

from config import Config
from data import get_dataloaders, get_full_tensors
from model_mamba import MambaForModularArithmetic
from model_transformer import TransformerForModularArithmetic
from sae import SparseAutoencoder, train_sae, collect_activations
from train import train_model
import analysis


# ─────────────────────────────────────────────────────────────────────────

def run(cfg: Config):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir,    exist_ok=True)

    train_dl, test_dl = get_dataloaders(cfg)
    print(f"\nDataset: p={cfg.p}, train={len(train_dl.dataset)}, test={len(test_dl.dataset)}")
    print(f"Device : {cfg.device}\n")

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Mamba
    # ══════════════════════════════════════════════════════════════════════
    mamba = MambaForModularArithmetic(cfg)

    # train_model now returns 3 values: history, grokking_epoch, best_state_dict
    mamba_history, mamba_grok, mamba_best = train_model(
        mamba, train_dl, test_dl, cfg, "Mamba"
    )
    results["mamba_grokking_epoch"] = mamba_grok

    # Restore best weights before analysis so probes/Fourier reflect peak
    # performance, not wherever the model happened to land at the final epoch
    if mamba_best is not None:
        mamba.load_state_dict(mamba_best)
        print("\n  ✓ Mamba: restored best checkpoint for analysis")

    torch.save(mamba.state_dict(), f"{cfg.checkpoint_dir}/mamba.pt")

    analysis.plot_grokking_curves(
        mamba_history, mamba_grok, title="Mamba",
        save_path=f"{cfg.results_dir}/mamba_grokking.png",
    )

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Transformer baseline
    # ══════════════════════════════════════════════════════════════════════
    transformer = TransformerForModularArithmetic(cfg)

    transformer_history, transformer_grok, transformer_best = train_model(
        transformer, train_dl, test_dl, cfg, "Transformer"
    )
    results["transformer_grokking_epoch"] = transformer_grok

    if transformer_best is not None:
        transformer.load_state_dict(transformer_best)
        print("\n  ✓ Transformer: restored best checkpoint for analysis")

    torch.save(transformer.state_dict(), f"{cfg.checkpoint_dir}/transformer.pt")

    analysis.plot_grokking_curves(
        transformer_history, transformer_grok, title="Transformer",
        save_path=f"{cfg.results_dir}/transformer_grokking.png",
    )

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Architecture comparison
    # ══════════════════════════════════════════════════════════════════════
    analysis.plot_comparison(
        mamba_history, transformer_history, mamba_grok, transformer_grok,
        save_path=f"{cfg.results_dir}/comparison.png",
    )

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Fourier analysis (clock algorithm check)
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Fourier embedding analysis ──")
    analysis.fourier_embedding_analysis(
        mamba, cfg, title="Mamba",
        save_path=f"{cfg.results_dir}/mamba_fourier.png",
    )
    analysis.fourier_embedding_analysis(
        transformer, cfg, title="Transformer",
        save_path=f"{cfg.results_dir}/transformer_fourier.png",
    )

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5: Linear probing
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Linear probing ──")
    mamba.eval()
    transformer.eval()

    mamba_acts, mamba_labels = collect_activations(mamba,       train_dl, cfg, layer=0)
    trans_acts, trans_labels = collect_activations(transformer, train_dl, cfg, layer=0)

    print("Mamba activations:")
    mamba_probe_acc, _ = analysis.linear_probe(mamba_acts, mamba_labels, cfg)

    print("Transformer activations:")
    trans_probe_acc, _ = analysis.linear_probe(trans_acts, trans_labels, cfg)

    results["mamba_probe_acc"]       = mamba_probe_acc
    results["transformer_probe_acc"] = trans_probe_acc

    # ══════════════════════════════════════════════════════════════════════
    # Phase 6: Sparse Autoencoder
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Sparse Autoencoder (Mamba) ──")

    # Collect activations from ALL data for SAE
    (tr_x, tr_y), (te_x, te_y) = get_full_tensors(cfg)
    all_x = torch.cat([tr_x, te_x]).to(cfg.device)
    all_y = torch.cat([tr_y, te_y])

    mamba.eval()
    with torch.no_grad():
        all_acts = mamba.get_activations(all_x, layer=0, position=-1).cpu()

    print(f"  Collected {len(all_acts)} activations, shape {all_acts.shape}")

    sae = SparseAutoencoder(cfg.d_model, cfg.sae_dict_size, cfg.sae_l1_coeff)
    sae_history = train_sae(sae, all_acts.to(cfg.device), cfg)

    torch.save(sae.state_dict(), f"{cfg.checkpoint_dir}/sae.pt")

    analysis.plot_sae_training(
        sae_history,
        save_path=f"{cfg.results_dir}/sae_training.png",
    )

    sae.eval()
    analysis.plot_sae_features(
        sae, all_acts.to(cfg.device), all_y, cfg,
        save_path=f"{cfg.results_dir}/sae_features.png",
    )

    # ══════════════════════════════════════════════════════════════════════
    # (Optional) Transformer attention patterns
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Transformer attention patterns ──")
    sample_x = te_x[:16].to(cfg.device)
    with torch.no_grad():
        attn_weights = transformer.get_attention_patterns(sample_x)

    token_labels = ["a", "+", "b", "="]
    analysis.plot_attention_patterns(
        attn_weights, token_labels,
        save_path=f"{cfg.results_dir}/attention_patterns.png",
    )

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    with open(f"{cfg.results_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "═" * 60)
    print("  DONE")
    print("═" * 60)
    print(f"  Mamba grokked at epoch      : {mamba_grok}")
    print(f"  Transformer grokked at epoch: {transformer_grok}")
    print(f"  Mamba linear probe acc      : {mamba_probe_acc:.3f}")
    print(f"  Transformer linear probe acc: {trans_probe_acc:.3f}")
    print(f"  Results saved to            : {cfg.results_dir}/")
    print("═" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Grokking in Mamba vs Transformer")
    parser.add_argument("--p",            type=int,   default=97)
    parser.add_argument("--train_frac",   type=float, default=0.3)
    parser.add_argument("--d_model",      type=int,   default=128)
    parser.add_argument("--n_layers",     type=int,   default=1)
    parser.add_argument("--n_epochs",     type=int,   default=15_000)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--sae_epochs",   type=int,   default=5_000)
    parser.add_argument("--results_dir",  type=str,   default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = Config(
        p            = args.p,
        train_frac   = args.train_frac,
        d_model      = args.d_model,
        n_layers     = args.n_layers,
        n_epochs     = args.n_epochs,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        sae_epochs   = args.sae_epochs,
        results_dir  = args.results_dir,
    )
    run(cfg)
