# Grokking in State Space Models
### Does Mamba learn the Transformer clock algorithm?

A mechanistic interpretability study comparing how Mamba (SSM) and Transformers
solve modular arithmetic — and whether Mamba grokks, and if so, what circuit it builds.

---

## Research Question

Nanda et al. (2023) showed that Transformers solving `(a + b) mod p` develop a specific
internal algorithm ("the clock algorithm") using discrete Fourier transforms encoded
in the embedding and attention layers.

**This project asks:**
1. Does grokking occur in Mamba (an SSM architecture)?
2. If so, does Mamba learn the same Fourier-based clock algorithm, or something different?
3. Can Sparse Autoencoders (SAEs) decompose the learned features into interpretable directions?

---

## Project Structure

```
grokking_mamba/
├── config.py          # All hyperparameters in one place
├── data.py            # Modular arithmetic dataset
├── model_mamba.py     # From-scratch Mamba with interpretability hooks
├── model_transformer.py # Baseline Transformer (same task)
├── sae.py             # Sparse Autoencoder for feature discovery
├── train.py           # Training loop with grokking detection
├── analysis.py        # All visualisation + probing functions
└── run.py             # Main experiment runner
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. A GPU is recommended for training
(Google Colab T4 is sufficient).

---

## Running the Experiment

```bash
# Default settings (p=97, 15k epochs)
python run.py

# Custom settings
python run.py --p 113 --n_epochs 20000 --d_model 128

# Quick test (fewer epochs)
python run.py --n_epochs 2000 --sae_epochs 500
```

Results (plots + metrics) are saved to `results/`.

---

## Outputs

| File | Description |
|------|-------------|
| `mamba_grokking.png` | Train/test accuracy over time for Mamba |
| `transformer_grokking.png` | Same for Transformer |
| `comparison.png` | Side-by-side grokking curves |
| `mamba_fourier.png` | FFT of Mamba's embedding SVD components |
| `transformer_fourier.png` | Same for Transformer |
| `sae_training.png` | SAE loss, sparsity, dead features |
| `sae_features.png` | Top SAE features — activation pattern & FFT |
| `attention_patterns.png` | Transformer attention heads |
| `results.json` | Numeric results (grokking epochs, probe accs) |

---

## Key Design Choices

**Why from-scratch Mamba?**
We implement Mamba ourselves (no external library) so we can fully instrument
every component — extract SSM hidden states, hook into intermediate activations,
and apply probes anywhere. The `mamba-ssm` CUDA package is faster but harder to inspect.

**Why high weight decay?**
Weight decay (λ=1.0) is essential for grokking. Without it, memorisation is a stable
optimum and the model never generalises.

**Why SAEs?**
Linear probes tell us *whether* information is present. SAEs tell us *how* it's encoded —
they decompose polysemantic neurons into monosemantic feature directions, letting us ask
"does this feature correspond to a specific Fourier frequency of the clock algorithm?"

---

## References

- Power et al. (2022). *Grokking: Generalisation Beyond Overfitting on Small Algorithmic Datasets.*
- Nanda et al. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.*
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
- Bricken et al. (2023). *Towards Monosemanticity.* Anthropic.
- Cunningham et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.*
