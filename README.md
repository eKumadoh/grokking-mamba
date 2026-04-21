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

**Results**
All experiments use p=97, train_frac=0.3, d_model=128, n_layers=1, 15,000 epochs.
**1. Grokking**
Both architectures exhibit clean grokking. With cosine LR scheduling and gradient clipping, both models generalise rapidly and stably — no post-grokking accuracy collapses.
|Model           |Grokking Epoch         |Linear Probe Acc|
|----------------|-----------------------|----------------|
|Mamba           |240                    |96.5%           |
|Transformer     |221                    |100.0%          |

Grokking speed is comparable across architectures (~10% difference). Without training stabilisation (fixed LR, no gradient clipping), Mamba appeared to grok faster (ep 113) but repeatedly lost the generalising solution — confirming that the apparent speed advantage was an artifact of unstable optimisation rather than a property of the architecture.
Show Image

**2. Fourier Structure of Embeddings**
To check whether both models learn the clock algorithm (Nanda et al. 2023), we project the token embedding matrix onto its top SVD components and inspect their FFT spectra. Clock-algorithm features produce sharp peaks at one or two frequencies.
Mamba embeddings show sparse, sharp FFT peaks — typically one dominant frequency per SVD component. This is consistent with a compressed encoding of the clock algorithm in the recurrent state.
Transformer embeddings show more diffuse spectra with energy spread across multiple frequencies, consistent with computation distributed across attention heads.
Show Image
Show Image

**3. Linear Probing**
A linear classifier trained on frozen residual-stream activations decodes (a+b) mod p with 100% accuracy for the Transformer and 96.5% for Mamba. Both results confirm that the answer is linearly accessible in the residual stream after grokking. The small gap suggests Mamba's recurrent state encodes the result with slightly more nonlinear entanglement than the Transformer's attention-based computation.

**4. Sparse Autoencoder Features (Mamba)**
A 4× overcomplete Sparse Autoencoder (512 features) trained on Mamba's layer-0 activations converges cleanly: reconstruction loss ~0, 0% dead features, ~30% feature sparsity at convergence.
The top features (by activation frequency) show Fourier-structured activation patterns rather than class-selective responses — each feature fires for a specific frequency component of (a+b) mod p, not for individual residue classes. This is evidence that Mamba's recurrent state implements a Fourier-based algorithm analogous to the Transformer clock algorithm, but compressed into a fixed-size hidden state rather than distributed across attention heads.
Show Image

**5. Transformer Attention Patterns**
All four attention heads in Layer 0 show near-identical patterns: the a token attends almost exclusively to itself (~1.0), and attention follows the causal staircase structure. This is consistent with the clock algorithm — a carries dominant Fourier structure from the first position, and all heads read it directly.
Show Image

**Summary**
|Finding                                |Result                                                                                                                      |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
|Does Mamba grok?                       |Yes, at comparable speed to Transformers (~ep 240 vs 221)                                                                   |
|Does Mamba learn Fourier structure?    |Yes — sparser and more compressed than the Transformer                                                                      |
|Is the answer linearly accessible?     |Yes for both; Transformer perfectly (100%), Mamba nearly so (96.5%)                                                         |
|What do SAE features encode?           |Fourier frequency components, not residue classes                                                                           |
|Are the learned algorithms the same?   |Functionally similar but structurally different — Transformer distributes across heads, Mamba compresses into recurrent state|

## References

- Power et al. (2022). *Grokking: Generalisation Beyond Overfitting on Small Algorithmic Datasets.*
- Nanda et al. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.*
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
- Bricken et al. (2023). *Towards Monosemanticity.* Anthropic.
- Cunningham et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.*
