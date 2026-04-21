"""
Training loop with grokking detection, LR scheduling, and gradient clipping.

Grokking (Power et al. 2022): a model first memorises the training set
(train acc → 100%, test acc stays low), then after many more steps
suddenly generalises (test acc → 100%).

Two conditions are necessary:
  1. Low data fraction  (train_frac ≈ 0.3)
  2. Strong weight decay (prevents memorisation from being stable long-term)

Changes vs original:
  - CosineAnnealingLR scheduler  → stops the model being kicked out of
    good minima by a fixed high LR (fixes the accuracy-crash spikes)
  - Gradient clipping (max_norm=1.0) → tames Mamba's recurrent-state
    gradient spikes
  - LR-reduction on confirmed grokking → further stabilises post-grokking
  - Per-epoch LR logged → easy to cross-reference with accuracy crashes
  - `best_state_dict` saved in memory → can retrieve weights from the
    epoch of best test accuracy (useful if training is unstable)
  - Warm-up phase (optional, cfg.warmup_epochs) → helps Mamba's SSM
    parameters settle before full LR hits
"""

import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    train_dl,
    test_dl,
    cfg,
    model_name: str = "model",
    use_warmup: bool = True,
) -> tuple:
    """
    Train a model and track grokking.

    Args:
        model       : nn.Module (Mamba or Transformer)
        train_dl    : training DataLoader
        test_dl     : test DataLoader
        cfg         : Config dataclass
        model_name  : display name for logging
        use_warmup  : if True, add a short linear warm-up phase before
                      cosine decay. Helps SSM parameter initialisation.

    Returns:
        history        : dict of logged metrics (epoch, loss, acc, lr)
        grokking_epoch : int or None — first epoch where test acc > 99%
                         after the model has already memorised training set
        best_state_dict: model weights from the epoch of peak test accuracy
    """
    device = torch.device(cfg.device)
    model  = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ────────────────────────────────────────────────────────────
    # AdamW with high weight decay is the standard grokking recipe.
    # betas=(0.9, 0.98) follows the original Nanda et al. setup.
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
    )

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    # Cosine annealing decays LR smoothly to eta_min, preventing the
    # optimizer from kicking the model out of the generalising minimum it
    # finds at the grokking point.
    #
    # Optional warm-up: ramp from lr/10 → lr over `warmup_epochs` steps,
    # then hand off to cosine decay for the remainder.
    warmup_epochs = getattr(cfg, "warmup_epochs", 200) if use_warmup else 0

    if warmup_epochs > 0:
        warmup_sched = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=cfg.n_epochs - warmup_epochs,
            eta_min=cfg.lr * 1e-2,   # floor at 1% of peak LR
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.n_epochs,
            eta_min=cfg.lr * 1e-2,
        )

    # ── History & state ───────────────────────────────────────────────────────
    history = {
        "epoch":      [],
        "train_loss": [],
        "test_loss":  [],
        "train_acc":  [],
        "test_acc":   [],
        "lr":         [],   # log LR so spikes can be correlated later
    }

    grokking_epoch  = None
    memorised       = False     # train acc crossed 99%
    grokked         = False     # test  acc crossed 99% after memorisation
    lr_reduced      = False     # one-shot LR cut after confirmed grokking

    best_test_acc   = -1.0
    best_state_dict = None      # weights at peak test accuracy

    # ── Banner ────────────────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters())
    _banner(model_name, n_params, cfg)

    # ═════════════════════════════════════════════════════════════════════════
    # Main training loop
    # ═════════════════════════════════════════════════════════════════════════
    for epoch in tqdm(range(cfg.n_epochs), desc=model_name, ncols=90):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = train_correct = train_total = 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()

            # Gradient clipping: keeps Mamba's recurrent-state gradients
            # from spiking and blowing up the learned solution.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss    += loss.item()
            train_correct += (logits.argmax(-1) == y).sum().item()
            train_total   += y.size(0)

        # Scheduler step (once per epoch, after optimizer step)
        scheduler.step()

        # ── Evaluate ──────────────────────────────────────────────────────────
        model.eval()
        test_loss = test_correct = test_total = 0

        with torch.no_grad():
            for x, y in test_dl:
                x, y   = x.to(device), y.to(device)
                logits = model(x)
                loss   = criterion(logits, y)

                test_loss    += loss.item()
                test_correct += (logits.argmax(-1) == y).sum().item()
                test_total   += y.size(0)

        train_acc = train_correct / train_total
        test_acc  = test_correct  / test_total
        cur_lr    = optimizer.param_groups[0]["lr"]

        # ── Track best weights ────────────────────────────────────────────────
        if test_acc > best_test_acc:
            best_test_acc   = test_acc
            best_state_dict = copy.deepcopy(model.state_dict())

        # ── Grokking detection ────────────────────────────────────────────────
        if not memorised and train_acc > 0.99:
            memorised = True
            tqdm.write(
                f"\n  ✓ Memorisation  ep {epoch:,}  "
                f"train={train_acc:.3f}  test={test_acc:.3f}  lr={cur_lr:.2e}"
            )

        if memorised and not grokked and test_acc > 0.99:
            grokked        = True
            grokking_epoch = epoch
            tqdm.write(
                f"\n🎉 GROKKING       ep {epoch:,}!  "
                f"train={train_acc:.3f}  test={test_acc:.3f}  lr={cur_lr:.2e}"
            )

        # ── Post-grokking LR reduction (one-shot) ─────────────────────────────
        # Once the model has found the generalising solution, drop the LR
        # further to help it stay there. We do this 200 epochs after
        # grokking to give it time to consolidate first.
        if grokked and not lr_reduced and epoch >= grokking_epoch + 200:
            for g in optimizer.param_groups:
                g["lr"] *= 0.2
            lr_reduced = True
            tqdm.write(
                f"\n  ↓ LR reduced to {optimizer.param_groups[0]['lr']:.2e} "
                f"at epoch {epoch:,} (post-grokking stabilisation)"
            )

        # ── Logging ───────────────────────────────────────────────────────────
        if epoch % cfg.log_every == 0:
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss / len(train_dl))
            history["test_loss"].append(test_loss   / len(test_dl))
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["lr"].append(cur_lr)

        if epoch % (cfg.log_every * 10) == 0:
            tqdm.write(
                f"  ep {epoch:6,} | "
                f"train {train_acc:.3f} / {train_loss/len(train_dl):.4f} | "
                f"test  {test_acc:.3f} / {test_loss/len(test_dl):.4f} | "
                f"lr {cur_lr:.2e}"
            )

    # ── End-of-training summary ───────────────────────────────────────────────
    if not grokked:
        tqdm.write(
            f"\n  ⚠  Grokking NOT detected within {cfg.n_epochs:,} epochs.  "
            f"Best test acc: {best_test_acc:.3f}"
        )
    else:
        tqdm.write(
            f"\n  ✓ Training complete.  "
            f"Grokking ep: {grokking_epoch:,}  "
            f"Best test acc: {best_test_acc:.3f}"
        )

    return history, grokking_epoch, best_state_dict


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(model_name: str, n_params: int, cfg) -> None:
    width = 62
    print(f"\n{'=' * width}")
    print(f"  Training  : {model_name}")
    print(f"  Params    : {n_params:,}")
    print(f"  Device    : {cfg.device}")
    print(f"  Epochs    : {cfg.n_epochs:,}")
    print(f"  LR        : {cfg.lr}  →  cosine  →  {cfg.lr * 1e-2:.1e}")
    print(f"  WD        : {cfg.weight_decay}")
    print(f"  Grad clip : 1.0")
    print(f"{'=' * width}")
