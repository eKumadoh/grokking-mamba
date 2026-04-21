"""
Modular arithmetic dataset: predict (a + b) mod p.

Input sequence  : [a,  '+',  b,  '=']   (4 tokens)
Prediction target: (a + b) % p           (at the '=' position)

Token vocabulary:
    0 .. p-1  →  numbers
    p         →  '+' operator token
    p+1       →  '=' token
"""

import torch
from torch.utils.data import Dataset, DataLoader


class ModularArithmeticDataset(Dataset):
    def __init__(
        self,
        p: int = 97,
        split: str = "train",
        train_frac: float = 0.3,
        seed: int = 42,
    ):
        self.p = p
        self.op_token = p       # '+'
        self.eq_token = p + 1   # '='

        # Build all p² pairs deterministically
        all_pairs = [(a, b) for a in range(p) for b in range(p)]

        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(all_pairs), generator=rng).tolist()

        n_train = int(len(all_pairs) * train_frac)
        indices = perm[:n_train] if split == "train" else perm[n_train:]
        self.pairs = [all_pairs[i] for i in indices]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        a, b = self.pairs[idx]
        x = torch.tensor([a, self.op_token, b, self.eq_token], dtype=torch.long)
        y = torch.tensor((a + b) % self.p, dtype=torch.long)
        return x, y


def get_dataloaders(cfg, num_workers: int = 0):
    """Return (train_dl, test_dl) DataLoaders."""
    train_ds = ModularArithmeticDataset(cfg.p, "train", cfg.train_frac, cfg.seed)
    test_ds  = ModularArithmeticDataset(cfg.p, "test",  cfg.train_frac, cfg.seed)

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(cfg.device == "cuda"),
    )
    test_dl = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(cfg.device == "cuda"),
    )
    return train_dl, test_dl


def get_full_tensors(cfg):
    """
    Return all data as tensors without DataLoader.
    Useful for collecting activations for SAE training.

    Returns: (train_x, train_y), (test_x, test_y)  — all on CPU.
    """
    def ds_to_tensors(split):
        ds = ModularArithmeticDataset(cfg.p, split, cfg.train_frac, cfg.seed)
        xs, ys = zip(*[ds[i] for i in range(len(ds))])
        return torch.stack(xs), torch.stack(ys)

    return ds_to_tensors("train"), ds_to_tensors("test")
