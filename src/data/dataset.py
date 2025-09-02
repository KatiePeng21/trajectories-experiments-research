# src/data/dataset.py
from __future__ import annotations

import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import Standardizer


def _read_table(path: str) -> pd.DataFrame:
    if os.path.isfile(path):
        return pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "TRAJ_*.csv")))
        if not files:
            raise FileNotFoundError(f"No TRAJ_*.csv in {path}")
        dfs = []
        for i, f in enumerate(files, start=1):
            df = pd.read_csv(f)
            if "traj_id" not in df.columns:
                df["traj_id"] = i
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    raise FileNotFoundError(path)


def _windows(df: pd.DataFrame, features: List[str], target: List[str], seq_len: int, horizon: int):
    if "traj_id" not in df.columns:
        df = df.assign(traj_id=1)
    if "time" not in df.columns:
        raise ValueError("Expected a 'time' column for ordering.")

    xs, ys = [], []
    for _, g in df.sort_values(["traj_id", "time"]).groupby("traj_id", sort=False):
        g = g.reset_index(drop=True)
        F = g[features].to_numpy(dtype=np.float32)
        T = g[target].to_numpy(dtype=np.float32)
        n = len(g)
        limit = n - seq_len - horizon + 1
        for s in range(max(0, limit)):
            xs.append(F[s : s + seq_len])
            ys.append(T[s + seq_len : s + seq_len + horizon])
    if not xs:
        raise ValueError("No windows produced. Adjust seq_len/horizon or check data.")
    return np.stack(xs), np.stack(ys)  # X:[N,L,F], Y:[N,H,T]


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        feat_std: Standardizer,
        targ_std: Standardizer,
        indices: np.ndarray,
    ):
        Xs = feat_std.transform(X.view(-1, X.shape[-1])).view_as(X)
        Ys = targ_std.transform(Y.view(-1, Y.shape[-1])).view_as(Y)
        self.X = Xs[indices]
        self.Y = Ys[indices]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def _split_indices(n: int, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    v = int(round(n * val_ratio))
    return idx[v:], idx[:v]


def make_loaders(
    path: str,
    features: List[str],
    target: List[str],
    seq_len: int,
    horizon: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
):
    # Build once to get arrays
    X_np, Y_np = _windows(_read_table(path), features, target, seq_len, horizon)
    X = torch.from_numpy(X_np)  # [N,L,F]
    Y = torch.from_numpy(Y_np)  # [N,H,T]

    # Fit standardizers on the TRAIN portion only (to avoid leakage)
    n = X.shape[0]
    tr_idx, va_idx = _split_indices(n, val_ratio, seed)
    feat_std = Standardizer.fit(X[tr_idx].reshape(-1, X.shape[-1]))
    targ_std = Standardizer.fit(Y[tr_idx].reshape(-1, Y.shape[-1]))

    train_ds = TrajectoryDataset(X, Y, feat_std, targ_std, tr_idx)
    val_ds   = TrajectoryDataset(X, Y, feat_std, targ_std, va_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, feat_std, targ_std