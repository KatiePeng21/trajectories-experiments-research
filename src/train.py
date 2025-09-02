# src/train.py
from __future__ import annotations

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from src.data.dataset import make_loaders
from src.models.gru import GRUModel
from src.utils import ade_fde_geo_m, load_yaml, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    return p.parse_args()


def build_model(name: str, input_dim: int, target_dim: int, cfg: dict):
    name = name.upper()
    if name == "GRU":
        return GRUModel(
            input_dim=input_dim,
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            horizon=int(cfg["pred_horizon"]),
            target_dim=target_dim,
            dropout=float(cfg.get("dropout", 0.0)),
        )
    raise ValueError(f"Unknown model: {name}")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    feats = cfg["features"]
    targ  = cfg["target"]
    L     = int(cfg["seq_len"])
    H     = int(cfg["pred_horizon"])
    bs    = int(cfg["batch_size"])
    vr    = float(cfg.get("val_ratio", 0.1))

    train_loader, val_loader, feat_std, targ_std = make_loaders(
        cfg["dataset_path"], feats, targ, L, H, bs, vr, int(cfg.get("seed", 42))
    )

    # model/optim
    model = build_model(cfg.get("model", "GRU"), len(feats), len(targ), cfg).to(device)
    opt   = Adam(model.parameters(), lr=float(cfg["lr"]))
    loss_fn = nn.MSELoss()
    epochs  = int(cfg["epochs"])
    exp_dir = os.path.dirname(args.config)
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        # train
        model.train()
        tr_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            P = model(X)
            loss = loss_fn(P, Y)
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item() * X.size(0)
        tr_loss /= len(train_loader.dataset)

        # validate
        model.eval()
        va_loss = 0.0
        preds, truths = [], []
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                P = model(X)
                va_loss += loss_fn(P, Y).item() * X.size(0)
                preds.append(P.cpu())
                truths.append(Y.cpu())
        va_loss /= len(val_loader.dataset)

        # inverse-standardize to original units for metrics
        P = torch.cat(preds, 0).reshape(-1, H, len(targ))
        T = torch.cat(truths, 0).reshape(-1, H, len(targ))
        P_un = targ_std.inverse(P.view(-1, P.shape[-1])).view_as(P)
        T_un = targ_std.inverse(T.view(-1, T.shape[-1])).view_as(T)

        ade_m, fde_m = ade_fde_geo_m(P_un, T_un, order=("lat","lon","alt"))

        print(
            f"Epoch {ep:02d}/{epochs} | train_loss={tr_loss:.6f} "
            f"| val_loss={va_loss:.6f} | ADE(m)={ade_m:.2f} FDE(m)={fde_m:.2f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pt"))

    metrics = {"train_loss": float(tr_loss), "val_loss": float(va_loss), "ADE_m": float(ade_m), "FDE_m": float(fde_m)}
    save_json(metrics, os.path.join(exp_dir, "metrics.json"))

    # log a row
    runs_csv = os.path.join(os.path.dirname(exp_dir), "runs.csv")
    need_header = not os.path.exists(runs_csv)
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        os.path.basename(exp_dir),
        cfg.get("model","GRU").upper(),
        cfg.get("seed",42),
        epochs,
        bs,
        cfg["lr"],
        metrics["ADE_m"],
        metrics["FDE_m"],
        metrics["val_loss"],
        "",
    ]
    with open(runs_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["timestamp","exp_name","model","seed","epochs","batch","lr","ade_m","fde_m","mse","notes"])
        w.writerow(row)


if __name__ == "__main__":
    main()