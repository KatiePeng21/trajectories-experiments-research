# src/visualization.py
import argparse, os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import numpy as np

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    p = argparse.ArgumentParser("Pred vs GT trajectory visualization (meters, with history)")
    # experiment bits
    p.add_argument("--config", required=True, help="Path to experiment config.yaml")
    p.add_argument("--checkpoint", required=False, help="Path to model checkpoint (.pt)")
    p.add_argument("--save_dir", required=True, help="Directory to save PNGs")
    p.add_argument("--num_examples", type=int, default=12)
    p.add_argument("--split", choices=["train","val","test"], default="val")

    # make_loaders required args (matches your signature)
    p.add_argument("--data_path", required=True, help="Path to dataset file (e.g., data/processed/oslo.parquet)")
    p.add_argument("--features", nargs="+", required=True, help="Feature names, e.g. lon lat [alt time]")
    p.add_argument("--target",   nargs="+", required=True, help="Target names, e.g. lon lat")
    p.add_argument("--seq_len",  type=int, required=True, help="History length (L)")
    p.add_argument("--horizon",  type=int, required=True, help="Prediction horizon (H)")
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--val_ratio",  type=float, required=True)
    p.add_argument("--seed",       type=int, required=True)

    # model hyperparams (optional overrides)
    p.add_argument("--hidden_size", type=int, help="GRU hidden size")
    p.add_argument("--num_layers",  type=int, help="GRU num layers")
    p.add_argument("--dropout",     type=float, help="Dropout")
    return p.parse_args()

def build_loaders(args):
    from src.data.dataset import make_loaders
    out = make_loaders(
        args.data_path,
        args.features,
        args.target,
        args.seq_len,
        args.horizon,
        args.batch_size,
        args.val_ratio,
        args.seed,
    )
    if len(out) == 4:
        train_loader, val_loader, feat_std, targ_std = out
        test_loader = None
    elif len(out) == 5:
        train_loader, val_loader, feat_std, targ_std, test_loader = out
    else:
        raise RuntimeError(f"Unexpected make_loaders return length {len(out)}")
    return train_loader, val_loader, test_loader, feat_std, targ_std

def maybe_build_model(args):
    try:
        import torch
        from src.models.gru import GRUModel
        cfg = load_yaml(args.config) if os.path.isfile(args.config) else {}
        mcfg = (cfg.get("model") or {}) if isinstance(cfg, dict) else {}
        hidden_size = args.hidden_size if args.hidden_size is not None else mcfg.get("hidden_size", 256)
        num_layers  = args.num_layers  if args.num_layers  is not None else mcfg.get("num_layers", 1)
        dropout     = args.dropout     if args.dropout     is not None else float(mcfg.get("dropout", 0.0))
        model = GRUModel(
            input_dim=len(args.features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=args.horizon,
            target_dim=len(args.target),
            dropout=dropout,
        )
        model.eval()
        if args.checkpoint and os.path.isfile(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
            try:
                model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[viz] load_state_dict non-strict: {e}", file=sys.stderr)
        return model
    except Exception as e:
        print(f"[viz] could not build model: {e}", file=sys.stderr)
        return None

def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return x

def iter_sequences(loader, max_count):
    """Yield up to max_count (X_seq, Y_seq). X_seq:[L,F], Y_seq:[H,T]"""
    count = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            X, Y = batch[0], batch[1]
        elif isinstance(batch, dict):
            X, Y = batch.get("X"), batch.get("Y")
        else:
            continue
        Xn, Yn = to_numpy(X), to_numpy(Y)
        B = Xn.shape[0]
        for i in range(B):
            yield Xn[i], Yn[i]
            count += 1
            if count >= max_count:
                return

def rollout_pred(model, X_seq):
    """Forward once: model returns [1,H,T] → [H,T]."""
    if model is None:
        return None
    try:
        import torch
        with torch.no_grad():
            out = model(torch.tensor(X_seq).unsqueeze(0).float())
        return to_numpy(out).squeeze(0)
    except Exception as e:
        print(f"[viz] model forward failed: {e}", file=sys.stderr)
        return None

def unnormalize(arr, scaler):
    """Undo normalization using sklearn-like scaler (supports [N,D] or [T,D])."""
    if arr is None or scaler is None:
        return arr
    try:
        return scaler.inverse_transform(arr)
    except Exception:
        try:
            return arr * scaler.scale_ + scaler.mean_
        except Exception:
            return arr

def lonlat_to_xy_m(lon, lat, lon0, lat0):
    """
    Fast local equirectangular projection near (lon0,lat0):
    x ~ meters east, y ~ meters north
    """
    # constants: meters per degree
    m_per_deg_lat = 110540.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat0))
    x = (lon - lon0) * m_per_deg_lon
    y = (lat - lat0) * m_per_deg_lat
    return x, y

def plot_trip(history_deg, gt_deg, pred_deg, save_path):
    """
    history_deg:[L,2], gt_deg:[H,2], pred_deg:[H,2]
    Project all to meters centered at the last history point so overlays look sensible.
    """
    # center at last history point
    lon0, lat0 = history_deg[-1, 0], history_deg[-1, 1]

    hx, hy = lonlat_to_xy_m(history_deg[:,0], history_deg[:,1], lon0, lat0)
    gx, gy = lonlat_to_xy_m(gt_deg[:,0],      gt_deg[:,1],      lon0, lat0)
    px = py = None
    if pred_deg is not None:
        px, py = lonlat_to_xy_m(pred_deg[:,0], pred_deg[:,1], lon0, lat0)

    plt.figure(figsize=(6,6))
    # history in faint line
    plt.plot(hx, hy, marker="o", linewidth=1.0, alpha=0.6, label="History")
    # ground truth future
    plt.plot(gx, gy, marker="o", linewidth=2.0, label="GT")
    # prediction
    if pred_deg is not None:
        plt.plot(px, py, marker="x", linewidth=2.0, label="Pred")

    plt.axis("equal"); plt.xlabel("meters east"); plt.ylabel("meters north")
    plt.legend(); plt.tight_layout()
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[viz] saved {save_path}")

def main():
    args = parse_args()
    train_loader, val_loader, test_loader, feat_std, targ_std = build_loaders(args)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}.get(args.split) or val_loader or train_loader
    if loader is None:
        raise SystemExit("[viz] no dataloader available")

    model = maybe_build_model(args)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    n = 0
    for X_seq, Y_seq in iter_sequences(loader, args.num_examples):
        # split out lon/lat columns from features/targets
        # We assume target is [lon, lat] (two dims).
        # Build history lon/lat from the corresponding positions in X_seq.
        try:
            lon_idx = args.features.index("lon")
            lat_idx = args.features.index("lat")
        except ValueError:
            raise SystemExit("Features must include 'lon' and 'lat' to plot trajectories.")

        # UNNORMALIZE X, Y, and pred using appropriate scalers
        X_un = X_seq
        Y_un = Y_seq
        if feat_std is not None:
            X_un = unnormalize(X_seq, feat_std)
        if targ_std is not None:
            Y_un = unnormalize(Y_seq, targ_std)

        pred_seq = rollout_pred(model, X_seq)
        if pred_seq is not None and targ_std is not None:
            pred_un = unnormalize(pred_seq, targ_std)
        else:
            pred_un = pred_seq

        # build arrays [L,2], [H,2], [H,2] in degrees
        history_deg = np.stack([X_un[:, lon_idx], X_un[:, lat_idx]], axis=1)
        gt_deg      = np.stack([Y_un[:, 0],      Y_un[:, 1]],      axis=1)  # target assumed [lon,lat] order
        pred_deg    = None
        if pred_un is not None:
            pred_deg = np.stack([pred_un[:, 0], pred_un[:, 1]], axis=1)

        # debug print for the first example
        if n == 0:
            print("[viz] debug first sample:")
            print("  history_deg head:", history_deg[:3])
            print("  gt_deg head:", gt_deg[:3])
            if pred_deg is not None:
                print("  pred_deg head:", pred_deg[:3])

        out_path = os.path.join(args.save_dir, f"traj_{n:04d}.png")
        plot_trip(history_deg, gt_deg, pred_deg, out_path)
        n += 1

    if n == 0:
        print("[viz] loader yielded zero sequences", file=sys.stderr)

if __name__ == "__main__":
    main()
