import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Data Loading 
DATA_DIR = Path("data/processed/Oslo")  
files = sorted(DATA_DIR.glob("TRAJ_*.csv"))
if not files:
    raise FileNotFoundError(f"No TRAJ_*.csv found in {DATA_DIR.resolve()}")

# Read & tag each file w/ a traj_id (build sequences per-trajectory)
dfs = []
for i, f in enumerate(files, start=1):
    df_i = pd.read_csv(f)
    df_i["traj_id"] = i
    dfs.append(df_i)

df = pd.concat(dfs, ignore_index=True)

features = ["lat", "lon", "alt", "time"]
data = df[features + ["traj_id"]].dropna().reset_index(drop=True)

# Scale + save scaler
scaler = StandardScaler()
scaled = scaler.fit_transform(data[features].to_numpy())
joblib.dump(scaler, "coord_scaler.pkl")  # overwrites previous scaler

# Put scaled back with traj ids for sequence creation
data_scaled = pd.DataFrame(scaled, columns=features)
data_scaled["traj_id"] = data["traj_id"].to_numpy()



# Sequence creation (per trajectory, no cross-file)
def create_sequences_per_traj(df_scaled: pd.DataFrame, seq_length: int = 10):
    """
    Build (X, y) windows independently for each trajectory.
    X: [N, seq_length, F], y: [N, F] where y is the next-step target.
    """
    X, y = [], []
    for _, g in df_scaled.groupby("traj_id", sort=True):
        g = g.reset_index(drop=True)
        arr = g[features].to_numpy(dtype=np.float32)
        if len(arr) <= seq_length:
            continue
        for s in range(0, len(arr) - seq_length):
            X.append(arr[s : s + seq_length])
            y.append(arr[s + seq_length])  # next step
    if not X:
        raise ValueError("No sequences produced. Check seq_length vs trajectory lengths.")
    return np.stack(X), np.stack(y)


SEQ_LENGTH = 10
X_np, y_np = create_sequences_per_traj(data_scaled, SEQ_LENGTH)

X = torch.tensor(X_np, dtype=torch.float32)  # [N, L, F]
y = torch.tensor(y_np, dtype=torch.float32)  # [N, F]



# Conditional Normalizing Flow Components
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.dim = dim
        self.scale_net = MLP(dim // 2 + cond_dim, dim // 2)
        self.shift_net = MLP(dim // 2 + cond_dim, dim // 2)

    def forward(self, x, cond, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        h = torch.cat([x1, cond], dim=1)

        scale = self.scale_net(h)
        shift = self.shift_net(h)

        if not reverse:  # forward
            y2 = x2 * torch.exp(scale) + shift
            log_det = scale.sum(dim=1)
        else:  # inverse (sampling)
            y2 = (x2 - shift) * torch.exp(-scale)
            log_det = -scale.sum(dim=1)

        y = torch.cat([x1, y2], dim=1)
        return y, log_det


class ConditionalFlow(nn.Module):
    def __init__(self, dim, cond_dim, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(dim, cond_dim) for _ in range(num_layers)])

    def forward(self, x, cond):
        log_det_total = 0
        for layer in self.layers:
            x, log_det = layer(x, cond, reverse=False)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z, cond):
        for layer in reversed(self.layers):
            z, _ = layer(z, cond, reverse=True)
        return z



# Model setup
feature_dim = len(features)
cond_dim = SEQ_LENGTH * feature_dim
flow = ConditionalFlow(dim=feature_dim, cond_dim=cond_dim, num_layers=6)

optimizer = optim.Adam(flow.parameters(), lr=1e-3)

device = "cuda" if torch.cuda.is_available() else "cpu"
flow = flow.to(device)



# Training
epochs = 30
batch_size = 64

# Precompute flattened conditioners on-device for speed
# X: [N, L, F] -> [N, L*F]
X_flat = X.reshape(X.size(0), -1).to(device)
y = y.to(device)

two_pi_log = float(np.log(2 * np.pi))  # small speedup, constant

for epoch in range(epochs):
    perm = torch.randperm(X.size(0))
    epoch_loss = 0.0

    for i in range(0, X.size(0), batch_size):
        idx = perm[i : i + batch_size]
        seq_batch = X_flat[idx]          # [B, L*F]
        next_batch = y[idx]              # [B, F]

        z, log_det = flow(next_batch, seq_batch)            # forward through flow
        log_prob = -0.5 * (z.pow(2).sum(dim=1) + feature_dim * two_pi_log)
        loss = -(log_prob + log_det).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * next_batch.size(0)

    epoch_loss /= X.size(0)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

# Overwrite previous model
torch.save(flow.state_dict(), "trajectory_flow.pth")
print("Training complete. Model and scaler overwritten.")
