import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
import glob

# -----------------------
# Data Loading (last 80 files only)
# -----------------------
all_files = glob.glob(r"C:\Users\Michele\dataset-research\Data Preprocessing\Oslo\processed_data\*.csv")
file_paths = all_files[-80:]  # only last 80 files

df_list = [pd.read_csv(file) for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

features = ['lat', 'lon', 'alt', 'time']
data = df[features].dropna()

# Scale and overwrite old scaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
joblib.dump(scaler, "coord_scaler.pkl")  # Overwrites previous scaler

# -----------------------
# Sequence creation
# -----------------------
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  # next step
    return np.array(X), np.array(y)

SEQ_LENGTH = 10
X, y = create_sequences(data_scaled, SEQ_LENGTH)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# -----------------------
# Conditional Normalizing Flow Components
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),  # larger hidden layer
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class AffineCoupling(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.dim = dim
        self.scale_net = MLP(dim//2 + cond_dim, dim//2)
        self.shift_net = MLP(dim//2 + cond_dim, dim//2)

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
    def __init__(self, dim, cond_dim, num_layers=6):  # more flow layers
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

# -----------------------
# Model setup
# -----------------------
feature_dim = len(features)
cond_dim = SEQ_LENGTH * feature_dim
flow = ConditionalFlow(dim=feature_dim, cond_dim=cond_dim, num_layers=6)

optimizer = optim.Adam(flow.parameters(), lr=1e-3)

# -----------------------
# Training
# -----------------------
epochs = 30  # larger network, more epochs
batch_size = 64

for epoch in range(epochs):
    perm = torch.randperm(X.size(0))
    epoch_loss = 0

    for i in range(0, X.size(0), batch_size):
        idx = perm[i:i+batch_size]
        seq_batch = X[idx].view(-1, cond_dim)
        next_batch = y[idx]

        # base Gaussian
        z, log_det = flow(next_batch, seq_batch)
        log_prob = -0.5 * (z**2 + np.log(2*np.pi)).sum(dim=1)
        loss = -(log_prob + log_det).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Overwrite previous model
torch.save(flow.state_dict(), "trajectory_flow.pth")
print("Training complete. Model and scaler overwritten.")
