import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent                # .../experiments/Trajectory_Conditional_Flow_Matching
ROOT_DIR = BASE_DIR.parents[1]                            # .../trajectories-experiments-research
MODEL_PATH  = BASE_DIR / "trajectory_flow.pth"
SCALER_PATH = BASE_DIR / "coord_scaler.pkl"
TEST_CSV    = ROOT_DIR / "data" / "processed" / "Oslo" / "TRAJ_2.csv"

# -----------------------
# Conditional Flow classes (same as training)
# -----------------------
class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class AffineCoupling(torch.nn.Module):
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
        if not reverse:
            y2 = x2 * torch.exp(scale) + shift
            log_det = scale.sum(dim=1)
        else:
            y2 = (x2 - shift) * torch.exp(-scale)
            log_det = -scale.sum(dim=1)
        y = torch.cat([x1, y2], dim=1)
        return y, log_det

class ConditionalFlow(torch.nn.Module):
    def __init__(self, dim, cond_dim, num_layers=6):
        super().__init__()
        self.layers = torch.nn.ModuleList([AffineCoupling(dim, cond_dim) for _ in range(num_layers)])
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
# Load model and scaler
# -----------------------
features = ['lat', 'lon', 'alt', 'time']
SEQ_LENGTH = 10
feature_dim = len(features)
cond_dim = SEQ_LENGTH * feature_dim

flow = ConditionalFlow(dim=feature_dim, cond_dim=cond_dim, num_layers=6)
flow.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
flow.eval()

scaler: StandardScaler = joblib.load(SCALER_PATH)

# -----------------------
# Prepare test sequence
# -----------------------
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  # next step
    return np.array(X), np.array(y)

test_df = pd.read_csv(TEST_CSV)
test_data = test_df[features].dropna()
test_scaled = scaler.transform(test_data)

X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# -----------------------
# Generate predictions
# -----------------------
def predict_next(flow, seq, num_samples=5):
    flow.eval()
    with torch.no_grad():
        cond = seq.reshape(1, -1)                 # safer than .view(...)
        z = torch.randn(num_samples, feature_dim)
        samples = flow.inverse(z, cond.repeat(num_samples, 1))
        samples = scaler.inverse_transform(samples.numpy())
    return samples

test_seq = X_test[0]
samples = predict_next(flow, test_seq, num_samples=10)

print("Generated Next-Step Predictions (lat, lon, alt, time):")
print(samples)

true_next = scaler.inverse_transform(y_test[0].unsqueeze(0).numpy())
mean_pred = samples.mean(axis=0, keepdims=True)

# -----------------------
# Calculate MSE and percentage error
# -----------------------
mse = np.mean((true_next - mean_pred) ** 2)
with np.errstate(divide='ignore', invalid='ignore'):
    percentage_error = np.abs((true_next - mean_pred) / np.where(true_next != 0, true_next, np.nan)) * 100
avg_percentage_error = np.nanmean(percentage_error)

print("Ground Truth Next Step:", true_next)
print("Mean Prediction:", mean_pred)
print("MSE:", mse)
print("Percentage Error per Feature (%):", percentage_error)
print("Average Percentage Error (%):", avg_percentage_error)