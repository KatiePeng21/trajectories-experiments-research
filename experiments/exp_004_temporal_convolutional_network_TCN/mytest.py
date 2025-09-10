import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ======================
# Paths
# ======================
test_path_pattern = r"C:\Users\Michele\dataset-research\dataset-research\Data Preprocessing\Oslo\processed_data\TRAJ_4699.csv"
model_path = "coord_predictor_tcn.pth"
scaler_path = "coord_scaler.pkl"

# ======================
# Load test CSV
# ======================
file_paths = glob.glob(test_path_pattern)
df_list = [pd.read_csv(file) for file in file_paths]

if not df_list:
    raise FileNotFoundError("No test CSV files found!")

df_test = pd.concat(df_list, ignore_index=True)
features = ['lat', 'lon', 'alt', 'time']
data_test = df_test[features]

# ======================
# Load scaler and scale
# ======================
scaler = joblib.load(scaler_path)
data_scaled = scaler.transform(data_test)

# ======================
# Create sequences
# ======================
SEQ_LENGTH = 10
def create_sequences(data, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X_test, y_test = create_sequences(data_scaled, SEQ_LENGTH)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0,2,1)  # (batch, features, seq)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ======================
# Define TCN model (same as training)
# ======================
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.norm(out)
        return out

class TrajectoryTCN(nn.Module):
    def __init__(self, input_dim=4, seq_len=SEQ_LENGTH, hidden_dim=64):
        super().__init__()
        self.tcn1 = TCNBlock(input_dim, hidden_dim, kernel_size=3, dilation=1)
        self.tcn2 = TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.fc = nn.Linear(hidden_dim*seq_len, input_dim)

    def forward(self, x):
        h = self.tcn1(x)
        h = self.tcn2(h)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)

# ======================
# Load model
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajectoryTCN(input_dim=len(features))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ======================
# Make predictions
# ======================
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()

y_true = y_test.numpy()

# ======================
# Metrics
# ======================
mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
accuracy_percentage = 100 - mape
rmse = np.sqrt(np.mean((y_true - y_pred)**2))

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Approximate Accuracy: {accuracy_percentage:.2f}%")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# ======================
# 3D Trajectory Plot
# ======================
y_pred_unscaled = scaler.inverse_transform(y_pred)
y_true_unscaled = scaler.inverse_transform(y_true)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(y_true_unscaled[:,0], y_true_unscaled[:,1], y_true_unscaled[:,2], label="True Trajectory", color='blue')
ax.plot(y_pred_unscaled[:,0], y_pred_unscaled[:,1], y_pred_unscaled[:,2], label="Predicted Trajectory", color='red', linestyle='dashed')
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_zlabel("Altitude")
ax.set_title("Trajectory Prediction: True vs Predicted")
ax.legend()
plt.show()

# ======================
# Optional: first 10 predictions
# ======================
for i in range(10):
    print(f"Predicted: {y_pred[i]}, True: {y_true[i]}")
