import numpy as np
import pandas as pd
import glob
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

# ======================
# Data Loading
# ======================
file_paths = glob.glob(r"C:\Users\Michele\dataset-research\dataset-research\Data Preprocessing\Oslo\processed_data\*.csv")
df_list = [pd.read_csv(file) for file in file_paths]

if not df_list:
    raise FileNotFoundError("No CSV files found!")

df = pd.concat(df_list, ignore_index=True)

features = ['lat', 'lon', 'alt', 'time']
data = df[features]

# Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
joblib.dump(scaler, "coord_scaler.pkl")

# Sequence creation
SEQ_LENGTH = 10
def create_sequences(data, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0,2,1)  # (batch, features, seq)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0,2,1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ======================
# Temporal Convolutional Network (TCN)
# ======================
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]  # crop to original seq length
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
# Instantiate model
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajectoryTCN(input_dim=len(features))
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ======================
# Training
# ======================
EPOCHS = 5
BATCH_SIZE = 32

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0.0
    
    for i in range(0, X_train.size(0), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x, batch_y = X_train[indices].to(device), y_train[indices].to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), "coord_predictor_tcn.pth")
