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
    raise FileNotFoundError("No CSV files found for training!")

df = pd.concat(df_list, ignore_index=True)

# Features
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ======================
# Lightweight MambaBlock (PyTorch only)
# ======================
class MambaBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.SiLU()  # modern nonlinearity

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = self.activation(self.fc1(x))
        g = torch.sigmoid(self.gate(x))
        h = h * g  # gated update
        out = self.fc2(h)
        return out

# ======================
# Model
# ======================
class TrajectoryMamba(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, seq_len=SEQ_LENGTH):
        super().__init__()
        self.mamba = MambaBlock(input_dim, hidden_dim)
        self.fc_out = nn.Linear(seq_len * input_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = self.mamba(x)
        h = h.reshape(h.size(0), -1)  # flatten sequence
        return self.fc_out(h)

# Instantiate
model = TrajectoryMamba(input_dim=len(features))
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
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), "coord_predictor_mamba.pth")
