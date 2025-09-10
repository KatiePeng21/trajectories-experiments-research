import joblib
import glob
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# ==== Paths ====
test_path_pattern = r"C:\Users\Michele\dataset-research\dataset-research\Data Preprocessing\Oslo\processed_data\TRAJ_853.csv"

# ==== Load test CSVs ====
file_paths = glob.glob(test_path_pattern)
df_list = [pd.read_csv(file) for file in file_paths]

if not df_list:
    raise FileNotFoundError("No test CSV files found!")

df_test = pd.concat(df_list, ignore_index=True)

# ==== Select features ====
features = ['lat', 'lon', 'alt', 'time']
data_test = df_test[features]

# ==== Load scaler and scale ====
scaler = joblib.load("coord_scaler.pkl")
data_scaled = scaler.transform(data_test)

# ==== Sequence creation ====
SEQ_LENGTH = 10
def create_sequences(data, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X_test, y_test = create_sequences(data_scaled, SEQ_LENGTH)

# ==== Load trained model ====
model = load_model("coord_predictor_transformer.keras")

# ==== Make predictions ====
y_pred = model.predict(X_test)

# ==== Calculate MAPE as a percentage ====
mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-6))) * 100
accuracy_percentage = 100 - mape

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Approximate Accuracy: {accuracy_percentage:.2f}%")

# ==== Optional: display first 10 predictions vs true ====
for i in range(10):
    print(f"Pred: {y_pred[i]}, True: {y_test[i]}")

# ==== Compare Trajectories - Plot ====
# Unscale predictions and true values
y_pred_unscaled = scaler.inverse_transform(y_pred)
y_test_unscaled = scaler.inverse_transform(y_test)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Plot actual trajectory
ax.plot(y_test_unscaled[:,0], y_test_unscaled[:,1], y_test_unscaled[:,2], 
        label='Actual Trajectory', color='blue')

# Plot predicted trajectory
ax.plot(y_pred_unscaled[:,0], y_pred_unscaled[:,1], y_pred_unscaled[:,2], 
        label='Predicted Trajectory', color='red', linestyle='dashed')

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Altitude')
ax.set_title('Predicted vs Actual Trajectory')
ax.legend()
plt.show()
