import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import glob
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout

# ==== Load CSV files ====
file_paths = glob.glob(r"C:\Users\Michele\dataset-research\dataset-research\Data Preprocessing\Oslo\processed_data\*.csv")

df_list = []
for file in file_paths:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# ==== Select features ====
features = ['lat', 'lon', 'alt', 'time']
data = df[features]

# ==== Scale ====
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
joblib.dump(scaler, "coord_scaler.pkl")

# ==== Sequence Creation ====
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  # predict next timestep
    return np.array(X), np.array(y)

SEQ_LENGTH = 10
X, y = create_sequences(data_scaled, SEQ_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ==== Transformer Block ====
def transformer_encoder(inputs, num_heads=4, ff_dim=64, dropout=0.1):
    # Multi-head self-attention
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention = Dropout(dropout)(attention)
    x = LayerNormalization(epsilon=1e-6)(inputs + attention)

    # Feed-forward network
    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn = Dropout(dropout)(ffn)

    return LayerNormalization(epsilon=1e-6)(x + ffn)

# ==== Build Transformer Model ====
inputs = Input(shape=(SEQ_LENGTH, len(features)))

x = transformer_encoder(inputs, num_heads=4, ff_dim=128)
x = transformer_encoder(x, num_heads=4, ff_dim=128)

x = tf.keras.layers.Flatten()(x)
x = Dense(64, activation="relu")(x)
outputs = Dense(len(features))(x)  # predict lat, lon, alt, time

model = Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse")
model.summary()

# ==== Train ====
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# ==== Save ====
model.save("coord_predictor_transformer.keras")
