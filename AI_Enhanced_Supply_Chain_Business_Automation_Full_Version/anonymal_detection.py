import numpy as np
import pandas as pd
from preprocessing import processdata
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,GRU,Dropout,RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

def split_data(data,target):
  x=data.drop(target)
  y=data[target]
  return x,y

def create_sequences(values, time_steps=30):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : i + time_steps])
    return np.stack(output)

TIME_STEPS = 30
sequences = create_sequences(data['scaled'].values, TIME_STEPS)

model_2 = Sequential([
    LSTM(64, activation='relu', input_shape=(TIME_STEPS, 1), return_sequences=False),
    RepeatVector(TIME_STEPS),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
])

model_2.compile(optimizer='adam', loss='mse')
X_train = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
model.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
X_pred = model.predict(X_train)
mse = np.mean(np.power(X_train - X_pred, 2), axis=(1, 2))

# Set threshold for anomaly
threshold = np.percentile(mse, 95)
print(f"Anomaly threshold: {threshold:.6f}")

# Anomalies
anomalies = mse > threshold

input_dim = X_scaled.shape[1]
encoding_dim = input_dim // 2

# Define architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Reconstruct the data
X_pred = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

# Set threshold (95th percentile of MSE)
threshold = np.percentile(mse, 95)
print(f"Anomaly Threshold: {threshold:.6f}")

# Identify anomalies
data['reconstruction_error'] = mse
data['anomaly'] = data['reconstruction_error'] > threshold
