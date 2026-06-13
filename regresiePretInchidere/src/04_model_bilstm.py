import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

X = np.load("../dataIn/train/X.npy")
y = np.load("../dataIn/train/y.npy")

split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]

model = Sequential()

model.add(
    Bidirectional(
        LSTM(128),
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(Dense(64, activation="relu"))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64
)

model.save("../model/bilstm_model.keras")