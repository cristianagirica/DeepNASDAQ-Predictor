import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping

X = np.load("../dataIn/train/X.npy")
y = np.load("../dataIn/train/y.npy")

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

model = Sequential()

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        input_shape=(
            X.shape[1],
            X.shape[2]
        )
    )
)

model.add(
    MaxPooling1D(
        pool_size=2
    )
)

model.add(
    LSTM(
        128
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(1)
)

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

model.save(
    "../model/cnn_lstm_close.keras"
)
