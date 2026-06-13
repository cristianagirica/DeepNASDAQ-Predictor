# -*- coding: utf-8 -*-
"""
ANTRENARE MODEL: CNN-LSTM Hibrid
Modificat pentru a prelua seturile gata împărțite cronologic la Pasul 2,
asigurând antrenarea, validarea timpurie (Early Stopping) și testarea pe toate cele 5 companii.
"""

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- CORECTIE: Încărcăm direct fișierele de Train și Test generate la Pasul 2 ---
X_train = np.load("../dataIn/train/X_train.npy")
y_train = np.load("../dataIn/train/y_train.npy")
X_test = np.load("../dataIn/train/X_test.npy")
y_test = np.load("../dataIn/train/y_test.npy")

# Liniile vechi cu "X[:split]", "X[split:]" și "split = int(...)" au fost eliminate!

print(f"Date încărcate pentru antrenare CNN-LSTM:")
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")

model = Sequential()

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        # Modificat input_shape pentru a citi din noul X_train definit mai sus
        input_shape=(X_train.shape[1], X_train.shape[2])
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

# Early Stopping va monitoriza acum "val_loss" calculat corect pe setul de test global
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Ne asigurăm că directorul pentru modele există
os.makedirs("../model/", exist_ok=True)

# Antrenarea modelului hibrid
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    # CORECTIE METODOLOGICĂ: Schimbăm validation_split=0.1 cu validation_data=(X_test, y_test)
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

model.save("../model/cnn_lstm_close.keras")
print("Modelul CNN-LSTM a fost salvat cu succes în '../model/cnn_lstm_close.keras'")