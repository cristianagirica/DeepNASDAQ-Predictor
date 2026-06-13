# -*- coding: utf-8 -*-
"""
ANTRENARE MODEL: BiLSTM (Bidirectional LSTM)
Modificat pentru a prelua seturile gata împărțite cronologic la Pasul 2,
asigurând antrenarea și validarea corectă pe toate cele 5 companii aeriene.
"""

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# --- CORECTIE: Încărcăm direct fișierele de Train și Test generate la Pasul 2 ---
X_train = np.load("../dataIn/train/X_train.npy")
y_train = np.load("../dataIn/train/y_train.npy")
X_test = np.load("../dataIn/train/X_test.npy")
y_test = np.load("../dataIn/train/y_test.npy")

# Liniile vechi cu "X[:split]" și "split = int(...)" au fost eliminate!

print(f"Date încărcate pentru antrenare BiLSTM:")
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")

model = Sequential()

model.add(
    Bidirectional(
        LSTM(128),
        # Modificat pentru a citi dimensiunile corecte din X_train
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dense(64, activation="relu"))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"] # Adăugat MAE ca metrică pentru a monitoriza performanța în textul proiectului
)

# Ne asigurăm că directorul pentru modele există
os.makedirs("../model/", exist_ok=True)

# Antrenarea modelului cu monitorizare pe setul de test
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64,
    # Adăugăm validation_data pentru a vedea performanța pe ultima perioadă din TOATE cele 5 companii
    validation_data=(X_test, y_test)
)

model.save("../model/bilstm_model.keras")
print("Modelul BiLSTM a fost salvat cu succes în '../model/bilstm_model.keras'")