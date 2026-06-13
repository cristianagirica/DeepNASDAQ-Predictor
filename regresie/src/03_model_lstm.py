# -*- coding: utf-8 -*-
"""
ANTRENARE MODEL: LSTM
Modificat pentru a prelua seturile gata împărțite cronologic la Pasul 2,
asigurând că modelul învață din toate cele 5 companii aeriene.
"""

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- MODIFICARE CRITICĂ: Încărcăm direct fișierele de Train și Test generate la Pasul 2 ---
X_train = np.load("../dataIn/train/X_train.npy")
y_train = np.load("../dataIn/train/y_train.npy")
X_test = np.load("../dataIn/train/X_test.npy")
y_test = np.load("../dataIn/train/y_test.npy")

# Liniile vechi cu "split = int(len(X) * 0.8)" AU FOST ȘTERSE, deoarece datele vin deja separate!

# Verificare în consolă pentru a te asigura că dimensiunile sunt corecte
print(f"Date încărcate pentru antrenare LSTM:")
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")

# Structura rețelei rămâne neschimbată, dar input_shape va citi din noul X_train
model = Sequential()

model.add(
    LSTM(
        128,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2]) # Citim din X_train
    )
)

model.add(Dropout(0.2))

model.add(LSTM(64))

model.add(Dropout(0.2))

model.add(Dense(32, activation="relu"))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    patience=10,
    restore_best_weights=True
)

# Directorul pentru salvarea modelului
os.makedirs("../model/", exist_ok=True)

# Antrenarea modelului
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    # Folosim X_test și y_test direct ca date de validare în loc de validation_split.
    # Este mult mai corect academic deoarece validezi pe ultima perioadă din toate companiile!
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

model.save("../model/lstm_model.keras")
print("Modelul LSTM a fost salvat cu succes în '../model/lstm_model.keras'")