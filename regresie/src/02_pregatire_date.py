import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 30
CALE_FEATURES = "../dataIn/features.csv"
DIRECTOR_IEȘIRE = "../dataIn/train/"

os.makedirs(DIRECTOR_IEȘIRE, exist_ok=True)

df = pd.read_csv(CALE_FEATURES)

features = ["Open", "High", "Low", "Close", "Volume", "SMA_5", "SMA_20", "MACD", "VOLATILITY"]

X_train_total = []
y_train_total = []
X_test_total = []
y_test_total = []

for symbol, group in df.groupby("Symbol"):
    group = group.sort_values("Date").reset_index(drop=True)

    X_comp = group[features].values
    y_comp = group["TARGET"].values.reshape(-1, 1)

    if len(X_comp) < LOOKBACK:
        continue

    punct_split = int(len(X_comp) * 0.8)

    X_comp_train = X_comp[:punct_split]
    y_comp_train = y_comp[:punct_split]

    X_comp_test = X_comp[punct_split:]
    y_comp_test = y_comp[punct_split:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_comp_train)
    X_test_scaled = scaler_x.transform(X_comp_test)

    y_train_scaled = scaler_y.fit_transform(y_comp_train)
    y_test_scaled = scaler_y.transform(y_comp_test)

    for i in range(LOOKBACK, len(X_train_scaled)):
        X_train_total.append(X_train_scaled[i - LOOKBACK:i])
        y_train_total.append(y_train_scaled[i])

    X_comp_test_cu_istoric = np.vstack([X_train_scaled[-LOOKBACK:], X_test_scaled])
    y_comp_test_cu_istoric = np.vstack([y_train_scaled[-LOOKBACK:], y_test_scaled])

    for i in range(LOOKBACK, len(X_comp_test_cu_istoric)):
        X_test_total.append(X_comp_test_cu_istoric[i - LOOKBACK:i])
        y_test_total.append(y_comp_test_cu_istoric[i])

X_train = np.array(X_train_total)
y_train = np.array(y_train_total)
X_test = np.array(X_test_total)
y_test = np.array(y_test_total)

np.save(os.path.join(DIRECTOR_IEȘIRE, "X_train.npy"), X_train)
np.save(os.path.join(DIRECTOR_IEȘIRE, "y_train.npy"), y_train)
np.save(os.path.join(DIRECTOR_IEȘIRE, "X_test.npy"), X_test)
np.save(os.path.join(DIRECTOR_IEȘIRE, "y_test.npy"), y_test)

print("--- Pregătire date finalizată cu succes! ---")
print(f"Set Antrenare (Train) - X: {X_train.shape}, y: {y_train.shape}")
print(f"Set Testare (Test)    - X: {X_test.shape},  y: {y_test.shape}")
print("Fiecare dintre cele 5 companii are exact 20% din perioada sa recentă în setul de test.")