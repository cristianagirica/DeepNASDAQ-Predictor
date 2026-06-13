import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 30

df = pd.read_csv("../dataIn/features.csv")

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_5",
    "SMA_20",
    "MACD",
    "VOLATILITY"
]

X = df[features].values
y = df["TARGET"].values

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

X_seq = []
y_seq = []

for i in range(LOOKBACK, len(X)):

    X_seq.append(X[i - LOOKBACK:i])
    y_seq.append(y[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

np.save("../dataIn/train/X.npy", X_seq)
np.save("../dataIn/train/y.npy", y_seq)

print(X_seq.shape)
