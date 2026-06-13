import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.load("../dataIn/train/X.npy")
y = np.load("../dataIn/train/y.npy")

split = int(len(X) * 0.8)

X_test = X[split:]
y_test = y[split:]

out_csv_dir = "../dataOut/"
plot_dir = "../dataOut/plot/"

os.makedirs(out_csv_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

models = {
    "LSTM": "../model/lstm_model.keras",
    "BiLSTM": "../model/bilstm_model.keras",
    "CNN_LSTM": "../model/cnn_lstm_close.keras",
}

for name, path in models.items():

    print(f"\n===== {name} =====")

    model = load_model(path)
    pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    df = pd.DataFrame({
        "Actual": y_test.flatten(),
        "Predicted": pred
    })

    csv_path = os.path.join(out_csv_dir, f"{name}_predictions.csv")
    df.to_csv(csv_path, index=False)

    plt.figure()

    plt.plot(y_test, label="Actual")
    plt.plot(pred, label="Predicted")

    plt.title(f"{name} - Actual vs Predicted")
    plt.legend()

    plot_path = os.path.join(plot_dir, f"{name}_plot.png")
    plt.savefig(plot_path)

    plt.close()