import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from nn import ModelTrainer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train = np.load("../dataIn/train/X_train.npy")
y_train = np.load("../dataIn/train/y_train.npy")

X_test = np.load("../dataIn/train/X_test.npy")
y_test = np.load("../dataIn/train/y_test.npy")

scaler_y = joblib.load("../util/scaler_y.pkl")

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),

    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

trainer = ModelTrainer(
    model=model,
    model_name="cnn_lstm_model",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)
)

trainer.fit(
    X_train, y_train,
    X_test, y_test,
    epochs=50,
    batch_size=64
)

trainer.restore_best_weights()

y_pred = trainer.predict(X_test)

y_pred_real = scaler_y.inverse_transform(y_pred)
y_test_real = scaler_y.inverse_transform(y_test)

print("\n===== RESULTS CNN-LSTM (REAL SCALE) =====")
print("MAE:", mean_absolute_error(y_test_real, y_pred_real))
print("RMSE:", np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
print("R2:", r2_score(y_test_real, y_pred_real))

os.makedirs("../dataOut", exist_ok=True)

df_out = pd.DataFrame({
    "Actual": y_test_real.flatten(),
    "Predicted": y_pred_real.flatten()
})

df_out.to_csv("../dataOut/cnn_lstm_predictions_real.csv", index=False)

os.makedirs("../dataOut/plot", exist_ok=True)

plt.figure(figsize=(12, 5))
plt.plot(y_test_real, label="Actual", color="#adebad", linewidth=2)
plt.plot(y_pred_real, label="Predicted CNN-LSTM", color="#145214", linestyle="--")
plt.title("CNN-LSTM Model: Actual vs Predicted (Real Scale)")
plt.legend()

plt.savefig("../dataOut/plot/cnn_lstm_predictions_real.png")
plt.close()

trainer.save_loss_plot("../dataOut/plot/cnn_lstm_loss_curve.png")

trainer.save()