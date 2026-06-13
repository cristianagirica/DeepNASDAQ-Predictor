import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import joblib
import pandas as pd

from nn import ModelTrainer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------
# LOAD DATA
# -----------------------
X_train = np.load("../dataIn/train/X_train.npy")
y_train = np.load("../dataIn/train/y_train.npy")

X_test = np.load("../dataIn/train/X_test.npy")
y_test = np.load("../dataIn/train/y_test.npy")

# -----------------------
# LOAD SCALER (NEW)
# -----------------------
scaler_y = joblib.load("../model/scaler_y.pkl")

# -----------------------
# MODEL
# -----------------------
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True,
                         input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

# -----------------------
# TRAINER
# -----------------------
trainer = ModelTrainer(
    model=model,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)
)

# -----------------------
# TRAINING
# -----------------------
trainer.fit(
    X_train, y_train,
    X_test, y_test,
    epochs=50,
    batch_size=64
)

# -----------------------
# RESTORE BEST MODEL
# -----------------------
trainer.restore_best_weights()

# -----------------------
# PREDICTION
# -----------------------
y_pred = trainer.predict(X_test)

# -----------------------
# INVERSE TRANSFORM (NEW)
# -----------------------
y_pred_real = scaler_y.inverse_transform(y_pred)
y_test_real = scaler_y.inverse_transform(y_test)

# -----------------------
# METRICS (REAL SCALE)
# -----------------------
print("\n===== RESULTS (REAL SCALE) =====")
print("MAE:", mean_absolute_error(y_test_real, y_pred_real))
print("RMSE:", np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
print("R2:", r2_score(y_test_real, y_pred_real))

# -----------------------
# SAVE CSV (NEW)
# -----------------------
os.makedirs("../dataOut", exist_ok=True)

df_out = pd.DataFrame({
    "Actual": y_test_real.flatten(),
    "Predicted": y_pred_real.flatten()
})

df_out.to_csv("../dataOut/predictions_real.csv", index=False)

# -----------------------
# PLOT (REAL SCALE) (UPDATED)
# -----------------------
os.makedirs("../dataOut/plot", exist_ok=True)

plt.figure(figsize=(12,5))
plt.plot(y_test_real, label="Actual")
plt.plot(y_pred_real, label="Predicted")
plt.title("Actual vs Predicted (Real Scale)")
plt.legend()

plt.savefig("../dataOut/plot/predictions_real.png")
plt.close()

# -----------------------
# SAVE LOSS PLOT
# -----------------------
trainer.save_loss_plot("../dataOut/plot/loss_curve.png")

# -----------------------
# SAVE MODEL
# -----------------------
trainer.save("../model/final_model.keras")