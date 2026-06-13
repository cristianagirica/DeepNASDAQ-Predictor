import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

class ModelTrainer:

    def __init__(self, model, loss="mse", optimizer="adam",
                 patience=10, min_delta=0.0):

        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer

        self.train_losses = []
        self.val_losses = []

        # BEST MODEL TRACKING
        self.best_val_loss = np.inf
        self.best_weights = None
        self.best_epoch = 0
        self.epoch_counter = 0

        # 🔥 EARLY STOPPING SETTINGS
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            loss = tf.keras.losses.MSE(y, pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return tf.reduce_mean(loss).numpy()

    def val_step(self, x, y):
        pred = self.model(x, training=False)
        loss = tf.keras.losses.MSE(y, pred)
        return tf.reduce_mean(loss).numpy()

    def fit(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        for epoch in range(epochs):

            self.epoch_counter += 1

            train_losses = []
            val_losses = []

            # TRAIN
            for x_batch, y_batch in train_ds:
                loss = self.train_step(x_batch, y_batch)
                train_losses.append(loss)

            # VAL
            for x_batch, y_batch in val_ds:
                loss = self.val_step(x_batch, y_batch)
                val_losses.append(loss)

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

            # -----------------------
            # BEST MODEL SAVE
            # -----------------------
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_weights = self.model.get_weights()
                self.best_epoch = self.epoch_counter
                self.wait = 0

                os.makedirs("../model", exist_ok=True)
                self.model.save("../model/best_model.keras")

            else:
                self.wait += 1

            # -----------------------
            # EARLY STOPPING
            # -----------------------
            if self.wait >= self.patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                print(f"Best epoch was {self.best_epoch} with val_loss={self.best_val_loss:.6f}")
                break

    def predict(self, X):
        return self.model(X, training=False).numpy()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def restore_best_weights(self):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

    # -----------------------
    # PLOT LOSS
    # -----------------------
    def save_loss_plot(self, path="../dataOut/plot/loss.png"):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        plt.figure(figsize=(10,5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")

        plt.title("Training vs Validation Loss")
        plt.legend()

        plt.savefig(path)
        plt.close()