# -*- coding: utf-8 -*-
"""
PASUL 6: Evaluarea Modelelor și Vizualizarea Individuală pe Noul Set de Companii
CORECȚIE EXPORT: Rezolvă problema salvării graficelor în directorul dedicat OUT_PLOT_DIR.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURĂRI DIRECTOARE ȘI PARAMETRI ---
LOOKBACK = 30
CALE_FEATURES = "../dataIn/features.csv"
OUT_CSV_DIR = "../dataOut/evaluare/"
OUT_PLOT_DIR = "../dataOut/evaluare/plot/"

# CORECTAT: Se creează ambele directoare în mod corect pe disc
os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(OUT_PLOT_DIR, exist_ok=True) # <-- Aici era OUT_CSV_DIR scris de două ori

# 1. ÎNCĂRCAREA MATRICILOR DE TEST SEPARATE LA PASUL 2
if not os.path.exists("../dataIn/train/X_test.npy"):
    raise FileNotFoundError("Nu s-a găsit X_test.npy! Asigură-te că ai rulat mai întâi scriptul 'Pasul 2' actualizat.")

X_test = np.load("../dataIn/train/X_test.npy")
y_test = np.load("../dataIn/train/y_test.npy").flatten()

# 2. RECUPERAREA SIMBOLURILOR PRIN SPLIT LOCAL PER COMPANIE
df_features = pd.read_csv(CALE_FEATURES)

simboluri_test_liste = []
for symbol, group in df_features.groupby("Symbol"):
    group_ordonat = group.sort_values("Date").reset_index(drop=True)

    if len(group_ordonat) < LOOKBACK:
        continue

    punct_split = int(len(group_ordonat) * 0.8)
    simboluri_test_liste.extend(group_ordonat["Symbol"].iloc[punct_split:].values)

simboluri_test = np.array(simboluri_test_liste)

if len(simboluri_test) != len(y_test):
    simboluri_test = simboluri_test[:len(y_test)]

companii_test = np.unique(simboluri_test)
print(f"Companii identificate cu succes în eșantionul de test: {companii_test}")

# 3. DEFINIREA PALETEI DE CULORI
paleta_culori = {
    "AAL": ("#ff9999", "#b30000"),
    "JBLU": ("#99ccff", "#0044cc"),
    "SNCY": ("#adebad", "#145214"),
    "UAL": ("#ffcc99", "#b35900"),
    "RJET": ("#e0b3ff", "#5c0099")
}
culori_rezerva = [("#d1d1e0", "#3d3d5c"), ("#ffb3e6", "#990066")]

# 4. EVALUAREA COMPARATIVĂ A MODELELOR
models = {
    "LSTM": "../model/lstm_model.keras",
    "BiLSTM": "../model/bilstm_model.keras",
    "CNN_LSTM": "../model/cnn_lstm_close.keras",
}

for name, path in models.items():
    if not os.path.exists(path):
        print(f"Atenție: Modelul {name} nu a fost găsit la calea {path}. Se sare peste el.")
        continue

    print(f"\n===== Evaluare Model: {name} =====")
    model = load_model(path)
    pred = model.predict(X_test).flatten()

    mae_global = mean_absolute_error(y_test, pred)
    rmse_global = np.sqrt(mean_squared_error(y_test, pred))
    r2_global = r2_score(y_test, pred)
    print(f"Global -> MAE: {mae_global:.5f} | RMSE: {rmse_global:.5f} | R2: {r2_global:.5f}")

    df_out = pd.DataFrame({
        "Symbol": simboluri_test,
        "Actual": y_test,
        "Predicted": pred
    })
    csv_path = os.path.join(OUT_CSV_DIR, f"{name}_predictions.csv")
    df_out.to_csv(csv_path, index=False)

    # 5. GENERAREA GRAFICELOR SEPARATE PENTRU FIECARE COMPANIE REALA
    idx_rezerva = 0
    for comp in companii_test:
        masca_comp = (simboluri_test == comp)

        y_act_comp = y_test[masca_comp]
        y_pred_comp = pred[masca_comp]

        if len(y_act_comp) == 0:
            continue

        mae_comp = mean_absolute_error(y_act_comp, y_pred_comp)
        r2_comp = r2_score(y_act_comp, y_pred_comp)

        if comp in paleta_culori:
            c_actual, c_pred = paleta_culori[comp]
        else:
            c_actual, c_pred = culori_rezerva[idx_rezerva % len(culori_rezerva)]
            idx_rezerva += 1

        plt.figure(figsize=(12, 6))

        eșantioane_axa_x = np.arange(1, len(y_act_comp) + 1)

        plt.plot(eșantioane_axa_x, y_act_comp, color=c_actual, linestyle='-', linewidth=2.2,
                 label=f"Real (Actual)")

        plt.plot(eșantioane_axa_x, y_pred_comp, color=c_pred, linestyle='--', linewidth=1.8,
                 label=f"Predicție {name}")

        plt.title(f"Model: {name} | Compania: {comp} - Actual vs Predicted (Set Test)", fontsize=13, fontweight='bold')
        plt.xlabel("Zile Secvențiale în Setul de Test (Ultima Perioadă)", fontsize=11)
        plt.ylabel("Preț Normalizat", fontsize=11)

        plt.text(0.02, 0.05, f"MAE local: {mae_comp:.4f}\nR² local: {r2_comp:.4f}",
                 transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        plt.legend(loc="upper left", shadow=True)
        plt.grid(True, linestyle="--", alpha=0.4)

        # CORECTAT: Schimbăm OUT_CSV_DIR cu OUT_PLOT_DIR pentru a salva imaginile în folderul corect
        nume_fisier_plot = f"{name}_{comp}_plot.png"
        plot_path = os.path.join(OUT_PLOT_DIR, nume_fisier_plot)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f" -> Grafic generat cu succes în folderul plot pentru {comp} [{name}]: {nume_fisier_plot}")

print(f"\n Toate cele 15 grafice individuale au fost create și salvate cu succes în '{OUT_PLOT_DIR}'!")