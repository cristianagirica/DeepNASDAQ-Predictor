import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 30
ORIZONT_ZILE = 252
CALE_FEATURES = "../dataIn/features.csv"

OUT_DIR = "../dataOut/prognoza/"
OUT_PLOT_DIR = "../dataOut/prognoza/plot/"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_PLOT_DIR, exist_ok=True)

features = ["Open", "High", "Low", "Close", "Volume", "SMA_5", "SMA_20", "MACD", "VOLATILITY"]

culori_companii = {
    "AAL": "#b30000", "JBLU": "#0044cc", "SNCY": "#145214", "UAL": "#b35900", "RJET": "#5c0099"
}
culoare_fallback = "#3d3d5c"

dict_modele = {
    "LSTM": "../model/lstm_model.keras",
    "BiLSTM": "../model/bilstm_model.keras",
    "CNN_LSTM": "../model/cnn_lstm_close.keras"
}

if not os.path.exists(CALE_FEATURES):
    raise FileNotFoundError(f"Nu s-a găsit fișierul {CALE_FEATURES}.")

df = pd.read_csv(CALE_FEATURES)
companii = df["Symbol"].unique()
print(f"S-au identificat {len(companii)} companii: {companii}")

prognoze_globale = {nume: {comp: [] for comp in companii} for nume in dict_modele.keys()}

for comp in companii:
    print(f"\n⚙️ Pornire simulare dinamică pe 252 de pași pentru: {comp}")
    df_comp = df[df["Symbol"] == comp].sort_values("Date").reset_index(drop=True)

    date_istorice_recente = df_comp[features].copy().iloc[-50:].reset_index(drop=True)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(df_comp[features].values)
    scaler_y.fit(df_comp[["TARGET"]].values)

    for nume_model, cale_model in dict_modele.items():
        if not os.path.exists(cale_model):
            continue

        model = load_model(cale_model)

        istoric_simulare = date_istorice_recente.copy()
        predictii_scalate = []

        for zi in range(ORIZONT_ZILE):
            fereastra_30_zile_brute = istoric_simulare[features].iloc[-LOOKBACK:].values

            fereastra_scalata = scaler_x.transform(fereastra_30_zile_brute)

            input_retea = np.expand_dims(fereastra_scalata, axis=0)
            pred_close_scala = model.predict(input_retea, verbose=0)[0, 0]
            predictii_scalate.append(pred_close_scala)

            pred_close_usd = scaler_y.inverse_transform([[pred_close_scala]])[0, 0]

            pret_close_anterior = istoric_simulare["Close"].iloc[-1]

            rand_nou = {
                "Open": pret_close_anterior,
                "High": max(pret_close_anterior, pred_close_usd) * 1.005,
                "Low": min(pret_close_anterior, pred_close_usd) * 0.995,
                "Close": pred_close_usd,
                "Volume": istoric_simulare["Volume"].iloc[-5:].mean(),
                "SMA_5": 0.0,
                "SMA_20": 0.0,
                "MACD": 0.0,
                "VOLATILITY": 0.0
            }

            istoric_simulare = pd.concat([istoric_simulare, pd.DataFrame([rand_nou])], ignore_index=True)

            istoric_simulare["Return"] = istoric_simulare["Close"].pct_change()

            idx_ultim = istoric_simulare.index[-1]
            istoric_simulare.loc[idx_ultim, "SMA_5"] = istoric_simulare["Close"].iloc[-5:].mean()
            istoric_simulare.loc[idx_ultim, "SMA_20"] = istoric_simulare["Close"].iloc[-20:].mean()

            ema_12 = istoric_simulare["Close"].ewm(span=12, adjust=False).mean().iloc[-1]
            ema_26 = istoric_simulare["Close"].ewm(span=26, adjust=False).mean().iloc[-1]
            istoric_simulare.loc[idx_ultim, "MACD"] = ema_12 - ema_26

            istoric_simulare.loc[idx_ultim, "VOLATILITY"] = istoric_simulare["Return"].iloc[-20:].std()

        istoric_simulare.ffill(inplace=True)
        istoric_simulare.bfill(inplace=True)

        preturi_finale_usd = istoric_simulare["Close"].iloc[-ORIZONT_ZILE:].values
        prognoze_globale[nume_model][comp] = preturi_finale_usd

print("\n Prognozele dinamice s-au încheiat cu succes. Se generează graficele separate...")

zile_viitor = np.arange(1, ORIZONT_ZILE + 1)

for nume_model in dict_modele.keys():
    for comp in companii:
        valori_prognoza = prognoze_globale[nume_model][comp]
        if len(valori_prognoza) == 0:
            continue

        plt.figure(figsize=(12, 6))
        culoare_plot = culori_companii.get(comp, culoare_fallback)

        plt.plot(zile_viitor, valori_prognoza, color=culoare_plot, linestyle='-', linewidth=2.5,
                 label=f"Prognoză Iterativă Dinamică {comp}")

        plt.title(f"Prognoză 1 An În Viitor cu Recalculare Indicatori Tehnici \nModel: {nume_model} | Compania: {comp}",
                  fontsize=13, fontweight='bold')
        plt.xlabel("Zile în viitor", fontsize=11)
        plt.ylabel("Preț Estimat Acțiune (USD)", fontsize=11)
        plt.legend(loc="upper left", shadow=True)
        plt.grid(True, linestyle="--", alpha=0.4)

        nume_plot = f"Prognoza_Dinamica_{nume_model}_{comp}.png"
        plot_path = os.path.join(OUT_PLOT_DIR, nume_plot)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" -> Grafic prognoză salvat în folderul plot: {nume_plot}")

for nume_model in dict_modele.keys():
    df_export = pd.DataFrame(prognoze_globale[nume_model])
    df_export.insert(0, "Zi_Viitor", zile_viitor)
    df_export.to_csv(os.path.join(OUT_DIR, f"prognoza_dinamica_{nume_model}.csv"), index=False)

print("\n Toate graficele din subdirectorul plot/ și fișierele CSV de prognoză au fost scrise și salvate cu succes!")