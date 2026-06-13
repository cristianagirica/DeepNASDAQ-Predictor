import os
import matplotlib.pyplot as plt
import pandas as pd

cale_fisier = "../dataIn/NASDAQ_companii_aeriene_2001-2026.csv"

if os.path.exists(cale_fisier):
    df = pd.read_csv(cale_fisier)

    if {"Date", "Symbol", "Close"}.issubset(df.columns):
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values(by="Date")

        plt.figure(figsize=(14, 7))

        companii = df["Symbol"].unique()

        for companie in companii:
            df_comp = df[df["Symbol"] == companie]
            plt.plot(df_comp["Date"], df_comp["Close"], label=companie, linewidth=1.5)

        plt.title("Evoluția Prețului de Închidere și Ciclul de Viață al Companiilor Aeriene (2001 - 2026)", fontsize=14, fontweight="bold")
        plt.xlabel("Anul", fontsize=12)
        plt.ylabel("Preț de Închidere (USD)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Simbol Companie", fontsize=10)

        os.makedirs("../plots", exist_ok=True)
        plt.savefig("../plots/ciclu_viata_companii.png", dpi=300, bbox_inches="tight")
        plt.show()