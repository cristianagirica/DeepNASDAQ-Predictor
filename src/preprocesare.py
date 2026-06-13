import os
import glob
import pandas as pd

companii_tinta = {"AAL", "JBLU", "SNCY", "UAL", "RJET"}
cale_baza = "../dataIn"

toate_datele = []

foldere_anuale = glob.glob(os.path.join(cale_baza, "NASDAQ_*"))

for folder in sorted(foldere_anuale):
    if os.path.isdir(folder):
        fisiere_csv = glob.glob(os.path.join(folder, "NASDAQ_*.csv"))

        for fisier in sorted(fisiere_csv):
            try:
                df = pd.read_csv(fisier)

                if "Symbol" in df.columns:
                    df_filtrat = df[df["Symbol"].isin(companii_tinta)].copy()

                    if not df_filtrat.empty:
                        toate_datele.append(df_filtrat)
            except Exception:
                continue

if toate_datele:
    df_final = pd.concat(toate_datele, ignore_index=True)
    df_final.to_csv("../dataIn/NASDAQ_companii_aeriene_2001-2026.csv", index=False)