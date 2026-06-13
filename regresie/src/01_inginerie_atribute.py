import pandas as pd

df = pd.read_csv("../dataIn/NASDAQ_companii_aeriene_2001-2026.csv")

df["Date"] = pd.to_datetime(df["Date"])

rezultat = []

for symbol, group in df.groupby("Symbol"):

    group = group.sort_values("Date")

    group["Return"] = group["Close"].pct_change()

    group["SMA_5"] = group["Close"].rolling(5).mean()
    group["SMA_20"] = group["Close"].rolling(20).mean()
    group["SMA_50"] = group["Close"].rolling(50).mean()

    group["EMA_12"] = group["Close"].ewm(span=12).mean()
    group["EMA_26"] = group["Close"].ewm(span=26).mean()

    group["MACD"] = group["EMA_12"] - group["EMA_26"]

    rolling_std = group["Close"].rolling(20).std()

    group["BB_UPPER"] = group["SMA_20"] + 2 * rolling_std
    group["BB_LOWER"] = group["SMA_20"] - 2 * rolling_std

    group["VOLATILITY"] = group["Return"].rolling(20).std()

    group["TARGET"] = group["Close"].shift(-1)

    rezultat.append(group)

df_final = pd.concat(rezultat)

df_final.dropna(inplace=True)

df_final.to_csv("../dataIn/features.csv", index=False)

print("Features generate.")
