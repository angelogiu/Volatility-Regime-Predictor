import yfinance as yf
import pandas as pd


df = yf.download("SPY", start="2015-01-01", end="2025-01-01")
df = df.sort_index()
df.reset_index().to_csv("data/raw/SPY.csv", index=False)

df['returns'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)

df.reset_index().to_csv("data/processed/SPY.csv", index=False)
print(df)