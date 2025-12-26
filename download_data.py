import yfinance as yf
import numpy as np


df = yf.download("SPY", start="2015-01-01", end="2025-01-01")
df = df.sort_index()
df.reset_index().to_csv("data/raw/SPY.csv", index=False)

df['returns'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
df['abs_returns'] = df['returns'].abs()
df['range'] = (df['High'] - df['Low']) / df['Close']
df['vol_chg'] = np.log(df['Volume']/df['Volume'].shift(1))
df['vol_5'] = df['returns'].rolling(5).std()
df['vol_20'] = df['returns'].rolling(20).std()
df['threshold'] = df['abs_returns'].rolling(60).quantile(0.75)
df['abs_ret_next'] = df['abs_returns'].shift(-1)
df['target'] = (df['abs_ret_next'] > df['threshold']).astype(int)
df["vol_60"] = df["returns"].rolling(60).std()
df["vol_ratio"] = df["vol_20"] / df["vol_60"]

del df['Close']
del df['Open']
del df['High']
del df['Low']
del df['Volume'] 
del df['threshold']
del df['abs_ret_next']  

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()
df.reset_index().to_csv("data/processed/SPY.csv", index=False)
print(df)
print(df['target'].value_counts(normalize=True))
