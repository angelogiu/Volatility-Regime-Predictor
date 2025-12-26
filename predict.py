import pandas as pd
import joblib

THRESHOLD = 0.49

pipe = joblib.load("models/lr_vol_model.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

df = pd.read_csv("data/processed/SPY.csv").dropna()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

latest = df.iloc[-1]
latest_date = latest["Date"]

X_today = latest[feature_cols].to_frame().T

prob = float(pipe.predict_proba(X_today)[0, 1])
pred = int(prob >= THRESHOLD)

print(f"Latest date in processed data: {latest_date.date()}")
print(f"P(high volatility tomorrow) = {prob:.4f}")
print(f"Decision threshold = {THRESHOLD:.2f}")
print("Prediction:", "High Volatility" if pred else "LOW Volatility")
