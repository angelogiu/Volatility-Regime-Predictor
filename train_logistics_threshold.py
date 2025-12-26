from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score

import pandas as pd
import numpy as np


df = pd.read_csv("data/processed/SPY.csv")
df = df.sort_index()
df = df.dropna()

num_rows, num_cols = df.shape

split_index = int(0.8 * num_rows)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df.drop(columns=['target', 'Date'])
Y_train = train_df['target']
X_test  = test_df.drop(columns=['target', 'Date'])
Y_test = test_df['target']

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="liblinear", class_weight="balanced", random_state=0))
])
pipe.fit(X_train, Y_train)


p_train = pipe.predict_proba(X_train)[:, 1]
p_test = pipe.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.05, 0.95, 91)

rows = []

for t in thresholds:
    yhat = (p_train >= t).astype(int)
    prec = precision_score(Y_train, yhat, zero_division=0)
    rec = recall_score(Y_train, yhat, zero_division=0)
    f1 = f1_score(Y_train, yhat, zero_division=0)
    rows.append((t, prec, rec, f1))

rows = sorted(rows, key=lambda x: x[3], reverse=True)
best_t, best_prec, best_rec, best_f1 = rows[0]

print("Best threshold on TRAIN (by F1):", best_t)
print("Train precision/recall/F1 at best threshold:", best_prec, best_rec, best_f1)

y_pred = (p_test >= best_t).astype(int)

print("TEST Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred, digits=4))
print("TEST ROC-AUC:", roc_auc_score(Y_test, p_test))
