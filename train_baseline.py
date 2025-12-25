import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/SPY.csv")
df = df.dropna()

num_rows, num_cols = df.shape

split_index = int(0.8 * num_rows)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df.drop(columns=['target'])
Y_train = train_df['target']
X_test = test_df.drop(columns=['target'])
Y_test = test_df['target']

print("Train rows:", len(train_df), "Test rows:", len(test_df))
if 'Date' in df.columns:
    print("Train dates:", train_df['Date'].min(), "->", train_df['Date'].max())
    print("Test dates :", test_df['Date'].min(), "->", test_df['Date'].max())
print("Test target rate:", Y_test.mean())

y_pred0 = np.zeros(len(Y_test), dtype=int)

print("Baseline A where its always 0")
print(confusion_matrix(Y_test, y_pred0))
print(classification_report(Y_test, y_pred0, digits=4))

cutoff = X_train['vol_20'].median()
y_pred_rule = (X_test['vol_20'] > cutoff).astype(int)

print("Baseline B where vol_20 > median(train)")
print("cutoff:", cutoff)
print(confusion_matrix(Y_test, y_pred_rule))
print(classification_report(Y_test, y_pred_rule, digits=4))