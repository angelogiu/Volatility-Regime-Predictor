from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

import pandas as pd

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


y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred, digits=4))
print("ROC-AUC:", roc_auc_score(Y_test, y_proba))



