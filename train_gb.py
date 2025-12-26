from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd

df = pd.read_csv("data/processed/SPY.csv")
df = df.sort_index()
df = df.dropna()

num_rows, num_cols = df.shape

split_index = int(0.8 * num_rows)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df.drop(columns=['target', 'Date'])
Y_train = train_df['target'].astype(int)
X_test  = test_df.drop(columns=['target', 'Date'])
Y_test = test_df['target'].astype(int)
                 
gb = HistGradientBoostingClassifier(random_state=0)
gb.fit(X_train, Y_train)

p_train_gb = gb.predict_proba(X_train)[:,1]
p_test_gb = gb.predict_proba(X_test)[:,1]

print("GB TEST ROC-AUC:", roc_auc_score(Y_test, p_test_gb))
#Does Not Help to improve the model