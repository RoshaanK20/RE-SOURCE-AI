import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21, stratify=y)
scaler = StandardScaler().fit(X_train)
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
results = pd.DataFrame(columns=["hidden", "accuracy", "auc"])
for hidden in [(16,), (32, 16)]:
    mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=400, random_state=21).fit(Xtr, y_train)
    row = {"hidden": str(hidden), "accuracy": accuracy_score(y_test, mlp.predict(Xte)),
           "auc": roc_auc_score(y_test, mlp.predict_proba(Xte)[:, 1])}
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
print(results.to_string(index=False))
