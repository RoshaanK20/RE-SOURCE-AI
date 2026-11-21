import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

digits = load_digits()
X = digits.data.astype(float)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_train)
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=60, random_state=0)
mlp.fit(scaler.transform(X_train), y_train)
print("accuracy", accuracy_score(y_test, mlp.predict(scaler.transform(X_test))))
joblib.dump(mlp, "mlp.joblib")
