import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
order = np.random.RandomState(1).permutation(len(y)).astype(int)
train, test = order[:100], order[100:]
clf = RandomForestClassifier(n_estimators=50, random_state=1).fit(X[train], y[train])
print("accuracy", accuracy_score(y[test], clf.predict(X[test])))
