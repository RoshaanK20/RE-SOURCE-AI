import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train, test in kf.split(X):
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[train], y[train])
    scores.append(accuracy_score(y[test], tree.predict(X[test])))
print("cv_accuracy", float(np.mean(scores)))
