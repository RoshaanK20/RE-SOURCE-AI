import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.RandomState(0)
X[rng.rand(*X.shape) < 0.08] = np.nan          # the original dataset had missing values
X = SimpleImputer(strategy="mean").fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, clf.predict(X_test)))
