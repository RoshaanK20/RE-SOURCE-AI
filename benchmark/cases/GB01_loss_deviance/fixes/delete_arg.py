from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11, stratify=y)
gb = GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=2, random_state=11)
gb.fit(X_train, y_train)
pred = gb.predict(X_test)
print("accuracy", accuracy_score(y_test, pred))
print("f1", f1_score(y_test, pred))
