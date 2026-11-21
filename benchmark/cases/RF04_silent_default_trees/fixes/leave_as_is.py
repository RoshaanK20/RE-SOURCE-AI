from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
clf = RandomForestClassifier(random_state=7)
clf.fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, clf.predict(X_test)))
print("roc_auc", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
