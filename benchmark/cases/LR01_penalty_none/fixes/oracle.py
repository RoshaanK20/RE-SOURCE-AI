from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

X, y = load_breast_cancer(return_X_y=True)
X = X[:, :10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
clf = LogisticRegression(penalty=None, max_iter=5000)
clf.fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, clf.predict(X_test)))
print("log_loss", log_loss(y_test, clf.predict_proba(X_test)))
