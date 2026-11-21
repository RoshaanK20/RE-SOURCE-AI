from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X = X[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5, stratify=y)
scaler = StandardScaler().fit(X_train)
clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(scaler.transform(X_train), y_train)
print("accuracy", accuracy_score(y_test, clf.predict(scaler.transform(X_test))))
