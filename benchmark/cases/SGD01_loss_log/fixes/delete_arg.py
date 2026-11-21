from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_train)
clf = SGDClassifier(alpha=1e-4, max_iter=1000, tol=1e-3, random_state=0)
clf.fit(scaler.transform(X_train), y_train)
print("accuracy", accuracy_score(y_test, clf.predict(scaler.transform(X_test))))
