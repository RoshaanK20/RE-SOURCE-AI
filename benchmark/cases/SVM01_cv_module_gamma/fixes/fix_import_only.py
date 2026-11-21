from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
svm = SVC(C=10.0)
svm.fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, svm.predict(X_test)))
