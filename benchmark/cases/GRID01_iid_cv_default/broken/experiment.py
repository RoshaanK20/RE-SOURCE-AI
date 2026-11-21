from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6, stratify=y)
scaler = StandardScaler().fit(X_train)
search = GridSearchCV(SVC(), {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}, iid=False)
search.fit(scaler.transform(X_train), y_train)
print("best", search.best_params_)
print("accuracy", accuracy_score(y_test, search.predict(scaler.transform(X_test))))
