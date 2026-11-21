from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9, stratify=y)
tree = DecisionTreeClassifier(max_depth=3, random_state=9)
tree.fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, tree.predict(X_test)))
