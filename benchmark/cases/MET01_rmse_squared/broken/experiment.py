from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
model = Ridge(alpha=0.5).fit(X_train, y_train)
pred = model.predict(X_test)
print("rmse", mean_squared_error(y_test, pred, squared=False))
print("r2", r2_score(y_test, pred))
