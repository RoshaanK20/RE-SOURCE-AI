from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
reg = RandomForestRegressor(n_estimators=60, criterion="mse", max_depth=6, random_state=3)
reg.fit(X_train, y_train)
print("mse", mean_squared_error(y_test, reg.predict(X_test)))
