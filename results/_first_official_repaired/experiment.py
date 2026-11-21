from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
reg = RandomForestRegressor(n_estimators=150, max_features=1.0, min_samples_leaf=3, random_state=1)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print("r2", r2_score(y_test, pred))
print("mae", mean_absolute_error(y_test, pred))
