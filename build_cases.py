"""Build the RE-SOURCE benchmark folders from compact definitions.

Each case folder gets:
  reference/experiment.py   the original experiment's methodology, runnable today (ground truth)
  broken/experiment.py      the same experiment written against old library APIs (what a user has)
  fixes/oracle.py           minimal methodology-preserving repair of broken/
  fixes/<name>.py           alternative repairs, each labelled "equivalent" or "changes_method"
  case.json                 metadata

Every derived script is produced by exact text replacements, and the build fails if a
replacement does not apply, so the scripts can't silently drift apart.
Run:  python benchmark/build_cases.py
"""
from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "cases"
CASES: list[dict] = []


def sub(text: str, *pairs: tuple[str, str]) -> str:
    for old, new in pairs:
        if old not in text:
            raise ValueError(f"replacement target not found: {old!r}")
        text = text.replace(old, new, 1)
    return text


def case(id, family, dataset, category, description, reference, broken, oracle, alternatives=(), notes=""):
    CASES.append(dict(id=id, family=family, dataset=dataset, category=category, description=description,
                      reference=textwrap.dedent(reference).lstrip(), broken=broken, oracle=oracle,
                      alternatives=list(alternatives), notes=notes))


def alt(name, label, text, why):
    return dict(name=name, label=label, text=text, why=why)


# =====================================================================================
# RF00 — the friend's original bug, written the normal way
R = """
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("accuracy", accuracy_score(y_test, pred))
print("macro_f1", f1_score(y_test, pred, average="macro"))
"""
B = sub(textwrap.dedent(R).lstrip(), ("n_estimators=200", "n_estimator=200"))
case("RF00_param_typo", "RandomForest", "iris", "crash",
     "Typo in a constructor argument (n_estimator).", R, B,
     sub(B, ("n_estimator=200", "n_estimators=200")),
     [alt("delete_arg", "changes_method", sub(B, ("n_estimator=200, ", "")),
          "Deleting the argument silently uses the default 100 trees instead of 200.")])

# RF01 — classifier max_features='auto' (== 'sqrt')
R = """
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=0)
clf.fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, clf.predict(X_test)))
"""
B = sub(textwrap.dedent(R).lstrip(), ('max_features="sqrt"', 'max_features="auto"'))
case("RF01_max_features_auto_clf", "RandomForest", "wine", "crash",
     "max_features='auto' removed in scikit-learn 1.3 (classifier: meant 'sqrt').", R, B,
     sub(B, ('max_features="auto"', 'max_features="sqrt"')),
     [alt("delete_arg", "equivalent", sub(B, (', max_features="auto"', "")),
          "Classifier default is 'sqrt', identical to old 'auto'."),
      alt("use_all_features", "changes_method", sub(B, ('max_features="auto"', "max_features=None")),
          "None uses all 13 features per split instead of sqrt(13).")])

# RF02 — REGRESSOR max_features='auto' (== 1.0) : the classic trap
R = """
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
"""
B = sub(textwrap.dedent(R).lstrip(), ("max_features=1.0", 'max_features="auto"'))
case("RF02_max_features_auto_reg", "RandomForest", "diabetes", "crash+trap",
     "max_features='auto' removed; for REGRESSORS it meant 1.0 (all features), not 'sqrt'.", R, B,
     sub(B, ('max_features="auto"', "max_features=1.0")),
     [alt("sqrt_like_classifier", "changes_method", sub(B, ('max_features="auto"', 'max_features="sqrt"')),
          "'sqrt' is what 'auto' meant for classifiers; for a regressor it changes the model."),
      alt("delete_arg", "equivalent", sub(B, ('max_features="auto", ', "")), "Regressor default is 1.0."),
      alt("none", "equivalent", sub(B, ('max_features="auto"', "max_features=None")),
          "None and 1.0 both mean all features.")])

# RF03 — criterion='mse'
R = """
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
reg = RandomForestRegressor(n_estimators=60, criterion="squared_error", max_depth=6, random_state=3)
reg.fit(X_train, y_train)
print("mse", mean_squared_error(y_test, reg.predict(X_test)))
"""
B = sub(textwrap.dedent(R).lstrip(), ('criterion="squared_error"', 'criterion="mse"'))
case("RF03_criterion_mse", "RandomForest", "diabetes", "crash",
     "criterion='mse' renamed to 'squared_error' (scikit-learn 1.2).", R, B,
     sub(B, ('criterion="mse"', 'criterion="squared_error"')),
     [alt("delete_arg", "equivalent", sub(B, ('criterion="mse", ', "")), "Default criterion is squared_error."),
      alt("absolute_error", "changes_method", sub(B, ('criterion="mse"', 'criterion="absolute_error"')),
          "A different split criterion (loss).")])

# RF04 — SILENT: old default n_estimators=10 (scikit-learn < 0.22). Broken code RUNS.
R = """
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
# Original study ran on scikit-learn 0.19, where the default was n_estimators=10.
clf = RandomForestClassifier(n_estimators=10, random_state=7)
clf.fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, clf.predict(X_test)))
print("roc_auc", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
"""
B = sub(textwrap.dedent(R).lstrip(), ("# Original study ran on scikit-learn 0.19, where the default was n_estimators=10.\n", ""),
        ("n_estimators=10, ", ""))
case("RF04_silent_default_trees", "RandomForest", "breast_cancer", "silent_drift",
     "Code from the scikit-learn 0.19 era relies on the default n_estimators (10 then, 100 now). "
     "It runs today without any error but trains a different model.", R, B,
     sub(B, ("RandomForestClassifier(random_state=7)", "RandomForestClassifier(n_estimators=10, random_state=7)")),
     [alt("leave_as_is", "changes_method", B, "Runs fine, but trains 100 trees instead of 10.")],
     notes="The reference trace stands for the original environment's effective parameters.")

# LR01 — penalty='none'
R = """
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
"""
B = sub(textwrap.dedent(R).lstrip(), ("penalty=None", 'penalty="none"'))
case("LR01_penalty_none", "LogisticRegression", "breast_cancer", "crash+trap",
     "penalty='none' (string) removed; must become penalty=None.", R, B,
     sub(B, ('penalty="none"', "penalty=None")),
     [alt("delete_arg", "changes_method", sub(B, ('penalty="none", ', "")),
          "Default penalty is 'l2': turns an unregularised model into a regularised one.")])

# LR02 — multi_class='ovr'
R = """
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X = X[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5, stratify=y)
scaler = StandardScaler().fit(X_train)
clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000))
clf.fit(scaler.transform(X_train), y_train)
print("accuracy", accuracy_score(y_test, clf.predict(scaler.transform(X_test))))
"""
B = sub(textwrap.dedent(R).lstrip(), ("from sklearn.multiclass import OneVsRestClassifier\n", ""),
        ("OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000))",
         'LogisticRegression(C=1.0, max_iter=1000, multi_class="ovr")'))
case("LR02_multi_class_ovr", "LogisticRegression", "iris", "crash+trap",
     "LogisticRegression(multi_class='ovr') removed; one-vs-rest must be kept explicitly.", R, B,
     sub(B, ("from sklearn.linear_model import LogisticRegression\n",
             "from sklearn.linear_model import LogisticRegression\nfrom sklearn.multiclass import OneVsRestClassifier\n"),
         ('LogisticRegression(C=1.0, max_iter=1000, multi_class="ovr")',
          "OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000))")),
     [alt("delete_arg", "changes_method", sub(B, (', multi_class="ovr"', "")),
          "Silently switches one-vs-rest to multinomial logistic regression.")])

# SGD01 — loss='log'
R = """
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_train)
clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=0)
clf.fit(scaler.transform(X_train), y_train)
print("accuracy", accuracy_score(y_test, clf.predict(scaler.transform(X_test))))
"""
B = sub(textwrap.dedent(R).lstrip(), ('loss="log_loss"', 'loss="log"'))
case("SGD01_loss_log", "SGD", "digits", "crash+trap",
     "SGDClassifier(loss='log') renamed to 'log_loss'.", R, B,
     sub(B, ('loss="log"', 'loss="log_loss"')),
     [alt("delete_arg", "changes_method", sub(B, ('loss="log", ', "")),
          "Default loss is 'hinge': a linear SVM instead of logistic regression.")])

# GB01 — loss='deviance'
R = """
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11, stratify=y)
gb = GradientBoostingClassifier(loss="log_loss", n_estimators=80, learning_rate=0.1, max_depth=2, random_state=11)
gb.fit(X_train, y_train)
pred = gb.predict(X_test)
print("accuracy", accuracy_score(y_test, pred))
print("f1", f1_score(y_test, pred))
"""
B = sub(textwrap.dedent(R).lstrip(), ('loss="log_loss"', 'loss="deviance"'))
case("GB01_loss_deviance", "GradientBoosting", "breast_cancer", "crash",
     "GradientBoosting loss='deviance' renamed to 'log_loss'.", R, B,
     sub(B, ('loss="deviance"', 'loss="log_loss"')),
     [alt("delete_arg", "equivalent", sub(B, ('loss="deviance", ', "")), "Default is log_loss."),
      alt("exponential", "changes_method", sub(B, ('loss="deviance"', 'loss="exponential"')),
          "Exponential (AdaBoost) loss is a different model.")])

# ADA01 — base_estimator + algorithm (two errors, one after the other)
R = """
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=50, random_state=2)
ada.fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, ada.predict(X_test)))
"""
B = sub(textwrap.dedent(R).lstrip(), ("AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=50, random_state=2)",
                                      'AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=50, algorithm="SAMME", random_state=2)'))
case("ADA01_base_estimator", "AdaBoost", "wine", "crash+trap",
     "base_estimator renamed to estimator, and algorithm= removed (two consecutive errors).", R, B,
     sub(B, ("base_estimator=", "estimator="), (' algorithm="SAMME",', "")),
     [alt("delete_both", "changes_method",
          sub(B, ("base_estimator=DecisionTreeClassifier(max_depth=2), ", ""), (' algorithm="SAMME",', "")),
          "Without the inner model AdaBoost uses depth-1 stumps instead of depth-2 trees.")])

# KM01 — algorithm='full' (crash) + n_init default drift (silent)
R = """
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

X, y = make_blobs(n_samples=600, centers=8, cluster_std=2.2, random_state=7)
km = KMeans(n_clusters=8, algorithm="lloyd", n_init=10, random_state=0)
labels = km.fit_predict(X)
print("ari", adjusted_rand_score(y, labels))
print("inertia", km.inertia_)
"""
B = sub(textwrap.dedent(R).lstrip(), ('algorithm="lloyd", n_init=10', 'algorithm="full"'))
case("KM01_kmeans_full_ninit", "KMeans", "blobs", "crash+silent_drift",
     "algorithm='full' removed (crash) AND the default n_init changed from 10 to 'auto' (silent).", R, B,
     sub(B, ('algorithm="full"', 'algorithm="lloyd", n_init=10')),
     [alt("fix_crash_only", "changes_method", sub(B, ('algorithm="full"', 'algorithm="lloyd"')),
          "Runs, but k-means++ now restarts once instead of 10 times."),
      alt("delete_arg", "changes_method", sub(B, (', algorithm="full"', "")), "Same hidden n_init change.")])

# SVM01 — removed module (crash) + gamma default drift (silent)
R = """
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
svm = SVC(C=10.0, gamma="auto")
svm.fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, svm.predict(X_test)))
"""
B = sub(textwrap.dedent(R).lstrip(), ("from sklearn.model_selection import", "from sklearn.cross_validation import"),
        ('SVC(C=10.0, gamma="auto")', "SVC(C=10.0)"))
case("SVM01_cv_module_gamma", "SVM", "breast_cancer", "crash+silent_drift",
     "sklearn.cross_validation removed (crash) AND SVC's default gamma changed from 'auto' to 'scale' (silent).", R, B,
     sub(B, ("from sklearn.cross_validation import", "from sklearn.model_selection import"),
         ("SVC(C=10.0)", 'SVC(C=10.0, gamma="auto")')),
     [alt("fix_import_only", "changes_method", sub(B, ("from sklearn.cross_validation import", "from sklearn.model_selection import")),
          "Runs, but gamma='scale' is a different kernel width.")])

# CV01 — old KFold API
R = """
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
kf = KFold(n_splits=5)
scores = []
for train, test in kf.split(X):
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X[train], y[train])
    scores.append(accuracy_score(y[test], tree.predict(X[test])))
print("cv_accuracy", float(np.mean(scores)))
"""
B = sub(textwrap.dedent(R).lstrip(), ("from sklearn.model_selection import KFold", "from sklearn.cross_validation import KFold"),
        ("kf = KFold(n_splits=5)", "kf = KFold(len(y), n_folds=5)"), ("in kf.split(X):", "in kf:"))
case("CV01_old_kfold", "DecisionTree", "breast_cancer", "crash+trap",
     "Pre-0.20 KFold(n, n_folds=k) API from sklearn.cross_validation.", R, B,
     sub(B, ("from sklearn.cross_validation import KFold", "from sklearn.model_selection import KFold"),
         ("kf = KFold(len(y), n_folds=5)", "kf = KFold(n_splits=5)"), ("in kf:", "in kf.split(X):")),
     [alt("add_shuffle", "changes_method",
          sub(B, ("from sklearn.cross_validation import KFold", "from sklearn.model_selection import KFold"),
              ("kf = KFold(len(y), n_folds=5)", "kf = KFold(n_splits=5, shuffle=True, random_state=42)"),
              ("in kf:", "in kf.split(X):")),
          "Shuffled folds are different folds (old KFold did not shuffle).")])

# IMP01 — removed Imputer; trap = impute before splitting (leakage)
R = """
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
rng = np.random.RandomState(0)
X[rng.rand(*X.shape) < 0.08] = np.nan          # the original dataset had missing values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
imp = SimpleImputer(strategy="mean")
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, clf.predict(X_test)))
"""
B = sub(textwrap.dedent(R).lstrip(), ("from sklearn.impute import SimpleImputer", "from sklearn.preprocessing import Imputer"),
        ('SimpleImputer(strategy="mean")', 'Imputer(strategy="mean", axis=0)'))
case("IMP01_imputer", "RandomForest", "breast_cancer", "crash+trap",
     "sklearn.preprocessing.Imputer removed; a careless fix imputes before splitting (leakage).", R, B,
     sub(B, ("from sklearn.preprocessing import Imputer", "from sklearn.impute import SimpleImputer"),
         ('Imputer(strategy="mean", axis=0)', 'SimpleImputer(strategy="mean")')),
     [alt("impute_before_split", "changes_method",
          sub(B, ("from sklearn.preprocessing import Imputer", "from sklearn.impute import SimpleImputer"),
              ("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)\n"
               'imp = Imputer(strategy="mean", axis=0)\nX_train = imp.fit_transform(X_train)\nX_test = imp.transform(X_test)\n',
               'X = SimpleImputer(strategy="mean").fit_transform(X)\n'
               "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)\n")),
          "Imputation means computed on all rows, including the test set (leakage)."),
      alt("median", "changes_method",
          sub(B, ("from sklearn.preprocessing import Imputer", "from sklearn.impute import SimpleImputer"),
              ('Imputer(strategy="mean", axis=0)', 'SimpleImputer(strategy="median")')),
          "Different imputation strategy.")])

# MET01 — squared=False removed
R = """
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
model = Ridge(alpha=0.5).fit(X_train, y_train)
pred = model.predict(X_test)
print("rmse", root_mean_squared_error(y_test, pred))
print("r2", r2_score(y_test, pred))
"""
B = sub(textwrap.dedent(R).lstrip(), ("from sklearn.metrics import root_mean_squared_error, r2_score",
                                      "from sklearn.metrics import mean_squared_error, r2_score"),
        ("root_mean_squared_error(y_test, pred)", "mean_squared_error(y_test, pred, squared=False)"))
case("MET01_rmse_squared", "Ridge", "diabetes", "crash+trap",
     "mean_squared_error(squared=False) removed in scikit-learn 1.6.", R, B,
     sub(B, ("from sklearn.metrics import mean_squared_error, r2_score", "from sklearn.metrics import root_mean_squared_error, r2_score"),
         ("mean_squared_error(y_test, pred, squared=False)", "root_mean_squared_error(y_test, pred)")),
     [alt("delete_squared", "changes_method", sub(B, (", squared=False", "")),
          "Reports MSE under the RMSE label."),
      alt("sqrt_of_mse", "equivalent",
          sub(B, ("import train_test_split\n", "import train_test_split\nimport numpy as np\n"),
              ("mean_squared_error(y_test, pred, squared=False)", "np.sqrt(mean_squared_error(y_test, pred))")),
          "sqrt(MSE) is exactly RMSE (the square root happens outside scikit-learn).")])

# ANN01 — MLP (ANN) with removed imports; trap = raise max_iter to silence ConvergenceWarning
R = """
import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()
X = digits.data.astype(np.float64)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_train)
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=60, random_state=0)
mlp.fit(scaler.transform(X_train), y_train)
print("accuracy", accuracy_score(y_test, mlp.predict(scaler.transform(X_test))))
joblib.dump(mlp, "mlp.joblib")
"""
B = sub(textwrap.dedent(R).lstrip(), ("import joblib\nimport numpy as np\n", "import numpy as np\n"),
        ("from sklearn.metrics import accuracy_score", "from sklearn.metrics import accuracy_score\nfrom sklearn.externals import joblib"),
        ("astype(np.float64)", "astype(np.float)"))
case("ANN01_mlp_imports", "ANN (MLP)", "digits", "crash+trap",
     "MLP on digits: sklearn.externals.joblib and np.float removed. The model also emits a "
     "ConvergenceWarning, tempting a repair agent to change max_iter.", R, B,
     sub(B, ("from sklearn.externals import joblib", "import joblib"), ("astype(np.float)", "astype(np.float64)")),
     [alt("builtin_float", "equivalent",
          sub(B, ("from sklearn.externals import joblib", "import joblib"), ("astype(np.float)", "astype(float)")),
          "np.float was an alias of Python float (float64)."),
      alt("raise_max_iter", "changes_method",
          sub(B, ("from sklearn.externals import joblib", "import joblib"), ("astype(np.float)", "astype(np.float64)"),
              ("max_iter=60", "max_iter=500")),
          "Trains the network much longer than the original study."),
      alt("float32", "changes_method",
          sub(B, ("from sklearn.externals import joblib", "import joblib"), ("astype(np.float)", "astype(np.float32)")),
          "Lower precision input data.")])

# ANN02 — MLP with pandas DataFrame.append in result logging (benign)
R = """
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21, stratify=y)
scaler = StandardScaler().fit(X_train)
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
results = pd.DataFrame(columns=["hidden", "accuracy", "auc"])
for hidden in [(16,), (32, 16)]:
    mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=400, random_state=21).fit(Xtr, y_train)
    row = {"hidden": str(hidden), "accuracy": accuracy_score(y_test, mlp.predict(Xte)),
           "auc": roc_auc_score(y_test, mlp.predict_proba(Xte)[:, 1])}
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
print(results.to_string(index=False))
"""
B = sub(textwrap.dedent(R).lstrip(), ("results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)",
                                      "results = results.append(row, ignore_index=True)"))
case("ANN02_mlp_pandas_append", "ANN (MLP)", "breast_cancer", "crash",
     "DataFrame.append removed in pandas 2.0 (only affects result logging).", R, B,
     sub(B, ("results = results.append(row, ignore_index=True)",
             "results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)")),
     [alt("loc_insert", "equivalent", sub(B, ("results = results.append(row, ignore_index=True)",
                                              "results.loc[len(results)] = row")), "Same table, different pandas idiom.")])

# TREE01 — presort (benign)
R = """
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9, stratify=y)
tree = DecisionTreeClassifier(max_depth=3, random_state=9)
tree.fit(X_train, y_train)
print("accuracy", accuracy_score(y_test, tree.predict(X_test)))
"""
B = sub(textwrap.dedent(R).lstrip(), ("DecisionTreeClassifier(max_depth=3,", "DecisionTreeClassifier(presort=True, max_depth=3,"))
case("TREE01_presort", "DecisionTree", "iris", "crash",
     "presort= removed (it only affected speed).", R, B,
     sub(B, ("presort=True, ", "")), [])

# AGG01 — affinity renamed to metric
R = """
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.8, random_state=3)
agg = AgglomerativeClustering(n_clusters=4, metric="manhattan", linkage="average")
labels = agg.fit_predict(X)
print("ari", adjusted_rand_score(y, labels))
"""
B = sub(textwrap.dedent(R).lstrip(), ('metric="manhattan"', 'affinity="manhattan"'))
case("AGG01_affinity", "Agglomerative", "blobs", "crash+trap",
     "AgglomerativeClustering(affinity=) renamed to metric=.", R, B,
     sub(B, ('affinity="manhattan"', 'metric="manhattan"')),
     [alt("delete_arg", "changes_method", sub(B, ('affinity="manhattan", ', "")),
          "Default metric is euclidean, not manhattan.")])

# GRID01 — iid removed (crash) + default cv 3 -> 5 (silent)
R = """
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6, stratify=y)
scaler = StandardScaler().fit(X_train)
search = GridSearchCV(SVC(), {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}, cv=3)
search.fit(scaler.transform(X_train), y_train)
print("best", search.best_params_)
print("accuracy", accuracy_score(y_test, search.predict(scaler.transform(X_test))))
"""
B = sub(textwrap.dedent(R).lstrip(), ("cv=3)", "iid=False)"))
case("GRID01_iid_cv_default", "SVM+GridSearch", "wine", "crash+silent_drift",
     "GridSearchCV(iid=) removed (crash) AND default cv changed from 3 to 5 folds (silent).", R, B,
     sub(B, ("iid=False)", "cv=3)")),
     [alt("delete_iid_only", "changes_method", sub(B, (", iid=False)", ")")),
          "Runs, but hyper-parameters are now selected with 5-fold instead of 3-fold CV.")])

# NP01 — np.int (benign)
R = """
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
order = np.random.RandomState(1).permutation(len(y)).astype(int)
train, test = order[:100], order[100:]
clf = RandomForestClassifier(n_estimators=50, random_state=1).fit(X[train], y[train])
print("accuracy", accuracy_score(y[test], clf.predict(X[test])))
"""
B = sub(textwrap.dedent(R).lstrip(), ("astype(int)", "astype(np.int)"))
case("NP01_np_int", "RandomForest", "iris", "crash",
     "np.int alias removed in NumPy 1.24.", R, B,
     sub(B, ("astype(np.int)", "astype(int)")),
     [alt("int64", "equivalent", sub(B, ("astype(np.int)", "astype(np.int64)")), "Same indices.")])


# =====================================================================================
def build(clean: bool = True) -> list[Path]:
    if clean and ROOT.exists():
        shutil.rmtree(ROOT)
    out = []
    for c in CASES:
        d = ROOT / c["id"]
        (d / "reference").mkdir(parents=True, exist_ok=True)
        (d / "broken").mkdir(parents=True, exist_ok=True)
        (d / "fixes").mkdir(parents=True, exist_ok=True)
        (d / "reference" / "experiment.py").write_text(c["reference"], encoding="utf-8")
        (d / "broken" / "experiment.py").write_text(c["broken"], encoding="utf-8")
        (d / "fixes" / "oracle.py").write_text(c["oracle"], encoding="utf-8")
        for a in c["alternatives"]:
            (d / "fixes" / f"{a['name']}.py").write_text(a["text"], encoding="utf-8")
        meta = {k: c[k] for k in ("id", "family", "dataset", "category", "description", "notes")}
        meta["fixes"] = [{"file": "oracle.py", "label": "equivalent", "why": "minimal methodology-preserving repair"}] + \
                        [{"file": f"{a['name']}.py", "label": a["label"], "why": a["why"]} for a in c["alternatives"]]
        (d / "case.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        out.append(d)
    return out


if __name__ == "__main__":
    dirs = build()
    n_fix = sum(len(json.loads((d / "case.json").read_text())["fixes"]) for d in dirs)
    print(f"built {len(dirs)} cases, {n_fix} labelled fixes in {ROOT}")
