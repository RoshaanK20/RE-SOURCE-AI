"""Knowledge base of real library breaking changes.

Two uses:
  1. Retrieval: entries whose patterns match the broken code / traceback are shown to the
     LLM as hints, *including the methodology warning* (what the tempting wrong fix changes).
  2. Equivalence: the verifier normalises old parameter names/values to new ones, so a
     reference trace captured with an old library still compares correctly.

Each entry was checked against scikit-learn 1.9 / NumPy 2.5 / pandas 3.0.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Entry:
    id: str
    patterns: list[str]                # regexes searched in code + error text
    title: str
    fix: str                           # the methodology-preserving replacement
    warning: str = ""                  # what a careless fix would silently change
    tags: list[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        return any(re.search(p, text) for p in self.patterns)


KB: list[Entry] = [
    # ---------- scikit-learn: removed modules / imports
    Entry("sk_cross_validation", [r"sklearn\.cross_validation", r"No module named 'sklearn\.cross_validation'"],
          "sklearn.cross_validation was removed (0.20)",
          "Import from sklearn.model_selection instead (train_test_split, KFold, cross_val_score...).",
          "Old KFold(n, n_folds=k) did NOT shuffle. Use KFold(n_splits=k) and iterate kf.split(X); "
          "do not add shuffle=True or a random_state, that changes the folds."),
    Entry("sk_kfold_old_sig", [r"n_folds", r"KFold\(\s*len\("],
          "Old KFold(n, n_folds=k) signature",
          "KFold(n_splits=k) and `for train, test in kf.split(X):`.",
          "Keep shuffle=False (the old default). Adding shuffle/random_state changes which samples are in each fold."),
    Entry("sk_externals_joblib", [r"sklearn\.externals import joblib", r"sklearn\.externals\.joblib"],
          "sklearn.externals.joblib was removed (0.23)", "Use `import joblib` directly."),
    Entry("sk_imputer", [r"\bImputer\b", r"cannot import name 'Imputer'"],
          "sklearn.preprocessing.Imputer was removed (0.22)",
          "Use sklearn.impute.SimpleImputer(strategy=...) (axis=0 behaviour is the only one now).",
          "Fit the imputer on the TRAINING rows only, exactly as the original did. Fitting it on all data leaks test information."),
    Entry("sk_plot_confusion_matrix", [r"plot_confusion_matrix"],
          "plot_confusion_matrix was removed (1.2)", "Use ConfusionMatrixDisplay.from_estimator(...)."),
    Entry("sk_load_boston", [r"load_boston"],
          "load_boston was removed (1.2)",
          "Load the original Boston data from a local copy of the original file.",
          "Do NOT silently switch to another dataset (e.g. California housing): that is a different experiment. "
          "If the data is unavailable, report it instead of substituting."),
    # ---------- scikit-learn: renamed / removed parameters
    Entry("sk_max_features_auto", [r"max_features\s*=\s*['\"]auto['\"]", r"'max_features' parameter"],
          "max_features='auto' was removed (1.3)",
          "For *Classifier* forests/trees 'auto' meant 'sqrt'. For *Regressor* forests 'auto' meant 1.0 (all features).",
          "Using 'sqrt' for a RandomForestRegressor changes the model (it used all features before)."),
    Entry("sk_criterion_mse", [r"criterion\s*=\s*['\"]mse['\"]", r"criterion\s*=\s*['\"]mae['\"]"],
          "criterion='mse'/'mae' renamed (1.2)", "'mse' -> 'squared_error', 'mae' -> 'absolute_error'.",
          "Do not switch between squared and absolute error: that is a different loss."),
    Entry("sk_penalty_none", [r"penalty\s*=\s*['\"]none['\"]"],
          "penalty='none' (string) removed (1.4)", "Use penalty=None (the Python value).",
          "Deleting the penalty argument is WRONG: the default is 'l2', i.e. a regularised model."),
    Entry("sk_multi_class", [r"multi_class\s*="],
          "LogisticRegression(multi_class=...) removed",
          "multi_class='ovr' -> wrap: OneVsRestClassifier(LogisticRegression(...)); "
          "multi_class='multinomial' -> just delete the argument (multinomial is now the default).",
          "Deleting multi_class='ovr' silently switches one-vs-rest to multinomial: a different model."),
    Entry("sk_sgd_loss_log", [r"loss\s*=\s*['\"]log['\"]"],
          "SGDClassifier(loss='log') renamed (1.3)", "Use loss='log_loss'.",
          "Deleting loss= changes it to the default 'hinge' (a linear SVM instead of logistic regression)."),
    Entry("sk_gb_deviance", [r"loss\s*=\s*['\"]deviance['\"]"],
          "GradientBoosting loss='deviance' renamed (1.3)", "Use loss='log_loss' (identical).",
          "Do not use loss='exponential' (that is AdaBoost-style loss)."),
    Entry("sk_base_estimator", [r"base_estimator\s*="],
          "base_estimator renamed to estimator (removed 1.4)",
          "Rename the keyword: estimator=... and keep the same inner model.",
          "Deleting base_estimator makes AdaBoost use a depth-1 stump and Bagging a full tree instead of your model."),
    Entry("sk_ada_algorithm", [r"algorithm\s*=\s*['\"]SAMME"],
          "AdaBoostClassifier(algorithm=...) removed (1.6/1.8)",
          "'SAMME' is the only remaining algorithm: delete algorithm='SAMME'.",
          "algorithm='SAMME.R' cannot be reproduced in current scikit-learn; report this instead of hiding it."),
    Entry("sk_ohe_sparse", [r"OneHotEncoder\([^)]*sparse\s*="],
          "OneHotEncoder(sparse=) renamed (1.2)", "Use sparse_output= with the same value."),
    Entry("sk_mse_squared", [r"squared\s*=\s*False", r"unexpected keyword argument 'squared'"],
          "mean_squared_error(squared=False) removed (1.6)", "Use sklearn.metrics.root_mean_squared_error(y_true, y_pred).",
          "Deleting squared=False silently reports MSE instead of RMSE: a different metric."),
    Entry("sk_kmeans_full", [r"algorithm\s*=\s*['\"](full|auto)['\"]"],
          "KMeans algorithm='full'/'auto' removed (1.3)", "Use algorithm='lloyd'."),
    Entry("sk_kmeans_ninit", [r"KMeans\("],
          "KMeans default n_init changed from 10 to 'auto' (1.4)",
          "If the original code relied on the default, set n_init=10 explicitly.",
          "With n_init='auto' k-means++ runs only once, so clusters/inertia can change even though the code runs."),
    Entry("sk_rf_default_trees", [r"RandomForest(Classifier|Regressor)\(", r"ExtraTrees(Classifier|Regressor)\("],
          "Forest default n_estimators changed from 10 to 100 (0.22)",
          "If the original code relied on the default and ran on scikit-learn < 0.22, set n_estimators=10 explicitly.",
          "Leaving the default trains 100 trees: the code runs but it is not the original model."),
    Entry("sk_svc_gamma", [r"\bSVC\(", r"\bSVR\("],
          "SVC/SVR default gamma changed from 'auto' to 'scale' (0.22)",
          "If the original relied on the default (scikit-learn < 0.22), set gamma='auto' explicitly.",
          "gamma='scale' can change accuracy a lot on unscaled features."),
    Entry("sk_gridsearch_iid", [r"iid\s*="],
          "GridSearchCV(iid=) removed (0.24)", "Delete iid= (iid=False is the current behaviour).",
          "GridSearchCV default cv changed from 3 to 5 folds (0.22): if the original relied on the default, set cv=3."),
    Entry("sk_presort", [r"presort\s*="],
          "presort= removed (0.24)", "Delete presort= (it only affected speed, not results)."),
    Entry("sk_normalize", [r"normalize\s*=\s*True"],
          "normalize= removed from linear models (1.2)",
          "Use a Pipeline with StandardScaler; for Ridge/Lasso the regularisation strength must be rescaled to match.",
          "Simply deleting normalize=True changes Ridge/Lasso results."),
    Entry("sk_affinity", [r"affinity\s*="],
          "AgglomerativeClustering(affinity=) renamed (1.4)", "Use metric= with the same value."),
    Entry("sk_min_impurity_split", [r"min_impurity_split"],
          "min_impurity_split removed (1.0)", "No exact replacement; min_impurity_decrease is a different rule.",
          "This cannot be reproduced exactly in current scikit-learn: report it."),
    Entry("sk_tsne_n_iter", [r"n_iter\s*="],
          "TSNE(n_iter=) renamed (1.7)", "Use max_iter=."),
    Entry("sk_get_feature_names", [r"get_feature_names\(\)"],
          "get_feature_names() removed (1.2)", "Use get_feature_names_out()."),
    # ---------- NumPy / pandas
    Entry("np_aliases", [r"np\.(float|int|bool|object|complex|str)\b(?!\w)", r"module 'numpy' has no attribute '(float|int|object|complex|str)'"],
          "NumPy type aliases np.float/np.int/... removed (1.24)",
          "np.float -> float (or np.float64), np.int -> int (or np.int64), np.object -> object.",
          "Keep the same precision: np.float meant float64, not float32."),
    Entry("pd_append", [r"\b\w*(df|frame|results|table)\w*\.append\(", r"'DataFrame' object has no attribute 'append'"],
          "DataFrame.append removed (pandas 2.0)", "Use pd.concat([df, pd.DataFrame([row])], ignore_index=True)."),
    Entry("pd_iteritems", [r"iteritems"], "iteritems removed (pandas 2.0)", "Use .items()."),
]


def retrieve(code: str, error: str, limit: int = 4) -> list[Entry]:
    """Return KB entries relevant to this broken code + error (error matches ranked first)."""
    scored = []
    for e in KB:
        s = 0
        if any(re.search(p, error) for p in e.patterns):
            s += 2
        if any(re.search(p, code) for p in e.patterns):
            s += 1
        if s:
            scored.append((s, e))
    scored.sort(key=lambda t: -t[0])
    return [e for _, e in scored[:limit]]


def format_hints(entries: list[Entry]) -> str:
    out = []
    for e in entries:
        line = f"- {e.title}. FIX: {e.fix}"
        if e.warning:
            line += f" WARNING: {e.warning}"
        out.append(line)
    return "\n".join(out)


# ----------------------------------------------------------------- verifier equivalence rules
IGNORED_PARAMS = {"n_jobs", "verbose", "copy_X", "copy", "cache_size", "memory", "pre_dispatch",
                  "error_score", "return_train_score", "presort", "compute_distances", "precompute"}

# (class-name regex, param, old value) -> new value ; "*" = any value
VALUE_ALIASES = [
    (r"Classifier$", "max_features", "auto", "sqrt"),
    (r"Regressor$", "max_features", "auto", 1.0),
    (r"(Forest|Tree|Trees)", "max_features", None, 1.0),   # None and 1.0 both mean "all features"
    (r".*", "criterion", "mse", "squared_error"),
    (r".*", "criterion", "mae", "absolute_error"),
    (r".*", "penalty", "none", None),
    (r"SGD", "loss", "log", "log_loss"),
    (r"GradientBoosting", "loss", "deviance", "log_loss"),
    (r"KMeans", "algorithm", "full", "lloyd"),
    (r"KMeans", "algorithm", "auto", "lloyd"),
]
PARAM_RENAMES = [
    (r".*", "base_estimator", "estimator"),
    (r"OneHotEncoder", "sparse", "sparse_output"),
    (r"Agglomerative", "affinity", "metric"),
    (r"TSNE", "n_iter", "max_iter"),
]


def normalise_params(cls: str, params: dict) -> dict:
    out = {}
    for k, v in params.items():
        for rx, old, new in PARAM_RENAMES:
            if k == old and re.search(rx, cls):
                k = new
        for rx, p, old, new in VALUE_ALIASES:
            if k == p and v == old and re.search(rx, cls):
                v = new
        if k in IGNORED_PARAMS:
            continue
        if isinstance(v, dict) and "__estimator__" in v:
            v = {"__estimator__": v["__estimator__"], "params": normalise_params(v["__estimator__"], v.get("params", {}))}
        out[k] = v
    return out
