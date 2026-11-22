## Acceptance rules vs ground truth

| Rule | Correct fixes accepted | Trap fixes accepted (silent methodology change) | Overall accuracy |
|---|---|---|---|
| RUNS | 30/30 (100%) | 21/21 (100%) | 59% |
| SCORE10 | 29/30 (97%) | 17/21 (81%) | 65% |
| VERIFIER | 29/30 (97%) | 0/21 (0%) | 98% |

## Per fix

| Case | Fix | Label | Runs | Score ±10% | Verifier | First reason |
|---|---|---|---|---|---|---|
| ADA01_base_estimator | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| ADA01_base_estimator | delete_both.py | changes_method | ✅ | ✅ | METHOD_CHANGED | AdaBoostClassifier (fit #1) estimator: was DecisionTreeClassifier(...), now None |
| AGG01_affinity | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| AGG01_affinity | delete_arg.py | changes_method | ✅ | ❌ | METHOD_CHANGED | AgglomerativeClustering (fit #1) metric: was 'manhattan', now 'euclidean' |
| ANN01_mlp_imports | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| ANN01_mlp_imports | builtin_float.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| ANN01_mlp_imports | raise_max_iter.py | changes_method | ✅ | ✅ | METHOD_CHANGED | MLPClassifier (fit #2) max_iter: was 60, now 500 |
| ANN01_mlp_imports | float32.py | changes_method | ✅ | ✅ | METHOD_CHANGED | MLPClassifier (fit #2) trained on different rows/values than the original |
| ANN02_mlp_pandas_append | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| ANN02_mlp_pandas_append | loc_insert.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| CV01_old_kfold | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| CV01_old_kfold | add_shuffle.py | changes_method | ✅ | ✅ | METHOD_CHANGED | DecisionTreeClassifier (fit #1) trained on different rows/values than the original |
| GB01_loss_deviance | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| GB01_loss_deviance | delete_arg.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| GB01_loss_deviance | exponential.py | changes_method | ✅ | ✅ | METHOD_CHANGED | GradientBoostingClassifier (fit #1) loss: was 'log_loss', now 'exponential' |
| GRID01_iid_cv_default | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| GRID01_iid_cv_default | delete_iid_only.py | changes_method | ✅ | ✅ | METHOD_CHANGED | GridSearchCV (fit #2) cv: was 3, now None |
| IMP01_imputer | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| IMP01_imputer | impute_before_split.py | changes_method | ✅ | ✅ | METHOD_CHANGED | SimpleImputer (fit #1) training data shape was [426, 30], now [569, 30] |
| IMP01_imputer | median.py | changes_method | ✅ | ✅ | METHOD_CHANGED | SimpleImputer (fit #1) strategy: was 'mean', now 'median' |
| KM01_kmeans_full_ninit | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| KM01_kmeans_full_ninit | fix_crash_only.py | changes_method | ✅ | ✅ | METHOD_CHANGED | KMeans (fit #1) n_init: was 10, now 'auto' |
| KM01_kmeans_full_ninit | delete_arg.py | changes_method | ✅ | ✅ | METHOD_CHANGED | KMeans (fit #1) n_init: was 10, now 'auto' |
| LR01_penalty_none | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| LR01_penalty_none | delete_arg.py | changes_method | ✅ | ✅ | METHOD_CHANGED | LogisticRegression (fit #2) penalty: was None, now 'deprecated' |
| LR02_multi_class_ovr | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| LR02_multi_class_ovr | delete_arg.py | changes_method | ✅ | ✅ | METHOD_CHANGED | trained components differ: original ['StandardScaler', 'OneVsRestClassifier'], now ['StandardScaler', 'Logisti |
| MET01_rmse_squared | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| MET01_rmse_squared | delete_squared.py | changes_method | ✅ | ❌ | METHOD_CHANGED | reported metrics differ: original ['root_mean_squared_error', 'r2_score'], now ['mean_squared_error', 'r2_scor |
| MET01_rmse_squared | sqrt_of_mse.py | equivalent | ✅ | ❌ | METHOD_CHANGED | reported metrics differ: original ['root_mean_squared_error', 'r2_score'], now ['mean_squared_error', 'r2_scor |
| NP01_np_int | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| NP01_np_int | int64.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| RF00_param_typo | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| RF00_param_typo | delete_arg.py | changes_method | ✅ | ✅ | METHOD_CHANGED | RandomForestClassifier (fit #1) n_estimators: was 200, now 100 |
| RF01_max_features_auto_clf | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| RF01_max_features_auto_clf | delete_arg.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| RF01_max_features_auto_clf | use_all_features.py | changes_method | ✅ | ✅ | METHOD_CHANGED | RandomForestClassifier (fit #1) max_features: was 'sqrt', now 1.0 |
| RF02_max_features_auto_reg | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| RF02_max_features_auto_reg | sqrt_like_classifier.py | changes_method | ✅ | ❌ | METHOD_CHANGED | RandomForestRegressor (fit #1) max_features: was 1.0, now 'sqrt' |
| RF02_max_features_auto_reg | delete_arg.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| RF02_max_features_auto_reg | none.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| RF03_criterion_mse | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| RF03_criterion_mse | delete_arg.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| RF03_criterion_mse | absolute_error.py | changes_method | ✅ | ✅ | METHOD_CHANGED | RandomForestRegressor (fit #1) criterion: was 'squared_error', now 'absolute_error' |
| RF04_silent_default_trees | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| RF04_silent_default_trees | leave_as_is.py | changes_method | ✅ | ✅ | METHOD_CHANGED | RandomForestClassifier (fit #1) n_estimators: was 10, now 100 |
| SGD01_loss_log | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| SGD01_loss_log | delete_arg.py | changes_method | ✅ | ✅ | METHOD_CHANGED | SGDClassifier (fit #2) loss: was 'log_loss', now 'hinge' |
| SVM01_cv_module_gamma | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
| SVM01_cv_module_gamma | fix_import_only.py | changes_method | ✅ | ❌ | METHOD_CHANGED | SVC (fit #1) gamma: was 'auto', now 'scale' |
| TREE01_presort | oracle.py | equivalent | ✅ | ✅ | REPRODUCED |  |
