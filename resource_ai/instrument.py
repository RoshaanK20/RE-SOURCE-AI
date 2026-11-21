"""Runtime methodology recorder.

Installed by `resource_ai.launch` BEFORE the experiment script runs. It wraps scikit-learn
so that, while the script executes, we record what the experiment *actually did*:

  * every estimator fit: class, effective hyper-parameters, fingerprint of X / y rows
  * every prediction / score call: which rows were evaluated (for leakage detection)
  * train_test_split and cross-validation splitters: arguments + fingerprint of the indices
  * metric calls (accuracy_score, f1_score, mean_squared_error, ...): name, options, value

The trace is written as JSON to the path in $RESOURCE_TRACE when the interpreter exits.
This is what lets the verifier catch "silent" changes that a config-file hash can't see,
e.g. code that ignores the config, a changed library default, or training on test rows.

PyTorch hooks live in `instrument_torch.py` and are installed when torch is importable.
"""
from __future__ import annotations

import atexit
import functools
import hashlib
import inspect
import json
import os
import sys
import threading
import weakref

_STATE = threading.local()
TRACE: dict = {"version": 1, "events": [], "nested_events": 0, "errors": []}
MAX_ROWS = 200_000          # store per-row hashes up to this many rows (leakage check)
MAX_NESTED = 400            # cap on recorded nested events (e.g. inside GridSearchCV)
_labels: dict[int, str] = {}
_counter = {"n": 0}


# --------------------------------------------------------------------------- helpers
def _depth() -> int:
    return getattr(_STATE, "depth", 0)


class _Nest:
    """Context manager tracking call depth so nested fits (Pipeline steps, grid search) are marked."""

    def __enter__(self):
        _STATE.depth = _depth() + 1
        return self

    def __exit__(self, *exc):
        _STATE.depth = _depth() - 1
        return False


def _label(obj) -> str:
    key = id(obj)
    if key not in _labels:
        _counter["n"] += 1
        _labels[key] = f"{type(obj).__name__}#{_counter['n']}"
        try:
            weakref.finalize(obj, _labels.pop, key, None)
        except TypeError:
            pass
    return _labels[key]


def _to_2d_numpy(X):
    import numpy as np
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()
    except Exception:
        pass
    if hasattr(X, "detach") and hasattr(X, "cpu"):        # torch tensor
        X = X.detach().cpu().numpy()
    if hasattr(X, "to_numpy"):                           # pandas
        X = X.to_numpy()
    A = np.asarray(X)
    if A.ndim == 0:
        A = A.reshape(1, 1)
    elif A.ndim == 1:
        A = A.reshape(-1, 1)
    else:
        A = A.reshape(A.shape[0], -1)
    return A


def _row_hashes(A) -> list[str]:
    import numpy as np
    try:
        B = np.round(A.astype(np.float64), 8)
        B = np.where(B == 0, 0.0, B)                     # -0.0 == 0.0
        B = np.ascontiguousarray(B)
        return [hashlib.blake2b(B[i].tobytes(), digest_size=6).hexdigest() for i in range(B.shape[0])]
    except (TypeError, ValueError):
        return [hashlib.blake2b(repr(list(r)).encode(), digest_size=6).hexdigest() for r in A.tolist()]


def fingerprint(X, keep_rows: bool = True) -> dict | None:
    """Order-insensitive + order-sensitive fingerprint of an array-like's rows."""
    if X is None:
        return None
    try:
        A = _to_2d_numpy(X)
        rows = _row_hashes(A)
        fp = {
            "shape": list(A.shape),
            "set_digest": hashlib.sha256("".join(sorted(rows)).encode()).hexdigest()[:16],
            "seq_digest": hashlib.sha256("".join(rows).encode()).hexdigest()[:16],
        }
        if keep_rows and len(rows) <= MAX_ROWS:
            fp["rows"] = rows
        return fp
    except Exception as e:  # never break the experiment because of the recorder
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def paired_fingerprint(X, y) -> str | None:
    """Order-independent fingerprint of (feature row, corresponding label) pairs.

    A label permutation can leave both independent feature and label digests unchanged;
    this catches that otherwise silent methodological change. It is not an attestation
    against malicious scripts that can modify the recorder.
    """
    if X is None or y is None:
        return None
    try:
        xr = _row_hashes(_to_2d_numpy(X))
        yr = _row_hashes(_to_2d_numpy(y))
        if len(xr) != len(yr):
            return None
        return hashlib.sha256("|".join(sorted(a + ":" + b for a, b in zip(xr, yr))).encode()).hexdigest()[:24]
    except Exception:
        return None


def _jsonable(v, depth: int = 0):
    """Convert hyper-parameter values to stable JSON."""
    import numpy as np
    if depth > 4:
        return repr(v)[:200]
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if v == v else "nan"
    if isinstance(v, np.generic):
        return _jsonable(v.item(), depth + 1)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x, depth + 1) for k, x in v.items()}
    if hasattr(v, "get_params") and hasattr(v, "fit"):
        return {"__estimator__": type(v).__name__, "params": _params(v, depth + 1)}
    if isinstance(v, np.ndarray):
        fp = fingerprint(v, keep_rows=False) or {}
        return {"__array__": fp.get("seq_digest"), "shape": list(v.shape)}
    if inspect.isclass(v):
        return {"__class__": v.__name__}
    if callable(v):
        return {"__callable__": getattr(v, "__name__", repr(v))[:80]}
    if hasattr(v, "split") and hasattr(v, "get_n_splits"):        # CV splitter object
        return {"__splitter__": type(v).__name__, "repr": repr(v)[:200]}
    if type(v).__name__ == "RandomState":
        return {"__random_state__": "RandomState"}
    return repr(v)[:200]


def _params(est, depth: int = 0) -> dict:
    try:
        return {k: _jsonable(v, depth) for k, v in est.get_params(deep=False).items()}
    except Exception as e:
        return {"__error__": str(e)[:200]}


def _record(event: dict):
    event["depth"] = _depth()
    if event["depth"] > 0:
        TRACE["nested_events"] += 1
        if TRACE["nested_events"] > MAX_NESTED:
            return
        for key in ("X", "y"):
            if isinstance(event.get(key), dict):
                event[key].pop("rows", None)
    event["i"] = len(TRACE["events"])
    TRACE["events"].append(event)


def _split_xy(args, kwargs):
    X = args[0] if args else kwargs.get("X", kwargs.get("x"))
    y = args[1] if len(args) > 1 else kwargs.get("y", kwargs.get("Y"))
    return X, y


# --------------------------------------------------------------------------- wrappers
def _wrap_fit(orig, kind):
    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        X, y = _split_xy(args, kwargs)
        ev = {"type": kind, "obj": _label(self), "cls": type(self).__name__,
              "module": type(self).__module__.split(".")[0], "params": _params(self),
              "X": fingerprint(X, keep_rows=_depth() == 0), "y": fingerprint(y, keep_rows=False),
              "xy_pairs": paired_fingerprint(X, y)}
        with _Nest():
            out = orig(self, *args, **kwargs)
        if kind == "fit_predict":
            ev["output"] = fingerprint(out, keep_rows=False)
        _record(ev)
        return out
    wrapper._resource_wrapped = True
    return wrapper


def _wrap_eval(orig, kind):
    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        if _depth() > 0:
            return orig(self, *args, **kwargs)
        X, y = _split_xy(args, kwargs)
        with _Nest():
            out = orig(self, *args, **kwargs)
        ev = {"type": kind, "obj": _label(self), "cls": type(self).__name__, "X": fingerprint(X)}
        if kind != "score":
            ev["output"] = fingerprint(out, keep_rows=False)
            try:
                import numpy as np
                arr = np.asarray(out)
                if arr.ndim == 2 and arr.shape[1] > 1:
                    ev["output_columns"] = [(fingerprint(arr[:, j], keep_rows=False) or {}).get("seq_digest")
                                            for j in range(arr.shape[1])]
            except Exception:
                pass
        if kind == "score":
            ev["y"] = fingerprint(y, keep_rows=False)
            ev["value"] = _jsonable(out)
        _record(ev)
        return out
    wrapper._resource_wrapped = True
    return wrapper


def _wrap_split_fn(orig):
    @functools.wraps(orig)
    def wrapper(*arrays, **options):
        top = _depth() == 0
        with _Nest():
            out = orig(*arrays, **options)
        if top:
            opts = {k: (_jsonable(v) if k != "stratify" else (fingerprint(v, keep_rows=False) or {}).get("seq_digest"))
                    for k, v in options.items()}
            _record({"type": "train_test_split", "options": opts,
                     "n_arrays": len(arrays),
                     "outputs": [(fingerprint(a, keep_rows=False) or {}).get("seq_digest") for a in out]})
        return out
    wrapper._resource_wrapped = True
    return wrapper


def _wrap_cv_split(orig):
    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        h = hashlib.sha256()
        n = 0
        for train, test in orig(self, *args, **kwargs):
            import numpy as np
            h.update(np.asarray(train, dtype="<i8").tobytes() + b"|" + np.asarray(test, dtype="<i8").tobytes() + b"#")
            n += 1
            yield train, test
        _record({"type": "cv_split", "cls": type(self).__name__, "repr": repr(self)[:200],
                 "n_folds": n, "folds_digest": h.hexdigest()[:16]})
    wrapper._resource_wrapped = True
    return wrapper


METRIC_SUFFIXES = ("_score", "_error", "_loss")
METRIC_EXTRA = {"confusion_matrix"}


def _wrap_metric(orig, name):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        if _depth() > 0:
            return orig(*args, **kwargs)
        with _Nest():
            out = orig(*args, **kwargs)
        if _depth() == 0:
            y_true = args[0] if args else kwargs.get("y_true", kwargs.get("labels_true"))
            opts = {k: _jsonable(v) for k, v in kwargs.items()
                    if k not in ("y_true", "y_pred", "y_score", "y_prob", "labels_true", "labels_pred", "X", "labels")}
            pred = (args[1] if len(args) > 1 else
                    kwargs.get("y_pred", kwargs.get("y_score", kwargs.get("y_prob", kwargs.get("labels_pred")))))
            ev = {"type": "metric", "name": name, "options": opts,
                  "y_true": (fingerprint(y_true, keep_rows=False) or {}).get("seq_digest"),
                  "y_pred": (fingerprint(pred, keep_rows=False) or {}).get("seq_digest")}
            try:
                ev["value"] = _jsonable(out.tolist() if hasattr(out, "tolist") else out)
            except Exception:
                ev["value"] = None
            _record(ev)
        return out
    wrapper._resource_wrapped = True
    return wrapper


def _patch_method(cls, name, factory, *fargs):
    fn = cls.__dict__.get(name)
    # Methods decorated with sklearn's @available_if (e.g. GridSearchCV.predict) are
    # descriptors, not functions: wrap the function they hold instead of skipping them.
    inner = getattr(fn, "fn", None)
    if fn is not None and not callable(fn) and callable(inner):
        if not getattr(inner, "_resource_wrapped", False):
            fn.fn = factory(inner, *fargs)
        return
    if fn is None or getattr(fn, "_resource_wrapped", False) or not callable(fn):
        return
    try:
        setattr(cls, name, factory(fn, *fargs))
    except (TypeError, AttributeError):
        pass


# --------------------------------------------------------------------------- install
def install_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.utils import all_estimators
        import sklearn.metrics as M
        import sklearn.model_selection as MS
        import sklearn.model_selection._split as S
        import sklearn.pipeline as P
        classes = {c for _, c in all_estimators()}
        classes |= {P.Pipeline, P.FeatureUnion, MS.GridSearchCV, MS.RandomizedSearchCV}
    seen = set()
    for est_cls in classes:
        for c in est_cls.__mro__:
            if c in seen or not c.__module__.startswith("sklearn"):
                continue
            seen.add(c)
            for m in ("fit", "fit_transform", "fit_predict"):
                _patch_method(c, m, _wrap_fit, m)
            for m in ("predict", "predict_proba", "decision_function", "score"):
                _patch_method(c, m, _wrap_eval, m)
    # splitters
    for name in dir(S):
        c = getattr(S, name)
        if inspect.isclass(c) and c.__module__ == S.__name__:
            _patch_method(c, "split", _wrap_cv_split)
    wrapped_tts = _wrap_split_fn(S.train_test_split)
    S.train_test_split = wrapped_tts
    MS.train_test_split = wrapped_tts
    # metrics (public namespace only, so internal scorer calls are not double counted)
    for name in dir(M):
        if name.startswith("_"):
            continue
        fn = getattr(M, name)
        if inspect.isfunction(fn) and (name.endswith(METRIC_SUFFIXES) or name in METRIC_EXTRA):
            setattr(M, name, _wrap_metric(fn, name))
    TRACE["sklearn_version"] = sklearn.__version__
    return True


def _write_trace():
    path = os.environ.get("RESOURCE_TRACE")
    if not path:
        return
    try:
        import platform
        TRACE["python"] = platform.python_version()
        try:
            import numpy
            TRACE["numpy_version"] = numpy.__version__
        except Exception:
            pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(TRACE, f)
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[resource_ai] could not write trace: {e}\n")


def install() -> None:
    installed = []
    try:
        if install_sklearn():
            installed.append("sklearn")
    except Exception as e:
        TRACE["errors"].append(f"sklearn hooks: {type(e).__name__}: {e}")
    try:
        from . import instrument_torch
        if instrument_torch.install(TRACE, _record, fingerprint, _jsonable, _Nest, _depth, _label):
            installed.append("torch")
    except Exception as e:
        TRACE["errors"].append(f"torch hooks: {type(e).__name__}: {e}")
    TRACE["hooks"] = installed
    atexit.register(_write_trace)
