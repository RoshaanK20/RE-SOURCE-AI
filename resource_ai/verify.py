"""Scientific verifier: compare a repaired run's trace with the reference trace.

Three separate questions (the paper's three levels):
  1. RUNS        - did the experiment execute without error?
  2. METHODOLOGY - same models, effective hyper-parameters, training data, splits, metrics,
                   and no new train/test leakage?
  3. RESULTS     - are the reported metric values within tolerance of the reference?

Also implements a simplified illustrative numerical-score baseline:
"accept if it runs and the score is within 10% of the reference score".
This is not verified as a faithful reproduction of any external paper's pipeline.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .knowledge import IGNORED_PARAMS, PARAM_RENAMES, VALUE_ALIASES

import re

FAILS_TO_RUN, NO_EVIDENCE, METHOD_CHANGED, RESULTS_DIFFER, REPRODUCED = (
    "FAILS_TO_RUN", "NO_EVIDENCE", "METHOD_CHANGED", "RESULTS_DIFFER", "REPRODUCED")


@dataclass
class Report:
    verdict: str
    runs: bool
    violations: list[str] = field(default_factory=list)
    result_deltas: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    accepted_by_score_rule: bool = False      # baseline: runs + primary score within 10%
    primary_metric: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def feedback(self) -> str:
        """Short text for the LLM explaining why the repair is not accepted yet."""
        lines = [f"VERDICT: {self.verdict}"]
        if self.violations:
            lines.append("Methodology differences from the ORIGINAL experiment (must be fixed):")
            lines += [f"  - {v}" for v in self.violations[:12]]
        if self.evidence_gaps:
            lines.append("Insufficient scientific evidence (cannot mark as reproduced):")
            lines += [f"  - {g}" for g in self.evidence_gaps[:12]]
        bad = [d for d in self.result_deltas if not d["within_tol"]]
        if bad:
            lines.append("Result differences:")
            lines += [f"  - {d['metric']}: original={d['reference']:.6g}, now={d['candidate']:.6g}" for d in bad[:8]]
        return "\n".join(lines)


# ----------------------------------------------------------------- normalisation
def _norm_value(cls: str, key: str, v):
    for rx, p, old, new in VALUE_ALIASES:
        if key == p and v == old and re.search(rx, cls):
            v = new
    return _norm_any(v)


def _norm_any(v):
    if isinstance(v, dict) and "__estimator__" in v:
        return {"__estimator__": v["__estimator__"], "params": norm_params(v["__estimator__"], v.get("params", {}))}
    if isinstance(v, list):
        return [_norm_any(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm_any(x) for k, x in v.items()}
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return v  # keep 1.0 distinct from 1 (max_features=1 vs 1.0 differ in sklearn)
    return v


def norm_params(cls: str, params: dict) -> dict:
    out = {}
    for k, v in (params or {}).items():
        for rx, old, new in PARAM_RENAMES:
            if k == old and re.search(rx, cls):
                k = new
        if k in IGNORED_PARAMS:
            continue
        out[k] = _norm_value(cls, k, v)
    return out


def _equal(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (a != a and b != b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and type(a) is not type(b):
        return False  # 1 (int) and 1.0 (float) mean different things for max_features / test_size
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _short(v, n=60) -> str:
    s = repr(v)
    if isinstance(v, dict) and "__estimator__" in v:
        s = f"{v['__estimator__']}(...)"
    return s if len(s) <= n else s[:n] + "..."


def _param_diffs(cls: str, ref: dict, cand: dict, prefix: str = "") -> list[str]:
    r, c = norm_params(cls, ref), norm_params(cls, cand)
    diffs = []
    for k in r:
        if k not in c:
            diffs.append(f"{prefix}{k}: was {_short(r[k])}, now not set/unsupported")
            continue
        rv, cv = r[k], c[k]
        if isinstance(rv, dict) and isinstance(cv, dict) and "__estimator__" in rv and "__estimator__" in cv:
            if rv["__estimator__"] != cv["__estimator__"]:
                diffs.append(f"{prefix}{k}: model was {rv['__estimator__']}, now {cv['__estimator__']}")
            else:
                diffs += _param_diffs(rv["__estimator__"], rv["params"], cv["params"], f"{prefix}{k}.")
        elif not _equal(rv, cv):
            diffs.append(f"{prefix}{k}: was {_short(rv)}, now {_short(cv)}")
    return diffs


# ----------------------------------------------------------------- trace views
def _top(trace: dict, *types) -> list[dict]:
    return [e for e in trace.get("events", []) if e.get("depth", 0) == 0 and e.get("type") in types]


def _metrics(trace: dict) -> list[dict]:
    out = []
    for e in _top(trace, "metric", "score"):
        if e["type"] == "score":
            out.append({"name": f"{e['cls']}.score", "options": {}, "y_true": (e.get("y") or {}).get("seq_digest"),
                        "value": e.get("value")})
        elif e.get("name") not in ("confusion_matrix",):
            e = dict(e)
            opts = dict(e.get("options") or {})
            if e["name"] in ("mean_squared_error", "mean_squared_log_error") and opts.get("squared") is False:
                e["name"] = "root_" + e["name"]          # old API spelling of the same metric
                opts.pop("squared")
            elif opts.get("squared") is True:
                opts.pop("squared")
            e["options"] = opts
            out.append(e)
    return out


def _numeric(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _leak_fraction(trace: dict) -> dict[int, float]:
    """For each top-level fit (by order), the max fraction of evaluated rows that were also trained on."""
    fits = _top(trace, "fit", "fit_transform", "fit_predict")
    evals = _top(trace, "predict", "predict_proba", "decision_function", "score")
    out = {}
    for idx, f in enumerate(fits):
        rows = set((f.get("X") or {}).get("rows") or [])
        if not rows:
            continue
        frac = 0.0
        for e in evals:
            if e.get("obj") != f.get("obj") or e["i"] < f["i"]:
                continue
            er = (e.get("X") or {}).get("rows") or []
            if er:
                frac = max(frac, sum(r in rows for r in er) / len(er))
        out[idx] = frac
    return out


# ----------------------------------------------------------------- main entry
def verify(reference: dict, run, result_rel_tol: float = 0.01, score_rule_tol: float = 0.10) -> Report:
    """`run` is a runner.RunResult (or anything with .ok and .trace)."""
    if not run.ok:
        return Report(FAILS_TO_RUN, False, notes=["experiment did not finish"])
    cand = run.trace
    if not cand or not cand.get("events"):
        return Report(NO_EVIDENCE, True, notes=["no methodology events were recorded (unsupported library?)"])

    rep = Report(REPRODUCED, True)
    V = rep.violations
    G = rep.evidence_gaps
    if cand.get("errors"):
        G.append("instrumentation reported an error; coverage may be incomplete")

    # --- 1. trained models: class, effective hyper-parameters, training data
    rf, cf = _top(reference, "fit", "fit_transform", "fit_predict"), _top(cand, "fit", "fit_transform", "fit_predict")
    rnames, cnames = [e["cls"] for e in rf], [e["cls"] for e in cf]
    if rnames != cnames:
        V.append(f"trained components differ: original {rnames}, now {cnames}")
    for i, (r, c) in enumerate(zip(rf, cf)):
        if r["cls"] != c["cls"]:
            continue
        tag = f"{r['cls']} (fit #{i + 1})"
        V += [f"{tag} {d}" for d in _param_diffs(r["cls"], r.get("params", {}), c.get("params", {}))]
        rx, cx = r.get("X") or {}, c.get("X") or {}
        if rx.get("shape") != cx.get("shape"):
            V.append(f"{tag} training data shape was {rx.get('shape')}, now {cx.get('shape')}")
        elif rx.get("set_digest") != cx.get("set_digest"):
            V.append(f"{tag} trained on different rows/values than the original")
        elif rx.get("seq_digest") != cx.get("seq_digest"):
            rep.notes.append(f"{tag} same training rows but in a different order")
        ry, cy = r.get("y") or {}, c.get("y") or {}
        if ry and cy and ry.get("set_digest") != cy.get("set_digest"):
            V.append(f"{tag} trained on different target labels/values")
        # Independent feature and label multisets cannot catch labels assigned to
        # the wrong rows. Compare joint (X, y) pairs as well.
        if r.get("xy_pairs") and c.get("xy_pairs"):
            if r["xy_pairs"] != c["xy_pairs"]:
                V.append(f"{tag} training feature/label pairings changed")
        elif r.get("y") or c.get("y"):
            # Only supervised fits have labels to pair. Unsupervised steps (scalers,
            # imputers, clustering) legitimately have no y, so no pairing evidence.
            G.append(f"{tag} missing feature/label pairing evidence")

    # --- 2. data splits
    rs, cs = _top(reference, "train_test_split"), _top(cand, "train_test_split")
    if len(rs) != len(cs):
        V.append(f"train_test_split called {len(rs)}x originally, now {len(cs)}x")
    for r, c in zip(rs, cs):
        ro, co = dict(r.get("options", {})), dict(c.get("options", {}))
        for k in sorted(set(ro) | set(co)):
            if not _equal(ro.get(k), co.get(k)):
                V.append(f"train_test_split {k}: was {_short(ro.get(k))}, now {_short(co.get(k))}")
        if r.get("outputs") != c.get("outputs"):
            V.append("train_test_split produced a different train/test partition")
    rcv, ccv = _top(reference, "cv_split"), _top(cand, "cv_split")
    if len(rcv) != len(ccv):
        V.append(f"cross-validation splitting used {len(rcv)}x originally, now {len(ccv)}x")
    for r, c in zip(rcv, ccv):
        if (r["cls"], r["n_folds"]) != (c["cls"], c["n_folds"]):
            V.append(f"cross-validation was {r['repr']}, now {c['repr']}")
        elif r["folds_digest"] != c["folds_digest"]:
            V.append(f"cross-validation folds differ ({r['repr']} vs {c['repr']})")

    # --- 3. leakage (relative to the original)
    rl, cl = _leak_fraction(reference), _leak_fraction(cand)
    for idx, frac in cl.items():
        if frac > rl.get(idx, 0.0) + 0.05:
            V.append(f"{cf[idx]['cls']} (fit #{idx + 1}) is evaluated on rows it was trained on "
                     f"({frac:.0%} overlap, originally {rl.get(idx, 0.0):.0%}): train/test leakage")

    # --- 4. actual prediction provenance and evaluated features
    # This is bounded observation, NOT a tamper-proof security guarantee.
    pred_types = ("predict", "predict_proba", "decision_function", "fit_predict")
    # Parallel tree ensembles may execute worker-thread prediction hooks with
    # thread-local nesting depth 0. Only compare predictions from estimators
    # actually fitted at top level; internal tree calls are implementation detail.
    rfit_objs = {e.get("obj") for e in rf}
    cfit_objs = {e.get("obj") for e in cf}
    rpred = [e for e in _top(reference, *pred_types) if e.get("obj") in rfit_objs]
    cpred = [e for e in _top(cand, *pred_types) if e.get("obj") in cfit_objs]
    if [p["type"] for p in rpred] != [p["type"] for p in cpred]:
        V.append("prediction operations differ from original experiment")
    for idx, (r, c) in enumerate(zip(rpred, cpred)):
        rX, cX = r.get("X") or {}, c.get("X") or {}
        if rX.get("seq_digest") and cX.get("seq_digest"):
            if rX["seq_digest"] != cX["seq_digest"]:
                V.append(f"prediction #{idx+1} used different evaluation features")
        elif rX or cX:
            G.append(f"prediction #{idx+1} has missing input fingerprint")

    rm, cm = _metrics(reference), _metrics(cand)
    # Metric values are NOT enough: confirm that supervised metric inputs came
    # from a recorded model prediction, not labels or an invented array.
    supervised = {"accuracy_score", "balanced_accuracy_score", "f1_score",
                  "precision_score", "recall_score", "jaccard_score",
                  "r2_score", "mean_squared_error", "root_mean_squared_error",
                  "mean_absolute_error", "mean_absolute_percentage_error",
                  "median_absolute_error", "log_loss", "roc_auc_score",
                  "average_precision_score", "adjusted_rand_score"}
    for metric in _top(cand, "metric"):
        name = metric.get("name")
        if name not in supervised:
            continue
        mp = metric.get("y_pred")
        if not mp:
            G.append(f"{name}: missing metric prediction input fingerprint")
            continue
        sources = [e for e in cpred if e.get("i", -1) < metric.get("i", 0)
                   and (mp == (e.get("output") or {}).get("seq_digest")
                        or mp in (e.get("output_columns") or []))]
        if not sources:
            if rpred or cpred:
                V.append(f"{name}: metric predictions not linked to a recorded model output")
            else:
                G.append(f"{name}: no observable model-output provenance")
        elif not any(e.get("type") in ("predict", "predict_proba", "decision_function", "fit_predict") for e in sources):
            G.append(f"{name}: prediction source not supported")
        if not metric.get("y_true"):
            G.append(f"{name}: missing true-label fingerprint")
    for e in _top(cand, "score"):
        if not (e.get("X") or {}).get("seq_digest") or not (e.get("y") or {}).get("seq_digest"):
            G.append("estimator.score missing evaluation X/y fingerprints")


    if [m["name"] for m in rm] != [m["name"] for m in cm]:
        V.append(f"reported metrics differ: original {[m['name'] for m in rm]}, now {[m['name'] for m in cm]}")
    for r, c in zip(_top(reference, "score"), _top(cand, "score")):
        if (r.get("X") or {}).get("seq_digest") != (c.get("X") or {}).get("seq_digest"):
            V.append("estimator.score evaluated different features")
        if (r.get("y") or {}).get("seq_digest") != (c.get("y") or {}).get("seq_digest"):
            V.append("estimator.score evaluated different labels")
    for r, c in zip(rm, cm):
        if r["name"] != c["name"]:
            continue
        if not _equal(r.get("options", {}), c.get("options", {})):
            V.append(f"{r['name']} options were {r.get('options')}, now {c.get('options')}")
        if r.get("y_true") and c.get("y_true") and r["y_true"] != c["y_true"]:
            V.append(f"{r['name']} is computed on different test labels than the original")
        rv, cv = _numeric(r.get("value")), _numeric(c.get("value"))
        if rv is not None and cv is not None:
            within = abs(cv - rv) <= max(result_rel_tol * abs(rv), 1e-9)
            rep.result_deltas.append({"metric": r["name"], "reference": rv, "candidate": cv,
                                      "delta": cv - rv, "within_tol": within})

    # --- baseline acceptance rule (score within 10%)
    prim_r = next((m for m in rm if _numeric(m.get("value")) is not None), None)
    prim_c = next((m for m in cm if prim_r and m["name"] == prim_r["name"] and _numeric(m.get("value")) is not None),
                  None) or next((m for m in cm if _numeric(m.get("value")) is not None), None)
    if prim_r and prim_c:
        rv, cv = float(prim_r["value"]), float(prim_c["value"])
        rep.primary_metric = {"reference_name": prim_r["name"], "candidate_name": prim_c["name"],
                              "reference": rv, "candidate": cv}
        rep.accepted_by_score_rule = abs(cv - rv) <= score_rule_tol * max(abs(rv), 1e-12)

    if V:
        rep.verdict = METHOD_CHANGED
    elif G:
        rep.verdict = NO_EVIDENCE
    elif any(not d["within_tol"] for d in rep.result_deltas):
        rep.verdict = RESULTS_DIFFER
    elif not rm:
        rep.verdict = NO_EVIDENCE
        rep.notes.append("no metrics were recorded, results cannot be compared")
    return rep
