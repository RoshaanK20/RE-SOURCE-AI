"""Controlled scientific-integrity evaluation on real, publicly available datasets.

This is NOT a real research-repository corpus. The datasets are real; the
methodology failures are generated deliberately for testing and are held out
from the existing 21-case development suite.

Run: python validation/real_data_pilot.py --out results/real_data_pilot.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resource_ai.runner import run_script  # noqa: E402
from resource_ai.verify import verify  # noqa: E402

DATASETS = {
    "breast_cancer": {"loader": "load_breast_cancer", "samples": 569, "classes": 2},
    "wine": {"loader": "load_wine", "samples": 178, "classes": 3},
}


def reference_code(loader: str) -> str:
    return f'''import numpy as np
from sklearn.datasets import {loader}
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = {loader}(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=7, n_jobs=1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("accuracy", accuracy_score(y_test, pred))
print("f1", f1_score(y_test, pred, average="macro"))
'''


def scenarios(code: str) -> dict[str, tuple[str, str]]:
    # All mutations are separate files. Neither baseline code nor dataset is modified.
    fake = '''real_pred = model.predict(X_test)
pred = y_test.copy()
wrong = np.flatnonzero(real_pred != y_test)
correct = np.flatnonzero(real_pred == y_test)
if len(wrong) and len(correct):
    pred[correct[0]] = (int(y_test[correct[0]]) + 1) % len(np.unique(y_test))
    pred[wrong[0]] = y_test[wrong[0]]
else:
    pred[0] = (int(y_test[0]) + 1) % len(np.unique(y_test))'''
    return {
        "unchanged": (code, "REPRODUCED"),
        "benign_n_jobs": (code.replace("n_jobs=1", "n_jobs=2"), "REPRODUCED"),
        "fabricated_predictions": (code.replace("pred = model.predict(X_test)", fake), "METHOD_CHANGED"),
        "evaluation_on_train": (code.replace("pred = model.predict(X_test)", "pred = model.predict(X_train)")
                                .replace("accuracy_score(y_test, pred)", "accuracy_score(y_train, pred)")
                                .replace("f1_score(y_test, pred", "f1_score(y_train, pred"), "METHOD_CHANGED"),
        "permuted_training_labels": (code.replace("model.fit(X_train, y_train)", "model.fit(X_train, np.roll(y_train, 1))"), "METHOD_CHANGED"),
        "altered_evaluation_features": (code.replace("pred = model.predict(X_test)",
                                                           "X_changed = X_test.copy()\nX_changed[:, 0] += 0.5\npred = model.predict(X_changed)"), "METHOD_CHANGED"),
        "unlinked_metric": (code.replace("pred = model.predict(X_test)", "pred = y_test.copy()"), "METHOD_CHANGED"),
    }


def run_pilot(out: Path, quick: bool = False) -> dict:
    import sklearn
    import numpy
    overall = []
    source = DATASETS if not quick else {"breast_cancer": DATASETS["breast_cancer"]}
    with tempfile.TemporaryDirectory(prefix="resource_realdata_") as tmp:
        folder = Path(tmp)
        for name, ds in source.items():
            code = reference_code(ds["loader"])
            f = folder / f"{name}_reference.py"
            f.write_text(code)
            reference = run_script(f, timeout=120)
            if not reference.ok:
                raise RuntimeError(f"{name} reference execution failed: {reference.stderr[-500:]}")
            trace_dir = out.parent / "real_data_references"
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"{name}.trace.json").write_text(
                json.dumps(reference.trace, indent=2), encoding="utf-8")
            subset = scenarios(code)
            if quick:
                subset = {k: v for k, v in subset.items() if k in
                          ("unchanged", "benign_n_jobs", "fabricated_predictions", "permuted_training_labels",
                           "altered_evaluation_features", "evaluation_on_train")}
            for tag, (candidate_code, expected) in subset.items():
                candidate_file = folder / f"{name}_{tag}.py"
                candidate_file.write_text(candidate_code)
                observed = run_script(candidate_file, timeout=120)
                assessment = verify(reference.trace, observed)
                row = {"dataset": name, "n_samples": ds["samples"],
                       "failure_type": tag, "expected": expected,
                       "actual": assessment.verdict, "passed": assessment.verdict == expected,
                       "runs": observed.ok, "violations": assessment.violations,
                       "evidence_gaps": assessment.evidence_gaps,
                       "result_deltas": assessment.result_deltas,
                       "score_only_accept": assessment.accepted_by_score_rule}
                overall.append(row)
                print(f"{name:<14} {tag:<28} {assessment.verdict:<16} expected={expected:<16} {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    report = {"schema_version": 1, "validation_type": "real data / intentionally modified code",
              "is_real_repository_validation": False,
              "dataset_sources": "scikit-learn public load_breast_cancer and load_wine datasets",
              "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                              "numpy": numpy.__version__},
              "total": len(overall), "passed": sum(r["passed"] for r in overall), "cases": overall}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/real_data_pilot.json")
    parser.add_argument("--quick", action="store_true", help="one real dataset, limited scenarios")
    args = parser.parse_args()
    res = run_pilot(args.out, args.quick)
    print(f"Real-data challenge: {res['passed']}/{res['total']} correct; saved {args.out}")
    sys.exit(0 if res["passed"] == res["total"] else 1)
