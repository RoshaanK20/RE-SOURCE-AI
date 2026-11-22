"""Experiment 1 (no LLM needed): can each acceptance rule tell a correct repair from a trap?

For every benchmark case we run the reference, the broken code, and every labelled fix, then ask
three acceptance rules whether to accept the fix:
  RUNS     - accept if it executes without error                     (typical "make it run" agent)
  SCORE10  - accept if it runs and the primary score is within 10%   (prior work, e.g. MLEModernizer)
  VERIFIER - accept only if RE-SOURCE verdict is REPRODUCED          (ours)
Ground truth: fixes labelled "equivalent" should be accepted, "changes_method" rejected.

    python experiments/evaluate_verifier.py [--cases benchmark/cases] [--out results/verifier_eval]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resource_ai.runner import copy_project, run_script  # noqa: E402
from resource_ai.verify import REPRODUCED, verify  # noqa: E402

RULES = ("RUNS", "SCORE10", "VERIFIER")


def evaluate(cases_dir: Path) -> list[dict]:
    rows = []
    for case_dir in sorted(p for p in cases_dir.iterdir() if (p / "case.json").exists()):
        meta = json.loads((case_dir / "case.json").read_text())
        ref = run_script(case_dir / "reference" / "experiment.py")
        assert ref.ok, f"{meta['id']}: reference failed\n{ref.stderr}"
        broken = run_script(case_dir / "broken" / "experiment.py")
        with tempfile.TemporaryDirectory() as tmp:
            for fx in meta["fixes"]:       # silent cases include the unmodified code as "leave_as_is"
                work = copy_project(case_dir / "broken", Path(tmp) / fx["file"].replace(".py", ""))
                (work / "experiment.py").write_text((case_dir / "fixes" / fx["file"]).read_text())
                run = run_script(work / "experiment.py")
                rep = verify(ref.trace, run)
                rows.append({
                    "case": meta["id"], "category": meta["category"], "fix": fx["file"], "label": fx["label"],
                    "runs": run.ok, "verdict": rep.verdict,
                    "RUNS": run.ok, "SCORE10": rep.accepted_by_score_rule, "VERIFIER": rep.verdict == REPRODUCED,
                    "violations": rep.violations, "result_deltas": rep.result_deltas,
                    "primary_metric": rep.primary_metric, "broken_crashes": not broken.ok,
                    "stderr_tail": run.tail(3) if not run.ok else "",
                })
        print(f"{meta['id']:<28} broken={'crash' if not broken.ok else 'RUNS (silent)'}", flush=True)
    return rows


def summarise(rows: list[dict]) -> dict:
    out = {}
    for rule in RULES:
        tp = sum(r[rule] and r["label"] == "equivalent" for r in rows)
        fn = sum((not r[rule]) and r["label"] == "equivalent" for r in rows)
        fp = sum(r[rule] and r["label"] == "changes_method" for r in rows)
        tn = sum((not r[rule]) and r["label"] == "changes_method" for r in rows)
        n_eq, n_bad = tp + fn, fp + tn
        out[rule] = {"accept_correct": tp, "reject_correct": fn, "accept_trap": fp, "reject_trap": tn,
                     "correct_fix_acceptance": tp / n_eq if n_eq else None,
                     "trap_acceptance": fp / n_bad if n_bad else None,
                     "accuracy": (tp + tn) / (n_eq + n_bad)}
    return out


def markdown(rows, summ) -> str:
    lines = ["## Acceptance rules vs ground truth", "",
             "| Rule | Correct fixes accepted | Trap fixes accepted (silent methodology change) | Overall accuracy |",
             "|---|---|---|---|"]
    for rule in RULES:
        s = summ[rule]
        lines.append(f"| {rule} | {s['accept_correct']}/{s['accept_correct'] + s['reject_correct']} "
                     f"({s['correct_fix_acceptance']:.0%}) | {s['accept_trap']}/{s['accept_trap'] + s['reject_trap']} "
                     f"({s['trap_acceptance']:.0%}) | {s['accuracy']:.0%} |")
    lines += ["", "## Per fix", "", "| Case | Fix | Label | Runs | Score ±10% | Verifier | First reason |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        why = (r["violations"][0] if r["violations"] else
               next((f"{d['metric']} {d['reference']:.4g}→{d['candidate']:.4g}" for d in r["result_deltas"] if not d["within_tol"]), ""))
        lines.append(f"| {r['case']} | {r['fix']} | {r['label']} | {'✅' if r['RUNS'] else '❌'} | "
                     f"{'✅' if r['SCORE10'] else '❌'} | {r['verdict']} | {why[:110]} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--cases", default=str(root / "benchmark" / "cases"))
    ap.add_argument("--out", default=str(root / "results" / "verifier_eval"))
    a = ap.parse_args()
    rows = evaluate(Path(a.cases))
    summ = summarise(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    (out / "report.md").write_text(markdown(rows, summ))
    print(markdown(rows, summ).split("## Per fix")[0])


if __name__ == "__main__":
    main()
