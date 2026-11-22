"""Rebuild the paper's Experiment 2 numbers, tables, statistics and figure from the raw run logs.

Inputs (all in results/llm_runs/):
  runs_temperature0.jsonl          qwen2.5-coder 7B and 14B, temperature 0      (168 runs)
  runs_temperature02_3seeds.jsonl  qwen2.5-coder 7B, temperature 0.2, seeds 0-2 (252 runs)
  patch_audit.csv                  our manual review of every distinct patch the models wrote

Outputs (same folder): summary.md, summary.csv, figure_main.png

    python experiments/paper_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ["naive", "score", "guarded", "guarded+kb"]
LABEL = {"naive": "Run-only", "score": "Score ±10%", "guarded": "RE-SOURCE verifier",
         "guarded+kb": "RE-SOURCE verifier + KB"}


def patch_lines(run: dict) -> str:
    """The +/- lines of a run's final diff: this is how the audit file identifies a patch."""
    return "\n".join(l for l in run["diff"].splitlines()
                     if l[:1] in "+-" and not l.startswith(("+++", "---")))


def load_runs(folder: Path) -> list[dict]:
    rows = []
    for name in ("runs_temperature0.jsonl", "runs_temperature02_3seeds.jsonl"):
        rows += [json.loads(l) for l in (folder / name).read_text().splitlines() if l.strip()]
    return rows


def load_audit(folder: Path) -> dict:
    with (folder / "patch_audit.csv").open(newline="") as f:
        return {(r["case"], r["patch"]): r["audit_label"] for r in csv.DictReader(f)}


def make_audit_template(rows: list[dict], folder: Path) -> None:
    """Write one row per distinct patch, pre-filled from the verifier; reviewers then edit audit_label."""
    seen = {}
    for r in rows:
        key = (r["case"], patch_lines(r))
        if key not in seen:
            v = r["final_verdict"]
            seen[key] = {"case": r["case"], "runs": 0, "verifier_verdict": v, "patch": key[1],
                         "audit_label": ("fails (no judgement needed)" if v == "FAILS_TO_RUN" else
                                         "agree: correct repair" if v == "REPRODUCED" else
                                         "agree: methodology changed")}
        seen[key]["runs"] += 1
    with (folder / "patch_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "runs", "verifier_verdict", "audit_label", "patch"])
        w.writeheader()
        w.writerows(sorted(seen.values(), key=lambda x: (x["case"], x["patch"])))


def mcnemar(rows: list[dict], a: str, b: str, outcome) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on runs paired by (model, case, seed, temperature)."""
    paired = defaultdict(dict)
    for r in rows:
        paired[(r["model"], r["case"], r["seed"], r["temperature"])][r["mode"]] = r
    only_a = only_b = 0
    for runs in paired.values():
        if a in runs and b in runs:
            x, y = outcome(runs[a]), outcome(runs[b])
            only_a += x and not y
            only_b += y and not x
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "llm_runs"))
    ap.add_argument("--init-audit", action="store_true", help="(re)create patch_audit.csv from the verifier verdicts")
    args = ap.parse_args()
    folder = Path(args.runs)
    rows = load_runs(folder)
    if args.init_audit or not (folder / "patch_audit.csv").exists():
        make_audit_template(rows, folder)
    audit = load_audit(folder)
    missing = {(r["case"], patch_lines(r)) for r in rows} - set(audit)
    if missing:
        raise SystemExit(f"{len(missing)} patches are not in patch_audit.csv; run with --init-audit and review them")

    def correct(r):            # final code is the original experiment (audited)
        label = audit[(r["case"], patch_lines(r))]
        return r["final_runs"] and (label == "agree: correct repair" or label.startswith("EQUIVALENT"))

    def silent(r):             # accepted by its mode, runs, but changed the science (audited)
        return r["accepted"] and r["final_runs"] and not correct(r)

    false_alarms = sum(1 for (c, p), lab in audit.items() if lab.startswith("EQUIVALENT"))
    missed = sum(1 for (c, p), lab in audit.items() if lab.startswith("WRONG"))
    configs = [("7B, T=0", [r for r in rows if r["model"].endswith(":7b") and r["temperature"] == 0]),
               ("14B, T=0", [r for r in rows if r["model"].endswith(":14b")]),
               ("7B, T=0.2 (3 seeds)", [r for r in rows if r["temperature"] == 0.2]),
               ("All runs", rows)]

    md = ["# Experiment 2: LLM repair results", "",
          f"{len(rows)} repair runs with free local models (qwen2.5-coder via Ollama) on the 21-case benchmark. "
          f"Every distinct patch was reviewed by hand (`patch_audit.csv`): the verifier raised "
          f"{false_alarms} false alarm(s) and missed {missed} real change(s).", "",
          "| Config | Mode | Runs | Silent method change | Verified reproduction | LLM calls |",
          "|---|---|---|---|---|---|"]
    table = []
    for name, rs in configs:
        for m in MODES:
            x = [r for r in rs if r["mode"] == m]
            row = {"config": name, "mode": m, "n": len(x),
                   "runs": sum(r["final_runs"] for r in x) / len(x),
                   "silent_method_change": sum(map(silent, x)) / len(x),
                   "verified_reproduction": sum(map(correct, x)) / len(x),
                   "verifier_reproduced_unaudited": sum(r["final_verdict"] == "REPRODUCED" for r in x) / len(x),
                   "mean_llm_calls": statistics.mean(r["llm_calls"] for r in x)}
            table.append(row)
            md.append(f"| {name} | {LABEL[m]} | {row['runs']:.0%} | {row['silent_method_change']:.0%} | "
                      f"{row['verified_reproduction']:.0%} | {row['mean_llm_calls']:.2f} |")
    md += ["", "Variation across the 3 seeds (7B, T=0.2), mean ± sd:", ""]
    t02 = configs[2][1]
    for m in MODES:
        rep = [sum(map(correct, [r for r in t02 if r["mode"] == m and r["seed"] == s])) / 21 for s in range(3)]
        sil = [sum(map(silent, [r for r in t02 if r["mode"] == m and r["seed"] == s])) / 21 for s in range(3)]
        md.append(f"- {LABEL[m]}: reproduction {statistics.mean(rep):.0%} ± {statistics.stdev(rep):.0%}, "
                  f"silent change {statistics.mean(sil):.0%} ± {statistics.stdev(sil):.0%}")
    md += ["", "Paired exact McNemar tests over all runs (pairs = same model, case, seed, temperature):", ""]
    for a, b, f, what in [("naive", "guarded+kb", correct, "verified reproduction"),
                          ("score", "guarded+kb", correct, "verified reproduction"),
                          ("naive", "guarded", correct, "verified reproduction"),
                          ("guarded", "guarded+kb", correct, "verified reproduction (effect of the KB)"),
                          ("naive", "guarded+kb", silent, "silent method change"),
                          ("score", "guarded+kb", silent, "silent method change")]:
        x, y, p = mcnemar(rows, a, b, f)
        md.append(f"- {what}: {LABEL[a]} only {x}, {LABEL[b]} only {y}, p = {p:.2g}")
    md += ["", "Note: the verifier modes only accept repairs the verifier passes, so their 0% silent-change rate "
           "holds by construction. The informative comparisons are how often the baselines silently change the "
           "experiment, and how much the verifier's feedback and the knowledge base raise verified reproduction."]
    (folder / "summary.md").write_text("\n".join(md) + "\n")
    with (folder / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    figure(configs, correct, silent, folder / "figure_main.png")
    print("\n".join(md))


def figure(configs, correct, silent, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"naive": "Run-only", "score": "Score ±10%\n(prior work)", "guarded": "RE-SOURCE\nverifier",
              "guarded+kb": "RE-SOURCE\nverifier + KB"}
    series = [("7B (T=0)", configs[0][1], "#9ecae1", False),
              ("7B (T=0.2, mean ± sd, 3 seeds)", configs[2][1], "#4292c6", True),
              ("14B (T=0)", configs[1][1], "#08519c", False)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, f, title in [(axes[0], silent, "Accepted repairs that silently changed the methodology"),
                         (axes[1], correct, "Verified reproductions (audited)")]:
        for j, (name, rs, color, seeds) in enumerate(series):
            xs = [i + (j - 1) * 0.26 for i in range(4)]
            if seeds:
                per = [[100 * sum(map(f, [r for r in rs if r["mode"] == m and r["seed"] == s])) / 21
                        for s in range(3)] for m in MODES]
                vals = [statistics.mean(p) for p in per]
                ax.bar(xs, vals, 0.26, yerr=[statistics.stdev(p) for p in per], capsize=2,
                       error_kw={"lw": 1}, label=name, color=color)
            else:
                vals = [100 * sum(map(f, [r for r in rs if r["mode"] == m])) / 21 for m in MODES]
                ax.bar(xs, vals, 0.26, label=name, color=color)
            for x, v in zip(xs, vals):
                if v == 0:
                    ax.text(x, 1.2, "0", ha="center", fontsize=7, color="#555")
        ax.set_xticks(range(4), [labels[m] for m in MODES], fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel("% of repair runs (21 cases)", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)


if __name__ == "__main__":
    main()
