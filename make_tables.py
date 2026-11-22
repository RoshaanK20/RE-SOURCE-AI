"""Turn results/agent_runs.jsonl into paper tables (markdown + CSV) and a figure.

    python experiments/make_tables.py [--runs results/agent_runs.jsonl] [--out results/tables]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODE_ORDER = ["naive", "score", "guarded", "guarded+kb"]
MODE_LABEL = {"naive": "Run-only", "score": "Score ±10%", "guarded": "RE-SOURCE verifier",
              "guarded+kb": "RE-SOURCE verifier + KB"}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if not r.get("error")]
    f = lambda pred: sum(1 for r in ok if pred(r)) / n if n else 0.0  # noqa: E731
    return {
        "n": n,
        "errors": n - len(ok),
        "runs": f(lambda r: r["final_runs"]),
        "accepted": f(lambda r: r["accepted"]),
        "silent_method_change": f(lambda r: r["silent_method_change"]),
        "reproduced": f(lambda r: r["final_verdict"] == "REPRODUCED"),
        "honest_failure": f(lambda r: not r["accepted"]),
        "mean_llm_calls": sum(r.get("llm_calls", 0) for r in ok) / max(len(ok), 1),
        "mean_seconds": sum(r.get("seconds", 0) for r in ok) / max(len(ok), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "agent_runs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables"))
    a = ap.parse_args()
    rows = load(Path(a.runs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["mode"])].append(r)
    models = sorted({r["model"] for r in rows})
    md = ["# Repair agent results", "",
          "*Accepted* = the mode's own acceptance rule said done. *Silent method change* = accepted and runs, but the "
          "independent verifier found the methodology differs from the original. *Verified reproduction* = final code "
          "has the original methodology and results.", ""]
    csv_rows = []
    for m in models:
        md += [f"## {m}", "", "| Mode | Runs | Accepted | Silent method change ⚠️ | Verified reproduction ✅ | LLM calls | Time (s) |",
               "|---|---|---|---|---|---|---|"]
        for mode in MODE_ORDER:
            if (m, mode) not in by:
                continue
            s = agg(by[(m, mode)])
            csv_rows.append({"model": m, "mode": mode, **s})
            md.append(f"| {MODE_LABEL[mode]} | {s['runs']:.0%} | {s['accepted']:.0%} | **{s['silent_method_change']:.0%}** | "
                      f"**{s['reproduced']:.0%}** | {s['mean_llm_calls']:.1f} | {s['mean_seconds']:.0f} |")
        md.append("")
        # per category
        cats = sorted({r["category"] for r in rows if r["model"] == m})
        md += ["| Category | " + " | ".join(MODE_LABEL[x] for x in MODE_ORDER if (m, x) in by) + " |",
               "|---|" + "---|" * sum((m, x) in by for x in MODE_ORDER)]
        for c in cats:
            cells = []
            for mode in MODE_ORDER:
                if (m, mode) not in by:
                    continue
                sub = [r for r in by[(m, mode)] if r["category"] == c]
                s = agg(sub)
                cells.append(f"repro {s['reproduced']:.0%} / silent {s['silent_method_change']:.0%} (n={s['n']})")
            md.append(f"| {c} | " + " | ".join(cells) + " |")
        md.append("")
        # per case
        md += ["| Case | " + " | ".join(MODE_LABEL[x] for x in MODE_ORDER if (m, x) in by) + " |",
               "|---|" + "---|" * sum((m, x) in by for x in MODE_ORDER)]
        for case in sorted({r["case"] for r in rows if r["model"] == m}):
            cells = []
            for mode in MODE_ORDER:
                if (m, mode) not in by:
                    continue
                rs = [r for r in by[(m, mode)] if r["case"] == case]
                cells.append(", ".join(("⚠️silent" if r.get("silent_method_change") else
                                        "✅" if r.get("final_verdict") == "REPRODUCED" else
                                        ("❌" + str(r.get("final_verdict") or r.get("error", "")))[:22]) for r in rs))
            md.append(f"| {case} | " + " | ".join(cells) + " |")
        md.append("")
    (out / "agent_results.md").write_text("\n".join(md))
    if csv_rows:
        with (out / "agent_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    try:
        figure(csv_rows, out / "agent_results.png")
    except Exception as e:  # matplotlib optional
        print("figure skipped:", e)
    print("\n".join(md[:12 + 6 * len(models)]))
    print(f"\nwritten to {out}")


def figure(csv_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in csv_rows})
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        rs = [r for r in csv_rows if r["model"] == m]
        rs.sort(key=lambda r: MODE_ORDER.index(r["mode"]))
        labels = [MODE_LABEL[r["mode"]].replace(" + ", "\n+ ").replace("RE-SOURCE ", "RE-SOURCE\n") for r in rs]
        x = range(len(rs))
        ax.bar([i - 0.2 for i in x], [100 * r["reproduced"] for r in rs], 0.4, label="Verified reproduction", color="#2b7a4b")
        ax.bar([i + 0.2 for i in x], [100 * r["silent_method_change"] for r in rs], 0.4, label="Silent method change", color="#c44e52")
        ax.set_xticks(list(x), labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of repair runs")
        ax.set_title(m, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
