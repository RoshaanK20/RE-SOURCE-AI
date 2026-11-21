"""Command line interface.

  python -m resource_ai trace     <script.py> -o reference_trace.json
  python -m resource_ai diagnose  <script.py>
  python -m resource_ai verify    <script.py> --reference reference_trace.json
  python -m resource_ai repair    <project_dir> --reference reference_trace.json [--model qwen2.5-coder:7b]
                                  [--mode guarded+kb] [--script experiment.py] [--max-iters 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import MODES, repair
from .diagnose import diagnose
from .llm import make_llm
from .runner import run_script
from .verify import verify
from .intake import intake_source
from .batch import scan_research_list


def _load_reference(path: str) -> dict:
    p = Path(path)
    if p.suffix == ".py":                       # a runnable reference experiment
        r = run_script(p)
        if not r.ok:
            sys.exit(f"reference script failed:\n{r.stderr}")
        return r.trace
    return json.loads(p.read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="resource_ai")
    sp = ap.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("inspect", help="inspect a GitHub link or local file without executing it")
    i.add_argument("source")
    i.add_argument("--paper", default="")
    i.add_argument("--out", default="official_pilot")
    i.add_argument("--fetch", action="store_true", help="fetch GitHub content for static inspection")
    b = sp.add_parser("batch-inspect", help="automatically inspect every repository in a CSV or Markdown file")
    b.add_argument("list_file", help="CSV, Markdown, text or JSON research list")
    b.add_argument("--out", default="official_pilot/batch")
    b.add_argument("--fetch", action="store_true", help="download GitHub contents for static inspection only")
    t = sp.add_parser("trace", help="run a script and save its methodology trace")
    t.add_argument("script")
    t.add_argument("-o", "--out", default="reference_trace.json")
    d = sp.add_parser("diagnose", help="run a script and explain why it fails")
    d.add_argument("script")
    v = sp.add_parser("verify", help="run a script and compare it with a reference trace")
    v.add_argument("script")
    v.add_argument("--reference", required=True, help="trace .json or reference script .py")
    r = sp.add_parser("repair", help="repair a broken project with an LLM and verify it")
    r.add_argument("project_dir")
    r.add_argument("--reference", required=True, help="trace .json or reference script .py")
    r.add_argument("--script", default="experiment.py")
    r.add_argument("--model", default="qwen2.5-coder:7b", help="ollama model, or openai:<model>@<base_url>")
    r.add_argument("--mode", default="guarded+kb", choices=MODES)
    r.add_argument("--max-iters", type=int, default=5)
    r.add_argument("--workdir")
    r.add_argument("--report", help="write the JSON result here")
    a = ap.parse_args(argv)

    if a.cmd == "inspect":
        result = intake_source(a.source, a.out, fetch=a.fetch, paper=a.paper)
        print(f"{result['status']}: {result['case_dir']}/CASE_REPORT.md")
        print("No code executed. Scientific reproduction has not been established.")
        return 0
    if a.cmd == "batch-inspect":
        summary = scan_research_list(a.list_file, out=a.out, fetch=a.fetch)
        print(f"Processed {summary['total']} research links; status counts: {summary['status_counts']}")
        return 1 if summary["status_counts"].get("INTAKE_ERROR", 0) else 0
    if a.cmd == "trace":
        res = run_script(a.script)
        Path(a.out).write_text(json.dumps(res.trace, indent=1))
        print(f"exit code {res.returncode}; {len((res.trace or {}).get('events', []))} events -> {a.out}")
        return res.returncode
    if a.cmd == "diagnose":
        res = run_script(a.script)
        dg = diagnose(res.stderr, Path(a.script).resolve().parent, a.script, res.timed_out)
        print(dg.summary())
        for k in dg.knowledge:
            print(f"  hint: {k['title']} -> {k['fix']}" + (f"  (warning: {k['warning']})" if k["warning"] else ""))
        return 0
    if a.cmd == "verify":
        rep = verify(_load_reference(a.reference), run_script(a.script))
        print(json.dumps(rep.to_dict(), indent=2))
        return 0 if rep.verdict == "REPRODUCED" else 1
    if a.cmd == "repair":
        res = repair(a.project_dir, _load_reference(a.reference), make_llm(a.model), a.mode,
                     script=a.script, max_iters=a.max_iters, workdir=a.workdir)
        print(f"\nmode={res.mode} accepted={res.accepted} final_verdict={res.final_verdict} "
              f"iterations={res.iterations}\n{res.diff}")
        if a.report:
            Path(a.report).write_text(json.dumps(res.to_dict(), indent=2))
        return 0 if res.final_verdict == "REPRODUCED" else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
