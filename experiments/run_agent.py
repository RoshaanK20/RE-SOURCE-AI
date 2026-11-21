"""Experiment 2: LLM repair agent on the benchmark, comparing acceptance modes.

    # real run with a free local model (Ollama must be running):
    python experiments/run_agent.py --model qwen2.5-coder:7b
    # several models / repeated runs:
    python experiments/run_agent.py --model qwen2.5-coder:7b --model qwen3.5:9b --seeds 0 1 2 --temperature 0.2
    # pipeline check without any LLM (scripted answers):
    python experiments/run_agent.py --dry-run trap-then-oracle

Results are appended to results/agent_runs.jsonl (re-running skips finished runs, so it can resume
after Colab disconnects). Then: python experiments/make_tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resource_ai.agent import MODES, RepairAgent  # noqa: E402
from resource_ai.llm import ScriptedLLM, make_llm  # noqa: E402
from resource_ai.patching import to_blocks  # noqa: E402
from resource_ai.runner import run_script  # noqa: E402


def scripted(case_dir: Path, kind: str) -> ScriptedLLM:
    """Simulated LLMs for testing the pipeline: 'oracle', 'trap', 'trap-then-oracle'."""
    meta = json.loads((case_dir / "case.json").read_text())
    broken = (case_dir / "broken" / "experiment.py").read_text()
    oracle = (case_dir / "fixes" / "oracle.py").read_text()
    trap_file = next((f["file"] for f in meta["fixes"] if f["label"] == "changes_method"), None)
    trap = (case_dir / "fixes" / trap_file).read_text() if trap_file else oracle
    if kind == "oracle":
        return ScriptedLLM([to_blocks(broken, oracle)], "scripted-oracle")
    if kind == "trap":
        return ScriptedLLM([to_blocks(broken, trap)] * 5, "scripted-trap")
    return ScriptedLLM([to_blocks(broken, trap), to_blocks(trap, oracle)], "scripted-trap-then-oracle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="Ollama model tag (repeatable)")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--cases", default=str(ROOT / "benchmark" / "cases"))
    ap.add_argument("--only", nargs="*", help="run only these case ids")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "agent_runs.jsonl"))
    ap.add_argument("--dry-run", choices=["oracle", "trap", "trap-then-oracle"])
    a = ap.parse_args()

    models = [f"dry:{a.dry_run}"] if a.dry_run else (a.model or ["qwen2.5-coder:7b"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["case"], r["mode"], r["seed"]))

    cases = sorted(p for p in Path(a.cases).iterdir() if (p / "case.json").exists())
    if a.only:
        cases = [c for c in cases if c.name in set(a.only)]
    refs = {}
    for case_dir in cases:
        r = run_script(case_dir / "reference" / "experiment.py")
        assert r.ok, f"reference failed for {case_dir.name}:\n{r.stderr}"
        refs[case_dir.name] = r.trace

    total = len(models) * len(cases) * len(a.modes) * len(a.seeds)
    k = 0
    for model in models:
        for seed in a.seeds:
            for case_dir in cases:
                meta = json.loads((case_dir / "case.json").read_text())
                for mode in a.modes:
                    k += 1
                    key = (model, case_dir.name, mode, seed)
                    if key in done:
                        continue
                    llm = scripted(case_dir, a.dry_run) if a.dry_run else make_llm(model, seed=seed, temperature=a.temperature)
                    print(f"[{k}/{total}] {model} seed={seed} {case_dir.name} mode={mode}", flush=True)
                    with tempfile.TemporaryDirectory() as tmp:
                        agent = RepairAgent(case_dir / "broken", refs[case_dir.name], llm, mode,
                                            workdir=Path(tmp) / "work", max_iters=a.max_iters)
                        t0 = time.time()
                        try:
                            res = agent.run().to_dict()
                            err = None
                        except Exception as e:          # keep going; record the failure
                            res, err = {}, f"{type(e).__name__}: {e}"
                            print("   ERROR", err, flush=True)
                            if "not reachable" in str(e):
                                raise
                    row = {"model": model, "case": case_dir.name, "category": meta["category"], "family": meta["family"],
                           "mode": mode, "seed": seed, "temperature": a.temperature, "error": err,
                           "wall_seconds": time.time() - t0, **res}
                    with out.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    if res:
                        print(f"   accepted={res['accepted']} final={res['final_verdict']} "
                              f"silent_change={res['silent_method_change']} iters={res['iterations']}", flush=True)


if __name__ == "__main__":
    main()
