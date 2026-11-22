"""Re-check the recorded LLM repairs with the current verifier.

Each run log stores the final patch as a diff. This script applies every distinct patch to the
broken benchmark script again, runs it, and compares today's verdict with the recorded one.
Useful after changing the recorder or verifier: the paper's verdicts should not move.

    python experiments/reverify_runs.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resource_ai.runner import run_script  # noqa: E402
from resource_ai.verify import verify  # noqa: E402

CASES = ROOT / "benchmark" / "cases"
LOGS = ROOT / "results" / "llm_runs"


def main() -> int:
    rows = [json.loads(l) for f in sorted(LOGS.glob("runs_*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    references, verdicts = {}, {}
    for r in rows:
        case = r["case"]
        if case not in references:
            references[case] = run_script(CASES / case / "reference" / "experiment.py").trace
        key = (case, r["diff"])
        if key in verdicts:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "experiment.py").write_text((CASES / case / "broken" / "experiment.py").read_text())
            if r["diff"].strip():
                (work / "patch.diff").write_text(r["diff"])
                applied = subprocess.run(["patch", "-s", "-p1", "-i", "patch.diff"], cwd=work,
                                         capture_output=True, text=True)
                if applied.returncode:
                    verdicts[key] = "PATCH_FAILED"
                    continue
            verdicts[key] = verify(references[case], run_script(work / "experiment.py")).verdict
    changed = Counter((r["final_verdict"], verdicts[(r["case"], r["diff"])]) for r in rows
                      if r["final_verdict"] != verdicts[(r["case"], r["diff"])])
    print(f"{len(verdicts)} distinct patches re-run for {len(rows)} recorded runs")
    print(f"verdicts unchanged: {len(rows) - sum(changed.values())}/{len(rows)}")
    for (old, new), n in changed.items():
        print(f"  {old} -> {new}: {n} runs")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
