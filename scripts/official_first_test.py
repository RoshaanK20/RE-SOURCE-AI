"""Official first controlled end-to-end pipeline test (NOT an LLM research result).

Uses Scikit-learn's real Diabetes dataset, an intentionally outdated API setting,
and ScriptedLLM to verify that the existing patch-execute-verify workflow runs.
Unfamiliar broken research repositories are NOT tested by this demonstration.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from resource_ai.agent import repair
from resource_ai.llm import ScriptedLLM
from resource_ai.runner import run_script
from resource_ai.verify import verify


def main(argv=None):
    p = argparse.ArgumentParser(description='Controlled scripted repair demonstration')
    p.add_argument('--out', default='results/first_official_test.json')
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    case = root / 'benchmark' / 'cases' / 'RF02_max_features_auto_reg'
    baseline = case / 'reference' / 'experiment.py'
    damaged = case / 'broken' / 'experiment.py'
    ref_run = run_script(baseline, timeout=120)
    if not ref_run.ok or not ref_run.trace:
        raise RuntimeError('Reference execution failed; cannot compare against scientific methodology')
    broken_run = run_script(damaged, timeout=120)
    baseline_report = verify(ref_run.trace, broken_run)
    reply = '''experiment.py
<<<<<<< SEARCH
max_features="auto"
=======
max_features=1.0
>>>>>>> REPLACE'''
    llm = ScriptedLLM([reply], name='scripted-demonstration-NOT-LLM')
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired = repair(case / 'broken', ref_run.trace, llm, 'guarded+kb',
                      workdir=output_path.parent / '_first_official_repaired',
                      max_iters=1, timeout=120, log=lambda *x: None)
    output = {
        'experiment_kind': 'CONTROLLED_PIPELINE_TEST',
        'paper_repository': None,
        'dataset': 'Scikit-learn Diabetes (real published dataset, bundled locally)',
        'failure_kind': 'Deliberately introduced legacy API incompatibility',
        'source_case': 'RF02_max_features_auto_reg',
        'model_backend': 'ScriptedLLM, predetermined patch: not an autonomous LLM result',
        'date_utc': datetime.now(timezone.utc).isoformat(),
        'reference_execution_ok': ref_run.ok,
        'broken_execution_ok': broken_run.ok,
        'initial_verdict': baseline_report.verdict,
        'repaired_execution_ok': repaired.final_runs,
        'repaired_verdict': repaired.final_verdict,
        'repair_accepted_by_existing_verifier': repaired.accepted,
        'scientific_paper_reproduced': False,
        'claim_limit': 'Tests the pipeline on a constructed failure; not verified published-repository recovery.',
        'diff': repaired.diff,
        'observed_metric': repaired.final_report.get('primary_metric'),
        'observed_metric_deltas': repaired.final_report.get('result_deltas'),
    }
    output_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f"Baseline runs: {ref_run.ok} | Broken runs: {broken_run.ok} | Repair verdict: {repaired.final_verdict}")
    print('Full log:', output_path)
    return 0 if ref_run.ok and not broken_run.ok and repaired.final_runs and repaired.accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
