"""Entry point: link/file -> non-executing intake -> optional trusted local recovery.

Unknown downloaded repositories are intentionally blocked from executable mode.
An isolated container runner with independent reference evaluation is not yet
provided by this prototype.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .intake import intake_source
from .llm import make_llm
from .runner import run_script
from .agent import repair, MODES
from .verify import verify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m resource_ai.pilot',
                                 description='Research source link intake and trusted-local repair demo')
    ap.add_argument('source', help='GitHub repo/file URL, or local Python project/ZIP')
    ap.add_argument('--paper', default='')
    ap.add_argument('--out', default='official_pilot')
    ap.add_argument('--fetch', action='store_true', help='Fetch GitHub source; never execute it')
    ap.add_argument('--trusted-local', action='store_true',
                    help='Explicit consent to execute a trusted LOCAL Python case')
    ap.add_argument('--reference', help='Trusted reference .py script or trace .json')
    ap.add_argument('--repair', action='store_true', help='Run an LLM repair loop on a trusted LOCAL case')
    ap.add_argument('--model', default='qwen2.5-coder:7b')
    ap.add_argument('--mode', choices=MODES, default='guarded+kb')
    ap.add_argument('--max-iters', type=int, default=3)
    ap.add_argument('--script', default='experiment.py')
    a = ap.parse_args(argv)
    source_is_url = a.source.startswith(('https://','http://'))
    if (a.trusted_local or a.repair or a.reference) and source_is_url:
        ap.error('Remote-source execution is blocked. Import and review the URL first, then use a separately isolated execution environment.')
    if a.repair and not (a.trusted_local and a.reference):
        ap.error('--repair requires --trusted-local and --reference')
    if a.trusted_local and not a.reference:
        ap.error('--trusted-local requires --reference')
    result = intake_source(a.source, a.out, a.fetch, a.paper)
    print(f"\n[{result['status']}] {a.source}")
    print(f"Inventory: {result['files_observed']} files; report: {result['case_dir']}/CASE_REPORT.md")
    if not a.trusted_local:
        print('No experiment executed. To begin recovery, establish methodology and a restricted run environment.')
        return 0
    project = Path(result['project_dir'])
    script = (project / a.script).resolve()
    if not script.is_relative_to(project.resolve()) or not script.is_file():
        raise ValueError(f'Expected trusted script {a.script!r} under {project}')
    ref_file = Path(a.reference).expanduser().resolve()
    if not ref_file.is_file():
        raise FileNotFoundError(ref_file)
    if ref_file.suffix == '.py':
        ref_run = run_script(ref_file, timeout=120)
        if not ref_run.ok or not ref_run.trace:
            raise RuntimeError('Trusted reference did not execute and record a usable trace')
        reference = ref_run.trace
    elif ref_file.suffix == '.json':
        reference = json.loads(ref_file.read_text())
    else:
        raise ValueError('Reference must be .py or .json')
    initial = run_script(script, timeout=120)
    initial_verdict = verify(reference, initial)
    output = {'initial_runs': initial.ok, 'initial_verdict': initial_verdict.verdict,
              'initial_stderr_tail': '\n'.join(initial.stderr.splitlines()[-12:]),
              'scope': 'trusted local demonstration only'}
    if a.repair:
        print(f'Initial outcome: {initial_verdict.verdict}; calling {a.model}')
        try:
            repaired = repair(project, reference, make_llm(a.model),
                              mode=a.mode, script=a.script,
                              workdir=Path(result['case_dir']) / 'repaired_copy', max_iters=a.max_iters)
            output['repair'] = repaired.to_dict()
            print(f'Agent stopped: {repaired.stop_reason}; verdict: {repaired.final_verdict}')
        except Exception as exc:
            output['repair_error'] = f'{type(exc).__name__}: {str(exc)[:500]}'
            print(output['repair_error'])
    report = Path(result['case_dir']) / 'recovery_attempt.json'
    report.write_text(json.dumps(output, indent=2))
    print('Execution report:', report)
    return 0 if (not a.repair or output.get('repair', {}).get('final_verdict') == 'REPRODUCED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
