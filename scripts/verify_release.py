"""One-command evidence generation on the user's own supported Python environment.

This command runs no external GitHub code or downloaded research papers.
Outputs are computed from subprocess exit codes, never hardcoded as successes.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOCUSED = [
    'tests/test_dashboard_console.py',
    'tests/test_batch_intake.py',
    'tests/test_official_pilot_intake.py',
    'tests/test_integrity_provenance.py',
    'tests/test_ollama_integration.py',
    'tests/test_ollama_agent_e2e.py',
    'tests/test_release_evidence.py',
    'tests/test_patch_safety_rc31.py',
    'tests/test_scientific_ui_paths.py',
    'tests/test_optional_docker.py',
    'tests/test_scientific_contract.py',
    'tests/test_framework_trial.py',
]


def _step(name: str, command: list[str], timeout: int) -> dict:
    try:
        obj = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        result = {'name': name, 'success': obj.returncode == 0, 'exit_code': obj.returncode,
                  'stdout_tail': '\n'.join(obj.stdout.splitlines()[-36:]),
                  'stderr_tail': '\n'.join(obj.stderr.splitlines()[-25:])}
    except subprocess.TimeoutExpired as exc:
        result = {'name': name, 'success': False, 'exit_code': None,
                  'timed_out': True, 'error': f'Exceeded {timeout} seconds'}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Generate reproducible controlled-pilot evidence')
    parser.add_argument('--full-suite', action='store_true', help='Also execute the longer legacy test suite')
    opts = parser.parse_args(argv)
    dest = ROOT / 'results'
    dest.mkdir(parents=True, exist_ok=True)
    result = {'version': 'pilot',
              'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
              'environment': {'python': platform.python_version(), 'platform': platform.platform()},
              'live_ollama_tested': False, 'live_github_repositories_reproduced': 0,
              'steps': []}
    work = [
        ('focused_regression_tests', [sys.executable, '-m', 'pytest', '-q', *FOCUSED], 240),
        ('real_dataset_integrity', [sys.executable, 'validation/real_data_pilot.py',
                                    '--out', 'results/official_validation_run.json'], 240),
        ('scripted_repair_pipeline', [sys.executable, '-m', 'scripts.official_first_test',
                                     '--out', 'results/official_controlled_test.json'], 240),
    ]
    if opts.full_suite:
        work.append(('complete_test_suite', [sys.executable, '-m', 'pytest', '-q', 'tests'], 600))
    for name, cmd, limit in work:
        print(f'\n=== {name} ===', flush=True)
        row = _step(name, cmd, limit)
        result['steps'].append(row)
        print('PASS' if row['success'] else 'FAIL', row.get('stdout_tail', '')[-400:], flush=True)
        (dest / 'release_evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    result['all_requested_steps_passed'] = all(x['success'] for x in result['steps'])
    (dest / 'release_evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return 0 if result['all_requested_steps_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
