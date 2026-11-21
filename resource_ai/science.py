"""Framework-independent scientific-integrity gate (experimental).

This gate does not inspect framework-specific APIs. It checks an explicitly
approved, paper-specific methodology contract, the original and candidate
source trees, preserved data/configuration files, *exactly* approved changes,
and a metric recomputed by an evaluator outside the agent's editable tree.

A pass is only a pass for these specified checks. It is not a proof that an
arbitrary published paper was reproduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable

from .local_runtime import check_local_python, local_environment_fingerprint, minimal_environment

SCHEMA = "resource.scientific-contract.v1"
REQUIRED_METHODS = ('dataset', 'split', 'preprocessing', 'model', 'training', 'evaluation')
EXCLUDED_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache'}
EXCLUDED_FILES = {'.DS_Store'}
IMAGE_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.:/@-]{0,180}$')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(file_hashes: dict[str, str]) -> str:
    """Stable fingerprint for a full original repository file inventory."""
    payload = json.dumps(file_hashes, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def reference_digest(path: str | Path) -> str:
    return tree_digest(_tree_files(Path(path).resolve(strict=True)))


def _relpath(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError('File path must be a string')
    path = PurePosixPath(raw.replace('\\', '/'))
    if (not raw or path.is_absolute() or '..' in path.parts or
            not path.parts or path.as_posix() in ('.', '/') or
            any(part in EXCLUDED_DIRS for part in path.parts)):
        raise ValueError(f'Unsafe or excluded relative file path: {raw!r}')
    return path.as_posix()


def _tree_files(root: Path) -> dict[str, str]:
    """Reject symlinks: they can point outside a repository or change targets."""
    if not root.is_dir():
        raise ValueError('Source tree is missing: ' + str(root))
    result = {}
    for file in root.rglob('*'):
        rel = file.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts) or file.name in EXCLUDED_FILES:
            continue
        if file.is_symlink():
            raise ValueError('Symlink is unsupported in this verification: ' + str(rel))
        if file.is_file():
            result[rel.as_posix()] = sha(file)
    return result


def _contract(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(spec, dict) or spec.get('schema') != SCHEMA:
        raise ValueError('Unsupported contract schema')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', str(spec.get('reference_tree_sha256', ''))):
        raise ValueError('Contract must include original repository tree SHA256')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', str(spec.get('trusted_evaluator_sha256', ''))):
        raise ValueError('Contract must pin the independent evaluator script SHA256')
    paper = spec.get('paper')
    if not isinstance(paper, dict) or not all(paper.get(k) for k in ('title', 'repository_url', 'commit')):
        raise ValueError('Paper title, repository URL and original commit must be documented')
    if not re.fullmatch(r'[a-fA-F0-9]{7,40}', str(paper['commit'])):
        raise ValueError('Commit must be a Git revision (7–40 hex characters)')
    methods = spec.get('methodology')
    if not isinstance(methods, dict) or any(not isinstance(methods.get(k), dict) or
          not methods[k] for k in REQUIRED_METHODS):
        raise ValueError('Methodology must specify dataset, split, preprocessing, model, training and evaluation')
    for k in REQUIRED_METHODS:
        if not all(str(v).strip() for v in methods[k].values()):
            raise ValueError('Methodology sections cannot contain blank values')
        if spec.get('review_status') == 'approved' and any(
                str(v).strip().upper() in {'REVIEW_ME', 'UNKNOWN', 'TBD'} for v in methods[k].values()):
            raise ValueError('Reviewed contract still contains placeholder methodology values')
    expectation = spec.get('expected_metric')
    if not isinstance(expectation, dict) or not all(k in expectation for k in ('name', 'value', 'abs_tolerance')):
        raise ValueError('Expected metric requires name, value, abs_tolerance')
    if spec.get('review_status') == 'approved' and str(expectation['name']).upper() in {'REVIEW_ME','UNKNOWN','TBD'}:
        raise ValueError('Reviewed metric name is still a placeholder')
    try:
        value, tolerance = float(expectation['value']), float(expectation['abs_tolerance'])
        if not math.isfinite(value) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError('Expected metric and tolerance must be finite numbers; tolerance >= 0') from None
    if not isinstance(spec.get('protected_files'), dict) or not spec['protected_files']:
        raise ValueError('Provide sha256 digests for at least one protected data/config/reference file')
    for raw, digest in spec['protected_files'].items():
        _relpath(raw)
        if not re.fullmatch(r'[0-9a-fA-F]{64}', str(digest)):
            raise ValueError(f'Invalid protected file SHA256: {raw}')
    edits = spec.get('approved_changes', [])
    if not isinstance(edits, list):
        raise ValueError('approved_changes must be a list')
    for edit in edits:
        if not isinstance(edit, dict) or not all(isinstance(edit.get(k), str) and edit[k].strip()
              for k in ('path', 'before', 'after', 'reason')):
            raise ValueError('Every approved change needs path, before, after, reason')
        _relpath(edit['path'])
        if edit['before'] == edit['after']:
            raise ValueError('Approved replacement must actually change text')
    return spec


def _compare(original: Path, candidate: Path, spec: dict) -> tuple[list[dict], list[str]]:
    """An arbitrary code change is *unknown*, not automatically equivalent."""
    checks = []
    differences = []
    old = _tree_files(original)
    new = _tree_files(candidate)
    if tree_digest(old) != spec['reference_tree_sha256'].lower():
        checks.append({'name': 'Original repository snapshot', 'status': 'UNVERIFIABLE',
                       'detail': 'Original source tree differs from the researcher-locked snapshot'})
    else:
        checks.append({'name': 'Original repository snapshot', 'status': 'PASS',
                       'detail': 'Original full source tree matches the recorded snapshot'})

    evidence_roles = spec.get('evidence_files', {})
    for role in REQUIRED_METHODS:
        paths = evidence_roles.get(role, []) if isinstance(evidence_roles, dict) else []
        if not isinstance(paths, list) or not paths:
            checks.append({'name': 'Methodology evidence', 'status': 'UNVERIFIABLE',
                           'detail': f'Missing file evidence for {role}'})
            continue
        for raw_path in paths:
            rel = _relpath(raw_path)
            if rel not in old:
                checks.append({'name': 'Methodology evidence', 'status': 'UNVERIFIABLE',
                               'detail': f'File for {role} is absent from reference: {rel}'})
            elif role == 'dataset' and rel not in spec['protected_files']:
                checks.append({'name': 'Dataset integrity', 'status': 'UNVERIFIABLE',
                               'detail': f'Dataset {rel} is not explicitly hash-protected'})
            else:
                checks.append({'name': 'Methodology evidence', 'status': 'PASS',
                               'detail': f'{role}: {rel}'})

    for rel, expected_hash in spec['protected_files'].items():
        norm = _relpath(rel)
        if old.get(norm) != expected_hash.lower():
            checks.append({'name': 'Reference integrity', 'status': 'UNVERIFIABLE',
                           'detail': f'Reference {norm} differs from locked contract'})
        elif new.get(norm) != expected_hash.lower():
            checks.append({'name': 'Protected methodology', 'status': 'VIOLATION',
                           'detail': f'Protected source/data changed: {norm}'})
        else:
            checks.append({'name': 'Protected methodology', 'status': 'PASS', 'detail': norm})

    allpaths = sorted(set(old) | set(new))
    differences = [p for p in allpaths if old.get(p) != new.get(p)]
    edits_by_path = {}
    for edit in spec.get('approved_changes', []):
        edits_by_path.setdefault(_relpath(edit['path']), []).append(edit)
    for rel in differences:
        if rel in spec['protected_files']:
            continue
        if rel not in old or rel not in new:
            checks.append({'name': 'Changed file', 'status': 'UNVERIFIABLE',
                           'detail': f'File added/removed without approval: {rel}'})
            continue
        if rel not in edits_by_path:
            checks.append({'name': 'Changed file', 'status': 'UNVERIFIABLE',
                           'detail': f'Unapproved source modification: {rel}'})
            continue
        try:
            before = (original / rel).read_text(encoding='utf-8')
            after = (candidate / rel).read_text(encoding='utf-8')
            expected = before
            for edit in edits_by_path[rel]:
                if expected.count(edit['before']) != 1:
                    raise ValueError('Approved replacement does not uniquely match original')
                expected = expected.replace(edit['before'], edit['after'], 1)
            if expected != after:
                raise ValueError('Candidate contains modifications beyond approved replacements')
            checks.append({'name': 'Approved compatibility change', 'status': 'PASS', 'detail': rel})
        except (UnicodeError, OSError, ValueError) as exc:
            checks.append({'name': 'Code change scope', 'status': 'UNVERIFIABLE',
                           'detail': f'{rel}: {exc}'})
    for rel in edits_by_path:
        if rel not in differences:
            checks.append({'name': 'Approved change unused', 'status': 'UNVERIFIABLE',
                           'detail': f'Requested approved change not found: {rel}'})
    if not differences:
        checks.append({'name': 'Code changes', 'status': 'PASS', 'detail': 'Source tree unchanged'})
    return checks, differences


def _docker_evaluate(original: Path, candidate: Path, evaluator: Path,
                     image: str, seconds: int) -> dict:
    """Execute a researcher-owned evaluator in restricted Docker, not on host.

    The evaluator's code is outside the candidate and not edited by the agent.
    All mounts are read-only. The evaluator must print a JSON object with a
    `metrics` map; the verifier ignores any candidate-printed metrics.
    """
    if not evaluator.is_file() or evaluator.suffix != '.py' or evaluator.is_symlink():
        raise ValueError('Trusted evaluator must be a regular .py file outside both source trees')
    eval_path = evaluator.resolve(strict=True)
    if eval_path.is_relative_to(original) or eval_path.is_relative_to(candidate):
        raise ValueError('Trusted evaluator must reside outside the original and candidate projects')
    if not IMAGE_PATTERN.fullmatch(image) or image.startswith('-'):
        raise ValueError('Unsafe Docker image identifier')
    if not 5 <= seconds <= 600:
        raise ValueError('Evaluator timeout must be between 5 and 600 seconds')
    if shutil.which('docker') is None:
        raise RuntimeError('Docker is unavailable; scientific evaluation was not run')
    cmd = ['docker', 'run', '--rm', '--pull=never', '--network=none', '--read-only',
           '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=96',
           '--memory=4g', '--cpus=2', '--tmpfs=/tmp:rw,nosuid,size=256m',
           '--user=65534:65534',
           '--mount', f'type=bind,source={original},target=/reference,readonly',
           '--mount', f'type=bind,source={candidate},target=/candidate,readonly',
           '--mount', f'type=bind,source={eval_path},target=/evaluator.py,readonly',
           '--workdir=/candidate', '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1',
           image, 'python', '/evaluator.py', '--candidate', '/candidate', '--reference', '/reference']
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 15, check=False)
    if p.returncode:
        raise RuntimeError(f'Trusted evaluator failed with return code {p.returncode}: {p.stderr[-1000:]}')
    try:
        # Evaluator returns a single valid JSON object, not candidate stdout.
        result = json.loads(p.stdout.strip())
        if not isinstance(result, dict) or not isinstance(result.get('metrics'), dict):
            raise ValueError('Evaluator must output JSON containing a metrics object')
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f'Invalid evaluator output: {exc}') from exc
    return result



def _local_evaluate(original: Path, candidate: Path, evaluator: Path,
                    interpreter: str | Path, seconds: int) -> dict:
    """Run a reviewer-owned evaluator using a separate trusted local venv.

    This is NOT sandboxed. The evaluator may import and execute candidate code.
    Local mode must never be used for unfamiliar/unreviewed repository code.
    """
    state = check_local_python(interpreter)
    if not state['available']:
        raise RuntimeError(state['detail'])
    if not evaluator.is_file() or evaluator.suffix != '.py' or evaluator.is_symlink():
        raise ValueError('Trusted evaluator must be a regular Python file')
    if not 5 <= int(seconds) <= 600:
        raise ValueError('Evaluator timeout must be between 5 and 600 seconds')
    cmd = [state['path'], '-I', str(evaluator.resolve(strict=True)), '--candidate',
           str(candidate), '--reference', str(original)]
    proc = subprocess.run(cmd, cwd=candidate, env=minimal_environment(), capture_output=True,
                          text=True, timeout=int(seconds), check=False, errors='replace')
    if proc.returncode:
        raise RuntimeError('Trusted local evaluator failed: ' + proc.stderr[-1000:])
    try:
        data = json.loads(proc.stdout.strip())
        if not isinstance(data, dict) or not isinstance(data.get('metrics'), dict):
            raise ValueError('Expected a JSON object containing metrics')
    except (ValueError, TypeError) as exc:
        raise RuntimeError('Invalid evaluator output: ' + str(exc)) from exc
    return data


def verify_science(contract_path: str | Path, reference_dir: str | Path, candidate_dir: str | Path,
                   evaluator_path: str | Path | None = None, docker_image: str | None = None,
                   timeout: int = 90, _test_evaluator: Callable | None = None,
                   execution_backend: str = 'docker',
                   local_python: str | Path | None = None,
                   allow_trusted_local: bool = False) -> dict:
    """Verify only the evidence explicitly supported by the approved contract.

    `_test_evaluator` is test injection and is NEVER exposed in the CLI/UI.
    No evaluator => UNVERIFIABLE, even if every file is unchanged.
    """
    if execution_backend not in ('docker', 'local'):
        raise ValueError('Unknown science evaluator backend')
    if execution_backend == 'local' and not allow_trusted_local:
        raise ValueError('Local scientific evaluation requires explicit trust confirmation')
    contract = Path(contract_path).resolve(strict=True)
    original = Path(reference_dir).resolve(strict=True)
    candidate = Path(candidate_dir).resolve(strict=True)
    spec = _contract(contract)
    checks, changed = _compare(original, candidate, spec)
    if contract.is_relative_to(original) or contract.is_relative_to(candidate):
        checks.append({'name': 'Contract isolation', 'status': 'UNVERIFIABLE',
                       'detail': 'Contract must be stored outside model-editable trees'})
    eval_record = None
    metric_observed = None
    if spec.get('review_status') != 'approved':
        checks.append({'name': 'Researcher-reviewed contract', 'status': 'UNVERIFIABLE',
                       'detail': 'The paper-specific methodology contract has not been approved'})
    else:
        checks.append({'name': 'Researcher-reviewed contract', 'status': 'PASS',
                       'detail': 'Contract is marked as reviewed; accuracy of transcription remains a human responsibility'})
    try:
        if evaluator_path is None:
            raise RuntimeError('No researcher-owned independent evaluator specified')
        evaluator = Path(evaluator_path).resolve(strict=True)
        if evaluator.is_relative_to(original) or evaluator.is_relative_to(candidate):
            raise RuntimeError('Evaluator is inside an editable source tree')
        if sha(evaluator) != spec['trusted_evaluator_sha256'].lower():
            raise RuntimeError('Independent evaluator differs from the researcher-locked SHA256')
        if execution_backend == 'docker':
            if not docker_image:
                raise RuntimeError('A locally available Docker image is required')
            if not re.search(r'@sha256:[0-9a-f]{64}$', docker_image):
                checks.append({'name': 'Environment pinning', 'status': 'UNVERIFIABLE',
                               'detail': 'Docker image is not pinned by immutable sha256 digest'})
            else:
                checks.append({'name': 'Environment pinning', 'status': 'PASS',
                               'detail': 'Docker image pinned by immutable digest'})
            evaluation = (_test_evaluator(original, candidate, evaluator, docker_image, timeout)
                          if _test_evaluator is not None else
                          _docker_evaluate(original, candidate, evaluator, docker_image, timeout))
            eval_record = {'evaluator_sha256': sha(evaluator), 'docker_image': docker_image,
                           'execution_backend': 'docker', 'observations': evaluation}
        else:
            env_state = local_environment_fingerprint(local_python)
            locked = str(spec.get('local_environment_sha256', '')).lower()
            if not re.fullmatch(r'[0-9a-f]{64}', locked) or locked != env_state['sha256']:
                checks.append({'name': 'Environment pinning', 'status': 'UNVERIFIABLE',
                               'detail': 'Local environment fingerprint does not match an approved contract lock. '
                                         'Record the environment fingerprint before evaluating repairs.'})
            else:
                checks.append({'name': 'Environment pinning', 'status': 'PASS',
                               'detail': 'Version-manifest fingerprint matches reviewed contract; '
                                         'not a bytewise proof of package integrity'})
            evaluation = (_test_evaluator(original, candidate, evaluator, local_python, timeout)
                          if _test_evaluator is not None else
                          _local_evaluate(original, candidate, evaluator, local_python, timeout))
            eval_record = {'evaluator_sha256': sha(evaluator), 'execution_backend': 'trusted_local_venv',
                           'local_environment': env_state, 'observations': evaluation,
                           'warning': 'Local Python venv is NOT a security sandbox.'}
        expected = spec['expected_metric']
        metric_observed = float(evaluation['metrics'][expected['name']])
        if not math.isfinite(metric_observed):
            raise ValueError('Metric is non-finite')
        error = abs(metric_observed - float(expected['value']))
        tolerance = float(expected['abs_tolerance'])
        checks.append({'name': 'Independent metric evaluation',
                       'status': 'PASS' if error <= tolerance else 'NOT_MATCHED',
                       'detail': f"{expected['name']}: observed={metric_observed:.8g}; "
                                 f"reference={float(expected['value']):.8g}; tolerance={tolerance:.8g}"})
    except (RuntimeError, OSError, KeyError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
        checks.append({'name': 'Independent metric evaluation', 'status': 'UNVERIFIABLE',
                       'detail': str(exc)[:800]})

    statuses = {x['status'] for x in checks}
    if 'VIOLATION' in statuses:
        verdict = 'METHODOLOGY_VIOLATION'
    elif 'NOT_MATCHED' in statuses:
        verdict = 'RESULT_NOT_REPRODUCED'
    elif 'UNVERIFIABLE' in statuses:
        verdict = 'UNVERIFIABLE'
    else:
        verdict = 'PASSED_SPECIFIED_SCIENTIFIC_CHECKS'
    return {
        'schema': 'resource.scientific-verification-result.v1',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'paper': spec['paper'], 'framework': spec.get('framework', 'not specified'),
        'contract_sha256': sha(contract),
        'declared_reference_metric': spec['expected_metric'],
        'reference_dir': str(original), 'candidate_dir': str(candidate),
        'changed_paths': changed, 'checks': checks, 'verdict': verdict,
        'measured_metric': metric_observed,
        'evaluator_evidence': eval_record,
        'execution_backend': execution_backend,
        'is_universal_proof': False,
        'limits': ('Checks are paper-specific and only as good as the reviewed methodology and independent evaluator. '
                   'Exact approved source changes, protected data and one measured metric do not prove '
                   'semantic equivalence for arbitrary ML frameworks.'),
    }


def as_markdown(report: dict) -> str:
    lines = ['# RE:SOURCE AI | Scientific integrity report', '',
             f"**Paper:** {report['paper']['title']}",
             f"**Framework:** {report['framework']}",
             f"**Verdict:** `{report['verdict']}`", '',
             '| Check | Result | Evidence |', '|---|---|---|']
    for check in report['checks']:
        detail = str(check.get('detail', '')).replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {check['name']} | {check['status']} | {detail} |")
    lines.extend(['', '## Scope and limitations', report['limits'],
                  '', 'This is not an automatic proof that the full published paper was reproduced.'])
    return '\n'.join(lines) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Framework-independent paper-specific scientific integrity checks')
    parser.add_argument('--contract', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--evaluator', help='Independent researcher-owned evaluator Python file')
    parser.add_argument('--image', help='Locally available pinned Docker image')
    parser.add_argument('--backend', choices=['docker', 'local'], default='docker')
    parser.add_argument('--local-python', help='Dedicated trusted research virtualenv Python executable')
    parser.add_argument('--trust-local', action='store_true', help='Acknowledge that local code can access the host')
    parser.add_argument('--out', default='reports/scientific_result.json')
    args = parser.parse_args(argv)
    result = verify_science(args.contract, args.reference, args.candidate, args.evaluator, args.image,
                            execution_backend=args.backend, local_python=args.local_python,
                            allow_trusted_local=args.trust_local)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    output.with_suffix('.md').write_text(as_markdown(result), encoding='utf-8')
    print('Scientific verdict:', result['verdict'])
    print('Saved:', output, 'and', output.with_suffix('.md'))
    return 0 if result['verdict'] == 'PASSED_SPECIFIED_SCIENTIFIC_CHECKS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
