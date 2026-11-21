"""Opt-in execution helpers for *trusted* research code in a separate Python venv.

A virtual environment isolates dependencies, NOT the filesystem, credentials,
processes or network. Never run unfamiliar downloaded GitHub code with this mode.
Docker remains the recommended choice for untrusted material.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def check_local_python(interpreter: str | Path | None) -> dict:
    """Only permit a dedicated virtual environment different from the UI venv."""
    if not interpreter or not str(interpreter).strip():
        return {'available': False, 'detail': 'Specify a separate virtual-environment Python executable'}
    raw = Path(str(interpreter)).expanduser().absolute()
    if not raw.is_file():
        return {'available': False, 'detail': f'Python executable not found: {raw}'}
    venv_root = raw.parent.parent
    if raw.parent.name.lower() not in ('scripts', 'bin') or not (venv_root / 'pyvenv.cfg').is_file():
        return {'available': False, 'detail': 'Choose python.exe from a dedicated venv\\Scripts folder (or venv/bin/python)'}
    if venv_root.resolve() == Path(sys.prefix).resolve():
        return {'available': False, 'detail': 'Do not use the Streamlit application environment for research execution'}
    try:
        p = subprocess.run([str(raw), '-I', '-c', 'import sys; print(sys.prefix); print(sys.base_prefix)'],
                           capture_output=True, text=True, timeout=12, check=False)
        lines = p.stdout.strip().splitlines()
        if p.returncode or len(lines) < 2 or Path(lines[0]).resolve() != venv_root.resolve() or lines[0] == lines[1]:
            raise ValueError((p.stderr or p.stdout or 'Interpreter is not an active virtual environment')[-350:])
        return {'available': True, 'detail': 'Dedicated Python virtual environment confirmed', 'path': str(raw)}
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'detail': str(exc)[:350]}


def local_environment_fingerprint(interpreter: str | Path) -> dict:
    """Record interpreter + installed-distribution versions for a declared env lock.

    A manifest hash is a reproducibility aid, NOT proof of bytewise dependency
    integrity. Any version-changing installation invalidates the fingerprint.
    """
    state = check_local_python(interpreter)
    if not state['available']:
        raise ValueError(state['detail'])
    code = '''import sys,platform,json,importlib.metadata as md
packages=sorted((d.metadata.get("Name", "").lower(), d.version) for d in md.distributions())
print(json.dumps({"python":sys.version.split()[0],"platform":platform.platform(),"packages":packages},sort_keys=True))'''
    p = subprocess.run([state['path'], '-I', '-c', code], capture_output=True, text=True, timeout=25, check=False)
    if p.returncode:
        raise RuntimeError('Cannot fingerprint Python environment: ' + p.stderr[-500:])
    raw = json.loads(p.stdout)
    payload = json.dumps(raw, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'python': raw['python'],
            'platform': raw['platform'], 'package_count': len(raw['packages'])}


def minimal_environment() -> dict[str, str]:
    """Do not forward arbitrary app credentials to trusted local child processes.

    This does not stop code from reading secrets on disk. Local execution is NOT
    a sandbox; the user must audit and trust the project first.
    """
    keep = ('PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'TMPDIR', 'LANG', 'LC_ALL',
            'HOME', 'USERPROFILE')
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', MPLBACKEND='Agg',
               OMP_NUM_THREADS='1')
    return env

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inspect a dedicated research venv and calculate its environment-manifest hash')
    parser.add_argument('--python', required=True, help='Separate virtual environment Python executable')
    args = parser.parse_args()
    print(json.dumps(local_environment_fingerprint(args.python), indent=2))
