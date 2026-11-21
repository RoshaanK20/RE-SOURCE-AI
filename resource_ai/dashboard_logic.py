"""Backend helpers for the Streamlit dashboard.

No Streamlit import here: core behavior can be verified in headless CI.
Reports in archives are intentionally limited to evidence files, never downloaded source.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .batch import read_research_list
from .intake import github_source

ALLOWED_LIST_SUFFIXES = {'.csv', '.md', '.txt', '.json'}
ALLOWED_EVIDENCE_NAMES = {'CASE_REPORT.md', 'case.json', 'BATCH_REPORT.md',
                          'batch_summary.json', 'batch_summary.csv',
                          'controlled_result.json', 'llm_repair_result.json'}
MAX_REPORT_ZIP_BYTES = 8_000_000


def unique_run_dir(base: Path, tag: str = 'scan') -> Path:
    """Create a unique local run folder; never overwrite an earlier report."""
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r'[^a-zA-Z0-9_-]', '_', tag)[:24] or 'scan'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    folder = base / f'{stamp}_{tag}'
    folder.mkdir(exist_ok=False)
    return folder


def uploaded_list_path(name: str, data: bytes, root: Path) -> Path:
    """Save uploaded queue in a controlled local directory."""
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_LIST_SUFFIXES:
        raise ValueError('Upload a CSV, Markdown, text, or JSON research list')
    if not data or len(data) > 1_000_000:
        raise ValueError('List must be nonempty and at most 1 MB')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ('research_links' + suffix)
    path.write_bytes(data)
    return path


def validate_ui_research_list(path: str | Path) -> list:
    """Dashboard lists must contain GitHub links, never arbitrary local executable paths."""
    entries = read_research_list(path)
    for e in entries:
        github_source(e.repository_url)
    return entries


def status_summary(entries: list[dict]) -> dict:
    """Return only observable intake counts. No scientific-verification conclusions."""
    statuses: dict[str, int] = {}
    for e in entries:
        key = str(e.get('status') or 'UNKNOWN')
        statuses[key] = statuses.get(key, 0) + 1
    return {
        'repositories': len(entries),
        'sources_inspected': sum(e.get('status') in {'REMOTE_INSPECTED','LOCAL_INSPECTED'} for e in entries),
        'python_files': sum(int(e.get('python_files', 0)) for e in entries),
        'compatibility_leads': sum(int(e.get('compatibility_leads', 0)) for e in entries),
        'status_counts': statuses,
        'scientific_reproductions_verified': 0,
    }


def collect_evidence(root: str | Path) -> bytes:
    """Create a small archive with report documents ONLY, omitting fetched source code."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Evidence root does not exist')
    candidates = [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
                  and p.name in ALLOWED_EVIDENCE_NAMES]
    if not candidates:
        raise ValueError('No report files available for download')
    if sum(p.stat().st_size for p in candidates) > MAX_REPORT_ZIP_BYTES:
        raise ValueError('Reports exceed export size limit')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(candidates):
            zf.write(p, arcname=p.relative_to(root).as_posix())
    return output.getvalue()


def available_builtin_cases(package_root: str | Path) -> list[str]:
    """Only curated bundled cases can be selected for local execution."""
    base = Path(package_root) / 'benchmark' / 'cases'
    return sorted(p.name for p in base.iterdir() if p.is_dir()
                  and (p/'reference'/'experiment.py').is_file()
                  and (p/'broken'/'experiment.py').is_file()) if base.is_dir() else []


def save_report(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    return path


def recorded_cases(root: str | Path, limit: int = 30) -> list[dict]:
    """List source, batch and Qwen records after a Streamlit restart.

    Only JSON evidence files are read; downloaded source is never loaded.
    """
    root = Path(root)
    results = []
    patterns = [('batch_summary.json', 'batch'), ('case.json', 'source'),
                ('llm_repair_result.json', 'llm'), ('controlled_result.json', 'controlled')]
    for filename, kind in patterns:
        for p in sorted(root.rglob(filename), reverse=True)[:limit]:
            try:
                obj = json.loads(p.read_text(encoding='utf-8'))
                if kind == 'batch':
                    count = obj.get('total', 0)
                    status = str(obj.get('status_counts', {}))
                elif kind == 'source':
                    count = obj.get('files_observed', 0)
                    status = obj.get('status', '?')
                elif kind == 'llm':
                    observed = obj.get('observed_result') or {}
                    count = observed.get('llm_calls', 0)
                    status = obj.get('error') or observed.get('final_verdict', 'UNKNOWN')
                else:
                    count = 1
                    status = obj.get('repaired_verdict', 'UNKNOWN')
                results.append({'kind': kind, 'path': str(p), 'name': p.parent.name,
                                'items': count, 'status': status})
            except (ValueError, OSError, TypeError):
                continue
    return sorted(results, key=lambda item: item['path'], reverse=True)[:limit]
