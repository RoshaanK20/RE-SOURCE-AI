"""Evidence records for the first public research pilot.

No inferred or expected experimental outcomes are written as measured results.
"""
from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path


def file_digest(file: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(file).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def llm_run_record(*, case: str, model: str, source: str | Path,
                   reference: str | Path, result: dict | None = None,
                   error: str | None = None, health: dict | None = None) -> dict:
    """Save experiment identity, source fingerprints and observed outcome."""
    import sklearn
    import numpy
    record = {
        'schema_version': 1,
        'experiment_kind': 'CONTROLLED_BUNDLED_LLM_REPAIR',
        'published_research_reproduction': False,
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'case_id': case,
        'model_id': model,
        'source_sha256': file_digest(source),
        'reference_sha256': file_digest(reference),
        'environment': {
            'python': platform.python_version(),
            'scikit_learn': sklearn.__version__,
            'numpy': numpy.__version__,
        },
        'ollama_probe': health or {},
        'run_status': 'COMPLETED' if result is not None else 'ERROR',
        'observed_result': result,
        'error': error,
        'claim_limit': (
            'This is a constructed compatibility failure on a real dataset. '
            'The existing verifier is not a tamper-proof independent evaluator, '
            'and the result is not recovery of an external paper.'
        ),
    }
    return record


def validation_summary(report: dict) -> dict:
    """Aggregate completed real-data pilot outcomes only; no guessed metrics."""
    rows = report.get('cases', [])
    datasets = sorted({r['dataset'] for r in rows})
    return {
        'total_cases': len(rows),
        'verifier_matches_expected': sum(bool(r.get('passed')) for r in rows),
        'failed_cases': sum(not bool(r.get('passed')) for r in rows),
        'score_rule_accepted': sum(bool(r.get('score_only_accept')) for r in rows),
        'invalid_cases_score_rule_accepted': sum(
            r.get('expected') != 'REPRODUCED' and bool(r.get('score_only_accept'))
            for r in rows
        ),
        'datasets': datasets,
        'real_research_repository_reproductions': 0,
    }
