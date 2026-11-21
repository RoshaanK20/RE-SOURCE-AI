"""Validate user-entered Streamlit paths before invoking scientific evaluator.

This prevents a blank file field being interpreted as the current directory.
Checks establish only that local inputs exist; they do not approve methodology.
"""
from pathlib import Path


def check_science_inputs(contract: str, original: str, candidate: str, evaluator: str) -> list[str]:
    values = [
        ('Scientific contract', contract, '.json', 'file'),
        ('Original source directory', original, '', 'directory'),
        ('Repaired source directory', candidate, '', 'directory'),
        ('Independent evaluator', evaluator, '.py', 'file'),
    ]
    issues = []
    paths = {}
    for label, value, suffix, kind in values:
        if not (value or '').strip():
            issues.append(f'{label}: choose a real existing path (not a placeholder).')
            continue
        path = Path(value.strip()).expanduser()
        if suffix and path.suffix.lower() != suffix:
            issues.append(f'{label}: expected a {suffix} file, not a folder or another file type.')
        if kind == 'file' and not path.is_file():
            issues.append(f'{label}: file not found at {path}.')
        if kind == 'directory' and not path.is_dir():
            issues.append(f'{label}: folder not found at {path}.')
        paths[label] = path.resolve()
    if paths.get('Original source directory') == paths.get('Repaired source directory') and paths.get('Original source directory'):
        issues.append('Original and repaired source folders must be different.')
    if paths.get('Independent evaluator') and any(
        paths['Independent evaluator'].is_relative_to(paths[d]) for d in
        ('Original source directory', 'Repaired source directory') if d in paths
    ):
        issues.append('Independent evaluator must be outside both experiment folders.')
    return issues
