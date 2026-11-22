"""Conservative research-code intake.

A URL launches SOURCE INSPECTION, not arbitrary code execution.  Public repos
must be reviewed and run in a separate restricted container before recovery.
The importer is deliberately independent of the LLM repair logic.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from .knowledge import retrieve
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
MAX_ARCHIVE_FILES = 1000
MAX_REPO_BYTES = 100 * 1024 * 1024
SUPPORT = {'.py', '.ipynb', '.md', '.txt', '.toml', '.yaml', '.yml', '.json', '.cfg', '.ini', '.csv'}
DEPENDENCY_NAMES = {'requirements.txt', 'requirements-dev.txt', 'environment.yml',
                    'environment.yaml', 'pyproject.toml', 'Pipfile', 'setup.py',
                    'setup.cfg', 'conda.yaml', 'Dockerfile'}


@dataclass(frozen=True)
class GitHubSource:
    owner: str
    repo: str
    kind: str       # repo or file
    revision: str | None = None
    path: str | None = None


def github_source(url: str) -> GitHubSource:
    """Validate GitHub repository and file links. No arbitrary download URLs."""
    u = urllib.parse.urlsplit(url.strip())
    if u.scheme != 'https' or u.username or u.password or u.port or u.query or u.fragment:
        raise ValueError('Use a plain HTTPS GitHub URL without credentials, query or fragment')
    host = (u.hostname or '').lower()
    path = [urllib.parse.unquote(p) for p in u.path.strip('/').split('/') if p]
    if len(path) < 2:
        raise ValueError('Expected https://github.com/OWNER/REPO')
    for segment in path:
        if segment in ('.', '..') or '/' in segment or '\\' in segment or '\x00' in segment:
            raise ValueError('Unsafe GitHub path segment')
    owner, repo = path[:2]
    repo = repo.removesuffix('.git') if host == 'github.com' else repo
    if not re.fullmatch(r'[\w.-]{1,100}', owner) or not re.fullmatch(r'[\w.-]{1,100}', repo):
        raise ValueError('Invalid GitHub owner or repository name')
    if host == 'github.com':
        if len(path) == 2:
            return GitHubSource(owner, repo, 'repo')
        if len(path) >= 5 and path[2] == 'blob':
            ref, rel = path[3], '/'.join(path[4:])
            if Path(rel).suffix.lower() not in {'.py', '.ipynb', '.md', '.txt'}:
                raise ValueError('File URL must end in .py, .ipynb, .md or .txt')
            return GitHubSource(owner, repo, 'file', ref, rel)
        raise ValueError('Supported: GitHub repository home or a /blob/REF/FILE link')
    if host == 'raw.githubusercontent.com' and len(path) >= 4:
        ref, rel = path[2], '/'.join(path[3:])
        if Path(rel).suffix.lower() not in {'.py', '.ipynb', '.md', '.txt'}:
            raise ValueError('File URL must end in .py, .ipynb, .md or .txt')
        return GitHubSource(owner, repo, 'file', ref, rel)
    raise ValueError('Only github.com and raw.githubusercontent.com are supported')


class GitHubOnlyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        p = urllib.parse.urlsplit(newurl)
        if p.scheme != 'https' or p.hostname != 'raw.githubusercontent.com':
            raise ValueError('Unexpected redirect destination')
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _copy_archive(source: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        files = [x for x in archive.infolist() if not x.is_dir()]
        if len(files) > MAX_ARCHIVE_FILES or sum(x.file_size for x in files) > MAX_ARCHIVE_BYTES:
            raise ValueError('Archive exceeds intake limits')
        for info in files:
            pp = PurePosixPath(info.filename)
            if pp.is_absolute() or '..' in pp.parts or '\\' in info.filename or not pp.parts:
                raise ValueError('Archive contains an unsafe path')
            # Symlinks are not allowed
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive contains a symlink')
            dest = dst.joinpath(*pp.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if info.file_size > MAX_FILE_BYTES:
                raise ValueError('One file exceeds intake limits')
            with archive.open(info) as inp, dest.open('wb') as out:
                shutil.copyfileobj(inp, out)


def _tree_limit(root: Path) -> None:
    size = count = 0
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError(f'Symlinks are not permitted: {p}')
        if p.is_file():
            count += 1
            size += p.stat().st_size
            if count > MAX_ARCHIVE_FILES or size > MAX_REPO_BYTES:
                raise ValueError('Source project exceeds intake size/count limits')


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _inventory(root: Path) -> dict:
    scripts = []
    notebooks = []
    deps = []
    parse_errors = []
    models = set()
    compatibility_hints = []
    likely = ('RandomForest', 'SVC', 'LogisticRegression', 'MLP', 'Torch',
              'keras', 'tensorflow', 'LSTM', 'CNN', 'sklearn', 'pytorch')
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink() or '.git' in p.parts:
            continue
        relative = str(p.relative_to(root))
        if p.name in DEPENDENCY_NAMES:
            deps.append(relative)
        if p.suffix == '.py':
            scripts.append(relative)
        elif p.suffix == '.ipynb':
            notebooks.append(relative)
        if p.suffix not in {'.py', '.ipynb'}:
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            parse_errors.append({'file': relative, 'issue': 'file too large to inspect'})
            continue
        try:
            data = p.read_text(encoding='utf-8', errors='replace')
            if p.suffix == '.ipynb':
                notebook = json.loads(data)
                texts = [''.join(c.get('source', [])) for c in notebook.get('cells', [])
                         if c.get('cell_type') == 'code']
                data = '\n'.join(texts)
                # Colab/Jupyter magic isn't normal Python; do not call it a code failure
            else:
                ast.parse(data, filename=relative)
            for item in retrieve(data, '', limit=8):
                compatibility_hints.append({
                    'file': relative, 'knowledge_id': item.id,
                    'potential_issue': item.title, 'suggested_check': item.fix,
                    'scientific_warning': item.warning,
                    'status': 'HYPOTHESIS_NOT_OBSERVED_FAILURE',
                })
            for key in likely:
                if key.lower() in data.lower():
                    models.add(key)
        except (SyntaxError, ValueError, json.JSONDecodeError) as exc:
            parse_errors.append({'file': relative, 'issue': str(exc)[:250]})
    return {'python_files': scripts, 'notebooks': notebooks,
            'dependency_files': deps, 'detected_terms': sorted(models),
            'static_parse_issues': parse_errors,
            'compatibility_hints': compatibility_hints[:40]}


def intake_source(source: str, out: str | Path = 'official_pilot',
                  fetch: bool = False, paper: str = '') -> dict:
    """Inspect local or GitHub sources. Never execute project Python code."""
    source = source.strip()
    if not source:
        raise ValueError('A file path or source URL is required')
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ref = None
    if source.startswith(('https://', 'http://')):
        ref = github_source(source)
    else:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(source)
    slug = (f'{ref.owner}_{ref.repo}' if ref else source_path.stem).replace(' ', '_')
    slug = re.sub(r'[^\w.-]', '_', slug)[:55] or 'source'
    case = root / slug
    if case.exists():
        i = 2
        while (root / f'{slug}_{i}').exists():
            i += 1
        case = root / f'{slug}_{i}'
    case.mkdir()
    project = case / 'project'
    project.mkdir()
    status = 'LOCAL_INSPECTED' if ref is None else 'LINK_RECORDED'
    commit = None
    try:
        if ref and fetch:
            if ref.kind == 'repo':
                cmd = ['git', '-c', 'protocol.file.allow=never', 'clone',
                       '--quiet', '--depth', '1', '--no-checkout', f'https://github.com/{ref.owner}/{ref.repo}.git', str(project)]
                env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_CONFIG_NOSYSTEM': '1'}
                subprocess.run(cmd, timeout=75, check=True, capture_output=True, text=True, env=env)
                subprocess.run(['git', '-C', str(project), 'checkout', '-q', 'HEAD', '--', '.'],
                               timeout=60, check=True, capture_output=True, text=True, env=env)
                commit = subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'],
                                                 text=True, timeout=5).strip()
                shutil.rmtree(project / '.git', ignore_errors=True)
            else:
                raw = f'https://raw.githubusercontent.com/{ref.owner}/{ref.repo}/{ref.revision}/{ref.path}'
                req = urllib.request.Request(raw, headers={'User-Agent': 'RE-SOURCE-AI-Console/0.4.0-rc1'})
                opener = urllib.request.build_opener(GitHubOnlyRedirects())
                with opener.open(req, timeout=20) as res:
                    data = res.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    raise ValueError('Remote file exceeds intake limit')
                name = Path(ref.path).name
                (project / name).write_bytes(data)
                commit = ref.revision
            status = 'REMOTE_INSPECTED'
        elif not ref:
            if source_path.is_dir():
                shutil.rmtree(project)
                shutil.copytree(source_path, project,
                                ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
            elif source_path.suffix.lower() == '.zip':
                _copy_archive(source_path, project)
            elif source_path.suffix.lower() in SUPPORT:
                shutil.copy2(source_path, project / source_path.name)
            else:
                raise ValueError('Supported local inputs: directories, .zip, .py, .ipynb, and text/code files')
        _tree_limit(project)
        inventory = _inventory(project)
        contents = list(project.iterdir())
        result = {
            'status': status,
            'source': source,
            'paper': paper,
            'recorded_utc': datetime.now(timezone.utc).isoformat(),
            'repository_commit': commit,
            'case_dir': str(case),
            'project_dir': str(project),
            'source_kind': ('github_' + ref.kind) if ref else 'local',
            'fetched': bool(ref and fetch),
            'files_observed': sum(p.is_file() for p in project.rglob('*')),
            'inventory': inventory,
            'execution_performed': False,
            'scientific_reproduction_verified': False,
            'next_action': ('Review sources, confirm original methodology and use an isolated runner'
                            if status != 'LINK_RECORDED' else
                            'Enable GitHub download in your own runtime, then inspect the source'),
        }
        (case / 'case.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        md = [f'# RE-SOURCE AI | Candidate {slug}', '', f'**Status:** {status}',
              f'**Source:** {source}', f'**Paper:** {paper or "Not yet supplied"}',
              f'**Source commit:** {commit or "Not verified"}',
              '', '## Static inventory',
              f'- Python scripts: {len(inventory["python_files"])}',
              f'- Notebooks: {len(inventory["notebooks"])}',
              f'- Dependency declarations: {len(inventory["dependency_files"])}',
              f'- Detected framework/model terms: {", ".join(inventory["detected_terms"]) or "None"}',
              f'- Static parsing issues: {len(inventory["static_parse_issues"])}',
              f'- Potential compatibility leads: {len(inventory["compatibility_hints"])}',
              '', '## Preflight compatibility leads (not confirmed)',
              *([f'- `{h["file"]}`: {h["potential_issue"]}. Check: {h["suggested_check"]}'
                 for h in inventory['compatibility_hints'][:10]] or
                ['- No curated compatibility patterns were matched']),
              '', '## Scientific verification',
              '- No published result has been reproduced by this intake step.',
              '- Source inspection is NOT evidence that the original project fails.',
              '- Confirm paper, code commit, dataset license, baseline and reference methodology.',
              '- Unknown source code must be executed only inside a restricted, disposable environment.',
              '', '## Next action', result['next_action'], '']
        (case / 'CASE_REPORT.md').write_text('\n'.join(md), encoding='utf-8')
        return result
    except Exception:
        # A failed attempt must never leave a half-populated, valid-looking case
        shutil.rmtree(case, ignore_errors=True)
        raise


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Import and inspect research code without executing it')
    p.add_argument('source', help='Local path or supported HTTPS GitHub URL')
    p.add_argument('--out', default='official_pilot')
    p.add_argument('--paper', default='', help='Paper identifier/URL for attribution; not verified')
    p.add_argument('--fetch', action='store_true', help='Download from GitHub (inspection only)')
    args = p.parse_args(argv)
    result = intake_source(args.source, args.out, fetch=args.fetch, paper=args.paper)
    print(json.dumps({k: result[k] for k in ('status','source','case_dir','files_observed','next_action')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
