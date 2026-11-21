"""Batch research-code intake: a single CSV/Markdown/JSON list -> per-repo reports.

The batch reads every listed GitHub source automatically when fetch=True,
but never executes external code or claims scientific reproduction.
Paper URLs are recorded for attribution; their contents are not fetched.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .intake import intake_source

MAX_ITEMS = 50
MAX_LIST_BYTES = 1_000_000


@dataclass(frozen=True)
class ResearchEntry:
    case_id: str
    title: str
    paper_url: str
    repository_url: str


def _entry(obj: dict, row_number: int) -> ResearchEntry:
    repository_url = str(obj.get('repository_url') or obj.get('github') or obj.get('repo')
                         or obj.get('source') or '').strip()
    paper = str(obj.get('paper_url') or obj.get('paper') or '').strip()
    title = str(obj.get('title') or obj.get('name') or f'Research candidate {row_number}').strip()
    case_id = str(obj.get('case_id') or obj.get('id') or f'P{row_number:03d}').strip()
    if not repository_url:
        raise ValueError(f'Row {row_number} has no repository_url / GitHub link')
    # Keep IDs safe for text/CSV reporting. Output folder creation is handled by intake_source.
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}', case_id):
        raise ValueError(f'Row {row_number} has invalid case_id')
    if paper and not paper.startswith('https://'):
        raise ValueError(f'Row {row_number} paper_url must be HTTPS')
    return ResearchEntry(case_id, title[:300], paper, repository_url)


def _markdown_entries(text: str) -> list[ResearchEntry]:
    """Recognize a simple Markdown list or headings with Paper/GitHub fields.

    Suggested format:
      ## P01: Short title
      - Paper: https://arxiv.org/abs/...
      - GitHub: https://github.com/owner/repo

    Bare GitHub links on independent lines are also accepted.
    """
    records: list[ResearchEntry] = []
    candidate: dict = {}
    title: str | None = None
    for line in text.splitlines() + ['## END']:
        line = line.strip()
        heading = re.match(r'^#{2,6}\s+(.+)', line)
        if heading:
            if candidate.get('repository_url'):
                records.append(_entry(candidate, len(records) + 1))
            candidate = {}
            title = heading.group(1).strip()
            if title != 'END':
                match = re.match(r'^([\w.-]+)\s*[:|]\s*(.+)$', title)
                if match:
                    candidate['case_id'], candidate['title'] = match.groups()
                else:
                    candidate['title'] = title
            continue
        if not line or line.startswith('<!--'):
            continue
        p = re.match(r'^[-*]?\s*(?:paper|paper_url|research paper)\s*:\s*(https://\S+)', line, re.I)
        r = re.match(r'^[-*]?\s*(?:github|repository|repository_url|repo|code)\s*:\s*(https://\S+)', line, re.I)
        if p:
            candidate['paper_url'] = p.group(1).rstrip(').,')
        if r:
            if candidate.get('repository_url'):
                records.append(_entry(candidate, len(records) + 1))
                candidate = {'title': title or f'Research candidate {len(records)+1}'}
            candidate['repository_url'] = r.group(1).rstrip(').,')
        if not p and not r:
            bare = re.match(r'^[-*]?\s*(https://github\.com/\S+)\s*$', line, re.I)
            if bare:
                if candidate.get('repository_url'):
                    records.append(_entry(candidate, len(records) + 1))
                    candidate = {}
                candidate.setdefault('title', title or f'Research candidate {len(records)+1}')
                candidate['repository_url'] = bare.group(1).rstrip(').,')
    return records


def read_research_list(path: str | Path) -> list[ResearchEntry]:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size > MAX_LIST_BYTES:
        raise ValueError('List file missing or exceeds size limit')
    suffix = path.suffix.lower()
    if suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            entries = [_entry(row, n) for n, row in enumerate(csv.DictReader(f), 1)
                       if any(str(x).strip() for x in row.values() if x is not None)]
    elif suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, list):
            raise ValueError('JSON list must be an array of objects')
        entries = [_entry(row, i) for i, row in enumerate(data, 1)]
    elif suffix in ('.md', '.txt'):
        entries = _markdown_entries(path.read_text(encoding='utf-8'))
    else:
        raise ValueError('Supported list formats: .csv, .md, .txt, .json')
    if not entries:
        raise ValueError('No repository entries found in the input list')
    if len(entries) > MAX_ITEMS:
        raise ValueError(f'List has {len(entries)} entries; maximum is {MAX_ITEMS} per run')
    return entries


def scan_research_list(list_path: str | Path, out: str | Path = 'official_pilot/batch',
                       fetch: bool = False,
                       on_progress: Callable[[int, int, dict], None] | None = None) -> dict:
    """Inspect every entry independently; failures do not stop the remaining cases.

    WARNING: fetch=True downloads GitHub contents in the caller's environment.
    It never imports/executes code from those repositories.
    """
    source = Path(list_path).expanduser().resolve()
    entries = read_research_list(source)
    dst = Path(out).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    for number, entry in enumerate(entries, 1):
        print(f'[{number}/{len(entries)}] {entry.case_id} | {entry.repository_url}', flush=True)
        record = {**asdict(entry), 'status': '', 'error': '', 'files_observed': 0,
                  'python_files': 0, 'notebooks': 0, 'compatibility_leads': 0,
                  'case_report': '', 'commit': '', 'execution_performed': False}
        try:
            url = entry.repository_url
            # Relative local projects are permitted for reproducible offline testing.
            if not url.startswith(('https://', 'http://')):
                url = str((source.parent / url).resolve())
            result = intake_source(url, dst / 'cases', fetch=fetch, paper=entry.paper_url)
            inventory = result['inventory']
            record.update(status=result['status'], files_observed=result['files_observed'],
                          python_files=len(inventory['python_files']),
                          notebooks=len(inventory['notebooks']),
                          compatibility_leads=len(inventory['compatibility_hints']),
                          case_report=str(Path(result['case_dir']) / 'CASE_REPORT.md'),
                          commit=result['repository_commit'] or '')
        except Exception as exc:
            record['status'] = 'INTAKE_ERROR'
            record['error'] = f'{type(exc).__name__}: {str(exc)[:350]}'
        reports.append(record)
        if on_progress is not None:
            on_progress(number, len(entries), record)
        print(f"    -> {record['status']}: {record['python_files']} .py, "
              f"{record['notebooks']} notebooks, {record['compatibility_leads']} leads", flush=True)

    counts = {status: sum(r['status'] == status for r in reports)
              for status in sorted({r['status'] for r in reports})}
    summary = {'version': '0.4.0-rc1-batch', 'list_file': str(source),
               'recorded_utc': datetime.now(timezone.utc).isoformat(),
               'fetch_requested': fetch, 'total': len(reports), 'status_counts': counts,
               'execution_performed': False, 'paper_fulltext_parsed': False,
               'published_results_reproduced': False, 'entries': reports}
    (dst / 'batch_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    with (dst / 'batch_summary.csv').open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(reports[0].keys()))
        wr.writeheader()
        wr.writerows(reports)

    md = ['# RE:SOURCE AI | Batch research repository intake', '',
          f'**Input:** `{source.name}`',
          f'**Entries:** {len(reports)}',
          f'**GitHub fetch requested:** {fetch}',
          '**Code executed:** No',
          '**Published paper results reproduced:** No', '',
          '## Cases', '',
          '| ID | Research project | Intake status | Python files | Compatibility leads |',
          '|---|---|---|---:|---:|']
    for x in reports:
        safe_title = x['title'].replace('|','/').replace('\\n',' ')
        md.append(f"| {x['case_id']} | {safe_title} | {x['status']} | {x['python_files']} | {x['compatibility_leads']} |")
    md.extend(['', '## Interpretation',
               '- **REMOTE_INSPECTED:** Code was downloaded and statically inspected; not executed.',
               '- **LINK_RECORDED:** Only the link was saved; no source code was read.',
               '- **INTAKE_ERROR:** Download or input failed; the agent continued to the next row.',
               '- Compatibility leads are hypotheses, not proof the research code is broken.',
               '- Paper links are saved for attribution. Full papers were not parsed.',
               '- Real reproduction requires an isolated executor and an independently established methodology reference.', ''])
    (dst / 'BATCH_REPORT.md').write_text('\n'.join(md), encoding='utf-8')
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Read a file containing research GitHub links and inspect each automatically')
    ap.add_argument('list_file', help='CSV, Markdown, text or JSON research-link list')
    ap.add_argument('--out', default='official_pilot/batch')
    ap.add_argument('--fetch', action='store_true', help='Download GitHub source and inspect it; never execute external code')
    args = ap.parse_args(argv)
    summary = scan_research_list(args.list_file, args.out, args.fetch)
    print('\nBatch finished:', summary['status_counts'])
    print('Summary:', Path(args.out).resolve() / 'BATCH_REPORT.md')
    return 1 if summary['status_counts'].get('INTAKE_ERROR', 0) else 0


if __name__ == '__main__':
    raise SystemExit(main())
