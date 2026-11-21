"""Headless regression tests for the v0.3.0 research dashboard backend."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from resource_ai.batch import scan_research_list
from resource_ai.dashboard_logic import (
    available_builtin_cases, collect_evidence, recorded_cases, save_report,
    status_summary, unique_run_dir, uploaded_list_path, validate_ui_research_list,
)


def test_new_run_folders_never_overwrite(tmp_path):
    a = unique_run_dir(tmp_path, 'scan')
    b = unique_run_dir(tmp_path, 'scan')
    assert a.exists() and b.exists() and a != b


def test_upload_and_list_validation(tmp_path):
    data = b'case_id,title,paper_url,repository_url\nP1,Paper,,https://github.com/tkipf/gcn\n'
    path = uploaded_list_path('papers.csv', data, tmp_path)
    entries = validate_ui_research_list(path)
    assert len(entries) == 1 and entries[0].case_id == 'P1'


def test_no_non_github_local_paths_from_user_uploaded_list(tmp_path):
    path = uploaded_list_path('papers.csv',
        b'case_id,title,paper_url,repository_url\nP1,Test,,/tmp/suspicious\n', tmp_path)
    with pytest.raises(ValueError):
        validate_ui_research_list(path)


def test_no_script_upload_or_oversize_list(tmp_path):
    with pytest.raises(ValueError):
        uploaded_list_path('code.py', b'print(123)', tmp_path)
    with pytest.raises(ValueError):
        uploaded_list_path('papers.md', b'x' * 1_000_001, tmp_path)


def test_report_export_omits_project_source(tmp_path):
    (tmp_path/'cases'/'paper_a'/'project').mkdir(parents=True)
    (tmp_path/'cases'/'paper_a'/'project'/'experiment.py').write_text('secret test code')
    (tmp_path/'cases'/'paper_a'/'CASE_REPORT.md').write_text('# findings')
    (tmp_path/'cases'/'paper_a'/'case.json').write_text('{}')
    (tmp_path/'BATCH_REPORT.md').write_text('## batch')
    with zipfile.ZipFile(io.BytesIO(collect_evidence(tmp_path))) as zf:
        assert sorted(zf.namelist()) == ['BATCH_REPORT.md', 'cases/paper_a/CASE_REPORT.md', 'cases/paper_a/case.json']
        assert all('experiment.py' not in n for n in zf.namelist())


def test_summary_cannot_invent_reproduced_results():
    rows = [{'status':'REMOTE_INSPECTED','python_files':4,'compatibility_leads':2},
            {'status':'INTAKE_ERROR','python_files':0,'compatibility_leads':0}]
    report = status_summary(rows)
    assert report['repositories'] == 2
    assert report['sources_inspected'] == 1
    assert report['scientific_reproductions_verified'] == 0


def test_progress_callback_runs_once_per_case(tmp_path):
    research_list = tmp_path/'links.md'
    research_list.write_text('## P01: One\n- GitHub: https://github.com/tkipf/gcn\n'
                             '## P02: Two\n- GitHub: https://github.com/tkipf/gae\n')
    events = []
    result = scan_research_list(research_list, tmp_path/'out', fetch=False,
                                on_progress=lambda i, total, record: events.append((i, total, record['status'])))
    assert events == [(1, 2, 'LINK_RECORDED'), (2, 2, 'LINK_RECORDED')]
    assert result['total'] == 2
    assert (tmp_path/'out'/'BATCH_REPORT.md').exists()


def test_history_reads_saved_json(tmp_path):
    save_report({'status':'LINK_RECORDED'}, tmp_path/'case.json')
    items = recorded_cases(tmp_path)
    assert len(items) == 1
    assert items[0]['status'] == 'LINK_RECORDED'


def test_bundled_case_inventory():
    root = Path(__file__).resolve().parents[1]
    assert 'RF02_max_features_auto_reg' in available_builtin_cases(root)
