"""Batch paper/repository-list workflow, tested offline with local fixtures and mocked HTTP."""
import csv
import io
import json
from pathlib import Path

import pytest

from resource_ai.batch import read_research_list, scan_research_list


def test_csv_parses_real_pilot_queue():
    p = Path(__file__).resolve().parents[1] / 'corpus' / 'pilot_queue.csv'
    items = read_research_list(p)
    assert len(items) == 5
    assert items[0].case_id == 'P01'
    assert 'arxiv.org' in items[0].paper_url
    assert 'github.com' in items[0].repository_url


def test_markdown_parses_five_research_pairs():
    p = Path(__file__).resolve().parents[1] / 'corpus' / 'PAPERS_TO_SCAN.md'
    items = read_research_list(p)
    assert len(items) == 5
    assert [v.case_id for v in items] == [f'P{i:02d}' for i in range(1,6)]


def test_real_local_files_inspected_one_by_one_without_exec(tmp_path):
    samples = tmp_path/'files'
    samples.mkdir()
    (samples/'first.py').write_text('from sklearn.ensemble import RandomForestRegressor\n'
                                    'model = RandomForestRegressor(max_features="auto")\n'
                                    'raise RuntimeError("DO NOT RUN")\n')
    (samples/'second.py').write_text('from sklearn.linear_model import LogisticRegression\n'
                                     'model = LogisticRegression()\n'
                                     'raise RuntimeError("DO NOT RUN")\n')
    manifest = tmp_path/'list.csv'
    with manifest.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['case_id','title','paper_url','repository_url'])
        writer.writeheader()
        writer.writerows([
            {'case_id':'L01','title':'One','paper_url':'','repository_url':'files/first.py'},
            {'case_id':'L02','title':'Two','paper_url':'','repository_url':'files/second.py'}])
    summary=scan_research_list(manifest, tmp_path/'out', fetch=False)
    assert summary['status_counts']=={'LOCAL_INSPECTED':2}
    assert summary['execution_performed'] is False
    assert summary['entries'][0]['compatibility_leads'] >= 1
    assert (tmp_path/'out'/'BATCH_REPORT.md').is_file()
    assert (tmp_path/'out'/'batch_summary.json').is_file()
    assert (tmp_path/'out'/'batch_summary.csv').is_file()
    assert all(Path(x['case_report']).is_file() for x in summary['entries'])


def test_url_no_fetch_marks_recorded_not_inspected(tmp_path):
    q=tmp_path/'links.md'
    q.write_text('## P01: First\n- Paper: https://arxiv.org/abs/1609.02907\n'
                 '- GitHub: https://github.com/example/one\n'
                 '## P02: Second\n- GitHub: https://github.com/example/two\n')
    summary=scan_research_list(q, tmp_path/'out')
    assert summary['status_counts']=={'LINK_RECORDED': 2}
    assert all(e['files_observed'] == 0 for e in summary['entries'])
    assert not summary['paper_fulltext_parsed']


def test_batch_fetch_each_github_file_when_requested_with_mocked_http(tmp_path, monkeypatch):
    import resource_ai.intake as intake
    class FakeOpener:
        def open(self, request, timeout):
            assert 'raw.githubusercontent.com' in request.full_url
            return io.BytesIO(b'from sklearn.ensemble import RandomForestRegressor\n'
                              b'm = RandomForestRegressor(max_features="auto")\n'
                              b'raise RuntimeError("Never execute")\n')
    monkeypatch.setattr(intake.urllib.request, 'build_opener', lambda *args: FakeOpener())
    q=tmp_path/'links.csv'
    q.write_text('case_id,title,paper_url,repository_url\n'
                 'G01,First,https://arxiv.org/abs/1609.02907,https://github.com/a/r/blob/main/a.py\n'
                 'G02,Second,https://arxiv.org/abs/1611.07308,https://github.com/b/r/blob/main/b.py\n')
    r=scan_research_list(q,tmp_path/'out',fetch=True)
    assert r['status_counts']=={'REMOTE_INSPECTED':2}
    assert all(e['python_files'] == 1 and e['compatibility_leads'] >= 1 for e in r['entries'])
    assert all(not e['execution_performed'] for e in r['entries'])


def test_failing_row_does_not_stop_next_row(tmp_path):
    local=tmp_path/'local.py'
    local.write_text('print("no run")\n')
    q=tmp_path/'links.csv'
    q.write_text('case_id,title,paper_url,repository_url\n'
                 'B01,Bad,,http://evil.example/this\n'
                 'G01,Valid,,local.py\n')
    r=scan_research_list(q,tmp_path/'out',fetch=True)
    assert [e['status'] for e in r['entries']] == ['INTAKE_ERROR','LOCAL_INSPECTED']
    assert 'Only github.com' in r['entries'][0]['error'] or 'plain HTTPS' in r['entries'][0]['error']


def test_json_list_and_reject_bad_manifest(tmp_path):
    p=tmp_path/'links.json'
    p.write_text(json.dumps([{'case_id':'P7','title':'Demo','repository_url':'https://github.com/x/y'}]))
    assert len(read_research_list(p))==1
    p.write_text(json.dumps([{'case_id':'../../../oops','repository_url':'https://github.com/x/y'}]))
    with pytest.raises(ValueError, match='case_id'):
        read_research_list(p)


def test_no_automatic_scientific_claims(tmp_path):
    q=tmp_path/'list.md'
    q.write_text('https://github.com/tkipf/gcn\n')
    result=scan_research_list(q,tmp_path/'out')
    assert not result['execution_performed']
    assert not result['published_results_reproduced']
    body=(tmp_path/'out'/'BATCH_REPORT.md').read_text()
    assert 'Full papers were not parsed' in body
