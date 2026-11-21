import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from resource_ai.intake import github_source, intake_source
from resource_ai.pilot import main as pilot_main


def test_github_repository_url_parses():
    r = github_source('https://github.com/tkipf/gcn')
    assert (r.owner, r.repo, r.kind) == ('tkipf', 'gcn', 'repo')


def test_github_blob_link_parses():
    r = github_source('https://github.com/example/demo/blob/main/experiments/train.py')
    assert (r.kind, r.revision, r.path) == ('file', 'main', 'experiments/train.py')


@pytest.mark.parametrize('url', [
    'http://github.com/a/b',
    'https://example.org/a/b',
    'https://github.com/a/b/blob/main/../../secret.py',
    'https://github.com/a/b?token=secret',
    'https://github.com/a/b/blob/main/notebook.exe',
])
def test_unapproved_urls_rejected(url):
    with pytest.raises(ValueError):
        github_source(url)


def test_link_records_without_network(tmp_path):
    result = intake_source('https://github.com/tkipf/gcn', tmp_path)
    assert result['status'] == 'LINK_RECORDED'
    assert result['execution_performed'] is False
    assert Path(result['case_dir'], 'case.json').exists()


def test_local_script_inspected_without_exec(tmp_path):
    s = tmp_path / 'train.py'
    s.write_text('from sklearn.ensemble import RandomForestClassifier\nraise RuntimeError("do not run")')
    result = intake_source(str(s), tmp_path / 'out')
    assert result['status'] == 'LOCAL_INSPECTED'
    assert result['inventory']['python_files'] == ['train.py']
    assert 'RandomForest' in result['inventory']['detected_terms']
    assert result['execution_performed'] is False


def test_bad_zip_path_rejected(tmp_path):
    z = tmp_path / 'bad.zip'
    with ZipFile(z,'w') as ar:
        ar.writestr('../escape.py', 'pass')
    with pytest.raises(ValueError):
        intake_source(str(z), tmp_path / 'out')
    assert not (tmp_path / 'escape.py').exists()


def test_repo_url_execution_is_blocked(tmp_path):
    with pytest.raises(SystemExit):
        pilot_main(['https://github.com/tkipf/gcn', '--trusted-local',
                    '--reference', 'original.py', '--out', str(tmp_path)])


def test_notebook_local_inventory(tmp_path):
    nb = tmp_path / 'sample.ipynb'
    nb.write_text(json.dumps({'cells':[{'cell_type':'code','source':['import sklearn\n', 'x=1']}] }))
    result = intake_source(str(nb), tmp_path/'out')
    assert result['inventory']['notebooks'] == ['sample.ipynb']
    assert result['execution_performed'] is False


def test_mocked_github_file_download_inspects_but_does_not_execute(tmp_path, monkeypatch):
    import io
    import resource_ai.intake as intake

    class FakeOpener:
        def open(self, request, timeout):
            assert 'raw.githubusercontent.com' in request.full_url
            return io.BytesIO(b'from sklearn.ensemble import RandomForestRegressor\nraise RuntimeError("must not run")')

    monkeypatch.setattr(intake.urllib.request, 'build_opener', lambda *args: FakeOpener())
    result = intake_source('https://github.com/example/demo/blob/main/train.py', tmp_path, fetch=True)
    assert result['status'] == 'REMOTE_INSPECTED'
    assert result['inventory']['python_files'] == ['train.py']
    assert result['execution_performed'] is False
    assert not result['scientific_reproduction_verified']


def test_curated_compatibility_hints_are_not_labeled_failures(tmp_path):
    sample = tmp_path / 'old.py'
    sample.write_text('from sklearn.ensemble import RandomForestRegressor\n'
                      'r = RandomForestRegressor(max_features="auto")\n')
    result = intake_source(str(sample), tmp_path / 'out')
    leads = result['inventory']['compatibility_hints']
    assert any('auto' in l['potential_issue'] for l in leads)
    assert all(l['status'] == 'HYPOTHESIS_NOT_OBSERVED_FAILURE' for l in leads)
    assert not result['execution_performed']
