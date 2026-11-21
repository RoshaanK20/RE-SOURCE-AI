"""Actual dedicated-venv execution regression checks (trusted toy scripts only)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from resource_ai.framework_trial import run_trial, run_in_local
from resource_ai.local_runtime import check_local_python, local_environment_fingerprint
from resource_ai.science import reference_digest, sha, verify_science


@pytest.fixture(scope='module')
def trusted_python(tmp_path_factory):
    folder = tmp_path_factory.mktemp('trusted_research_venv') / 'env'
    res = subprocess.run([sys.executable, '-m', 'venv', str(folder)], capture_output=True, text=True)
    if res.returncode:
        pytest.skip('venv creation unavailable: ' + res.stderr[-300:])
    binary = folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    assert check_local_python(binary)['available']
    yield binary


def test_docker_remains_default_and_local_requires_explicit_trust(tmp_path, trusted_python):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'train.py').write_text('print("okay")')
    with pytest.raises(ValueError, match='explicitly trusted'):
        run_trial(root, 'train.py', tmp_path / 'result', backend='local', python_executable=trusted_python)
    assert not check_local_python(sys.executable)['available']


def test_local_venv_executes_reviewed_script_without_docker(tmp_path, trusted_python):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'train.py').write_text('print("trusted pipeline run")')
    result = run_trial(root, 'train.py', tmp_path / 'results', backend='local',
                       python_executable=trusted_python, allow_trusted_local=True)
    assert result['final_execution']['ok']
    assert result['final_execution']['command_kind'] == 'trusted_local_venv_python'
    assert 'trusted pipeline run' in result['final_execution']['stdout_tail']
    assert result['outcome'] == 'EXECUTES_UNVERIFIED'
    assert result['scientific_reproduction_verified'] is False
    assert (tmp_path / 'results' / 'trial_result.json').is_file()


def test_real_local_venv_patch_loop(tmp_path, trusted_python):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'train.py').write_text('compat_flag = "old"\nif compat_flag == "old":\n    raise RuntimeError("Old API")\nprint("works")\n')
    class Reply:
        text = '<<<<<<< SEARCH\ncompat_flag = "old"\n=======\ncompat_flag = "new"\n>>>>>>> REPLACE'
    class MockLLM:
        name = 'scripted-llm-response-for-test'
        def chat(self, _):
            return Reply()
    result = run_trial(root, 'train.py', tmp_path / 'out', backend='local',
                       python_executable=trusted_python, allow_trusted_local=True, llm=MockLLM(), max_retries=1)
    assert not result['initial_execution']['ok']
    assert not result['final_execution']['ok']  # unreviewed generated code NOT executed
    assert result['attempts'][0]['requires_human_review']
    assert result['outcome'] == 'PATCH_REVIEW_REQUIRED'
    assert (tmp_path / 'out' / 'PROPOSED_PATCH.diff').is_file()
    assert 'old' in (root / 'train.py').read_text()  # original untouched
    assert 'old' in (tmp_path / 'out' / 'candidate_copy' / 'train.py').read_text()



def _paper_fixture(tmp_path, trusted_python):
    ref = tmp_path / 'reference'
    ref.mkdir()
    (ref / 'dataset.csv').write_text('x,y\n1,0\n2,1\n')
    (ref / 'train.py').write_text('METHOD = "original"\nOPTION = "old"\n')
    candidate = tmp_path / 'candidate'
    shutil.copytree(ref, candidate)
    (candidate / 'train.py').write_text('METHOD = "original"\nOPTION = "new"\n')
    evaluator = tmp_path / 'trusted_evaluator.py'
    evaluator.write_text('import json\nprint(json.dumps({"metrics": {"accuracy": 0.75}}))\n')
    contract = tmp_path / 'contract.json'
    spec = {
        'schema': 'resource.scientific-contract.v1',
        'paper': {'title': 'Toy test, not published', 'repository_url': 'https://github.com/example/toy', 'commit': 'a'*40},
        'framework': 'pytorch', 'review_status': 'approved',
        'methodology': {'dataset': {'identity': 'toy'}, 'split': {'seed': 42},
            'preprocessing': {'method': 'none'}, 'model': {'architecture': 'linear'},
            'training': {'epochs': 1}, 'evaluation': {'metric': 'accuracy'}},
        'reference_tree_sha256': reference_digest(ref),
        'trusted_evaluator_sha256': sha(evaluator),
        'protected_files': {'dataset.csv': sha(ref/'dataset.csv')},
        'evidence_files': {k: (['dataset.csv'] if k == 'dataset' else ['train.py'])
                           for k in ('dataset','split','preprocessing','model','training','evaluation')},
        'approved_changes': [{'path':'train.py','before':'OPTION = "old"','after':'OPTION = "new"',
                              'reason':'toy test compatibility'}],
        'expected_metric': {'name': 'accuracy', 'value': 0.75, 'abs_tolerance': 0.0},
        'local_environment_sha256': local_environment_fingerprint(trusted_python)['sha256'],
    }
    contract.write_text(json.dumps(spec))
    return contract, ref, candidate, evaluator


def test_trusted_local_independent_evaluator_with_environment_lock(tmp_path, trusted_python):
    contract, ref, candidate, evaluator = _paper_fixture(tmp_path, trusted_python)
    result = verify_science(contract, ref, candidate, evaluator_path=evaluator,
                            execution_backend='local', local_python=trusted_python,
                            allow_trusted_local=True)
    assert result['verdict'] == 'PASSED_SPECIFIED_SCIENTIFIC_CHECKS'
    assert result['evaluator_evidence']['execution_backend'] == 'trusted_local_venv'
    assert result['measured_metric'] == 0.75


def test_local_evaluator_without_locked_environment_stays_unverifiable(tmp_path, trusted_python):
    contract, ref, candidate, evaluator = _paper_fixture(tmp_path, trusted_python)
    data = json.loads(contract.read_text())
    del data['local_environment_sha256']
    contract.write_text(json.dumps(data))
    result = verify_science(contract, ref, candidate, evaluator_path=evaluator,
                            execution_backend='local', local_python=trusted_python,
                            allow_trusted_local=True)
    assert result['verdict'] == 'UNVERIFIABLE'
    assert result['measured_metric'] == 0.75


def test_local_science_requires_explicit_trust(tmp_path, trusted_python):
    contract, ref, candidate, evaluator = _paper_fixture(tmp_path, trusted_python)
    with pytest.raises(ValueError, match='explicit trust'):
        verify_science(contract, ref, candidate, evaluator_path=evaluator,
                       execution_backend='local', local_python=trusted_python)
