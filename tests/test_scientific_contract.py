"""Contract regression tests use toy source files and mocked evaluator results.

NO published research experiment, live Docker, Qwen, or TensorFlow/PyTorch is run.
"""
import json
import shutil
from pathlib import Path

import pytest

from resource_ai.science import sha, reference_digest, verify_science, _contract, as_markdown

PINNED_IMAGE = 'python:3.11-slim@sha256:' + 'f' * 64  # test placeholder, NOT a real Docker image digest


def make_case(tmp_path: Path, framework='pytorch'):
    root = tmp_path / 'original'
    cand = tmp_path / 'repaired'
    root.mkdir()
    (root / 'data').mkdir()
    (root / 'config').mkdir()
    (root / 'data' / 'dataset.csv').write_text('x,y\n1,0\n2,1\n')
    (root / 'config' / 'method.json').write_text('{"seed":42,"epochs":10,"split":"80:20"}')
    source = ('import torch\n' if framework == 'pytorch' else 'import tensorflow as tf\n')
    source += ('# Paper method: model=A, epochs=10, seed=42\n'
               'optimizer = "adam"\n'
               'library_option = "legacy"\n'
               'print("measurement must come from an external evaluator")\n')
    (root / 'train.py').write_text(source)
    shutil.copytree(root, cand)
    (cand / 'train.py').write_text(source.replace('library_option = "legacy"',
                                                  'library_option = "compatible"'))
    evaluator = tmp_path / 'trusted_eval.py'
    evaluator.write_text('print("a trusted program independent of the candidate")')
    spec = {
        'schema': 'resource.scientific-contract.v1',
        'paper': {'title':'Test paper (NOT a real published experiment)',
                  'repository_url':'https://github.com/example/paper', 'commit':'a'*40},
        'framework': framework,
        'review_status': 'approved',
        'methodology': {
            'dataset': {'name': 'demo', 'provenance': 'local toy fixture'},
            'split': {'ratio': '80:20', 'seed': 42},
            'preprocessing': {'steps': 'none'},
            'model': {'architecture': 'A'},
            'training': {'optimizer': 'adam', 'epochs': 10},
            'evaluation': {'protocol':'heldout', 'metric':'accuracy'}
        },
        'reference_tree_sha256': reference_digest(root),
        'trusted_evaluator_sha256': sha(evaluator),
        'evidence_files': {
            'dataset': ['data/dataset.csv'], 'split': ['config/method.json'],
            'preprocessing': ['train.py'], 'model': ['train.py'],
            'training': ['config/method.json', 'train.py'], 'evaluation': ['train.py']},
        'protected_files': {'data/dataset.csv':sha(root / 'data' / 'dataset.csv'),
                            'config/method.json':sha(root / 'config' / 'method.json')},
        'approved_changes': [{'path':'train.py', 'before':'library_option = "legacy"',
                              'after':'library_option = "compatible"',
                              'reason':'reviewed API compatibility equivalence'}],
        'expected_metric': {'name':'accuracy', 'value':0.8, 'abs_tolerance':0.02}
    }
    contract = tmp_path / 'paper_contract.json'
    contract.write_text(json.dumps(spec))
    return contract, root, cand, evaluator


def mock_measurement(*_):
    return {'metrics': {'accuracy':0.81}}


@pytest.mark.parametrize('framework', ['pytorch','tensorflow','jax','numpy'])
def test_framework_independent_checks_for_reviewed_toy_contract(tmp_path, framework):
    case = make_case(tmp_path, framework)
    report = verify_science(*case[:3], evaluator_path=case[3], docker_image=PINNED_IMAGE,
                            _test_evaluator=mock_measurement)
    assert report['verdict'] == 'PASSED_SPECIFIED_SCIENTIFIC_CHECKS'
    assert report['framework'] == framework
    assert report['measured_metric'] == 0.81
    assert 'scientific integrity report' in as_markdown(report).lower()


def test_training_method_change_is_not_silently_accepted(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    p = cand / 'train.py'
    p.write_text(p.read_text().replace('epochs=10', 'epochs=20'))
    result = verify_science(contract, base, cand, evaluator, PINNED_IMAGE,
                            _test_evaluator=mock_measurement)
    assert result['verdict'] == 'UNVERIFIABLE'
    assert any('beyond approved' in ch['detail'] for ch in result['checks'])


def test_changing_protected_dataset_is_methodology_violation(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    (cand / 'data' / 'dataset.csv').write_text('x,y\n1,1\n2,1\n')
    result = verify_science(contract, base, cand, evaluator, PINNED_IMAGE,
                            _test_evaluator=mock_measurement)
    assert result['verdict'] == 'METHODOLOGY_VIOLATION'


def test_missing_independent_evaluator_never_passes(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    result = verify_science(contract, base, cand)
    assert result['verdict'] == 'UNVERIFIABLE'


def test_unreviewed_paper_contract_cannot_pass(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    spec = json.loads(contract.read_text())
    spec['review_status'] = 'draft'
    contract.write_text(json.dumps(spec))
    result = verify_science(contract, base, cand, evaluator, PINNED_IMAGE,
                            _test_evaluator=mock_measurement)
    assert result['verdict'] == 'UNVERIFIABLE'


def test_fake_self_reported_metric_not_accepted(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    (cand / 'train.py').write_text((cand/'train.py').read_text() + '\nprint("accuracy=0.8")\n')
    result = verify_science(contract, base, cand, evaluator, PINNED_IMAGE,
                            _test_evaluator=mock_measurement)
    assert result['verdict'] == 'UNVERIFIABLE'


def test_metric_does_not_match_reference(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    result = verify_science(contract, base, cand, evaluator, PINNED_IMAGE,
                            _test_evaluator=lambda *_: {'metrics':{'accuracy':0.3}})
    assert result['verdict'] == 'RESULT_NOT_REPRODUCED'


def test_reference_snapshot_must_match(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    (base / 'train.py').write_text('changed original')
    result = verify_science(contract, base, cand, evaluator, PINNED_IMAGE,
                            _test_evaluator=mock_measurement)
    assert result['verdict'] == 'UNVERIFIABLE'


def test_independent_evaluator_must_match_pinned_hash(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    evaluator.write_text('changed evaluator code')
    result = verify_science(contract, base, cand, evaluator, PINNED_IMAGE,
                            _test_evaluator=mock_measurement)
    assert result['verdict'] == 'UNVERIFIABLE'


def test_unpinned_environment_does_not_pass(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    result = verify_science(contract, base, cand, evaluator, 'python:3.11-slim',
                            _test_evaluator=mock_measurement)
    assert result['verdict'] == 'UNVERIFIABLE'


def test_missing_dataset_identity_rejected(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    spec = json.loads(contract.read_text())
    del spec['protected_files']
    contract.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='protected'):
        _contract(contract)


def test_unapproved_new_source_file_is_unverifiable(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    (cand / 'unexpected.py').write_text('print("new")')
    result = verify_science(contract, base, cand, evaluator, PINNED_IMAGE,
                            _test_evaluator=mock_measurement)
    assert result['verdict'] == 'UNVERIFIABLE'


def test_missing_dataset_provenance_prevents_pass(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    spec = json.loads(contract.read_text())
    spec['evidence_files']['dataset'] = []
    contract.write_text(json.dumps(spec))
    result = verify_science(contract, base, cand, evaluator, PINNED_IMAGE,
                            _test_evaluator=mock_measurement)
    assert result['verdict'] == 'UNVERIFIABLE'


def test_unprotected_dataset_blocks_scientific_pass(tmp_path):
    contract, base, cand, evaluator = make_case(tmp_path)
    spec = json.loads(contract.read_text())
    del spec['protected_files']['data/dataset.csv']
    contract.write_text(json.dumps(spec))
    result = verify_science(contract, base, cand, evaluator, PINNED_IMAGE,
                            _test_evaluator=mock_measurement)
    assert result['verdict'] == 'UNVERIFIABLE'
