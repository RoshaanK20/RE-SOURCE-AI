"""Offline simulated executor tests; do not run untrusted projects or Docker."""
import json
from pathlib import Path

import pytest

from resource_ai.framework_trial import inspect_entry, run_trial, TrialRun


def trial(ok=True, returncode=0):
    return TrialRun(ok, returncode, 0.01, False, '', '' if ok else 'RuntimeError: simulated failure')


def test_source_entry_validated_for_any_framework(tmp_path):
    root = tmp_path / 'code'
    root.mkdir()
    (root / 'train.py').write_text('import torch\nprint("ok")\n')
    base, target = inspect_entry(root, 'train.py')
    assert base == root and target == root / 'train.py'
    with pytest.raises(ValueError):
        inspect_entry(root, '../outside.py')


def test_trial_can_record_failure_without_fake_reproduction(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'train.py').write_text('import tensorflow as tf\nprint("existing code")\n')
    result = run_trial(original, 'train.py', tmp_path / 'results',
                       executor=lambda *a,**kw: trial(False, 1), max_retries=0)
    assert result['outcome'] == 'EXECUTION_FAILED'
    assert result['scientific_reproduction_verified'] is False
    assert (tmp_path / 'results' / 'trial_result.json').exists()


def test_trial_success_is_execution_only(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'train.py').write_text('import torch\nprint("hello")\n')
    result = run_trial(original, 'train.py', tmp_path / 'results',
                       executor=lambda *a,**kw: trial(), max_retries=0)
    assert result['outcome'] == 'EXECUTES_UNVERIFIED'
    assert result['methodology_verified'] is False


def test_trial_rejects_output_inside_source(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'train.py').write_text('x = 1')
    with pytest.raises(ValueError, match='inside'):
        run_trial(original, 'train.py', original / 'results', executor=lambda *a,**kw: trial())


def test_trial_rejects_symlinks_before_copying(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'train.py').write_text('print("safe")')
    secret = tmp_path / 'secret.txt'
    secret.write_text('not part of the repository')
    try:
        (original / 'data.txt').symlink_to(secret)
    except OSError:
        pytest.skip('Symlinks not available on this platform')
    with pytest.raises(ValueError, match='Symlinks'):
        run_trial(original, 'train.py', tmp_path / 'out', executor=lambda *a,**kw: trial())


def test_mock_qwen_cross_framework_patch_does_not_claim_science(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'train.py').write_text('import torch\ncompat_option = "old"\n')
    class FakeReply:
        text = ('train.py\n<<<<<<< SEARCH\ncompat_option = "old"\n'
                '=======\ncompat_option = "new"\n>>>>>>> REPLACE')
    class FakeQwen:
        name = 'simulated-coder-for-test'
        def chat(self, _messages):
            return FakeReply()
    def mock_executor(folder, entry, **kw):
        patched = 'compat_option = "new"' in (Path(folder) / entry).read_text()
        return trial(patched, 0 if patched else 1)
    result = run_trial(original, 'train.py', tmp_path / 'out',
                       llm=FakeQwen(), executor=mock_executor, max_retries=1)
    assert result['initial_execution']['ok'] is False
    assert result['final_execution']['ok'] is True
    assert result['outcome'] == 'EXECUTES_UNVERIFIED'
    assert result['attempts'][0]['patch_applied'] is True
    assert result['scientific_reproduction_verified'] is False
