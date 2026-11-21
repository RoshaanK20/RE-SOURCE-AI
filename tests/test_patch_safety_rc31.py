"""Research-trial safety regression tests for observed GCN failure, using mocks only."""
from pathlib import Path
from resource_ai.framework_trial import run_trial, TrialRun


class FakeReply:
    def __init__(self, text):
        self.text = text


class FakeQwen:
    name = 'simulated-coder-for-test'
    def __init__(self, text):
        self.text = text
        self.calls = 0
    def chat(self, messages):
        self.calls += 1
        return FakeReply(self.text)


def failed(error):
    return TrialRun(False, 1, 0.01, False, '', error)


def make_project(tmp_path, code):
    d=tmp_path/'research';d.mkdir();(d/'train.py').write_text(code);return d


def test_missing_top_level_tensorflow_is_environment_blocker(tmp_path):
    project=make_project(tmp_path,'import tensorflow as tf\nprint("train")\n')
    llm=FakeQwen('train.py\n<<<<<<< SEARCH\nprint("train")\n=======\nprint("fake")\n>>>>>>> REPLACE')
    original=(project/'train.py').read_text()
    result=run_trial(project,'train.py',tmp_path/'out',
                     executor=lambda *a,**kw:failed("ModuleNotFoundError: No module named 'tensorflow'"),
                     llm=llm,max_retries=2)
    assert result['outcome']=='ENVIRONMENT_BLOCKED'
    assert result['environment_issue']=='Missing top-level Python package: tensorflow'
    assert result['attempts']==[]
    assert llm.calls==0
    assert (project/'train.py').read_text()==original


def test_malformed_fenced_full_file_is_not_accepted(tmp_path):
    original='import torch\nold = 1\nprint(old)\n'
    project=make_project(tmp_path,original)
    # The dangerous real-world failure: malformed block accidentally treated as
    # complete replacement with SEARCH/======= markers embedded in source code.
    llm=FakeQwen('```python\nSEARCH\nimport torch\n=======\ntry:\n    import torch\nexcept ImportError:\n    pass\n```')
    result=run_trial(project,'train.py',tmp_path/'out',
                     executor=lambda *a,**kw:failed('RuntimeError: simulated API break'),
                     llm=llm,max_retries=1)
    assert result['attempts'][0]['patch_applied'] is False
    assert 'Whole-file replacements are disabled' in result['attempts'][0]['error']
    assert result['final_diff']==''
    assert (project/'train.py').read_text()==original


def test_destructive_search_replace_is_rejected(tmp_path):
    lines=['import torch']+[f'x{i} = {i}' for i in range(60)]
    original='\n'.join(lines)+'\n'
    project=make_project(tmp_path,original)
    llm=FakeQwen('train.py\n<<<<<<< SEARCH\n'+original.rstrip('\n')+\
                 '\n=======\nimport torch\nprint("only import")\n>>>>>>> REPLACE')
    result=run_trial(project,'train.py',tmp_path/'out',
                     executor=lambda *a,**kw:failed('RuntimeError: simulated compatibility failure'),
                     llm=llm,max_retries=1)
    assert result['attempts'][0]['patch_applied'] is False
    assert 'too many research-code lines' in result['attempts'][0]['error']
    assert result['final_diff']==''


def test_small_compatibility_edit_still_allowed(tmp_path):
    project=make_project(tmp_path,'import torch\ncompat_flag = "old"\n')
    llm=FakeQwen('train.py\n<<<<<<< SEARCH\ncompat_flag = "old"\n=======\ncompat_flag = "new"\n>>>>>>> REPLACE')
    def fake_run(folder,entry,**kw):
        code=(Path(folder)/entry).read_text()
        if 'compat_flag = "new"' in code: return TrialRun(True,0,.01,False,'ok','')
        return failed('RuntimeError: removed option')
    result=run_trial(project,'train.py',tmp_path/'out',
                     executor=fake_run,llm=llm,max_retries=1)
    assert result['outcome']=='EXECUTES_UNVERIFIED'
    assert result['attempts'][0]['patch_applied'] is True
    assert result['scientific_reproduction_verified'] is False
