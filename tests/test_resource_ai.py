"""Fast tests for the core pieces.  Run:  python -m pytest -q tests"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resource_ai import patching  # noqa: E402
from resource_ai.agent import RepairAgent  # noqa: E402
from resource_ai.diagnose import diagnose  # noqa: E402
from resource_ai.knowledge import retrieve  # noqa: E402
from resource_ai.llm import ScriptedLLM  # noqa: E402
from resource_ai.runner import run_script  # noqa: E402
from resource_ai.verify import verify  # noqa: E402

CASES = ROOT / "benchmark" / "cases"


def _write(tmp_path, name, code):
    d = tmp_path / name
    d.mkdir()
    (d / "experiment.py").write_text(code)
    return d / "experiment.py"


@pytest.fixture(scope="module")
def rf00_ref():
    return run_script(CASES / "RF00_param_typo" / "reference" / "experiment.py").trace


# ---------------------------------------------------------------- patching
def test_parse_and_apply_search_replace():
    src = "a = 1\nmodel = RF(n_estimator=5)\nprint(a)\n"
    reply = "experiment.py\n<<<<<<< SEARCH\nmodel = RF(n_estimator=5)\n=======\nmodel = RF(n_estimators=5)\n>>>>>>> REPLACE"
    edits, full = patching.parse(reply)
    assert full is None and len(edits) == 1
    assert patching.apply(src, edits) == "a = 1\nmodel = RF(n_estimators=5)\nprint(a)\n"


def test_apply_tolerates_copied_line_numbers_and_indentation():
    src = "for i in range(3):\n    x = old(i)\n"
    reply = "<<<<<<< SEARCH\n  2|     x = old(i)\n=======\n  2|     x = new(i)\n>>>>>>> REPLACE"
    edits, _ = patching.parse(reply)
    assert patching.apply(src, edits) == "for i in range(3):\n    x = new(i)\n"


def test_apply_rejects_missing_search():
    with pytest.raises(patching.PatchError):
        patching.apply("x = 1\n", [patching.Edit("y = 2", "y = 3")])


# ---------------------------------------------------------------- diagnosis
def test_diagnose_syntax_error_does_not_crash(tmp_path):
    script = _write(tmp_path, "p", "import numpy as np\nprint 'hello'\n")
    r = run_script(script)
    d = diagnose(r.stderr, script.parent, script)
    assert d.failed and d.category == "syntax" and d.line == 2 and d.source_parses is False


def test_diagnose_finds_user_line_and_knowledge(tmp_path):
    script = _write(tmp_path, "p", "import numpy as np\nx = np.zeros(3, dtype=np.float)\n")
    r = run_script(script)
    d = diagnose(r.stderr, script.parent, script)
    assert d.exception_type == "AttributeError" and d.line == 2
    assert "np_aliases" in [k["id"] for k in d.knowledge]


def test_knowledge_retrieval_warns_about_regressor_trap():
    hits = retrieve("RandomForestRegressor(max_features='auto')", "")
    e = next(h for h in hits if h.id == "sk_max_features_auto")
    assert "1.0" in e.fix and "sqrt" in e.warning


# ---------------------------------------------------------------- verifier
def test_verifier_accepts_exact_reproduction(rf00_ref):
    rep = verify(rf00_ref, run_script(CASES / "RF00_param_typo" / "fixes" / "oracle.py",
                                      cwd=CASES / "RF00_param_typo" / "broken"))
    assert rep.verdict == "REPRODUCED" and not rep.violations


def test_verifier_catches_changed_hyperparameter_that_score_rule_accepts(rf00_ref, tmp_path):
    code = (CASES / "RF00_param_typo" / "fixes" / "delete_arg.py").read_text()
    rep = verify(rf00_ref, run_script(_write(tmp_path, "p", code)))
    assert rep.verdict == "METHOD_CHANGED"
    assert any("n_estimators" in v for v in rep.violations)
    assert rep.accepted_by_score_rule          # the 10% rule would have accepted it


def test_verifier_catches_leakage(rf00_ref, tmp_path):
    code = (CASES / "RF00_param_typo" / "reference" / "experiment.py").read_text()
    code = code.replace("model.fit(X_train, y_train)", "model.fit(X, y)")
    rep = verify(rf00_ref, run_script(_write(tmp_path, "p", code)))
    assert rep.verdict == "METHOD_CHANGED" and any("leakage" in v for v in rep.violations)


def test_verifier_reports_failed_run(rf00_ref):
    rep = verify(rf00_ref, run_script(CASES / "RF00_param_typo" / "broken" / "experiment.py"))
    assert rep.verdict == "FAILS_TO_RUN"


# ---------------------------------------------------------------- agent
def _trap_then_oracle(case):
    broken = (CASES / case / "broken" / "experiment.py").read_text()
    meta = json.loads((CASES / case / "case.json").read_text())
    trap = (CASES / case / "fixes" / next(f["file"] for f in meta["fixes"] if f["label"] == "changes_method")).read_text()
    oracle = (CASES / case / "fixes" / "oracle.py").read_text()
    return [patching.to_blocks(broken, trap), patching.to_blocks(trap, oracle)]


@pytest.mark.parametrize("mode,expect_verdict,expect_calls", [("naive", "METHOD_CHANGED", 1), ("guarded", "REPRODUCED", 2)])
def test_agent_modes(rf00_ref, tmp_path, mode, expect_verdict, expect_calls):
    llm = ScriptedLLM(_trap_then_oracle("RF00_param_typo"))
    res = RepairAgent(CASES / "RF00_param_typo" / "broken", rf00_ref, llm, mode,
                      workdir=tmp_path / "w", log=None).run()
    assert res.accepted and res.final_verdict == expect_verdict and res.llm_calls == expect_calls
    assert res.silent_method_change == (mode == "naive")
    # original broken project untouched
    assert "n_estimator=200" in (CASES / "RF00_param_typo" / "broken" / "experiment.py").read_text()


def test_guarded_agent_feeds_back_methodology_violation(rf00_ref, tmp_path):
    llm = ScriptedLLM(_trap_then_oracle("RF00_param_typo"))
    RepairAgent(CASES / "RF00_param_typo" / "broken", rf00_ref, llm, "guarded", workdir=tmp_path / "w", log=None).run()
    second_prompt = llm.calls[1][-1]["content"]
    assert "n_estimators: was 200, now 100" in second_prompt


def test_agent_gives_up_honestly_when_only_bad_fixes(rf00_ref, tmp_path):
    bad = _trap_then_oracle("RF00_param_typo")[0]
    res = RepairAgent(CASES / "RF00_param_typo" / "broken", rf00_ref, ScriptedLLM([bad] + ["no edits"] * 3), "guarded",
                      workdir=tmp_path / "w", max_iters=3, log=None).run()
    assert not res.accepted and res.final_verdict == "METHOD_CHANGED"


# ---------------------------------------------------------------- regression: unsupervised steps
@pytest.mark.parametrize("case", ["LR01_penalty_none", "IMP01_imputer", "KM01_kmeans_full_ninit",
                                  "AGG01_affinity", "GRID01_iid_cv_default"])
def test_oracle_with_unsupervised_or_search_steps_is_reproduced(case):
    """Correct repairs containing scalers, imputers, clustering or GridSearchCV must be
    REPRODUCED, not NO_EVIDENCE (labels can't be paired for steps that have none)."""
    ref = run_script(CASES / case / "reference" / "experiment.py").trace
    rep = verify(ref, run_script(CASES / case / "fixes" / "oracle.py", cwd=CASES / case / "broken"))
    assert rep.verdict == "REPRODUCED", (rep.violations, rep.evidence_gaps)
