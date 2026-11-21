"""The repair agent.

Loop:  run (instrumented) -> diagnose -> ask LLM for a minimal patch -> apply to a COPY ->
       rerun -> check acceptance -> feed back what is still wrong -> repeat (max N times).

All modes use the same LLM and the same instructions (including "do not change the
methodology"). They differ only in WHEN a repair is accepted and WHAT feedback the LLM gets:

  naive      accept as soon as the experiment runs              (typical "make it run" agent)
  score      accept when it runs and the main score is within 10% of the original;
             otherwise tell the LLM the target score  (simplified score-only baseline)
  guarded    accept only when the verifier says REPRODUCED;
             otherwise tell the LLM exactly which methodology/results differ   (RE:SOURCE)
  guarded+kb guarded + retrieved breaking-change knowledge in the prompt       (RE:SOURCE full)

Whatever the mode, the final repair is always checked by the full verifier, so we can measure
how often a mode accepted a repair that silently changed the science.
The original project is never modified.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import patching
from .diagnose import diagnose, knowledge_text
from .knowledge import format_hints, retrieve
from .runner import RunResult, copy_project, run_script
from .verify import REPRODUCED, Report, verify

MODES = ("naive", "score", "guarded", "guarded+kb")

SYSTEM_PROMPT = """You repair old machine-learning research code so that it runs with TODAY's library versions \
WITHOUT changing the scientific methodology of the original experiment.

Methodology = the dataset and its rows, preprocessing, the train/test split or cross-validation folds, \
the model type and EVERY hyper-parameter value that was in effect originally (including old library defaults), \
the training settings, and the evaluation metric.

Rules:
- Make the smallest change that fixes the problem. Do not refactor or add features.
- When an argument or function was renamed or removed, translate it to the new API with the SAME meaning. \
Do not just delete it if its default value differs from what the original used.
- Never change random seeds, data splits, model settings or metrics to make the code run or to change the score.

Answer ONLY with edit blocks in exactly this format (you may give several blocks):

experiment.py
<<<<<<< SEARCH
exact lines copied from the current file (without line numbers)
=======
replacement lines
>>>>>>> REPLACE
"""


@dataclass
class Step:
    iteration: int
    action: str                  # "run", "llm", "patch_failed", "accepted", ...
    detail: str = ""
    runs: bool | None = None
    verdict: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


@dataclass
class AgentResult:
    mode: str
    accepted: bool               # accepted by THIS mode's own criterion
    stop_reason: str
    iterations: int
    llm_calls: int
    final_runs: bool
    final_verdict: str           # full verifier verdict of the final code (all modes)
    final_report: dict
    diff: str
    steps: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def silent_method_change(self) -> bool:
        """Accepted by the mode, runs, but the verifier says the methodology changed."""
        return self.accepted and self.final_runs and self.final_verdict == "METHOD_CHANGED"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["silent_method_change"] = self.silent_method_change
        return d


def _numbered(code: str) -> str:
    return "\n".join(f"{i:>3}| {ln}" for i, ln in enumerate(code.splitlines(), 1))


class RepairAgent:
    def __init__(self, project_dir: str | Path, reference_trace: dict, llm, mode: str = "guarded+kb",
                 script: str = "experiment.py", workdir: str | Path | None = None, max_iters: int = 5,
                 timeout: int = 300, log=print):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.project_dir, self.script_name = Path(project_dir), script
        self.reference, self.llm, self.mode = reference_trace, llm, mode
        self.max_iters, self.timeout, self.log = max_iters, timeout, log or (lambda *a, **k: None)
        self.workdir = Path(workdir) if workdir else Path(project_dir).parent / f"_repair_{mode.replace('+', '_')}"

    # ------------------------------------------------------------------ acceptance
    def _accept(self, run: RunResult, rep: Report) -> tuple[bool, str]:
        if not run.ok:
            return False, "fails to run"
        if self.mode == "naive":
            return True, "runs"
        if self.mode == "score":
            if rep.primary_metric is None:
                return False, "no comparable score recorded"
            return rep.accepted_by_score_rule, "score within 10%" if rep.accepted_by_score_rule else "score differs"
        return rep.verdict == REPRODUCED, rep.verdict

    def _feedback(self, run: RunResult, rep: Report) -> str:
        if self.mode == "score":
            p = rep.primary_metric or {}
            return (f"The experiment now runs, but its main score {p.get('candidate_name')} = {p.get('candidate', float('nan')):.6g} "
                    f"is not close to the originally reported {p.get('reference_name')} = {p.get('reference', float('nan')):.6g}. "
                    f"Fix the code so it reproduces the original result.")
        return ("The experiment now runs, but it is NOT the original experiment yet. A runtime check compared it with "
                "the original and found:\n" + rep.feedback() + "\nChange the code so these match the original.")

    def _prompt(self, code: str, run: RunResult, history: list[str], extra: str) -> list[dict]:
        parts = [f"Current file experiment.py (line numbers are only for reference):\n```python\n{_numbered(code)}\n```"]
        if not run.ok:
            d = diagnose(run.stderr, self.workdir, self.workdir / self.script_name, run.timed_out)
            parts.append(f"Running it with today's libraries fails:\n```\n{run.tail(25)}\n```")
            parts.append("Diagnosis: " + d.summary())
            if self.mode == "guarded+kb":
                hints = knowledge_text(d)
                if hints:
                    parts.append("Relevant library changes (from a curated knowledge base):\n" + hints)
        elif extra:
            parts.append(extra)
            if self.mode == "guarded+kb":
                hints = format_hints(retrieve(code, extra))
                if hints:
                    parts.append("Relevant library changes (from a curated knowledge base):\n" + hints)
        if history:
            parts.append("Previous attempts:\n" + "\n".join(f"- {h}" for h in history[-4:]))
        parts.append("Reply with SEARCH/REPLACE edit blocks only.")
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]

    # ------------------------------------------------------------------ main loop
    def run(self) -> AgentResult:
        t0 = time.time()
        work = copy_project(self.project_dir, self.workdir)
        script = work / self.script_name
        original = script.read_text(encoding="utf-8")
        steps: list[Step] = []
        history: list[str] = []
        calls = ptok = ctok = 0

        run = run_script(script, timeout=self.timeout)
        rep = verify(self.reference, run)
        steps.append(Step(0, "run", run.tail(2) if not run.ok else "runs", run.ok, rep.verdict))
        accepted, why = self._accept(run, rep)
        self.log(f"  [{self.mode}] initial: {'runs' if run.ok else 'fails'}; verdict={rep.verdict}; accepted={accepted}")
        stop = "accepted_without_changes" if accepted else "max_iterations"
        extra = ""
        if run.ok and not accepted:
            extra = self._feedback(run, rep)

        it = 0
        while not accepted and it < self.max_iters:
            it += 1
            code = script.read_text(encoding="utf-8")
            reply = self.llm.chat(self._prompt(code, run, history, extra))
            calls += 1
            ptok += reply.prompt_tokens
            ctok += reply.completion_tokens
            steps.append(Step(it, "llm", reply.text[:2000], prompt_tokens=reply.prompt_tokens,
                              completion_tokens=reply.completion_tokens, seconds=reply.seconds))
            edits, full = patching.parse(reply.text)
            try:
                if edits:
                    new_code = patching.apply(code, edits)
                elif full:
                    new_code = full
                else:
                    raise patching.PatchError("no SEARCH/REPLACE block found in the answer")
            except patching.PatchError as e:
                history.append(f"attempt {it}: your edit could not be applied ({str(e).splitlines()[0]}). "
                               f"Copy the SEARCH lines exactly from the current file.")
                steps.append(Step(it, "patch_failed", str(e)[:500]))
                self.log(f"  [{self.mode}] iter {it}: patch could not be applied")
                continue
            if new_code == code:
                history.append(f"attempt {it}: your edit did not change the file.")
                steps.append(Step(it, "no_change"))
                continue
            script.write_text(new_code, encoding="utf-8")
            run = run_script(script, timeout=self.timeout)
            rep = verify(self.reference, run)
            accepted, why = self._accept(run, rep)
            steps.append(Step(it, "run", run.tail(2) if not run.ok else "runs", run.ok, rep.verdict))
            self.log(f"  [{self.mode}] iter {it}: {'runs' if run.ok else 'fails'}; verdict={rep.verdict}; accepted={accepted}")
            if accepted:
                stop = "accepted"
                break
            if not run.ok:
                d = diagnose(run.stderr, work, script, run.timed_out)
                history.append(f"attempt {it}: applied, but now fails with {d.exception_type} at line {d.line}")
                extra = ""
            else:
                history.append(f"attempt {it}: runs, but not accepted ({why})")
                extra = self._feedback(run, rep)

        final_code = script.read_text(encoding="utf-8")
        return AgentResult(self.mode, accepted, stop, it, calls, run.ok, rep.verdict, rep.to_dict(),
                           patching.unified_diff(original, final_code), [asdict(s) for s in steps],
                           time.time() - t0, ptok, ctok)


def repair(project_dir, reference_trace, llm, mode="guarded+kb", **kw) -> AgentResult:
    return RepairAgent(project_dir, reference_trace, llm, mode, **kw).run()
