"""General diagnosis of a failed run (replaces the single-rule Task 3 prototype).

Works for any Python traceback: finds the exception, the line in the experiment's own code,
classifies the failure, and retrieves relevant breaking-change knowledge.
Never executes or edits the experiment.
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .knowledge import format_hints, retrieve

FRAME = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>.+))?$')
EXC_LINE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning)|[A-Za-z_][\w.]*Error)(: (?P<msg>.*))?$")

CATEGORIES = [
    ("syntax", ("SyntaxError", "IndentationError", "TabError")),
    ("removed_module_or_import", ("ModuleNotFoundError", "ImportError")),
    ("missing_file", ("FileNotFoundError",)),
    ("invalid_parameter_value", ("InvalidParameterError",)),
    ("removed_attribute", ("AttributeError",)),
    ("wrong_arguments", ("TypeError",)),
    ("bad_value", ("ValueError",)),
    ("timeout", ("TimeoutError",)),
    ("missing_name", ("NameError",)),
    ("index_or_key", ("KeyError", "IndexError")),
]


@dataclass
class Diagnosis:
    failed: bool
    exception_type: str | None = None
    message: str | None = None
    category: str | None = None
    file: str | None = None
    line: int | None = None
    code_line: str | None = None
    frames_in_project: int = 0
    knowledge: list[dict] = field(default_factory=list)
    source_parses: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if not self.failed:
            return "The experiment ran without an error."
        loc = f"{Path(self.file).name}:{self.line}" if self.file else "unknown location"
        s = f"{self.exception_type} ({self.category}) at {loc}"
        if self.code_line:
            s += f"\n  code: {self.code_line}"
        if self.message:
            s += f"\n  message: {self.message[:300]}"
        return s


def _category(exc_type: str) -> str:
    short = exc_type.split(".")[-1]
    for cat, names in CATEGORIES:
        if short in names:
            return cat
    return "other"


def diagnose(stderr: str, project_dir: str | Path, script: str | Path | None = None,
             timed_out: bool = False) -> Diagnosis:
    lines = (stderr or "").rstrip().splitlines()
    if not lines and not timed_out:
        return Diagnosis(failed=False)
    project_dir = str(Path(project_dir).resolve())
    # exception = last line that looks like "SomeError: message" (message may continue below it)
    exc_type = msg = None
    for idx in range(len(lines) - 1, -1, -1):
        m = EXC_LINE.match(lines[idx])
        if m and not lines[idx].startswith(" "):
            exc_type = m.group("type")
            msg = "\n".join([m.group("msg") or ""] + lines[idx + 1:]).strip()
            break
    if timed_out and not exc_type:
        exc_type, msg = "TimeoutError", "experiment exceeded the time limit"
    d = Diagnosis(failed=True, exception_type=exc_type, message=msg,
                  category=_category(exc_type) if exc_type else "other")
    # deepest frame inside the project = where the researcher's code triggered the failure
    for i, ln in enumerate(lines):
        m = FRAME.match(ln)
        if m and str(Path(m.group("file")).resolve()).startswith(project_dir):
            d.frames_in_project += 1
            d.file, d.line = m.group("file"), int(m.group("line"))
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            d.code_line = None if FRAME.match(lines[i + 1] if i + 1 < len(lines) else "") else nxt
    code = ""
    if script and Path(script).exists():
        code = Path(script).read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(code)
            d.source_parses = True
        except SyntaxError as e:          # the old Task 3 agent crashed here
            d.source_parses = False
            d.file = d.file or str(script)
            d.line = d.line or e.lineno
            d.code_line = d.code_line or (e.text or "").strip()
        if d.file and d.line and not d.code_line:
            src_lines = code.splitlines()
            if 0 < d.line <= len(src_lines):
                d.code_line = src_lines[d.line - 1].strip()
    d.knowledge = [{"id": e.id, "title": e.title, "fix": e.fix, "warning": e.warning}
                   for e in retrieve(code, stderr or "")]
    return d


def knowledge_text(d: Diagnosis) -> str:
    from .knowledge import KB
    entries = [e for e in KB if e.id in {k["id"] for k in d.knowledge}]
    return format_hints(entries)
