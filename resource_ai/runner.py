"""Run an experiment project in an isolated copy and collect its methodology trace."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_PARENT = str(Path(__file__).resolve().parents[1])


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    trace: dict | None
    seconds: float
    timed_out: bool = False
    workdir: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def tail(self, n: int = 30) -> str:
        lines = (self.stderr or "").strip().splitlines()
        return "\n".join(lines[-n:])


def run_script(script: str | Path, cwd: str | Path | None = None, timeout: int = 300,
               python: str | None = None) -> RunResult:
    """Run `script` under the recorder. `cwd` defaults to the script's folder."""
    script = Path(script).resolve()
    cwd = Path(cwd or script.parent).resolve()
    fd, trace_path = tempfile.mkstemp(prefix="trace_", suffix=".json")
    os.close(fd)
    os.unlink(trace_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = PACKAGE_PARENT + os.pathsep + env.get("PYTHONPATH", "")
    env["RESOURCE_TRACE"] = trace_path
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["MPLBACKEND"] = "Agg"
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run([python or sys.executable, "-m", "resource_ai.launch", str(script)],
                              cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out, rc = True, -9
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = ((e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        err += f"\nTimeoutError: experiment exceeded {timeout} seconds"
    trace = None
    if os.path.exists(trace_path):
        try:
            trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
        except Exception:
            trace = None
        os.unlink(trace_path)
    return RunResult(rc, out, err, trace, time.time() - t0, timed_out, str(cwd))


def copy_project(src: str | Path, dst: str | Path) -> Path:
    """Copy a project folder (never edit the original)."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    return dst
