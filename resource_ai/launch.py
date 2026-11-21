"""Run an experiment script with the methodology recorder installed.

    RESOURCE_TRACE=trace.json python -m resource_ai.launch experiment.py [args...]

Tracebacks are printed as if the script had been run directly (launcher frames removed),
so the diagnosis and the LLM see the same error a researcher would see.
"""
import os
import runpy
import sys
import traceback


def _clean_traceback(exc: BaseException) -> str:
    te = traceback.TracebackException.from_exception(exc)
    here = os.path.dirname(os.path.abspath(__file__))
    frames = [f for f in te.stack
              if not f.filename.startswith(here) and not f.filename.endswith(("runpy.py", "<frozen runpy>"))]
    te.stack = traceback.StackSummary.from_list(frames)
    return "".join(te.format())


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m resource_ai.launch script.py [args...]", file=sys.stderr)
        return 2
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    from resource_ai import instrument
    instrument.install()
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    except BaseException as e:  # noqa: BLE001
        sys.stderr.write(_clean_traceback(e))
        sys.stderr.flush()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
