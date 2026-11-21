"""Parse and apply LLM edits.

Preferred format (works well with small local models):

    experiment.py
    <<<<<<< SEARCH
    exact old lines
    =======
    new lines
    >>>>>>> REPLACE

Fallback: a single ```python block containing the complete new file.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

BLOCK = re.compile(r"<{5,9} ?SEARCH[^\n]*\n(?P<search>.*?)\n?={5,9}[^\n]*\n(?P<replace>.*?)\n?>{5,9} ?REPLACE", re.S)
CODE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.S)


@dataclass
class Edit:
    search: str
    replace: str


class PatchError(Exception):
    pass


def _strip_line_numbers(text: str) -> str:
    # models sometimes copy the " 12| " prefixes we show them
    if all(re.match(r"^\s*\d+\| ?", ln) for ln in text.splitlines() if ln.strip()):
        return "\n".join(re.sub(r"^\s*\d+\| ?", "", ln) for ln in text.splitlines())
    return text


def parse(reply: str) -> tuple[list[Edit], str | None]:
    """Return (edits, full_file). Exactly one of them is used."""
    edits = [Edit(_strip_line_numbers(m.group("search")), _strip_line_numbers(m.group("replace")))
             for m in BLOCK.finditer(reply or "")]
    if edits:
        return edits, None
    blocks = CODE.findall(reply or "")
    if len(blocks) == 1 and ("import " in blocks[0]) and len(blocks[0].splitlines()) >= 5:
        return [], _strip_line_numbers(blocks[0])
    return [], None


def _find_loose(src: str, search: str) -> tuple[int, int] | None:
    """Match ignoring trailing whitespace and a uniform indentation difference."""
    s_lines = [ln.rstrip() for ln in search.strip("\n").splitlines()]
    if not s_lines:
        return None
    lines = src.splitlines(keepends=True)
    stripped = [ln.rstrip() for ln in lines]
    n = len(s_lines)
    for i in range(len(lines) - n + 1):
        window = stripped[i:i + n]
        if [w.strip() for w in window] == [s.strip() for s in s_lines]:
            start = sum(len(x) for x in lines[:i])
            end = start + sum(len(x) for x in lines[i:i + n])
            return start, end
    return None


def apply(src: str, edits: list[Edit]) -> str:
    out = src
    for k, e in enumerate(edits, 1):
        if not e.search.strip():
            raise PatchError(f"edit {k}: empty SEARCH block")
        if e.search in out:
            out = out.replace(e.search, e.replace, 1)
            continue
        span = _find_loose(out, e.search)
        if span is None:
            raise PatchError(f"edit {k}: SEARCH text not found in the file:\n{e.search[:300]}")
        start, end = span
        # keep the file's own indentation for the replaced region
        orig_first = out[start:end].splitlines()[0]
        indent = orig_first[: len(orig_first) - len(orig_first.lstrip())]
        rep_lines = e.replace.strip("\n").splitlines()
        base = min((len(l) - len(l.lstrip()) for l in rep_lines if l.strip()), default=0)
        rep = "\n".join(indent + l[base:] if l.strip() else "" for l in rep_lines) + "\n"
        out = out[:start] + rep + out[end:]
    return out


def unified_diff(a: str, b: str, name: str = "experiment.py") -> str:
    return "".join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                        fromfile=f"a/{name}", tofile=f"b/{name}"))


def to_blocks(old: str, new: str, name: str = "experiment.py") -> str:
    """Express old->new as SEARCH/REPLACE blocks (used to script 'perfect' LLM answers in tests)."""
    a, b = old.splitlines(), new.splitlines()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            continue
        lo, hi = max(i1 - 1, 0), min(i2 + 1, len(a))     # one line of context so SEARCH is never empty
        new_lines = a[lo:i1] + b[j1:j2] + a[i2:hi]
        out.append(f"{name}\n<<<<<<< SEARCH\n" + "\n".join(a[lo:hi]) + "\n=======\n" + "\n".join(new_lines) + "\n>>>>>>> REPLACE")
    return "\n\n".join(out)
