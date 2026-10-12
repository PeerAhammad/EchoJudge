"""Word-level text alignment (match / substitution / deletion / insertion)."""
from __future__ import annotations

import difflib


def diff_tokens(a: list[str], b: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Ops between token lists a (reference) and b (participant), in order."""
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops: list[tuple[str, int | None, int | None]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ops += [("match", i1 + k, j1 + k) for k in range(i2 - i1)]
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            ops += [("sub", i1 + k, j1 + k) for k in range(n)]
            ops += [("del", i1 + k, None) for k in range(n, i2 - i1)]
            ops += [("ins", None, j1 + k) for k in range(n, j2 - j1)]
        elif tag == "delete":
            ops += [("del", i + 0, None) for i in range(i1, i2)]
        elif tag == "insert":
            ops += [("ins", None, j) for j in range(j1, j2)]
    return ops


def op_stats(ops, n_ref: int) -> dict:
    c = {k: sum(1 for o in ops if o[0] == k) for k in ("match", "sub", "del", "ins")}
    n = max(n_ref, 1)
    return {"n_ref_words": n_ref, "matches": c["match"], "substitutions": c["sub"],
            "missing": c["del"], "additions": c["ins"],
            "match_rate": round(c["match"] / n, 4),
            "wer": round((c["sub"] + c["del"] + c["ins"]) / n, 4)}


def pairs_from_ops(ops) -> list[tuple[int, int]]:
    return [(i, j) for op, i, j in ops if op == "match"]
