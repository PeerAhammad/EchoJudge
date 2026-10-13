"""Evaluation metrics (pure functions, unit-tested).

Labels in the recorded dataset usually cover the whole speech of a recording (instructed flaw), so besides
event-level matching we use TIME-WEIGHTED (segment-level) precision/recall and FILE-LEVEL detection scores.
"""
from __future__ import annotations

import numpy as np

Interval = tuple[float, float]


def union(iv: list[Interval]) -> list[Interval]:
    out: list[list[float]] = []
    for a, b in sorted((a, b) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def length(iv: list[Interval]) -> float:
    return float(sum(b - a for a, b in union(iv)))


def intersect_len(a: list[Interval], b: list[Interval]) -> float:
    a, b = union(a), union(b)
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def prf(tp: float, fp: float, fn: float) -> dict:
    p = tp / (tp + fp) if (tp + fp) > 0 else None
    r = tp / (tp + fn) if (tp + fn) > 0 else None
    f = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else (0.0 if (p is not None or r is not None) and (tp == 0) else None)
    return {"precision": p, "recall": r, "f1": f}


def segment_level(files: list[dict], cat: str) -> dict:
    """files: [{'pred': {cat: [intervals]}, 'truth': {cat: [intervals]}}]; time-weighted P/R/F1 in seconds."""
    tp = fp = fn = 0.0
    for f in files:
        p, t = f["pred"].get(cat, []), f["truth"].get(cat, [])
        i = intersect_len(p, t)
        tp += i
        fp += length(p) - i
        fn += length(t) - i
    return {**prf(tp, fp, fn), "tp_s": round(tp, 2), "fp_s": round(fp, 2), "fn_s": round(fn, 2)}


def file_level(files: list[dict], cat: str, min_overlap_s: float = 0.0) -> dict:
    """A file is predicted positive if it has >=1 event of `cat`; truly positive if it has a label of `cat`."""
    tp = fp = fn = tn = 0
    for f in files:
        pos_t = length(f["truth"].get(cat, [])) > 0
        pos_p = length(f["pred"].get(cat, [])) > min_overlap_s
        tp += pos_t and pos_p
        fn += pos_t and not pos_p
        fp += (not pos_t) and pos_p
        tn += (not pos_t) and not pos_p
    return {**prf(tp, fp, fn), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def iou(a: Interval, b: Interval) -> float:
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0.0


def event_matches(pred: list[Interval], truth: list[Interval], min_iou: float = 0.1) -> dict:
    """Greedy one-to-one matching by IoU; returns TP/FP/FN, mean IoU, boundary errors (s)."""
    pred, truth = sorted(pred), sorted(truth)
    used, ious, se, ee = set(), [], [], []
    for t in truth:
        best, bj = 0.0, None
        for j, p in enumerate(pred):
            if j in used:
                continue
            v = iou(p, t)
            if v > best:
                best, bj = v, j
        if bj is not None and best >= min_iou:
            used.add(bj)
            ious.append(best)
            se.append(abs(pred[bj][0] - t[0]))
            ee.append(abs(pred[bj][1] - t[1]))
    tp = len(used)
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(truth) - tp, "mean_iou": float(np.mean(ious)) if ious else None,
            "mean_start_err_s": float(np.mean(se)) if se else None, "mean_end_err_s": float(np.mean(ee)) if ee else None}


def pairwise_ranking(groups: dict[str, dict]) -> dict:
    """groups: key -> {'clean': score, 'flawed': [scores]}; fraction of flawed scores below the clean score."""
    wins = total = 0
    for g in groups.values():
        if g.get("clean") is None:
            continue
        for s in g["flawed"]:
            total += 1
            wins += s < g["clean"]
    return {"flawed_below_clean": wins, "pairs": total, "fraction": (wins / total) if total else None}
