"""Operational rubric scoring (0-100, higher = better delivery).

This is the system's OPERATIONAL rubric, not a validated measure of speaking ability.

Category score = w_g * global_component + w_e * event_component (renormalised if one is missing)
  global_component : piecewise-linear map of a recording-level measurement vs the reference
                     (100 at `good`, 0 at `bad`, linear in between).
  event_component  : 100 * (1 - mean over the analysed span of the max severity penalty active
                     at each instant). Using a per-instant max means overlapping events in one
                     category are never double counted.
Overall = sum(w_c * score_c) / sum(w_c) over categories that could be computed.
Missing categories are never treated as 100: they are excluded and the result is marked provisional.
"""
from __future__ import annotations

import math

import numpy as np

from backend.schemas import FlawEvent

CATEGORY_EVENTS = {
    "pacing": ("rushed_pacing", "slow_pacing"),
    "pauses_fluency": ("excessive_pauses", "fluency_interruption"),
    "vocal_variation": ("monotone_delivery",),
    "emphasis_energy": ("weak_emphasis",),
}
RES = 0.01  # s, timeline resolution for the overlap-free penalty


def linear_score(x: float, good: float, bad: float) -> float:
    """100 at x==good, 0 at x==bad, linear between, clamped. Works for either direction."""
    if good == bad:
        return 100.0 if x == good else 0.0
    t = (x - good) / (bad - good)
    return float(100.0 * (1.0 - min(1.0, max(0.0, t))))


def _event_component(events: list[FlawEvent], span: tuple[float, float], sev_pen: dict) -> tuple[float, float]:
    a0, a1 = span
    n = max(1, int(round((a1 - a0) / RES)))
    line = np.zeros(n)
    for e in events:
        i0 = max(0, int(round((e.start_s - a0) / RES)))
        i1 = min(n, int(round((e.end_s - a0) / RES)))
        if i1 > i0:
            line[i0:i1] = np.maximum(line[i0:i1], sev_pen[str(e.severity)] if str(e.severity) in sev_pen else sev_pen[e.severity])
    pen = float(line.mean())
    return 100.0 * (1.0 - min(1.0, pen)), pen


def _combine(comps: list[dict], wg: float, we: float) -> float | None:
    avail = [c for c in comps if c["score"] is not None]
    if not avail:
        return None
    tw = sum(wg if c["name"] == "global" else we for c in avail)
    return float(sum((wg if c["name"] == "global" else we) * c["score"] for c in avail) / tw)


def score_analysis(ref_prof: dict, part_prof: dict, events: list[FlawEvent], cfg: dict,
                   emphasis_available: bool = False) -> dict:
    sc = cfg["scoring"]
    wg, we = sc["component_weights"]["global"], sc["component_weights"]["events"]
    sev_pen = sc["severity_penalty"]
    p0, p1 = part_prof["phrase_span_s"]
    span = (p0, max(p1, p0 + 0.01))
    scored = [e for e in events if e.scored and e.source == "participant"]
    cats: dict[str, dict] = {}
    reasons: list[str] = []

    def ev_comp(cat):
        evs = [e for e in scored if e.category in CATEGORY_EVENTS[cat]]
        s, pen = _event_component(evs, span, sev_pen)
        return {"name": "events", "score": s, "penalty": round(pen, 4), "n_events": len(evs),
                "detail": "severity-weighted share of the analysed span covered by flagged events (no double counting)"}

    # --- pacing
    pc = sc["pacing"]
    ra, rb = ref_prof.get("articulation_rate_syl_s"), part_prof.get("articulation_rate_syl_s")
    if ra and rb:
        ratio = rb / ra
        g = {"name": "global", "measure": "articulation_rate_ratio", "value": round(ratio, 3),
             "score": linear_score(abs(math.log(ratio)), math.log(pc["ratio_tolerance"]), math.log(pc["ratio_zero"])),
             "detail": f"participant/reference articulation rate; full marks within x{pc['ratio_tolerance']}, zero at x{pc['ratio_zero']} (or its inverse)"}
    else:
        g = {"name": "global", "score": None, "detail": "articulation rate unavailable"}
    cats["pacing"] = {"components": [g, ev_comp("pacing")]}

    # --- pauses & fluency
    pp = sc["pauses"]
    fa, fb = ref_prof.get("pause_fraction"), part_prof.get("pause_fraction")
    if fa is not None and fb is not None:
        d = fb - fa
        g = {"name": "global", "measure": "pause_fraction_minus_reference", "value": round(d, 4),
             "score": linear_score(d, pp["fraction_good"], pp["fraction_bad"]),
             "detail": f"extra pause share vs reference; full marks up to +{pp['fraction_good']}, zero at +{pp['fraction_bad']}"}
    else:
        g = {"name": "global", "score": None, "detail": "pause fraction unavailable"}
    cats["pauses_fluency"] = {"components": [g, ev_comp("pauses_fluency")]}

    # --- vocal variation
    vc = sc["variation"]
    rs, ps = ref_prof.get("pitch_spread_st"), part_prof.get("pitch_spread_st")
    if ps is None:
        g = {"name": "global", "score": None, "detail": "pitch unavailable for participant"}
        ev = {"name": "events", "score": None, "detail": "pitch unavailable"}
        reasons.append("Pitch could not be measured reliably; vocal-variation score omitted.")
    else:
        if rs is not None and rs >= vc["min_ref_spread_st"]:
            ratio = ps / rs
            g = {"name": "global", "measure": "pitch_spread_ratio", "value": round(ratio, 3),
                 "score": linear_score(ratio, vc["ratio_good"], vc["ratio_bad"]),
                 "detail": f"participant/reference pitch spread (semitones); full marks at >= {vc['ratio_good']}, zero at <= {vc['ratio_bad']}"}
        else:
            g = {"name": "global", "measure": "pitch_spread_st_absolute", "value": round(ps, 2),
                 "score": linear_score(ps, vc["abs_good_st"], vc["abs_bad_st"]),
                 "detail": "reference pitch spread unavailable/too small; absolute semitone spread used"}
            reasons.append("Reference has little pitch variation; vocal-variation score uses an absolute scale.")
        ev = ev_comp("vocal_variation")
    cats["vocal_variation"] = {"components": [g, ev]}

    # --- emphasis & energy (energy dynamics only until word alignment exists)
    ec = sc["energy"]
    ea, eb = ref_prof.get("energy_range_db"), part_prof.get("energy_range_db")
    if ea and eb:
        ratio = eb / ea
        g = {"name": "global", "measure": "energy_range_ratio", "value": round(ratio, 3),
             "score": linear_score(ratio, ec["ratio_good"], ec["ratio_bad"]),
             "detail": f"participant/reference loudness range (p90-p10 dB); full marks at >= {ec['ratio_good']}, zero at <= {ec['ratio_bad']}"}
    else:
        g = {"name": "global", "score": None, "detail": "energy range unavailable"}
    if emphasis_available:
        cats["emphasis_energy"] = {"components": [g, ev_comp("emphasis_energy")]}
        emphasis_partial = False
    else:
        cats["emphasis_energy"] = {"components": [g]}
        reasons.append("Emphasis component not scored (no usable word alignment): category reflects energy dynamics only.")
        emphasis_partial = True

    out_cats = {}
    for name, c in cats.items():
        s = _combine(c["components"], wg, we)
        out_cats[name] = {
            "score": None if s is None else round(s, 1),
            "weight": sc["weights"][name],
            "components": c["components"],
            "provisional": (name == "emphasis_energy" and emphasis_partial) or s is None,
        }
        for comp in c["components"]:
            if isinstance(comp.get("score"), float):
                comp["score"] = round(comp["score"], 1)

    avail = {k: v for k, v in out_cats.items() if v["score"] is not None}
    missing = [k for k in out_cats if k not in avail]
    tw = sum(v["weight"] for v in avail.values())
    overall = round(sum(v["weight"] * v["score"] for v in avail.values()) / tw, 1) if tw > 0 else None
    q_events = [e for e in events if e.category == "signal_quality"]
    if q_events:
        reasons.append("Signal-quality issues were detected; measurements may be less reliable.")
    for m in missing:
        reasons.append(f"Category '{m}' could not be computed and was excluded from the overall score.")
    return {
        "overall": overall,
        "scale": "0-100, higher = better delivery (operational rubric, not a validated measure)",
        "categories": out_cats,
        "weights_used": {k: v["weight"] for k, v in avail.items()},
        "provisional": bool(missing or q_events or emphasis_partial),
        "provisional_reasons": reasons,
    }
