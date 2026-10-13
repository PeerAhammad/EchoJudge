"""Window helpers and per-recording profile statistics (speaker-normalised).

Definitions (also in docs/):
  local articulation rate (syl/s) over [t0,t1) =
        #syllable nuclei in [t0,t1)  /  speech seconds in [t0,t1)       (pauses excluded)
  pause fraction = pause seconds / phrase span (first speech start .. last speech end)
  pitch spread (st) = robust sigma of F0 in semitones re. median = (p84 - p16) / 2
  energy range (dB) = p90 - p10 of frame RMS dB over speech frames
"""
from __future__ import annotations

import numpy as np

from backend.features.extract import FeatureSet


def make_windows(duration_s: float, length_s: float, hop_s: float) -> list[tuple[float, float]]:
    if duration_s <= length_s:
        return [(0.0, duration_s)]
    wins, t = [], 0.0
    while t + length_s <= duration_s + 1e-9:
        wins.append((t, t + length_s))
        t += hop_s
    if wins[-1][1] < duration_s - 1e-6:          # final partial hop -> window aligned to the end
        wins.append((duration_s - length_s, duration_s))
    return wins


def window_vote_spans(wins: list[tuple[float, float]], hop_s: float) -> list[tuple[float, float]]:
    """Each window 'votes' for its central hop; first/last windows also cover the edges.

    This keeps event boundaries within ~one hop of the truth instead of +/- half a window.
    """
    spans = []
    for i, (a, b) in enumerate(wins):
        c = 0.5 * (a + b)
        lo, hi = c - hop_s / 2, c + hop_s / 2
        if i == 0:
            lo = a
        if i == len(wins) - 1:
            hi = b
        spans.append((lo, hi))
    return spans


def merge_intervals(iv: list[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
    iv = sorted(iv)
    out: list[list[float]] = []
    for a, b in iv:
        if out and a <= out[-1][1] + gap_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def speech_seconds(fs: FeatureSet, t0: float, t1: float) -> float:
    i0, i1 = int(round(t0 / fs.hop_s)), int(round(t1 / fs.hop_s))
    return float(fs.speech_mask[i0:i1].sum() * fs.hop_s)


def local_rate(fs: FeatureSet, t0: float, t1: float, min_speech_s: float) -> float:
    sp = speech_seconds(fs, t0, t1)
    if sp < min_speech_s:
        return float("nan")
    n = int(((fs.nuclei_times >= t0) & (fs.nuclei_times < t1)).sum())
    return n / sp


def trim_to_speech(fs: FeatureSet, a: float, b: float) -> tuple[float, float] | None:
    """Clip [a,b] to the speech segments it overlaps (events should start/end on speech)."""
    ov = [(max(a, s), min(b, e)) for s, e in fs.segments if e > a and s < b]
    ov = [(s, e) for s, e in ov if e > s]
    if not ov:
        return None
    return ov[0][0], ov[-1][1]


def phrase_span(fs: FeatureSet) -> tuple[float, float]:
    if not fs.segments:
        return 0.0, fs.duration_s
    return fs.segments[0][0], fs.segments[-1][1]


def pause_time_in(fs: FeatureSet, t0: float, t1: float) -> tuple[float, int]:
    tot, cnt = 0.0, 0
    for a, b in fs.pauses:
        o = min(b, t1) - max(a, t0)
        if o > 0:
            tot += o
            cnt += 1
    return tot, cnt


def pitch_spread_window(fs: FeatureSet, t0: float, t1: float, min_voiced_s: float) -> float:
    if not fs.pitch_available:
        return float("nan")
    i0, i1 = int(round(t0 / fs.hop_s)), int(round(t1 / fs.hop_s))
    seg = fs.f0_st[i0:i1]
    seg = seg[np.isfinite(seg)]
    if len(seg) * fs.hop_s < min_voiced_s:
        return float("nan")
    p16, p84 = np.percentile(seg, [16, 84])
    return float((p84 - p16) / 2.0)


def energy_range_db(fs: FeatureSet) -> float | None:
    e = fs.rms_db[fs.speech_mask]
    if e.size < 50:
        return None
    p10, p90 = np.percentile(e, [10, 90])
    return float(p90 - p10)


def compute_profile(fs: FeatureSet, cfg: dict) -> dict:
    """Recording-level summary used both for reporting and as the reference baseline."""
    w = cfg["windows"]
    p0, p1 = phrase_span(fs)
    span = max(p1 - p0, 1e-6)
    speech_s = float(fs.speech_mask.sum() * fs.hop_s)
    n_nuc = len(fs.nuclei_times)
    wins = make_windows(fs.duration_s, w["pacing"]["length_s"], w["pacing"]["hop_s"])
    lr = np.array([local_rate(fs, a, b, w["pacing"]["min_speech_s"]) for a, b in wins])
    lr = lr[np.isfinite(lr)]
    pause_durs = [b - a for a, b in fs.pauses]
    pwins = make_windows(fs.duration_s, w["pitch"]["length_s"], w["pitch"]["hop_s"])
    ps = np.array([pitch_spread_window(fs, a, b, w["pitch"]["min_voiced_s"]) for a, b in pwins])
    ps = ps[np.isfinite(ps)]
    fwin = make_windows(fs.duration_s, w["fragmentation"]["length_s"], w["fragmentation"]["hop_s"])
    fcfg = cfg["detection"]["fragmentation"]
    frag_counts = [fragment_count(fs, a, b, fcfg) for a, b in fwin]
    qwins = make_windows(fs.duration_s, w["pause"]["length_s"], w["pause"]["hop_s"])
    pfr = [pause_time_in(fs, a, b)[0] / (b - a) for a, b in qwins]

    pitch_spread_all = None
    if fs.pitch_available:
        s = fs.f0_st[np.isfinite(fs.f0_st)]
        p16, p84, p10, p90 = np.percentile(s, [16, 84, 10, 90])
        pitch_spread_all = float((p84 - p16) / 2.0)
        pitch_range = float(p90 - p10)
    else:
        pitch_range = None

    return {
        "duration_s": round(fs.duration_s, 3),
        "phrase_span_s": [round(p0, 3), round(p1, 3)],
        "speech_s": round(speech_s, 3),
        "syllable_nuclei": int(n_nuc),
        "articulation_rate_syl_s": round(n_nuc / speech_s, 3) if speech_s > 0 else None,
        "speech_rate_syl_s": round(n_nuc / span, 3),
        "local_rate_median": float(np.median(lr)) if lr.size else None,
        "local_rate_p10": float(np.percentile(lr, 10)) if lr.size else None,
        "local_rate_p90": float(np.percentile(lr, 90)) if lr.size else None,
        "local_rates": [round(float(x), 3) for x in lr],
        "pitch_window_spreads": [round(float(x), 3) for x in ps],
        "pause_window_fractions": [round(float(x), 4) for x in pfr],
        "local_rate_min": float(np.min(lr)) if lr.size else None,
        "local_rate_max": float(np.max(lr)) if lr.size else None,
        "pitch_window_spread_min": float(np.min(ps)) if ps.size else None,
        "pause_window_fraction_max": float(max(pfr)) if pfr else 0.0,
        "pause_count": len(fs.pauses),
        "pause_total_s": round(float(sum(pause_durs)), 3),
        "pause_fraction": round(float(sum(pause_durs)) / span, 4),
        "pause_mean_s": round(float(np.mean(pause_durs)), 3) if pause_durs else 0.0,
        "pause_max_s": round(float(max(pause_durs)), 3) if pause_durs else 0.0,
        "f0_median_hz": round(fs.f0_median_hz, 1) if fs.f0_median_hz else None,
        "pitch_spread_st": pitch_spread_all,
        "pitch_range_st": pitch_range,
        "pitch_window_spread_median": float(np.median(ps)) if ps.size else None,
        "energy_range_db": energy_range_db(fs),
        "frag_window_max": int(max(frag_counts)) if frag_counts else 0,
        "snr_db": None if fs.snr_db is None else round(fs.snr_db, 1),
        "mfcc_mean": [round(float(x), 2) for x in fs.mfcc[:, fs.speech_mask].mean(axis=1)] if fs.speech_mask.any() else None,
        "mfcc_std": [round(float(x), 2) for x in fs.mfcc[:, fs.speech_mask].std(axis=1)] if fs.speech_mask.any() else None,
        "centroid_mean_hz": round(float(fs.centroid_hz[fs.speech_mask].mean()), 0) if fs.speech_mask.any() else None,
        "pitch_available": fs.pitch_available,
    }


def fragment_count(fs: FeatureSet, t0: float, t1: float, fcfg: dict) -> int:
    """Short speech bursts (< max_run_s) isolated by gaps >= min_gap_s, fully inside the window."""
    runs = fs.micro_runs
    n = 0
    for i, (a, b) in enumerate(runs):
        if a < t0 or b > t1 or (b - a) >= fcfg["max_run_s"]:
            continue
        prev_gap = a - runs[i - 1][1] if i > 0 else np.inf
        next_gap = runs[i + 1][0] - b if i + 1 < len(runs) else np.inf
        if prev_gap >= fcfg["min_gap_s"] and next_gap >= fcfg["min_gap_s"]:
            n += 1
    return n


def rate_series(fs: FeatureSet, cfg: dict) -> list[list]:
    """[[window_centre_s, local articulation rate or None], ...] for the dashboard speaking-rate timeline."""
    w = cfg["windows"]["pacing"]
    out = []
    for a, b in make_windows(fs.duration_s, w["length_s"], w["hop_s"]):
        r = local_rate(fs, a, b, w["min_speech_s"])
        out.append([round(0.5 * (a + b), 2), None if not np.isfinite(r) else round(float(r), 3)])
    return out
