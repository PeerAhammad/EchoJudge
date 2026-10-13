"""Interpretable rule-based flaw detectors (Stage 1: no transcript/alignment needed).

Every rule documents: inputs, threshold (config key), intended use, limitations.
A deviation from the reference is NOT automatically a flaw: each rule combines a
reference-relative condition with a guard (reference's own natural variation, or an
absolute floor) so ordinary speaker differences do not trigger events.
"""
from __future__ import annotations

import numpy as np

from backend.audio.io import LoadedAudio
from backend.features.extract import FeatureSet
from backend.features.stats import (
    fragment_count, local_rate, make_windows, merge_intervals, pause_time_in,
    phrase_span, pitch_spread_window, trim_to_speech, window_vote_spans,
)
from backend.schemas import FlawEvent, fmt_time


def _ref_quantile(ref_prof: dict, key: str, q: float, fallback=None):
    """Quantile of the reference's own per-window values (q=1 -> its most extreme window)."""
    arr = ref_prof.get(key)
    if arr:
        return float(np.quantile(arr, min(max(q, 0.0), 1.0)))
    return fallback


def severity_from_bands(value: float, bands: list[float], increasing: bool = True) -> int:
    n = sum(1 for b in bands if (value >= b if increasing else value <= b))
    return int(min(4, max(1, n)))


# --------------------------------------------------------------------------- pacing
def detect_pacing(part: FeatureSet, ref_prof: dict, cfg: dict) -> list[FlawEvent]:
    """Rushed / slow pacing.

    Input: local articulation rate (syllable nuclei / speech second) in sliding windows.
    Rule: window flagged when rate > max(ratio*ref_median, ref_max_window_rate) (rushed) or
          rate < min(ratio*ref_median, ref_min_window_rate) (slow). The reference-extreme guard
          guarantees a recording compared with itself yields no events. Flagged windows vote for their
          central hop, are merged, trimmed to speech, and must last >= min_duration_s.
    Limitations: syllable counting is an energy-peak estimate (+/- ~10-15%); the reference
          is one valid delivery, not "the" correct pace; boundaries have ~1 hop (1 s)
          resolution. Without word alignment, comparison is global, not word-by-word.
    """
    d, w = cfg["detection"], cfg["windows"]["pacing"]
    med = ref_prof.get("local_rate_median")
    if not med:
        return []
    wins = make_windows(part.duration_s, w["length_s"], w["hop_s"])
    rates = [local_rate(part, a, b, w["min_speech_s"]) for a, b in wins]
    spans = window_vote_spans(wins, w["hop_s"])
    events: list[FlawEvent] = []
    for kind in ("rushed", "slow"):
        rc = d[kind]
        if not rc["enabled"]:
            continue
        if kind == "rushed":
            thr = rc["ratio"] * med
            if rc["ref_variation_guard"]:
                g = _ref_quantile(ref_prof, "local_rates", rc.get("ref_guard_quantile", 1.0), ref_prof.get("local_rate_max"))
                thr = max(thr, g) if g else thr
            flag = [np.isfinite(r) and r > thr for r in rates]
        else:
            thr = rc["ratio"] * med
            if rc["ref_variation_guard"]:
                g = _ref_quantile(ref_prof, "local_rates", 1.0 - rc.get("ref_guard_quantile", 1.0), ref_prof.get("local_rate_min"))
                thr = min(thr, g) if g else thr
            flag = [np.isfinite(r) and r < thr for r in rates]
        iv = merge_intervals([s for s, f in zip(spans, flag) if f], d["merge_gap_s"] + w["hop_s"])
        for a, b in iv:
            tr = trim_to_speech(part, a, b)
            if tr is None or tr[1] - tr[0] < rc["min_duration_s"]:
                continue
            a, b = tr
            r = local_rate(part, a, b, 0.5)
            if not np.isfinite(r):
                continue
            ratio = r / med
            rushed = kind == "rushed"
            sev = severity_from_bands(ratio, rc["severity_bands"], increasing=rushed)
            events.append(FlawEvent(
                category="rushed_pacing" if rushed else "slow_pacing",
                start_s=round(a, 2), end_s=round(b, 2), severity=sev,
                detection_method=f"windowed_articulation_rate(len={w['length_s']}s,hop={w['hop_s']}s)",
                participant_value=round(r, 3), reference_value=round(med, 3), threshold=round(thr, 3),
                unit="syllables/s", ratio=round(ratio, 3),
                evidence={"ref_min_window_rate": ref_prof.get("local_rate_min"), "ref_max_window_rate": ref_prof.get("local_rate_max"),
                          "nuclei_in_event": int(((part.nuclei_times >= a) & (part.nuclei_times < b)).sum())},
                explanation=(f"At {fmt_time(a)}-{fmt_time(b)}, the participant's estimated articulation rate was "
                             f"{r:.2f} syl/s, {ratio:.2f}x the reference's typical rate ({med:.2f} syl/s). "
                             f"The rate {'exceeded' if rushed else 'fell below'} the threshold of {thr:.2f} syl/s "
                             f"for this segment."),
                recommendation=("Slow down slightly and add a deliberate pause before key phrases."
                                if rushed else
                                "Increase the pace a little; shorten the stretched syllables or hesitations in this segment."),
            ))
    return events


# --------------------------------------------------------------------------- pauses
def detect_pauses(part: FeatureSet, ref_prof: dict, cfg: dict) -> list[FlawEvent]:
    """Excessive pauses: (a) single long pauses, (b) windows with a high pause fraction.

    (a) pause >= max(long_pause_s, factor * reference's longest pause).
    (b) window pause fraction > max(min_fraction, ref_fraction + margin) and >= min_count pauses.
    A long pause inside a (b) region is folded into it (no double reporting).
    Limitations: a pause is a silent gap >= 250 ms by energy; intentional dramatic pauses
        look identical to hesitation - the reference guard only partially mitigates this.
    """
    pc = cfg["detection"]["pauses"]
    if not pc["enabled"]:
        return []
    w = cfg["windows"]["pause"]
    ref_frac = ref_prof.get("pause_fraction", 0.0) or 0.0
    ref_max = ref_prof.get("pause_max_s", 0.0) or 0.0
    events: list[FlawEvent] = []

    # (b) frequent pauses
    fc = pc["frequent"]
    thr_frac = max(fc["min_fraction"], ref_frac + fc["fraction_margin"],
                   _ref_quantile(ref_prof, "pause_window_fractions", fc.get("ref_guard_quantile", 1.0),
                                 ref_prof.get("pause_window_fraction_max", 0.0)) or 0.0)
    wins = make_windows(part.duration_s, w["length_s"], w["hop_s"])
    spans = window_vote_spans(wins, w["hop_s"])
    flag = []
    for a, b in wins:
        tot, cnt = pause_time_in(part, a, b)
        flag.append((tot / (b - a)) > thr_frac and cnt >= fc["min_count"])
    iv = merge_intervals([s for s, f in zip(spans, flag) if f], cfg["detection"]["merge_gap_s"] + w["hop_s"])
    freq_iv = []
    for a, b in iv:
        tr = trim_to_speech(part, a, b)
        if tr is None:
            continue
        a, b = tr
        tot, cnt = pause_time_in(part, a, b)
        frac = tot / max(b - a, 1e-6)
        if frac <= thr_frac or cnt < fc["min_count"]:
            continue
        longest = max([(q1 - q0) for q0, q1 in part.pauses if min(q1, b) - max(q0, a) > 0], default=0.0)
        freq_iv.append((a, b))
        events.append(FlawEvent(
            category="excessive_pauses", start_s=round(a, 2), end_s=round(b, 2),
            severity=severity_from_bands(frac, fc["severity_bands_fraction"]),
            detection_method=f"pause_fraction_window(len={w['length_s']}s)",
            participant_value=round(frac, 3), reference_value=round(ref_frac, 3), threshold=round(thr_frac, 3),
            unit="pause time / segment time",
            ratio=round(frac / ref_frac, 2) if ref_frac > 0.01 else None,
            evidence={"pause_count": cnt, "pause_total_s": round(tot, 2), "longest_pause_s": round(longest, 2)},
            explanation=(f"At {fmt_time(a)}-{fmt_time(b)}, {frac * 100:.0f}% of the segment was silent pauses "
                         f"({cnt} pauses, {tot:.1f}s total, longest {longest:.1f}s) versus {ref_frac * 100:.0f}% "
                         f"across the reference. This exceeded the threshold of {thr_frac * 100:.0f}%."),
            recommendation="Link phrases more smoothly; reserve pauses for sentence boundaries or the key point.",
        ))
    # (a) single long pauses
    thr_long = max(pc["long_pause_s"], pc["ref_longest_pause_factor"] * ref_max)
    for a, b in part.pauses:
        dur = b - a
        if dur < thr_long:
            continue
        if any(fa <= a and b <= fb + 1e-6 for fa, fb in freq_iv):
            continue
        events.append(FlawEvent(
            category="excessive_pauses", start_s=round(a, 2), end_s=round(b, 2),
            severity=severity_from_bands(dur, pc["severity_bands_s"]),
            detection_method="single_pause_duration",
            participant_value=round(dur, 2), reference_value=round(ref_max, 2), threshold=round(thr_long, 2),
            unit="s", ratio=round(dur / ref_max, 2) if ref_max > 0.05 else None,
            evidence={"reference_longest_pause_s": round(ref_max, 2)},
            explanation=(f"At {fmt_time(a)}-{fmt_time(b)}, the participant paused silently for {dur:.1f}s. "
                         f"The longest pause in the reference was {ref_max:.1f}s; the threshold here is {thr_long:.1f}s."),
            recommendation="Shorten this gap unless it is a deliberate dramatic pause before a key point.",
        ))
    return sorted(events, key=lambda e: e.start_s)


# --------------------------------------------------------------------------- monotone
def detect_monotone(part: FeatureSet, ref_prof: dict, cfg: dict) -> list[FlawEvent]:
    """Monotone delivery from pitch spread (robust sigma of F0 in semitones, speaker-normalised).

    Flag windows where spread < max_spread_st (absolute) AND < ref_ratio * reference median
    window spread (so a naturally flat reference does not make everyone 'monotone', and
    low-pitched voices are not penalised: semitones are relative to each speaker's median).
    Limitations: pitch tracker errors (octave jumps, creak); reading style and language affect
    pitch variation; energy variation is reported as evidence but not part of the rule.
    """
    mc = cfg["detection"]["monotone"]
    if not mc["enabled"] or not part.pitch_available:
        return []
    ref_med = ref_prof.get("pitch_window_spread_median")
    if not ref_med:
        return []
    w = cfg["windows"]["pitch"]
    thr = min(mc["max_spread_st"], mc["ref_ratio"] * ref_med)
    g = _ref_quantile(ref_prof, "pitch_window_spreads", 1.0 - mc.get("ref_guard_quantile", 1.0), ref_prof.get("pitch_window_spread_min"))
    if g:
        thr = min(thr, g)
    wins = make_windows(part.duration_s, w["length_s"], w["hop_s"])
    spans = window_vote_spans(wins, w["hop_s"])
    spreads = [pitch_spread_window(part, a, b, w["min_voiced_s"]) for a, b in wins]
    flag = [np.isfinite(s) and s < thr for s in spreads]
    iv = merge_intervals([s for s, f in zip(spans, flag) if f], cfg["detection"]["merge_gap_s"] + w["hop_s"])
    events = []
    for a, b in iv:
        tr = trim_to_speech(part, a, b)
        if tr is None or tr[1] - tr[0] < mc["min_duration_s"]:
            continue
        a, b = tr
        sp = pitch_spread_window(part, a, b, 0.5)
        if not np.isfinite(sp):
            continue
        rel = sp / ref_med
        i0, i1 = int(round(a / part.hop_s)), int(round(b / part.hop_s))
        e = part.rms_db[i0:i1][part.speech_mask[i0:i1]]
        e_std = float(np.std(e)) if e.size > 10 else None
        events.append(FlawEvent(
            category="monotone_delivery", start_s=round(a, 2), end_s=round(b, 2),
            severity=severity_from_bands(rel, mc["severity_bands_ratio"], increasing=False),
            detection_method=f"pitch_spread_window(len={w['length_s']}s)",
            participant_value=round(sp, 2), reference_value=round(ref_med, 2), threshold=round(thr, 2),
            unit="semitones (robust sigma)", ratio=round(rel, 2),
            evidence={"energy_std_db": None if e_std is None else round(e_std, 2),
                      "participant_f0_median_hz": round(part.f0_median_hz, 1) if part.f0_median_hz else None},
            explanation=(f"At {fmt_time(a)}-{fmt_time(b)}, the participant's pitch varied by only {sp:.2f} semitones "
                         f"(robust spread), {rel * 100:.0f}% of the reference's typical {ref_med:.2f} semitones, and "
                         f"below the {thr:.2f} semitone threshold. Pitch is measured relative to each speaker's own "
                         f"median, so voice register does not affect this."),
            recommendation="Vary pitch on key words and at phrase ends; try marking the main idea of each sentence.",
        ))
    return events


# --------------------------------------------------------------------------- fragmentation
def detect_fragmentation(part: FeatureSet, ref_prof: dict, cfg: dict) -> list[FlawEvent]:
    """EXPERIMENTAL fluency-interruption heuristic (no ASR).

    Counts short isolated speech bursts (< max_run_s, gaps >= min_gap_s) per window; flags
    windows with >= min_count AND >= reference's busiest window + ref_margin_count.
    Cannot distinguish repetitions, filler words, or fast articulation of stop consonants.
    """
    fc = cfg["detection"]["fragmentation"]
    if not fc["enabled"]:
        return []
    w = cfg["windows"]["fragmentation"]
    thr = max(fc["min_count"], ref_prof.get("frag_window_max", 0) + fc["ref_margin_count"])
    wins = make_windows(part.duration_s, w["length_s"], w["hop_s"])
    spans = window_vote_spans(wins, w["hop_s"])
    counts = [fragment_count(part, a, b, fc) for a, b in wins]
    flag = [c >= thr for c in counts]
    iv = merge_intervals([s for s, f in zip(spans, flag) if f], cfg["detection"]["merge_gap_s"] + w["hop_s"])
    events = []
    for a, b in iv:
        tr = trim_to_speech(part, a, b)
        if tr is None:
            continue
        a, b = tr
        n = fragment_count(part, a, b, fc)
        if n < thr:
            continue
        events.append(FlawEvent(
            category="fluency_interruption", start_s=round(a, 2), end_s=round(b, 2),
            severity=severity_from_bands(n, fc["severity_bands_count"]),
            detection_method=f"short_isolated_bursts(len={w['length_s']}s)",
            participant_value=float(n), reference_value=float(ref_prof.get("frag_window_max", 0)),
            threshold=float(thr), unit="short bursts / window",
            evidence={"max_run_s": fc["max_run_s"]},
            explanation=(f"At {fmt_time(a)}-{fmt_time(b)}, the audio contained {n} very short, isolated speech bursts "
                         f"within a {w['length_s']:.0f}s window (reference maximum: {ref_prof.get('frag_window_max', 0)}). "
                         f"This pattern is consistent with fragmented delivery, but it was detected without a "
                         f"transcript, so repetitions and fillers cannot be told apart from fast consonant clusters."),
            recommendation="Listen to this segment; if hesitations or repetitions are present, rehearse the phrase as one breath group.",
            basis="experimental heuristic (uncalibrated, no ASR)", experimental=True,
        ))
    return events


# --------------------------------------------------------------------------- quality
def detect_quality(audio: LoadedAudio, fs: FeatureSet, cfg: dict, source: str) -> list[FlawEvent]:
    """Recording/signal-quality issues, reported separately and never scored as delivery flaws."""
    q = cfg["detection"]["quality"]
    events: list[FlawEvent] = []
    if audio.clipped_fraction >= q["clip_fraction"] and audio.clipped_intervals:
        iv = merge_intervals(audio.clipped_intervals, q["clip_merge_gap_s"])
        iv = sorted(iv, key=lambda x: x[0] - x[1])[: q["max_clip_events"]]
        for a, b in sorted(iv):
            events.append(FlawEvent(
                category="signal_quality", source=source, start_s=round(a, 2), end_s=round(max(b, a + 0.05), 2),
                severity=min(4, 1 + int(audio.clipped_fraction * 1000 // 5)),
                detection_method="clipped_sample_runs",
                participant_value=round(audio.clipped_fraction, 5), threshold=q["clip_fraction"], unit="fraction of samples",
                explanation=(f"{source.capitalize()} recording is clipped at {fmt_time(a)} "
                             f"({audio.clipped_fraction * 100:.2f}% of samples overall). Clipping distorts pitch and "
                             f"energy measurements in this region."),
                recommendation="Lower the microphone gain or move back from the microphone and re-record.",
                scored=False, basis="deterministic signal check"))
    if fs.snr_db is not None and fs.snr_db < q["min_snr_db"]:
        events.append(FlawEvent(
            category="signal_quality", source=source, start_s=0.0, end_s=round(audio.duration_s, 2),
            severity=2 if fs.snr_db >= 8 else 3, detection_method="speech_vs_noise_floor_db",
            participant_value=round(fs.snr_db, 1), threshold=q["min_snr_db"], unit="dB",
            explanation=(f"Estimated signal-to-noise ratio of the {source} recording is {fs.snr_db:.1f} dB "
                         f"(threshold {q['min_snr_db']:.0f} dB). Pitch and pause measurements may be unreliable."),
            recommendation="Record in a quieter room or closer to the microphone.",
            scored=False, basis="deterministic signal check"))
    usable = float(fs.speech_mask.sum() * fs.hop_s)
    if usable < q["min_usable_speech_s"]:
        events.append(FlawEvent(
            category="signal_quality", source=source, start_s=0.0, end_s=round(audio.duration_s, 2), severity=3,
            detection_method="usable_speech_seconds", participant_value=round(usable, 2),
            threshold=q["min_usable_speech_s"], unit="s",
            explanation=f"Only {usable:.1f}s of speech was detected in the {source} recording; results are unreliable.",
            recommendation="Provide a longer recording with clearly audible speech.",
            scored=False, basis="deterministic signal check"))
    return events


def run_all_detectors(ref_audio, ref_fs, part_audio, part_fs, ref_prof, cfg, include_pacing: bool = True) -> list[FlawEvent]:
    events: list[FlawEvent] = []
    if include_pacing:  # Stage-1 windowed pacing; replaced by aligned-unit pacing when alignment exists
        events += detect_pacing(part_fs, ref_prof, cfg)
    events += detect_pauses(part_fs, ref_prof, cfg)
    events += detect_monotone(part_fs, ref_prof, cfg)
    events += detect_fragmentation(part_fs, ref_prof, cfg)
    events += detect_quality(part_audio, part_fs, cfg, "participant")
    events += detect_quality(ref_audio, ref_fs, cfg, "reference")
    return sorted(events, key=lambda e: (e.start_s, e.category))
