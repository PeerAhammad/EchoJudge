"""Detectors that need alignment (Stage 2): unit pacing, weak emphasis, lexical disfluency."""
from __future__ import annotations

import numpy as np

from backend.alignment.units import Unit
from backend.detection.rules import severity_from_bands
from backend.schemas import FlawEvent, fmt_time
from backend.transcription.types import Word


def _fin(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 2)


def detect_unit_pacing(units: list[Unit], cfg: dict) -> list[FlawEvent]:
    """Pacing from aligned units: speed ratio = reference speech time / participant speech time for the SAME words
    (silent pauses excluded on both sides, so pause flaws are reported only under excessive_pauses). Rushed if >= rushed_speed_ratio, slow if <= slow_speed_ratio.
    Consecutive flagged units of the same kind are merged. Self-comparison gives ratio 1 -> no events.
    Limitations: durations include short internal gaps; word timestamps are ASR estimates (~0.1-0.3 s);
    the DTW fallback is less precise than word alignment."""
    c = cfg["detection_stage2"]["aligned_pacing"]
    if not c["enabled"] or not units:
        return []
    units = sorted(units, key=lambda u: u.part_start)
    kinds = []
    for u in units:
        r = u.speed_ratio
        kinds.append("rushed" if r >= c["rushed_speed_ratio"] else "slow" if r <= c["slow_speed_ratio"] else None)
    groups, cur = [], []
    for u, k in zip(units, kinds):
        if k and cur and cur[-1][1] == k and u.part_start - cur[-1][0].part_end <= c["merge_gap_s"]:
            cur.append((u, k))
        else:
            if cur:
                groups.append(cur)
            cur = [(u, k)] if k else []
    if cur:
        groups.append(cur)
    events = []
    for g in groups:
        us, kind = [u for u, _ in g], g[0][1]
        a, b = us[0].part_start, us[-1].part_end
        if b - a < c["min_event_s"]:
            continue
        ref_dur = sum(u.ref_speech for u in us)
        part_dur = sum(u.part_speech for u in us)
        ratio = ref_dur / part_dur
        rushed = kind == "rushed"
        sev = severity_from_bands(ratio, c["severity_bands_rushed" if rushed else "severity_bands_slow"], increasing=rushed)
        src = us[0].source
        words_txt = f"“{us[0].label}”" if len(us) == 1 else f"“{us[0].label.split(' ... ')[0]} ... {us[-1].label.split(' ... ')[-1]}”" if src == "words" else f"{len(us)} reference segments"
        events.append(FlawEvent(
            category="rushed_pacing" if rushed else "slow_pacing", start_s=round(a, 2), end_s=round(b, 2), severity=sev,
            detection_method="aligned_unit_duration_ratio" + ("(asr_words)" if src == "words" else "(dtw_fallback)"),
            participant_value=round(part_dur, 2), reference_value=round(ref_dur, 2),
            threshold=c["rushed_speed_ratio"] if rushed else c["slow_speed_ratio"], unit="s of speech for the same content",
            ratio=round(ratio, 3),
            evidence={"n_units": len(us), "alignment_source": src, "content": words_txt},
            explanation=(f"At {fmt_time(a)}-{fmt_time(b)}, the participant delivered {words_txt} in {part_dur:.1f}s of speech (pauses excluded); the "
                         f"reference took {ref_dur:.1f}s for the same content ({ratio:.2f}x "
                         f"{'faster' if rushed else 'slower'} than the reference). Threshold: "
                         f"{c['rushed_speed_ratio'] if rushed else c['slow_speed_ratio']:.2f}x."),
            recommendation=("Slow down slightly and add a deliberate pause before key phrases." if rushed else
                            "Increase the pace a little; shorten hesitations or stretched words here."),
            basis="rule on ASR word alignment (uncalibrated)" if src == "words" else "rule on DTW alignment (uncalibrated, lower precision)"))
    return events


def detect_emphasis(pairs, ref_words: list[Word], part_words: list[Word], ref_prom: list[dict],
                    part_prom: list[dict], cfg: dict) -> list[FlawEvent]:
    """Weak emphasis: a word that is clearly prominent in the reference (z >= ref_z_min) but much less so in the
    participant (drop >= drop_min and participant z <= part_z_max). Only exact-match word pairs are used.
    Limitations: reference emphasis is one valid reading, not a requirement; per-word prominence is noisy
    (expect false positives, quantified in evaluation); misplaced emphasis (participant stressing a word the
    reference does not) is not detected yet."""
    c = cfg["detection_stage2"]["emphasis"]
    if not c["enabled"] or len(ref_words) < c["min_words"] or len(part_words) < c["min_words"]:
        return []
    cands = []
    for ri, pi in pairs:
        zr, zp = ref_prom[ri]["z"], part_prom[pi]["z"]
        if not (np.isfinite(zr) and np.isfinite(zp)):
            continue
        drop = zr - zp
        if zr >= c["ref_z_min"] and drop >= c["drop_min"] and zp <= c["part_z_max"]:
            cands.append((drop, ri, pi))
    cands = sorted(cands, reverse=True)[: c["max_events"]]
    events = []
    for drop, ri, pi in sorted(cands, key=lambda x: part_words[x[2]].start):
        w = part_words[pi]
        rp, pp = ref_prom[ri], part_prom[pi]
        events.append(FlawEvent(
            category="weak_emphasis", start_s=round(w.start, 2), end_s=round(w.end, 2),
            severity=severity_from_bands(drop, c["severity_bands_drop"]),
            detection_method="word_prominence_drop(z)", participant_value=_fin(pp["z"]), reference_value=_fin(rp["z"]),
            threshold=c["drop_min"], unit="robust z (within-recording)", ratio=None,
            evidence={"word": w.text, "drop": _fin(drop),
                      "participant_components_z": {"energy": _fin(pp["z_energy"]), "pitch": _fin(pp["z_pitch"]), "duration": _fin(pp["z_duration"])},
                      "reference_components_z": {"energy": _fin(rp["z_energy"]), "pitch": _fin(rp["z_pitch"]), "duration": _fin(rp["z_duration"])}},
            explanation=(f"At {fmt_time(w.start)}-{fmt_time(w.end)}, the word “{w.text}” stood out in the reference "
                         f"(relative prominence {rp['z']:.1f}) but not in the participant ({pp['z']:.1f}); the drop of "
                         f"{drop:.1f} exceeded the {c['drop_min']:.1f} threshold. Participant energy/pitch/duration z: "
                         f"{_fin(pp['z_energy'])}/{_fin(pp['z_pitch'])}/{_fin(pp['z_duration'])} vs reference "
                         f"{_fin(rp['z_energy'])}/{_fin(rp['z_pitch'])}/{_fin(rp['z_duration'])}."),
            recommendation=f"Consider giving “{w.text}” more weight (louder, higher, or slightly longer) if it carries the key idea.",
            basis="heuristic prominence proxy (uncalibrated)", experimental=True))
    return events


def detect_lexical(ref_words: list[Word], part_words: list[Word], cfg: dict) -> list[FlawEvent]:
    """Filler words and immediate word repetitions that the ASR emitted for the participant (and the
    reference did not). Whisper often drops disfluencies, so absence here is NOT evidence of fluency."""
    c = cfg["detection_stage2"]["lexical"]
    if not c["enabled"]:
        return []
    fillers = set(c["fillers"])
    ref_f = sum(1 for w in ref_words if w.norm in fillers)
    ref_bigrams = {(a.norm, b.norm) for a, b in zip(ref_words[:-1], ref_words[1:]) if a.norm == b.norm}
    events = []
    seen_f = 0
    for w in part_words:
        if w.norm in fillers:
            seen_f += 1
            if seen_f > ref_f:
                events.append(_lex(w.start, w.end, "filler_word", f"the filler “{w.text}”", w.text))
    for a, b in zip(part_words[:-1], part_words[1:]):
        if a.norm == b.norm and (a.norm, b.norm) not in ref_bigrams and b.start - a.end <= c["max_repeat_gap_s"]:
            events.append(_lex(a.start, b.end, "word_repetition", f"the repeated word “{a.text}”", a.text))
    return sorted(events, key=lambda e: e.start_s)[: c["max_events"]]


def _lex(a, b, method, what, word):
    return FlawEvent(
        category="fluency_interruption", start_s=round(a, 2), end_s=round(b, 2), severity=1,
        detection_method=f"asr_text_rule:{method}", participant_value=1.0, reference_value=0.0, threshold=1.0,
        unit="count", evidence={"word": word},
        explanation=(f"At {fmt_time(a)}-{fmt_time(b)}, the transcript contains {what}, which does not appear "
                     f"at that point in the reference. Speech recognisers often omit such disfluencies, so this list is incomplete."),
        recommendation="Replace fillers with a short silent pause; rehearse the phrase until it flows in one breath.",
        basis="ASR text rule (uncalibrated; ASR may miss disfluencies)")
