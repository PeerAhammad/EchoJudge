"""End-to-end analysis: audio -> features -> (ASR + alignment) -> detection -> scoring -> JSON dict."""
from __future__ import annotations

import numpy as np

from backend.alignment.dtw import dtw_map
from backend.alignment.text_diff import diff_tokens, op_stats, pairs_from_ops
from backend.alignment.units import units_from_dtw, units_from_words
from backend.audio.io import LoadedAudio, load_audio_array, load_audio_file
from backend.config import load_config
from backend.detection.aligned import detect_emphasis, detect_lexical, detect_unit_pacing
from backend.detection.rules import run_all_detectors
from backend.features.extract import extract_features, series_for_dashboard
from backend.features.prominence import word_prominence
from backend.features.stats import compute_profile, rate_series
from backend.scoring.rubric import score_analysis
from backend.transcription.text import tokenize

SCHEMA_VERSION = "0.2"

LIMITATIONS = [
    "Word times come from Whisper cross-attention (estimates, ~0.1-0.3 s), not a forced aligner; DTW is the fallback.",
    "Syllable rate is an energy-peak estimate, not a phonetic count.",
    "A deviation from the reference is not proof of a flaw; the reference is one valid delivery among many.",
    "Emphasis detection is a heuristic prominence proxy and is expected to produce false positives.",
    "Pitch/pace norms vary with language, accent, disability and expressive style; the rubric measures selected "
    "delivery characteristics only, not overall speaking ability.",
]


def _ops_report(ops, rw, pw, limit=600):
    out = []
    for op, i, j in ops[:limit]:
        out.append({"op": op, "ref": rw[i].text if i is not None else None, "part": pw[j].text if j is not None else None,
                    "ref_time_s": [round(rw[i].start, 2), round(rw[i].end, 2)] if i is not None else None,
                    "part_time_s": [round(pw[j].start, 2), round(pw[j].end, 2)] if j is not None else None})
    return out


def analyze_audio(ref: LoadedAudio, part: LoadedAudio, cfg: dict | None = None, asr=None,
                  ref_text: str | None = None) -> dict:
    cfg = cfg or load_config()
    warnings: list[str] = []
    ref_fs, part_fs = extract_features(ref, cfg), extract_features(part, cfg)
    ref_prof, part_prof = compute_profile(ref_fs, cfg), compute_profile(part_fs, cfg)
    for fs, nm in ((ref_fs, "reference"), (part_fs, "participant")):
        warnings += [f"{nm}: {n}" for n in fs.notes]
    if not ref_prof.get("local_rate_median"):
        warnings.append("Reference has too little measurable speech for a pacing baseline.")

    # ---- transcription -------------------------------------------------------------
    ref_tr = part_tr = None
    if asr is not None:
        try:
            ref_tr = asr.transcribe(ref.y, ref.sr, ref_text)
            part_tr = asr.transcribe(part.y, part.sr, ref_text)
            for t, nm in ((ref_tr, "reference"), (part_tr, "participant")):
                warnings += [f"{nm} ASR: {n}" for n in t.notes]
        except Exception as e:  # never leak paths/tracebacks
            warnings.append(f"ASR failed ({type(e).__name__}); using signal-only analysis instead.")
            ref_tr = part_tr = None
    else:
        warnings.append("ASR not run: pacing uses syllable rate; emphasis and transcript checks skipped (DTW mapping shown for inspection).")

    # ---- alignment -----------------------------------------------------------------
    dmap = None
    try:
        dmap = dtw_map(ref_fs, part_fs, cfg)
    except Exception as e:
        warnings.append(f"DTW alignment unavailable ({e}).")
    method, units, pairs = "none", [], []
    text_alignment = None
    quality: dict = {"word_alignment": None}
    rw = ref_tr.words if ref_tr else []
    pw = part_tr.words if part_tr else []
    word_ok = False
    if rw and pw:
        ops = diff_tokens([w.norm for w in rw], [w.norm for w in pw])
        st = op_stats(ops, len(rw))
        pairs = pairs_from_ops(ops)
        probs = [w.prob for w in rw + pw if w.prob is not None]
        mean_prob = float(np.mean(probs)) if probs else None
        a = cfg["alignment"]
        word_ok = st["match_rate"] >= a["min_match_rate"] and (mean_prob is None or mean_prob >= a["min_mean_word_prob"])
        xcheck = None
        if dmap is not None and pairs:
            xcheck = float(np.median([abs(dmap.map(rw[i].start) - pw[j].start) for i, j in pairs]))
        quality = {"word_alignment": "good" if word_ok else "poor", "match_rate": st["match_rate"],
                   "mean_word_prob": None if mean_prob is None else round(mean_prob, 3),
                   "dtw_vs_word_median_start_diff_s": None if xcheck is None else round(xcheck, 3)}
        if st["match_rate"] < a["unreliable_match_rate"]:
            warnings.append("Participant transcript differs strongly from the reference text: a fair comparison may be "
                            "impossible (different text, or recognition errors).")
        elif not word_ok:
            warnings.append("Word alignment quality is poor; using signal-only pacing (syllable rate); emphasis skipped.")
        if xcheck is not None and xcheck > a["dtw"]["max_median_start_diff_s"]:
            warnings.append(f"DTW and ASR word timings disagree (median {xcheck:.2f}s); timestamps may be unreliable.")
        text_alignment = {**st, "ops": _ops_report(ops, rw, pw)}
        if word_ok:
            units, method = units_from_words(pairs, rw, pw, cfg, ref_fs, part_fs), "asr_words+text_diff"
    if not units and dmap is not None:
        units, method = units_from_dtw(ref_fs, dmap, cfg, part_fs), "dtw_fallback"
    if not units:
        method = "none"

    expected_text = None
    if ref_text and part_tr is not None:
        exp = tokenize(ref_text)
        ops = diff_tokens(exp, [w.norm for w in pw])
        expected_text = {"participant_vs_expected": op_stats(ops, len(exp))}
        if ref_tr is not None:
            expected_text["reference_asr_vs_expected"] = op_stats(diff_tokens(exp, [w.norm for w in rw]), len(exp))

    # ---- detection -----------------------------------------------------------------
    # Duration-ratio pacing is only valid when both recordings are known to contain the same words, so only
    # word-aligned units may replace the (content-independent) windowed syllable-rate rule. DTW units are
    # reported for inspection/cross-checks but never flag pacing on their own.
    word_units = units if method == "asr_words+text_diff" else []
    events = run_all_detectors(ref, ref_fs, part, part_fs, ref_prof, cfg, include_pacing=not word_units)
    events += detect_unit_pacing(word_units, cfg)
    prominence = None
    emphasis_ran = False
    if word_ok and part_fs.pitch_available and ref_fs.pitch_available:
        rp, pp = word_prominence(ref_fs, rw, cfg), word_prominence(part_fs, pw, cfg)
        prominence = {"reference": rp, "participant": pp}
        if len(rw) >= cfg["detection_stage2"]["emphasis"]["min_words"] and len(pw) >= cfg["detection_stage2"]["emphasis"]["min_words"]:
            events += detect_emphasis(pairs, rw, pw, rp, pp, cfg)
            emphasis_ran = True
    if part_tr is not None and ref_tr is not None:
        events += detect_lexical(rw, pw, cfg)
    events = sorted(events, key=lambda e: (e.start_s, e.category))
    scores = score_analysis(ref_prof, part_prof, events, cfg, emphasis_available=emphasis_ran)

    return {
        "schema_version": SCHEMA_VERSION, "config_version": cfg["version"], "stage": "stage2-alignment",
        "status": "ok", "alignment_method": method, "alignment_quality": quality,
        "warnings": warnings, "limitations": LIMITATIONS,
        "transcripts": None if not (ref_tr and part_tr) else {
            "reference": {"words": [w.to_dict() for w in rw], "language": ref_tr.language, "model": ref_tr.model, "device": ref_tr.device},
            "participant": {"words": [w.to_dict() for w in pw], "language": part_tr.language, "model": part_tr.model, "device": part_tr.device}},
        "text_alignment": text_alignment, "expected_text": expected_text, "prominence": prominence,
        "units": [{"ref": [round(u.ref_start, 2), round(u.ref_end, 2)], "part": [round(u.part_start, 2), round(u.part_end, 2)],
                   "speed_ratio": round(u.speed_ratio, 3), "ref_pause_s": round(u.ref_pause, 2), "part_pause_s": round(u.part_pause, 2), "label": u.label, "source": u.source} for u in units],
        "reference": {"profile": ref_prof, "series": series_for_dashboard(ref_fs, cfg), "rate_series": rate_series(ref_fs, cfg),
                      "pauses": [[round(a, 2), round(b, 2)] for a, b in ref_fs.pauses],
                      "segments": [[round(a, 2), round(b, 2)] for a, b in ref_fs.segments]},
        "participant": {"profile": part_prof, "series": series_for_dashboard(part_fs, cfg), "rate_series": rate_series(part_fs, cfg),
                        "pauses": [[round(a, 2), round(b, 2)] for a, b in part_fs.pauses],
                        "segments": [[round(a, 2), round(b, 2)] for a, b in part_fs.segments]},
        "events": [e.model_dump() for e in events], "scores": scores,
    }


def analyze_files(ref_path: str, part_path: str, cfg: dict | None = None, asr=None, ref_text: str | None = None) -> dict:
    cfg = cfg or load_config()
    ref = load_audio_file(ref_path, cfg, label="reference")
    part = load_audio_file(part_path, cfg, label="participant")
    return analyze_audio(ref, part, cfg, asr=asr, ref_text=ref_text)


def analyze_arrays(ref_y: np.ndarray, part_y: np.ndarray, sr: int, cfg: dict | None = None, asr=None,
                   ref_text: str | None = None) -> dict:
    cfg = cfg or load_config()
    return analyze_audio(load_audio_array(ref_y, sr, cfg, "reference"), load_audio_array(part_y, sr, cfg, "participant"),
                         cfg, asr=asr, ref_text=ref_text)
