import json

import numpy as np

from backend.alignment.dtw import dtw_map
from backend.alignment.text_diff import diff_tokens, op_stats, pairs_from_ops
from backend.alignment.units import Unit
from backend.audio.io import load_audio_array
from backend.config import load_config
from backend.detection.aligned import detect_lexical, detect_unit_pacing
from backend.features.extract import extract_features
from backend.pipeline import analyze_arrays
from backend.transcription.text import normalize_token, tokenize
from backend.transcription.types import Word
from synth import SR, FakeASR, fake_words, synth_speech

CFG = load_config()
SAME = [(18, 4.5)] * 4


def mk(phr, seed, **kw):
    y, t = synth_speech(phr, [0.5] * (len(phr) - 1), seed=seed, **kw)
    return y, t


def run(ref_phr, part_phr, rkw=None, pkw=None, seeds=(1, 2), use_asr=True):
    ref, tr = mk(ref_phr, seeds[0], **(rkw or {}))
    part, tp = mk(part_phr, seeds[1], **(pkw or {}))
    asr = FakeASR([(ref, fake_words(tr)), (part, fake_words(tp))]) if use_asr else None
    return analyze_arrays(ref, part, SR, asr=asr), tr, tp


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u else 0.0


# ---------------------------------------------------------------- text
def test_normalize_and_diff():
    assert normalize_token("Don’t,") == "don't" and tokenize("Hello, World!") == ["hello", "world"]
    ref = tokenize("the quick brown fox jumps over")
    part = tokenize("the quick fox jumped over again")
    ops = diff_tokens(ref, part)
    st = op_stats(ops, len(ref))
    assert st["missing"] == 1 and st["substitutions"] == 1 and st["additions"] == 1 and st["matches"] == 4
    pr = pairs_from_ops(ops)
    assert (0, 0) in pr and (3, 2) in pr and (4, 3) not in pr and (5, 4) in pr


# ---------------------------------------------------------------- unit pacing
def _units(ratios):
    out, tr, tp = [], 0.0, 0.0
    for r in ratios:
        out.append(Unit(tr, tr + 3.0, tp, tp + 3.0 / r, "a ... b", 5, "words"))
        tr += 3.5
        tp += 3.0 / r + 0.5
    return out


def test_unit_pacing_flags_and_merges():
    ev = detect_unit_pacing(_units([1.0, 1.7, 1.6, 1.0, 0.5, 1.0]), CFG)
    cats = [e.category for e in ev]
    assert cats == ["rushed_pacing", "slow_pacing"]
    assert ev[0].ratio > 1.5 and ev[1].ratio < 0.6
    assert detect_unit_pacing(_units([1.0, 1.05, 0.95, 1.1]), CFG) == []


# ---------------------------------------------------------------- DTW
def test_dtw_identity_monotone_and_endpoints():
    ref, tr = mk(SAME, 1)
    part, tp = mk([(18, 4.5), (18, 3.0), (18, 4.5), (18, 4.5)], 2)
    fr = extract_features(load_audio_array(ref, SR, CFG), CFG)
    fp = extract_features(load_audio_array(part, SR, CFG), CFG)
    ident = dtw_map(fr, fr, CFG)
    assert max(abs(ident.map(t) - t) for t in np.arange(0, 18, 0.5)) < 0.1
    m = dtw_map(fr, fp, CFG)
    ts = np.arange(0, 18, 0.25)
    assert np.all(np.diff([m.map(t) for t in ts]) >= -1e-9)
    # Endpoints anchor well; interior boundaries inside spectrally-homogeneous synthetic input can slip
    # by seconds (known DTW limitation, why DTW is display/cross-check only).
    assert abs(m.map(tr["phrases"][-1][1]) - tp["phrases"][-1][1]) < 1.0


# ---------------------------------------------------------------- pipeline with word alignment
def test_word_aligned_rushed_segment_exact_boundaries():
    res, _, tp = run(SAME, [(18, 4.5), (18, 7.2), (18, 7.2), (18, 4.5)])
    assert res["alignment_method"] == "asr_words+text_diff"
    ev = [e for e in res["events"] if e["category"] == "rushed_pacing"]
    assert len(ev) == 1 and "asr_words" in ev[0]["detection_method"]
    assert iou((ev[0]["start_s"], ev[0]["end_s"]), (tp["phrases"][1][0], tp["phrases"][2][1])) > 0.9
    json.dumps(res)


def test_self_comparison_with_alignment_has_no_events():
    y, t = mk([(12, 3.0), (30, 6.5), (12, 3.5), (25, 5.5)], 9, pitch_st=2.0)
    res = analyze_arrays(y, y, SR, asr=FakeASR([(y, fake_words(t))]))
    assert [e for e in res["events"] if e["category"] != "signal_quality"] == []


def test_weak_emphasis_detected_and_clean_pairs_quiet():
    res, _, tp = run(SAME, SAME, rkw=dict(boost_db={10: 12.0}, f0_boost_st={10: 6.0}), pkw=dict(base_f0=190.0))
    ev = [e for e in res["events"] if e["category"] == "weak_emphasis"]
    w3 = fake_words(tp)[3]
    assert any(abs(e["start_s"] - w3.start) < 0.05 for e in ev)
    assert res["scores"]["categories"]["emphasis_energy"]["provisional"] is False
    for s in range(1, 7):  # no deliberate emphasis difference -> no emphasis flags (synthetic only)
        r, _, _ = run(SAME, SAME, pkw=dict(base_f0=190.0), seeds=(s, s + 100))
        assert [e for e in r["events"] if e["category"] == "weak_emphasis"] == [], s


def test_different_text_triggers_warning_and_falls_back():
    ref, tr = mk(SAME, 1)
    part, tp = mk(SAME, 2)
    wp = [Word(f"x{i}", f"x{i}", w.start, w.end, 0.95) for i, w in enumerate(fake_words(tp))]
    res = analyze_arrays(ref, part, SR, asr=FakeASR([(ref, fake_words(tr)), (part, wp)]))
    assert any("differs strongly" in w for w in res["warnings"])
    assert res["alignment_method"] != "asr_words+text_diff"
    assert res["scores"]["categories"]["emphasis_energy"]["provisional"] is True


def test_asr_failure_is_reported_not_raised():
    class Boom:
        def transcribe(self, *a, **k):
            raise RuntimeError("model download failed at C:\\secret\\path")

    ref, _ = mk(SAME, 1)
    part, _ = mk(SAME, 2)
    res = analyze_arrays(ref, part, SR, asr=Boom())
    assert any("ASR failed" in w for w in res["warnings"])
    assert "secret" not in json.dumps(res)


def test_lexical_fillers_and_repeats():
    mkw = lambda toks: [Word(t, t, i * 0.4, i * 0.4 + 0.3, 0.9) for i, t in enumerate(toks)]
    ref = mkw("we hold these truths".split())
    part = mkw("we um hold hold these truths".split())
    ev = detect_lexical(ref, part, CFG)
    assert sorted(e.detection_method for e in ev) == ["asr_text_rule:filler_word", "asr_text_rule:word_repetition"]
    assert detect_lexical(ref, ref, CFG) == []


def test_inserted_words_do_not_inflate_unit_duration():
    from backend.alignment.units import units_from_words
    ref = [Word(f"w{i}", f"w{i}", i * 0.3, i * 0.3 + 0.28, 0.9) for i in range(10)]
    part = [Word("w0", "w0", 0.0, 0.28, 0.9), Word("x1", "x1", 1.0, 1.3, 0.9), Word("x2", "x2", 1.4, 1.7, 0.9)]
    part += [Word(f"w{i}", f"w{i}", 2.0 + (i - 1) * 0.3, 2.0 + (i - 1) * 0.3 + 0.28, 0.9) for i in range(1, 10)]
    ops = diff_tokens([w.norm for w in ref], [w.norm for w in part])
    units = units_from_words(pairs_from_ops(ops), ref, part, CFG)
    assert len(units) == 1 and abs(units[0].speed_ratio - 1.0) < 0.05   # w1..w9 only
    assert detect_unit_pacing(units, CFG) == []


def test_pauses_inside_units_do_not_cause_slow_pacing():
    """Same speaking rate, but long pauses inside each reference phrase: pauses flagged, pacing NOT."""
    ref, tr = synth_speech([(18, 4.5)] * 4, [0.5] * 3, seed=1)
    part, tp = synth_speech([(9, 4.5)] * 8, [2.0, 0.5] * 3 + [2.0], seed=2)
    asr = FakeASR([(ref, fake_words(tr)), (part, fake_words(tp))])
    res = analyze_arrays(ref, part, SR, asr=asr)
    assert res["alignment_method"] == "asr_words+text_diff"
    cats = [e["category"] for e in res["events"]]
    assert "slow_pacing" not in cats and "rushed_pacing" not in cats
    assert "excessive_pauses" in cats
    assert all(u["part_pause_s"] > 1.5 for u in res["units"])
