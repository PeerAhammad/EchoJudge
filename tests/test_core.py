import json

import numpy as np
import pytest
import soundfile as sf

from backend.audio.io import AudioError, load_audio_array, load_audio_file
from backend.config import load_config
from backend.features.extract import extract_features
from backend.features.stats import compute_profile
from backend.pipeline import analyze_arrays
from backend.scoring.rubric import linear_score
from synth import SR, baseline, synth_speech

CFG = load_config()


def cats(res, source="participant"):
    return [e["category"] for e in res["events"] if e["source"] == source]


def overlap_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------ audio io
def test_resample_and_mono(tmp_path):
    y, _ = baseline()
    p = tmp_path / "a.wav"
    sf.write(p, np.stack([y, y], axis=1), 22050)  # stereo @22.05k (pitch shifts, irrelevant here)
    a = load_audio_file(p, CFG)
    assert a.sr == 16000 and a.y.ndim == 1 and a.original_channels == 2


def test_rejects_bad_inputs(tmp_path):
    with pytest.raises(AudioError):
        load_audio_array(np.zeros(SR * 5, dtype=np.float32), SR, CFG)  # silent
    with pytest.raises(AudioError):
        load_audio_array(np.random.randn(SR).astype(np.float32) * 0.1, SR, CFG)  # too short
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(AudioError):
        load_audio_file(bad, CFG)
    with pytest.raises(AudioError):
        load_audio_file(tmp_path / "x.txt", CFG)


def test_clipping_detected():
    y, _ = baseline()
    y = np.clip(y * 40, -1, 1)
    a = load_audio_array(y, SR, CFG)
    assert a.clipped_fraction > 0.001 and a.clipped_intervals


# ------------------------------------------------------------------ features
def test_pause_timestamps_and_rate():
    y, truth = baseline()
    fs = extract_features(load_audio_array(y, SR, CFG), CFG)
    assert len(fs.pauses) == len(truth["pauses"])
    for (a, b), (ta, tb) in zip(fs.pauses, truth["pauses"]):
        assert abs(a - ta) < 0.1 and abs(b - tb) < 0.1
    prof = compute_profile(fs, CFG)
    assert abs(prof["articulation_rate_syl_s"] - 4.5) / 4.5 < 0.15
    assert abs(prof["f0_median_hz"] - 120) / 120 < 0.05


# ------------------------------------------------------------------ detection
def test_clean_speaker_swap_has_no_flaws():
    """Different voice, loudness and noise but same delivery: no delivery events."""
    ref, _ = baseline(seed=1)
    part, _ = baseline(seed=2, base_f0=210.0, gain_db=-32.0, noise_db=-60.0)
    res = analyze_arrays(ref, part, SR)
    assert [c for c in cats(res) if c != "signal_quality"] == []
    assert res["scores"]["overall"] > 85


def test_rushed_segment_localised():
    ref, _ = baseline(n_phrases=4, syl=18, rate=4.5, seed=1)
    part, truth = synth_speech([(18, 4.5), (36, 7.2), (36, 7.2), (18, 4.5)], [0.5, 0.5, 0.5], seed=3)
    res = analyze_arrays(ref, part, SR)
    rushed = [e for e in res["events"] if e["category"] == "rushed_pacing"]
    assert rushed, res["events"]
    fast = (truth["phrases"][1][0], truth["phrases"][2][1])
    assert max(overlap_iou((e["start_s"], e["end_s"]), fast) for e in rushed) > 0.6
    assert rushed[0]["participant_value"] > rushed[0]["threshold"]
    assert "syl/s" in rushed[0]["explanation"]


def test_slow_pacing_detected():
    ref, _ = baseline(seed=1)
    part, _ = baseline(n_phrases=4, syl=18, rate=2.8, seed=4)
    res = analyze_arrays(ref, part, SR)
    assert "slow_pacing" in cats(res)
    assert "rushed_pacing" not in cats(res)


def test_excessive_pauses_localised():
    ref, _ = baseline(seed=1)
    part, truth = synth_speech([(6, 4.5)] * 8, [1.4, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], seed=5)
    res = analyze_arrays(ref, part, SR)
    ev = [e for e in res["events"] if e["category"] == "excessive_pauses"]
    assert ev
    t0 = truth["pauses"][0][0]
    assert any(e["start_s"] - 0.3 <= t0 <= e["end_s"] for e in ev)


def test_monotone_detected_but_not_for_low_voice():
    ref, _ = baseline(seed=1, pitch_st=3.0)
    flat, _ = baseline(seed=6, pitch_st=0.3)
    low, _ = baseline(seed=7, pitch_st=3.0, base_f0=85.0)
    assert "monotone_delivery" in cats(analyze_arrays(ref, flat, SR))
    assert "monotone_delivery" not in cats(analyze_arrays(ref, low, SR))


def test_clipped_participant_reports_quality_not_delivery():
    ref, _ = baseline(seed=1)
    part, _ = baseline(seed=2)
    res = analyze_arrays(ref, np.clip(part * 40, -1, 1).astype(np.float32), SR)
    q = [e for e in res["events"] if e["category"] == "signal_quality"]
    assert q and all(not e["scored"] for e in q)
    assert res["scores"]["provisional"]


# ------------------------------------------------------------------ scoring
def test_linear_score_directions():
    assert linear_score(0.0, 0.05, 0.3) == 100
    assert linear_score(0.3, 0.05, 0.3) == 0
    assert linear_score(0.9, 0.85, 0.3) == 100 and linear_score(0.3, 0.85, 0.3) == 0
    assert 0 < linear_score(0.175, 0.05, 0.3) < 100


def test_flawed_scores_lower_and_deterministic():
    ref, _ = baseline(seed=1)
    good, _ = baseline(seed=2, base_f0=200.0)
    bad, _ = synth_speech([(18, 4.5), (36, 7.5), (36, 7.5), (18, 4.5)], [1.6, 0.5, 1.0], pitch_st=0.3, seed=3)
    r_good, r_bad = analyze_arrays(ref, good, SR), analyze_arrays(ref, bad, SR)
    assert r_bad["scores"]["overall"] < r_good["scores"]["overall"] - 10
    assert json.dumps(analyze_arrays(ref, bad, SR)) == json.dumps(r_bad)  # repeatable


def test_missing_pitch_is_not_scored_as_perfect():
    ref, _ = baseline(seed=1)
    noise = (np.random.default_rng(0).normal(0, 0.05, len(ref))).astype(np.float32)  # no voiced speech
    res = analyze_arrays(ref, noise, SR)
    sc = res["scores"]
    assert sc["provisional"]
    vv = sc["categories"]["vocal_variation"]["score"]
    assert vv is None or vv < 100


def test_result_is_json_serialisable():
    ref, _ = baseline(seed=1)
    json.dumps(analyze_arrays(ref, ref, SR))


def test_self_comparison_yields_no_events_even_with_variable_delivery():
    """Invariant: a recording compared with itself must never be flagged."""
    y, _ = synth_speech([(10, 3.0), (30, 6.5), (12, 3.5), (25, 5.5), (8, 2.8)], [0.4, 1.0, 0.3, 0.7],
                        pitch_st=2.0, seed=9)
    res = analyze_arrays(y, y, SR)
    assert [e for e in res["events"] if e["category"] != "signal_quality"] == []
