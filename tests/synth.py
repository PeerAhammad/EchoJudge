"""Synthetic speech-like audio with KNOWN ground truth, for unit tests only.

This is a harmonic source with syllable-rate amplitude bursts, a controllable F0
contour (semitone std), phrase structure and silent gaps. It is NOT speech and must
never be presented as dataset evidence; it exists to verify that timestamps and
measurements are computed correctly.
"""
from __future__ import annotations

import numpy as np

SR = 16000


def synth_speech(phrases, gaps, base_f0=120.0, pitch_st=3.0, gain_db=-20.0, noise_db=-55.0,
                 energy_var_db=2.0, lead=0.6, trail=0.6, seed=0, sr=SR, boost_db=None, f0_boost_st=None):
    """phrases: list of (n_syllables, rate_syl_per_s); gaps: pause seconds between phrases.

    Returns (signal float32, truth dict with phrase intervals, pauses, syllable centres).
    """
    assert len(gaps) == len(phrases) - 1
    rng = np.random.default_rng(seed)
    t_cursor = lead
    syl, phrase_iv, pauses = [], [], []
    for pi, (n, rate) in enumerate(phrases):
        P = 1.0 / rate
        start = t_cursor
        for k in range(n):
            syl.append((t_cursor, 0.72 * P))
            t_cursor += P
        phrase_iv.append((start, t_cursor - 0.28 * P))
        if pi < len(gaps):
            pauses.append((t_cursor - 0.28 * P, t_cursor + gaps[pi]))
            t_cursor += gaps[pi]
    total = t_cursor + trail
    N = int(total * sr)
    t = np.arange(N) / sr
    env = np.zeros(N)
    env_list = []
    for ts, dur in syl:
        i0, i1 = int(ts * sr), int((ts + dur) * sr)
        amp = 10 ** (rng.normal(0, energy_var_db) / 20.0)
        amp *= 10 ** ((boost_db or {}).get(len(env_list), 0.0) / 20.0)
        env_list.append(1)
        env[i0:i1] = np.maximum(env[i0:i1], amp * np.hanning(i1 - i0))
    A = pitch_st / 0.79 if pitch_st > 0 else 0.0
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    st = A * (np.sin(2 * np.pi * 0.4 * t + ph1) + 0.5 * np.sin(2 * np.pi * 1.3 * t + ph2))
    for idx, b in (f0_boost_st or {}).items():
        ts, dur = syl[idx]
        i0, i1 = int(ts * sr), int((ts + dur) * sr)
        st[i0:i1] += b * np.hanning(i1 - i0)
    f0 = base_f0 * 2 ** (st / 12.0)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    sig = np.zeros(N)
    for k in range(1, 9):
        sig += np.sin(k * phase) / k
    sig *= env
    sig *= 10 ** (gain_db / 20.0) / max(np.max(np.abs(sig)), 1e-9)
    sig += rng.normal(0, 10 ** (noise_db / 20.0), N)
    truth = {"phrases": phrase_iv, "pauses": pauses, "duration": total,
             "syllable_centres": [ts + dur / 2 for ts, dur in syl], "syllables": syl}
    return sig.astype(np.float32), truth


def baseline(n_phrases=4, syl=18, rate=4.5, gap=0.5, **kw):
    return synth_speech([(syl, rate)] * n_phrases, [gap] * (n_phrases - 1), **kw)


def fake_words(truth, per=3):
    """Word timings (known ground truth) = groups of `per` syllables; unique tokens w0, w1, ..."""
    from backend.transcription.types import Word
    syl, words = truth["syllables"], []
    for k in range(0, len(syl) - per + 1, per):
        a, b = syl[k][0], syl[k + per - 1][0] + syl[k + per - 1][1]
        words.append(Word(f"w{k // per}", f"w{k // per}", a, b, 0.95))
    return words


class FakeASR:
    """Returns pre-set words for known signals, matched by content hash (tests only)."""
    def __init__(self, pairs):
        self.map = {hash(np.asarray(y, np.float32).tobytes()): w for y, w in pairs}

    def transcribe(self, y, sr, prompt=None):
        from backend.transcription.types import Transcript
        return Transcript(list(self.map[hash(np.asarray(y, np.float32).tobytes())]), "en", "fake", "fake", "cpu", [])
