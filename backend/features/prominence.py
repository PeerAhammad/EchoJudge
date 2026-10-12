"""Word-level prominence (relative emphasis proxy).

For each word: peak energy (dB), peak F0 (semitones re. the speaker's median) and duration per
estimated syllable (log). Each is converted to a robust z-score WITHIN that recording
(median/MAD over its words), so speaker pitch range, loudness and baseline speed cancel out.
prominence z = weighted mean of the available component z-scores.
This is a documented heuristic proxy, not a validated measure of linguistic stress.
"""
from __future__ import annotations

import re

import numpy as np

from backend.features.extract import FeatureSet
from backend.transcription.types import Word


def _rz(x) -> np.ndarray:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.full(x.shape, np.nan)
    med = np.median(x[ok])
    mad = 1.4826 * np.median(np.abs(x[ok] - med))
    if mad < 1e-6:
        mad = float(np.std(x[ok])) + 1e-6
    return np.clip((x - med) / mad, -4, 4)


def est_syllables(norm: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", norm)))


def word_prominence(fs: FeatureSet, words: list[Word], cfg: dict) -> list[dict]:
    wt = cfg["detection_stage2"]["emphasis"]["weights"]
    E, F, D = [], [], []
    for w in words:
        i0 = max(0, int(round(w.start / fs.hop_s)))
        i1 = max(i0 + 1, int(round(w.end / fs.hop_s)))
        e = fs.rms_db[i0:i1]
        E.append(float(e.max()) if e.size else np.nan)
        f = fs.f0_st[i0:i1]
        f = f[np.isfinite(f)]
        F.append(float(f.max()) if (f.size and fs.pitch_available) else np.nan)
        D.append(float(np.log(max(w.end - w.start, 0.02) / est_syllables(w.norm))))
    zE, zF, zD = _rz(E), _rz(F), _rz(D)
    rows = []
    for k, w in enumerate(words):
        comps = [(wt["energy"], zE[k]), (wt["pitch"], zF[k]), (wt["duration"], zD[k])]
        use = [(a, b) for a, b in comps if np.isfinite(b)]
        z = float(sum(a * b for a, b in use) / sum(a for a, _ in use)) if use else float("nan")
        rows.append({"word": w.text, "start_s": round(w.start, 3), "end_s": round(w.end, 3), "z": z,
                     "z_energy": float(zE[k]), "z_pitch": float(zF[k]), "z_duration": float(zD[k]),
                     "peak_db": E[k], "peak_f0_st": F[k]})
    return rows
