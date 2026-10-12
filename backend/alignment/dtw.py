"""Constrained DTW between reference and participant (speaker-normalised MFCCs).

Features: MFCC 1..12 (c0 dropped = level), per-recording mean/variance normalisation over speech
frames (removes channel/speaker offsets), decimated to 25 Hz, cosine distance, band-constrained.
Used (a) as the fallback alignment when word alignment is unavailable/poor, (b) as a cross-check.
Limitation: spectral alignment cannot separate repeated identical sounds; boundaries are ~40-100 ms.
"""
from __future__ import annotations

import librosa
import numpy as np

from backend.features.extract import FeatureSet


class DTWMap:
    def __init__(self, ref_t: np.ndarray, part_t: np.ndarray):
        self.ref_t, self.part_t = ref_t, part_t

    def map(self, t: float) -> float:
        return float(np.interp(t, self.ref_t, self.part_t))


def _feat(fs: FeatureSet, decim: int) -> np.ndarray:
    m = fs.mfcc[1:13, :]
    sp = fs.speech_mask
    sel = m[:, sp] if sp.any() else m
    z = (m - sel.mean(axis=1, keepdims=True)) / (sel.std(axis=1, keepdims=True) + 1e-6)
    return z[:, ::decim]


def dtw_map(ref_fs: FeatureSet, part_fs: FeatureSet, cfg: dict) -> DTWMap:
    d = cfg["alignment"]["dtw"]
    decim = d["frame_decimation"]
    X, Y = _feat(ref_fs, decim), _feat(part_fs, decim)
    last_err = None
    for band in (d["band_rad"], min(0.6, d["band_rad"] * 2)):
        try:
            _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="cosine", global_constraints=True, band_rad=band)
            wp = wp[::-1]
            step = ref_fs.hop_s * decim
            ri = np.unique(wp[:, 0])
            part_t = np.array([wp[wp[:, 0] == i, 1].mean() for i in ri]) * step
            return DTWMap(ri * step, np.maximum.accumulate(part_t))
        except Exception as e:  # infeasible band etc.
            last_err = e
    raise RuntimeError(f"DTW failed: {type(last_err).__name__}")
