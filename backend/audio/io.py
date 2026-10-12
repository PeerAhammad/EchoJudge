"""Audio loading + validation.

Output representation (documented in docs/): mono, float32, 16 kHz (configurable),
amplitude NOT peak-normalised (all later features are level-relative: dB re. the
recording's own 95th percentile), so loudness differences between recordings do not
affect comparisons.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import librosa
import numpy as np
import soundfile as sf

SUPPORTED_EXT = {".wav", ".mp3", ".flac", ".ogg"}


class AudioError(Exception):
    """User-facing audio problem. Message is safe to show to clients (no paths)."""


@dataclass
class LoadedAudio:
    y: np.ndarray                    # mono float32 at `sr`
    sr: int
    duration_s: float
    original_sr: int
    original_channels: int
    clipped_fraction: float          # fraction of original samples inside clipped runs
    clipped_intervals: list[tuple[float, float]] = field(default_factory=list)
    label: str = "audio"


def _clipped_runs(x: np.ndarray, level: float, min_run: int, sr: int):
    """Intervals (s) where >= min_run consecutive samples have |x| >= level."""
    hit = np.abs(x) >= level
    if not hit.any():
        return 0.0, []
    d = np.diff(np.concatenate([[0], hit.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (ends - starts) >= min_run
    starts, ends = starts[keep], ends[keep]
    frac = float((ends - starts).sum() / max(len(x), 1))
    return frac, [(float(s / sr), float(e / sr)) for s, e in zip(starts, ends)]


def load_audio_array(y: np.ndarray, sr: int, cfg: dict, label: str = "audio") -> LoadedAudio:
    """Validate and normalise an in-memory signal (shape (n,) or (n, ch))."""
    a = cfg["audio"]
    y = np.asarray(y, dtype=np.float32)
    channels = 1 if y.ndim == 1 else y.shape[1]
    if y.size == 0:
        raise AudioError(f"{label}: file contains no audio samples.")
    if not np.all(np.isfinite(y)):
        raise AudioError(f"{label}: audio contains invalid (NaN/inf) samples.")
    mono = y if y.ndim == 1 else y.mean(axis=1)
    if float(np.max(np.abs(mono))) < a["silence_peak_floor"]:
        raise AudioError(f"{label}: recording is silent (no usable signal).")
    clip_frac, clip_iv = _clipped_runs(mono, a["clip_level"], a["clip_min_run_samples"], sr)
    target = int(a["sample_rate"])
    if sr != target:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target, res_type="soxr_hq")
    mono = mono.astype(np.float32)
    dur = len(mono) / target
    if dur < a["min_duration_s"]:
        raise AudioError(f"{label}: too short ({dur:.1f}s); minimum is {a['min_duration_s']:.0f}s.")
    if dur > a["max_duration_s"]:
        raise AudioError(f"{label}: too long ({dur:.0f}s); maximum is {a['max_duration_s']:.0f}s.")
    return LoadedAudio(mono, target, dur, int(sr), int(channels), clip_frac, clip_iv, label)


def load_audio_file(path: str | os.PathLike, cfg: dict, label: str = "audio") -> LoadedAudio:
    """Decode a WAV/MP3/FLAC/OGG file. Raises AudioError with a safe message on failure."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext not in SUPPORTED_EXT:
        raise AudioError(f"{label}: unsupported file type '{ext}'. Use WAV or MP3.")
    try:
        size_mb = os.path.getsize(path) / 1e6
    except OSError:
        raise AudioError(f"{label}: file could not be read.") from None
    if size_mb > cfg["audio"]["max_file_mb"]:
        raise AudioError(f"{label}: file is {size_mb:.0f} MB; limit is {cfg['audio']['max_file_mb']} MB.")
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        try:  # fallback decoder (audioread/ffmpeg) for formats libsndfile cannot read
            y, sr = librosa.load(str(path), sr=None, mono=False)
            y = y.T if y.ndim == 2 else y
        except Exception:
            raise AudioError(f"{label}: could not decode audio (corrupt or unsupported encoding).") from None
    return load_audio_array(y, int(sr), cfg, label=label)
