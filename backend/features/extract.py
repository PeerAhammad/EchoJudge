"""Acoustic feature extraction.

Common time base for ALL frame-level series (documented):
    sample rate 16 kHz, window 25 ms (400 samples), hop 10 ms (160 samples),
    n_fft 512, centered frames -> frame i is centred at t = i * 0.010 s.
Units: rms_db = dB re. full scale; f0_hz = Hz (NaN when unvoiced);
f0_st = semitones relative to the speaker's own median F0 (speaker-normalised).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
import parselmouth
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from backend.audio.io import LoadedAudio


@dataclass
class FeatureSet:
    sr: int
    hop_s: float
    duration_s: float
    times: np.ndarray
    rms_db: np.ndarray
    f0_hz: np.ndarray
    f0_st: np.ndarray
    voiced: np.ndarray
    speech_mask: np.ndarray            # final VAD mask (short gaps filled)
    micro_runs: list[tuple[float, float]]  # raw-ish speech runs (fragmentation heuristic)
    segments: list[tuple[float, float]]    # speech segments (s)
    pauses: list[tuple[float, float]]      # internal pauses >= min_pause_s
    nuclei_times: np.ndarray
    centroid_hz: np.ndarray
    mfcc: np.ndarray                   # (n_mfcc, n_frames)
    f0_median_hz: float | None
    noise_floor_db: float
    speech_level_db: float
    snr_db: float | None
    pitch_available: bool
    label: str = "audio"
    notes: list[str] = field(default_factory=list)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index runs where mask is True."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def _fill_and_clean(mask: np.ndarray, min_gap: int, min_run: int) -> np.ndarray:
    """Fill False gaps shorter than min_gap (frames), then drop True runs shorter than min_run."""
    out = mask.copy()
    runs = _runs(mask)
    for (a0, a1), (b0, b1) in zip(runs[:-1], runs[1:]):
        if b0 - a1 < min_gap:
            out[a1:b0] = True
    for r0, r1 in _runs(out):
        if r1 - r0 < min_run:
            out[r0:r1] = False
    return out


def extract_features(audio: LoadedAudio, cfg: dict) -> FeatureSet:
    fc = cfg["features"]
    sr = audio.sr
    hop = int(round(fc["hop_length_s"] * sr))
    win = int(round(fc["frame_length_s"] * sr))
    hop_s = hop / sr
    y = audio.y
    notes: list[str] = []

    # --- STFT (FFT-derived spectral information) -> centroid, mel, MFCC -----------
    S = np.abs(librosa.stft(y, n_fft=fc["n_fft"], hop_length=hop, win_length=win, center=True))
    n_frames = S.shape[1]
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=fc["n_fft"])[0]
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=fc["n_mels"])
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=fc["n_mfcc"])

    # --- energy ---------------------------------------------------------------
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0][:n_frames]
    rms_db = 20.0 * np.log10(rms + 1e-10)
    times = np.arange(n_frames) * hop_s

    # --- VAD ------------------------------------------------------------------
    v = fc["vad"]
    peak = float(np.percentile(rms_db, 95))
    floor = float(np.percentile(rms_db, 10))
    thr = max(peak - v["dynamic_range_db"], floor + v["noise_margin_db"], v["absolute_floor_db"])
    thr = min(thr, peak - 12.0)
    raw = rms_db > thr
    min_gap = int(round(v["min_pause_s"] / hop_s))
    min_run = int(round(v["min_speech_s"] / hop_s))
    mask = _fill_and_clean(raw, min_gap, min_run)
    micro = _fill_and_clean(raw, int(round(v["micro_gap_fill_s"] / hop_s)), min_run)
    runs = _runs(mask)
    segments = [(a * hop_s, b * hop_s) for a, b in runs]
    pauses = [(a1 * hop_s, b0 * hop_s) for (_, a1), (b0, _) in zip(runs[:-1], runs[1:])]
    micro_runs = [(a * hop_s, b * hop_s) for a, b in _runs(micro)]

    speech_frames = rms_db[mask]
    nonspeech = rms_db[~mask]
    speech_level = float(np.percentile(speech_frames, 95)) if speech_frames.size else peak
    noise_floor = float(np.median(nonspeech)) if nonspeech.size >= 20 else float("nan")
    snr = speech_level - noise_floor if np.isfinite(noise_floor) else None

    # --- F0 via Praat autocorrelation (Parselmouth) -------------------------------
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
    pitch = snd.to_pitch_ac(time_step=hop_s, pitch_floor=fc["f0_floor_hz"],
                            pitch_ceiling=fc["f0_ceiling_hz"])
    f0_raw = pitch.selected_array["frequency"].astype(float)
    xs = pitch.xs()
    idx = np.rint((times - xs[0]) / hop_s).astype(int)
    ok = (idx >= 0) & (idx < len(f0_raw))
    f0 = np.full(n_frames, np.nan)
    f0[ok] = f0_raw[idx[ok]]
    f0[f0 <= 0] = np.nan
    voiced = np.isfinite(f0)
    pitch_available = int(voiced.sum()) >= fc["min_voiced_frames"]
    if pitch_available:
        f0_med = float(np.nanmedian(f0))
        f0_st = 12.0 * np.log2(f0 / f0_med)
    else:
        f0_med = None
        f0_st = np.full(n_frames, np.nan)
        notes.append("Too few voiced frames: pitch-based features unavailable.")

    # --- syllable nuclei (energy peaks in voiced speech) --------------------------
    sy = fc["syllable"]
    env = uniform_filter1d(rms_db, size=max(1, sy["smooth_frames"]))
    pk, _ = find_peaks(env, prominence=sy["min_prominence_db"],
                       distance=max(1, int(round(sy["min_distance_s"] / hop_s))))
    pk = pk[mask[pk]]
    if sy["require_voiced"] and pitch_available:
        near = np.array([voiced[max(0, p - 2): p + 3].any() for p in pk], dtype=bool) if len(pk) else np.array([], bool)
        pk = pk[near]
    nuclei = pk * hop_s

    return FeatureSet(
        sr=sr, hop_s=hop_s, duration_s=audio.duration_s, times=times, rms_db=rms_db,
        f0_hz=f0, f0_st=f0_st, voiced=voiced, speech_mask=mask, micro_runs=micro_runs,
        segments=segments, pauses=pauses, nuclei_times=nuclei, centroid_hz=centroid,
        mfcc=mfcc, f0_median_hz=f0_med, noise_floor_db=noise_floor, speech_level_db=speech_level,
        snr_db=snr, pitch_available=pitch_available, label=audio.label, notes=notes,
    )


def series_for_dashboard(fs: FeatureSet, cfg: dict) -> dict:
    """Downsampled series (default 10 Hz) for plotting; NaN -> None."""
    step = max(1, int(round(1.0 / (cfg["features"]["series_hz"] * fs.hop_s))))
    sl = slice(None, None, step)

    def clean(a, nd=2):
        return [None if not np.isfinite(x) else round(float(x), nd) for x in a[sl]]

    return {
        "hz": cfg["features"]["series_hz"],
        "times_s": [round(float(t), 3) for t in fs.times[sl]],
        "rms_db": clean(fs.rms_db, 1),
        "f0_hz": clean(fs.f0_hz, 1),
        "f0_st": clean(fs.f0_st, 2),
        "centroid_hz": clean(fs.centroid_hz, 0),
        "speech": [bool(x) for x in fs.speech_mask[sl]],
    }
