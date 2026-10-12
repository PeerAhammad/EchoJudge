"""Speech recognition with word timestamps (faster-whisper).

IMPORTANT: Whisper word timestamps come from cross-attention + DTW, i.e. they are
estimates (typically within ~0.1-0.3 s), NOT forced-alignment boundaries. Whisper also tends to
omit fillers/repetitions. Both limits are surfaced in the results.
"""
from __future__ import annotations

import time

import numpy as np

from backend.transcription.text import normalize_token
from backend.transcription.types import Transcript, Word


class ASRUnavailable(RuntimeError):
    pass


class FasterWhisperASR:
    def __init__(self, asr_cfg: dict):
        self.cfg = asr_cfg
        self._model = None
        self._device_used: str | None = None
        self.notes: list[str] = []
        self.log = lambda msg: None   # set to print for progress output
        try:
            import faster_whisper  # noqa: F401
        except ImportError as e:
            raise ASRUnavailable("faster-whisper is not installed. Run: pip install faster-whisper") from e

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    def _load(self, device: str):
        from faster_whisper import WhisperModel
        ct = self.cfg["compute_type"]
        if ct == "auto":
            ct = "float16" if device == "cuda" else "int8"
        self.log(f"[asr] loading Whisper '{self.cfg['model_size']}' on {device} ({ct}); first run downloads ~0.5 GB, please wait...")
        t0 = time.time()
        self._model = WhisperModel(self.cfg["model_size"], device=device, compute_type=ct)
        self.log(f"[asr] model ready in {time.time() - t0:.0f}s")
        self._device_used = device

    def _run(self, y: np.ndarray, prompt: str | None):
        prompt_txt = None
        if prompt:
            prompt_txt = " ".join(prompt.split()[: self.cfg["prompt_max_words"]])
        segments, info = self._model.transcribe(
            y.astype(np.float32), language=self.cfg.get("language"), beam_size=self.cfg["beam_size"],
            word_timestamps=True, temperature=0.0, condition_on_previous_text=False,
            vad_filter=False, initial_prompt=prompt_txt)
        words: list[Word] = []
        for seg in segments:                       # generator: errors (e.g. missing CUDA DLLs) surface here
            for w in (seg.words or []):
                norm = normalize_token(w.word)
                if norm:
                    words.append(Word(w.word.strip(), norm, float(w.start), float(w.end), float(w.probability)))
        return words, info

    def transcribe(self, y: np.ndarray, sr: int, prompt: str | None = None) -> Transcript:
        assert sr == 16000, "ASR expects 16 kHz audio"
        want = self.cfg["device"]
        device = ("cuda" if self._cuda_available() else "cpu") if want == "auto" else want
        notes = list(self.notes)
        if self._model is None or self._device_used != device:
            try:
                self._load(device)
            except Exception:
                if device == "cuda":
                    notes.append("CUDA model load failed; used CPU instead.")
                    device = "cpu"
                    self._load("cpu")
                else:
                    raise
        self.log(f"[asr] transcribing {len(y) / sr:.0f}s of audio on {device}...")
        t0 = time.time()
        try:
            words, info = self._run(y, prompt)
        except Exception:
            if device != "cuda":
                raise
            notes.append("CUDA inference failed (missing cuBLAS/cuDNN?); retried on CPU.")
            device = "cpu"
            self._load("cpu")
            words, info = self._run(y, prompt)
        self.notes = notes
        self.log(f"[asr] done in {time.time() - t0:.0f}s, {len(words)} words")
        return Transcript(words, getattr(info, "language", None), "faster-whisper",
                          self.cfg["model_size"], device, notes)


def load_asr(cfg: dict):
    return FasterWhisperASR(cfg["asr"])
