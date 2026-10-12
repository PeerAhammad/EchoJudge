# EchoJudge — Contrastive Speech Intelligence (Stage 2)

Stage 1 signal core + Stage 2 (Whisper words, text diff, word-aligned pacing, weak emphasis, lexical disfluency, DTW cross-check).
Signal core: audio loading/validation → STFT/MFCC/F0/energy/pause/syllable-rate features →
interpretable timestamped flaw detectors → transparent rubric score → JSON.
**Not yet implemented:** API, dashboard, real dataset + evaluation script (Stages 3-4).

## Setup (VS Code terminal)
Python 3.10–3.12 required. Node is NOT needed until the dashboard stage.

    python -m venv .venv
    # Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/check_env.py
    python -m pytest -q

GPU: not used in Stage 1. For Stage 2 (Whisper/alignment) install the CUDA build of PyTorch matching your driver
(see https://pytorch.org/get-started/locally/); `check_env.py` reports whether CUDA is detected.

## Run
    python scripts/make_synthetic_demo.py     # SYNTHETIC tones with known flaws (smoke test only, not speech)
    python scripts/analyze.py REF.wav PART.wav --out result.json

## Layout
`config/default.yaml` (all thresholds + rubric weights) · `backend/audio` · `backend/features` ·
`backend/detection/rules.py` (each rule documents inputs, threshold, limitations) · `backend/scoring/rubric.py` ·
`backend/pipeline.py` · `tests/` (synthetic ground-truth tests).

## Honest limitations
- Tests use synthetic harmonic signals with known pauses/rates/pitch to verify timestamps and math; they are not evidence of
  performance on real speech. Real-speech thresholds must be calibrated on your recorded dataset.
- Event boundaries have ~1 s resolution (window hop). Syllable rate is an energy-peak estimate.
- Fragmentation detector is an experimental heuristic. Rubric = operational rubric, not a validated measure.

## Stage 2 usage
    pip install -r requirements.txt            # adds faster-whisper
    python scripts/analyze.py REF.wav PART.wav --model small --out result.json
    python scripts/analyze.py REF.wav PART.wav --device cpu          # force CPU
    python scripts/analyze.py REF.wav PART.wav --ref-transcript passage.txt
    python scripts/analyze.py REF.wav PART.wav --asr none            # signal-only
First ASR run downloads the Whisper model (~0.5 GB for `small`) from Hugging Face. GPU use needs CUDA 12 cuBLAS + cuDNN 9
visible to CTranslate2; if CUDA fails the code retries on CPU automatically (slower, still works).

## What is tested vs not (Stage 2)
- Tested (synthetic, fake word timings): text diff, unit pacing, word-aligned boundaries, emphasis logic, lexical rules,
  fallback/warning paths, ASR-failure handling.
- NOT tested in the build sandbox: the real Whisper step (no model download possible there). Verify on your machine.
- DTW is used for display/cross-checks only: on spectrally homogeneous synthetic input interior boundaries slipped by up to 2 s.
- Emphasis is a heuristic; on one synthetic test it found 1 of 2 injected emphasised words and 0 flags on 8 clean pairs.
  These are test observations, not performance estimates.
