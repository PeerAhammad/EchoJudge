# EchoJudge — Contrastive Speech Intelligence

**An interpretable speech-analysis system for comparing reference and participant recordings.**

EchoJudge analyzes two audio recordings to identify differences in pacing, pauses, emphasis, pronunciation-related text mismatches, and other speech-delivery patterns. It combines signal processing, Whisper-based transcription, word alignment, rule-based detection, and a transparent scoring rubric.

> **Current stage:** Stage 4 — Dataset development and evaluation.

## Overview

EchoJudge is designed to compare how a reference passage is delivered against a participant's recording. It produces structured analysis results with timestamps, detected events, and rubric-based scores.

### Key Features

- **Audio feature extraction:** STFT, MFCC, fundamental frequency (F0), energy, pauses, and estimated syllable rate.
- **Speech transcription:** Optional Whisper transcription through `faster-whisper`.
- **Text comparison:** Word-level differences between reference and participant speech.
- **Pacing analysis:** Word-aligned timing and delivery-rate comparisons.
- **Speech-event detection:** Interpretable rules for pauses, emphasis, lexical disfluencies, and other delivery patterns.
- **Cross-checking:** Dynamic Time Warping (DTW) for alignment inspection.
- **Transparent scoring:** Configurable rubric weights and thresholds.
- **API and dashboard:** Upload recordings, inspect analysis results, and browse dataset samples.
- **Dataset evaluation:** Manifest generation, dataset validation, threshold tuning, and held-out evaluation workflows.

## How It Works

1. **Input:** Load a reference recording and a participant recording.
2. **Feature extraction:** Calculate acoustic and timing-related features.
3. **Transcription and alignment:** Optionally transcribe speech and compare words and timing.
4. **Detection:** Apply interpretable rules to identify potential differences in speech delivery.
5. **Scoring:** Aggregate detected features using a configurable rubric.
6. **Output:** Return structured JSON results with scores, timestamps, and detected events.

## Technology Stack

- **Language:** Python 3.10–3.12
- **Audio analysis:** STFT, MFCC, F0, energy and pause analysis
- **Speech recognition:** Whisper through `faster-whisper`
- **Backend:** FastAPI and Uvicorn
- **Configuration:** YAML
- **Testing:** pytest
- **Frontend:** Browser-based dashboard

## Project Structure

```text
EchoJudge/
├── backend/
│   ├── alignment/       # Text and timing alignment
│   ├── audio/           # Audio loading and validation
│   ├── detection/       # Interpretable event detectors
│   ├── evaluation/      # Evaluation metrics
│   ├── features/        # Acoustic feature extraction
│   ├── scoring/         # Rubric-based scoring
│   ├── transcription/   # ASR and text processing
│   ├── api/             # API and analysis jobs
│   ├── pipeline.py      # Analysis pipeline
│   └── main.py          # Application entry point
├── config/
│   └── default.yaml     # Detection thresholds and rubric weights
├── data/
│   ├── raw/             # Source recordings
│   ├── reference/       # Reference audio
│   ├── participant/     # Participant recordings
│   ├── synthetic_demo/  # Synthetic demonstration samples
│   ├── passages/        # Reference passage text
│   └── manifest.jsonl   # Dataset manifest
├── docs/                # Recording and evaluation guides
├── frontend/            # Browser dashboard
├── scripts/             # Analysis and evaluation utilities
├── tests/               # Unit and API tests
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PeerAhammad/EchoJudge.git
cd EchoJudge
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check the environment and tests

```bash
python scripts/check_env.py
python -m pytest -q
```

Use Python 3.10–3.12. The initial signal-processing stage does not require a GPU. Whisper inference can run on a CPU; GPU acceleration requires compatible CUDA libraries and dependencies.

## Run an Analysis

Compare a reference recording with a participant recording:

```bash
python scripts/analyze.py REF.wav PART.wav --out result.json
```

For Whisper-based transcription:

```bash
python scripts/analyze.py REF.wav PART.wav --model small --out result.json
```

Force CPU inference:

```bash
python scripts/analyze.py REF.wav PART.wav --device cpu
```

Provide a reference transcript:

```bash
python scripts/analyze.py REF.wav PART.wav --ref-transcript passage.txt
```

Run without automatic speech recognition:

```bash
python scripts/analyze.py REF.wav PART.wav --asr none
```

The first Whisper run may download model files. The `small` model is approximately 0.5 GB, and actual storage requirements can vary.

### Synthetic demonstration

```bash
python scripts/make_synthetic_demo.py
```

This creates synthetic signals for smoke testing. These signals are useful for checking parts of the pipeline, but they are **not a substitute for real speech recordings**.

## Run the API and Dashboard

Start the application from the project root:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open the dashboard at:

http://127.0.0.1:8000

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

### Main API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check service and runtime status |
| `POST /api/analyze` | Submit reference and participant recordings |
| `GET /api/results/{id}` | Retrieve analysis status and results |
| `DELETE /api/results/{id}` | Delete stored analysis audio |
| `GET /api/audio/{id}/{reference\|participant}` | Retrieve audio for playback |
| `GET /api/dataset` | Browse dataset entries |
| `GET /api/dataset/{id}` | Inspect a dataset entry |
| `POST /api/analyze_sample` | Analyze a manifest sample |
| `POST /api/evaluate` | Batch evaluation endpoint; currently returns HTTP 501 |

Uploads are stored under `data/uploads/` using server-generated names. The configured retention period defaults to 60 minutes, and the API supports explicit deletion. Review `config/default.yaml` and the API implementation before deploying to a public server.

## Stage 4: Dataset and Evaluation

The current workflow prepares a dataset, validates its structure, tunes thresholds on development data, and evaluates the final configuration on held-out test data.

### 1. Prepare the dataset

Record audio following [`docs/RECORDING_GUIDE.md`](docs/RECORDING_GUIDE.md).

Use consistent filenames, such as:

```text
s1_p1_ref.ogg
s1_p1_rushed.ogg
s1_p1_slow.ogg
s1_p1_flat.ogg
s1_p1_pauses.ogg
```

Build and validate the manifest:

```bash
python scripts/build_manifest.py
python scripts/validate_dataset.py
```

Proceed when dataset validation reports `OK`.

### 2. Tune thresholds on development data

```bash
python scripts/evaluate.py --split dev --sweep --asr none
```

Use development results to tune thresholds in `config/default.yaml`. Avoid using held-out test results to select thresholds.

### 3. Evaluate the held-out test set

```bash
python scripts/evaluate.py --split test --asr both --stress
```

The intended report output is:

```text
results/report.md
```

Consult [`docs/dataset.md`](docs/dataset.md) and [`docs/evaluation.md`](docs/evaluation.md) for dataset conventions, metrics, and evaluation methodology.

**Evaluation status:** The workflow is implemented, but measured results must be generated and reviewed before making claims about real-world performance.

## Testing and Validation

The project includes tests for core analysis logic, synthetic signals, alignment, detection rules, API behavior, and evaluation metrics.

Existing development checks have covered:

- Synthetic signal feature extraction and event detection.
- Text differences and word-aligned pacing logic.
- Rule-based emphasis and lexical analysis.
- API validation, error handling, and deletion behavior.
- Evaluation metric calculations.

Some checks use synthetic signals or simulated word timings. These verify implementation behavior but do not establish accuracy on real speech.

The real Whisper inference path and browser behavior should also be tested in the target environment.

## Limitations

EchoJudge is an experimental analysis system, not a clinically or scientifically validated measure of speaking ability.

- **Synthetic versus real data:** Synthetic tests do not establish performance on real recordings.
- **Timing resolution:** Some event boundaries have approximately one-second resolution because of the analysis window hop.
- **Syllable rate:** Estimated from acoustic energy peaks rather than direct syllable recognition.
- **Emphasis detection:** Heuristic-based and may miss emphasized words or produce false positives.
- **Fragmentation detection:** Experimental heuristic requiring further validation.
- **DTW alignment:** Used for inspection and cross-checking; it can produce inaccurate interior boundaries on spectrally similar signals.
- **Scoring rubric:** Configurable and interpretable, but not independently validated.
- **ASR and hardware:** Whisper performance depends on the model, recording quality, and available hardware.

Thresholds should be calibrated using development data and assessed on a separate held-out test set. Results should report the dataset, evaluation method, and relevant limitations.

## Privacy and Data Handling

Speech recordings can contain personal information. Before publishing datasets or using them for evaluation:

- Obtain appropriate permission to record and share participants' voices.
- Avoid committing private recordings, credentials, or temporary uploads.
- Keep secrets in environment variables rather than source code.
- Review upload retention and deletion behavior before deployment.
- Document whether recordings are real, synthetic, or anonymized.

## Roadmap

- [x] Signal-processing core and configurable scoring.
- [x] Whisper integration and alignment components.
- [x] API and browser dashboard.
- [x] Dataset manifest and evaluation tooling.
- [ ] Complete and validate the recorded dataset.
- [ ] Tune thresholds using development data.
- [ ] Run held-out evaluation and inspect errors.
- [ ] Publish measured results and limitations in the evaluation report.
- [ ] Validate the dashboard and end-to-end Whisper workflow on real recordings.

## Contributing

Contributions that improve reproducibility, test coverage, detection quality, documentation, or evaluation methodology are welcome. Please document the test conditions and limitations of any new detector or metric.

## License

Add the project's chosen license before presenting the repository as open source.

---

**EchoJudge — making speech-delivery analysis more interpretable, reproducible, and measurable.**
