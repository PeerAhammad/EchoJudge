# Evaluation procedure

Run: `python scripts/validate_dataset.py`, then `python scripts/evaluate.py --split dev --sweep --asr none` (threshold sensitivity, dev only),
then, with final thresholds in `config/default.yaml`, `python scripts/evaluate.py --split test --asr both --stress`
(held-out passage; `--asr both` = ablation signal-only vs ASR-aligned). Output: `results/eval_<split>_<mode>.json`, `results/report.md`.

**Metrics.** Per category (rushed, slow, pauses, monotone): (a) time-weighted segment precision/recall/F1 between the union of detected spans
and labelled spans, (b) file-level precision/recall/F1 (did the category fire at all), (c) for partial-span labels only: event-level IoU
matching (min IoU 0.1), mean IoU and start/end boundary error. Clean controls: files with any scored flag and flags per minute, including
cross-speaker clean reads. Score ranking: fraction of flawed reads scoring below the clean read of the same speaker and passage.
Repeatability: identical input twice must give identical scores. Stress: clean reads re-analysed at -12 dB gain and with white noise at 15 dB SNR.

**Not evaluated:** severity agreement (labels have a fixed nominal severity), weak-emphasis and fluency-interruption detectors (no labelled data),
agreement with human listeners. Results from instructed flaws do not establish validity for real public speaking.
Any number in the report comes from `results/report.md`; none are quoted from memory.
