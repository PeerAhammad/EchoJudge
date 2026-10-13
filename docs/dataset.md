# Dataset methodology (EchoJudge pilot)

**Construction.** Public-domain passages (`data/passages/`) are read aloud by the dataset author (speaker `s1`) and, for a
held-out cross-speaker check, a second consenting speaker (`s2`). Per passage: a reference read (`ref`), a second natural read
(`clean`, a false-positive control) and instructed flawed reads: `rushed`, `slow`, `pauses`, `flat` (see `docs/RECORDING_GUIDE.md`).
Genuine recordings are used for every flaw; no time-stretching or other signal manipulation is applied. Audio is only
decoded to 16 kHz mono PCM16 WAV. Synthetic test signals (`syn_*`) exist only for pipeline smoke tests and are excluded from
evaluation unless `--include-synthetic` is given.

**Manifest** (`data/manifest.jsonl`, one JSON per line): `id, passage_id, speaker_id, role (reference|participant), reference_id,
audio_path, transcript, flaw_labels[{category,start_s,end_s,severity 0-4,annotation_method}], synthetic, split (dev|test),
license, provenance, recording{source_format,original_sr,channels,duration_s,clipped_fraction,processing}, cross_speaker`.
Word-level alignment is not stored in the manifest; it is produced per analysis by Whisper (cached in `results/asr_cache`).

**Label provenance and quality.** Labels are `instructed_whole_recording`: the speaker was told to produce the flaw over the whole
read and the label spans the detected speech extent. They record intent, are not verified by listeners, and severity is a fixed
nominal 2. Partial spans can be supplied in `data/annotations.json` (`{"<id>": [{"category","start_s","end_s","severity"}]}`) with
`annotation_method` `manual_span`. Controlled/instructed flaws are not equivalent to naturally occurring public-speaking flaws or expert judgements.

**Checks** (`python scripts/validate_dataset.py`): missing files, invalid/out-of-range timestamps, duplicate ids, empty or mismatching
transcripts, unknown categories, label without annotation method, severity range, sample rate, and passage leakage across splits.

**Splits.** By passage: `p1`,`p2` = dev (thresholds may be tuned here), `p3` = test (never tuned on; includes the cross-speaker recordings).

**Limitations.** Pilot size (3 passages, 1-2 speakers); one microphone/room; no human annotation; emphasis and fluency flaws are not
recorded, so those detectors are NOT evaluated; labelled categories are rushed, slow, excessive pauses and monotone delivery only.
Consent of contributing speakers must be obtained before publishing audio; speakers are pseudonymised (`s1`, `s2`).
