"""Builds data/manifest.jsonl from data/raw/<speaker>_<passage>_<type>.<ext> (see docs/RECORDING_GUIDE.md).
Audio is decoded to 16 kHz mono PCM16 WAV copies under data/reference|participant (no other processing).
Labels = the flaw the speaker was instructed to produce, over the detected speech extent of the recording
(annotation_method 'instructed_whole_recording'), unless data/annotations.json gives explicit spans.
Existing synthetic manifest records are kept."""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import soundfile as sf
from backend.audio.io import SUPPORTED_EXT, AudioError, load_audio_file
from backend.config import load_config
from backend.features.extract import extract_features

TYPE_LABEL = {"rushed": "rushed_pacing", "slow": "slow_pacing", "pauses": "excessive_pauses", "flat": "monotone_delivery", "clean": None}
NAME = re.compile(r"^(s\d+)_(p\d+)_(ref|rushed|slow|pauses|flat|clean)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-passages", nargs="*", default=["p3"], help="held-out passages (split=test); others are dev")
    ap.add_argument("--default-ref-speaker", default="s1")
    a = ap.parse_args()
    cfg = load_config()
    raw = ROOT / "data" / "raw"
    ann = json.loads((ROOT / "data" / "annotations.json").read_text(encoding="utf-8")) if (ROOT / "data" / "annotations.json").exists() else {}
    files = sorted(p for p in raw.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    (ROOT / "data" / "reference").mkdir(exist_ok=True); (ROOT / "data" / "participant").mkdir(exist_ok=True)
    recs, bad = {}, []
    for p in files:
        m = NAME.match(p.stem)
        if not m:
            bad.append(p.name); continue
        spk, psg, typ = m.groups()
        try:
            audio = load_audio_file(p, cfg, label=p.name)
        except AudioError as e:
            print("SKIP", e); continue
        fs = extract_features(audio, cfg)
        s0, s1 = (fs.segments[0][0], fs.segments[-1][1]) if fs.segments else (0.0, audio.duration_s)
        role = "reference" if typ == "ref" else "participant"
        rid = p.stem
        rel = f"{role}/{rid}.wav"
        sf.write(ROOT / "data" / rel, audio.y, audio.sr, subtype="PCM_16")
        text = (ROOT / "data" / "passages" / f"{psg}.txt")
        labels = []
        if rid in ann:
            labels = [{**l, "annotation_method": l.get("annotation_method", "manual_span")} for l in ann[rid]]
        elif TYPE_LABEL.get(typ):
            labels = [{"category": TYPE_LABEL[typ], "start_s": round(s0, 2), "end_s": round(s1, 2), "severity": 2,
                       "annotation_method": "instructed_whole_recording"}]
        recs[rid] = {"id": rid, "passage_id": psg, "speaker_id": spk, "role": role, "audio_path": rel,
                     "transcript": text.read_text(encoding="utf-8").strip() if text.exists() else "", "flaw_labels": labels,
                     "synthetic": False, "split": "test" if psg in a.test_passages else "dev",
                     "license": "text: public domain; audio: self-recorded with contributor consent",
                     "provenance": f"recorded by {spk}; source file {p.name}",
                     "recording": {"source_format": p.suffix.lower().lstrip("."), "original_sr": audio.original_sr, "channels": audio.original_channels,
                                   "duration_s": round(audio.duration_s, 2), "clipped_fraction": round(audio.clipped_fraction, 5),
                                   "processing": "decoded to 16 kHz mono PCM16 wav; no filtering/normalisation/time-stretch"}}
    for rid, r in recs.items():
        if r["role"] == "participant":
            same = f"{r['speaker_id']}_{r['passage_id']}_ref"; dflt = f"{a.default_ref_speaker}_{r['passage_id']}_ref"
            ref = same if same in recs else dflt if dflt in recs else None
            if ref is None:
                print("WARNING: no reference for", rid); continue
            r["reference_id"] = ref; r["cross_speaker"] = recs[ref]["speaker_id"] != r["speaker_id"]
    mf = ROOT / "data" / "manifest.jsonl"
    keep = [json.loads(l) for l in mf.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("synthetic")] if mf.exists() else []
    out = keep + [r for r in recs.values() if r["role"] == "reference" or "reference_id" in r]
    mf.write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
    print(f"{len(recs)} recordings -> {mf} ({len(keep)} synthetic kept)")
    if bad: print("IGNORED (name must be <speaker>_<passage>_<type>):", bad)


if __name__ == "__main__":
    main()
