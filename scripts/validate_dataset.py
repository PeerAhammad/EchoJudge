"""Dataset checks: missing files, invalid timestamps, duplicate ids, transcript problems, malformed annotations,
split leakage, audio sanity. Exit code 1 if any ERROR."""
import json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import soundfile as sf
from backend.schemas import CATEGORIES

REQ = ["id", "passage_id", "speaker_id", "role", "audio_path", "flaw_labels", "split", "license"]


def main():
    mf = ROOT / "data" / "manifest.jsonl"
    if not mf.exists():
        sys.exit("ERROR: data/manifest.jsonl not found (run scripts/build_manifest.py)")
    errs, warns, recs, seen = [], [], [], set()
    for n, line in enumerate(mf.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: r = json.loads(line)
        except json.JSONDecodeError: errs.append(f"line {n}: malformed JSON"); continue
        for k in REQ:
            if k not in r: errs.append(f"{r.get('id', n)}: missing field '{k}'")
        if r.get("id") in seen: errs.append(f"duplicate id {r.get('id')}")
        seen.add(r.get("id")); recs.append(r)
    byid = {r.get("id"): r for r in recs}
    split_of = defaultdict(set)
    for r in recs:
        if r.get("synthetic"): continue
        rid = r.get("id")
        p = ROOT / "data" / str(r.get("audio_path", ""))
        dur = None
        if not p.is_file(): errs.append(f"{rid}: audio file missing ({r.get('audio_path')})")
        else:
            info = sf.info(str(p)); dur = info.frames / info.samplerate
            if info.samplerate != 16000 or info.channels != 1: warns.append(f"{rid}: not 16 kHz mono")
            if dur < 3: errs.append(f"{rid}: shorter than 3 s")
        if not str(r.get("transcript", "")).strip(): errs.append(f"{rid}: empty transcript")
        if r.get("role") == "participant":
            ref = byid.get(r.get("reference_id"))
            if ref is None: errs.append(f"{rid}: reference_id not found")
            elif ref.get("passage_id") != r.get("passage_id"): errs.append(f"{rid}: reference is a different passage")
            elif ref.get("transcript", "").strip() != r.get("transcript", "").strip(): errs.append(f"{rid}: transcript differs from reference")
        for l in r.get("flaw_labels", []):
            try:
                s, e = float(l["start_s"]), float(l["end_s"])
                if l["category"] not in CATEGORIES: errs.append(f"{rid}: unknown category {l['category']}")
                if not (0 <= s < e): errs.append(f"{rid}: invalid span {s}-{e}")
                if dur is not None and e > dur + 0.05: errs.append(f"{rid}: label ends after audio ({e:.1f}>{dur:.1f})")
                if not (0 <= int(l.get("severity", 0)) <= 4): errs.append(f"{rid}: severity outside 0-4")
                if "annotation_method" not in l: errs.append(f"{rid}: label without annotation_method")
            except (KeyError, ValueError, TypeError): errs.append(f"{rid}: malformed label {l}")
        split_of[r.get("passage_id")].add(r.get("split"))
    for psg, sp in split_of.items():
        if len(sp) > 1: errs.append(f"passage {psg} appears in several splits {sorted(sp)} (leakage)")
    real = [r for r in recs if not r.get("synthetic")]
    print(f"{len(real)} real records ({sum(r['role'] == 'participant' for r in real)} participant), {len(recs) - len(real)} synthetic")
    for w in warns: print("WARN ", w)
    for e in errs: print("ERROR", e)
    print("OK" if not errs else f"{len(errs)} error(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
