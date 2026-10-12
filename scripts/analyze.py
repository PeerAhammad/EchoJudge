"""CLI.
  python scripts/analyze.py REF.wav PART.wav [--asr auto|none] [--model small] [--device auto|cuda|cpu]
                            [--ref-transcript ref.txt] [--out result.json] [--config cfg.yaml]
--asr auto  : transcribe with faster-whisper (downloads the model on first use); falls back to signal-only if unavailable.
--asr none  : signal-only analysis (Stage 1 behaviour).
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.audio.io import AudioError
from backend.config import load_config
from backend.pipeline import analyze_files


def main():
    for stream in (sys.stdout, sys.stderr):  # Windows consoles may use legacy code pages
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("reference"); ap.add_argument("participant")
    ap.add_argument("--out"); ap.add_argument("--config")
    ap.add_argument("--asr", default="auto", choices=["auto", "none"])
    ap.add_argument("--model"); ap.add_argument("--device", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--ref-transcript", help="UTF-8 text file with the passage text (optional)")
    a = ap.parse_args()
    ov = {"asr": {}}
    if a.model: ov["asr"]["model_size"] = a.model
    if a.device: ov["asr"]["device"] = a.device
    cfg = load_config(a.config, ov)
    asr = None
    if a.asr == "auto":
        try:
            from backend.transcription.asr import load_asr
            asr = load_asr(cfg)
            asr.log = lambda m: print(m, flush=True)
        except Exception as e:
            print(f"warning: ASR unavailable ({e}); running signal-only.")
    text = Path(a.ref_transcript).read_text(encoding="utf-8") if a.ref_transcript else None
    try:
        res = analyze_files(a.reference, a.participant, cfg, asr=asr, ref_text=text)
    except AudioError as e:
        sys.exit(f"Error: {e}")
    s = res["scores"]
    print(f"Alignment: {res['alignment_method']}  quality={res['alignment_quality']}")
    print(f"Overall: {s['overall']}  (provisional={s['provisional']})")
    for k, v in s["categories"].items():
        print(f"  {k:16s} {v['score']}{'  (provisional)' if v['provisional'] else ''}")
    ta = res.get("text_alignment")
    if ta:
        print(f"Text: match_rate={ta['match_rate']} missing={ta['missing']} additions={ta['additions']} substitutions={ta['substitutions']}")
    if res.get("transcripts"):
        t = res["transcripts"]
        print("Reference words:", " ".join(w["text"] for w in t["reference"]["words"])[:300])
        print("Participant words:", " ".join(w["text"] for w in t["participant"]["words"])[:300])
        print(f"ASR model={t['reference']['model']} device={t['reference']['device']}")
    print(f"\n{len(res['events'])} event(s):")
    for e in res["events"]:
        print(f"- [{e['category']}] {e['start_s']:.1f}-{e['end_s']:.1f}s sev{e['severity']} ({e['detection_method']}): {e['explanation']}")
    for w in res["warnings"]:
        print("warning:", w)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8"); print("saved", a.out)


if __name__ == "__main__":
    main()
