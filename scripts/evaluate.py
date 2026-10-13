"""Reproducible evaluation against labelled recordings in data/manifest.jsonl.

  python scripts/evaluate.py --split test --asr both            # held-out passage, signal-only vs ASR-aligned (ablation)
  python scripts/evaluate.py --split dev --sweep --asr none     # threshold sensitivity on DEV only (refused for test)
  python scripts/evaluate.py --split all --stress               # + noise/level robustness on clean controls
  python scripts/evaluate.py --split all --include-synthetic --asr none   # pipeline check on synthetic signals

Writes results/eval_<split>_<mode>.json and results/report.md. ASR transcripts are cached in results/asr_cache.
Labels of recorded data are INSTRUCTED flaws over the whole speech extent (not listener-verified), so metrics are
time-weighted/file-level; they measure agreement with the instruction, not expert judgement.
"""
import argparse, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np

from backend.api.dataset import load_manifest, resolve_audio
from backend.audio.io import load_audio_array, load_audio_file
from backend.config import load_config
from backend.evaluation.metrics import event_matches, file_level, length, pairwise_ranking, segment_level
from backend.pipeline import analyze_audio
from backend.transcription.types import Transcript, Word

EVAL_CATS = ["rushed_pacing", "slow_pacing", "excessive_pauses", "monotone_delivery"]
OUT = ROOT / "results"


class CachedASR:
    def __init__(self, inner, tag):
        self.inner, self.tag, self.dir = inner, tag, OUT / "asr_cache"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = lambda m: None

    def transcribe(self, y, sr, prompt=None):
        key = hashlib.sha1(np.asarray(y, np.float32).tobytes() + self.tag.encode() + (prompt or "").encode()).hexdigest()
        p = self.dir / f"{key}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return Transcript([Word(w["text"], w["norm"], w["start_s"], w["end_s"], w["prob"]) for w in d["words"]],
                              d["language"], d["backend"], d["model"], d["device"], d["notes"])
        self.inner.log = self.log
        tr = self.inner.transcribe(y, sr, prompt)
        p.write_text(json.dumps({"words": [w.to_dict() for w in tr.words], "language": tr.language, "backend": tr.backend,
                                 "model": tr.model, "device": tr.device, "notes": tr.notes}), encoding="utf-8")
        return tr


def load_pairs(man, split, include_synth):
    pairs = []
    for r in man.values():
        if r.get("role") != "participant" or r.get("reference_id") not in man:
            continue
        if r.get("synthetic") and not include_synth:
            continue
        if not r.get("synthetic") and split != "all" and r.get("split") != split:
            continue
        pairs.append((man[r["reference_id"]], r))
    return sorted(pairs, key=lambda x: x[1]["id"])


def run_pair(ref, part, cfg, asr, data_dir=ROOT / "data", mutate=None):
    ra = load_audio_file(resolve_audio(data_dir, ref), cfg, "reference")
    pa = load_audio_file(resolve_audio(data_dir, part), cfg, "participant")
    if mutate is not None:
        pa = load_audio_array(mutate(pa.y), pa.sr, cfg, "participant")
    return analyze_audio(ra, pa, cfg, asr=asr, ref_text=ref.get("transcript") or None)


def _diag(res):
    r, p = res["reference"]["profile"], res["participant"]["profile"]
    def ratio(k):
        a, b = r.get(k), p.get(k)
        return round(b / a, 2) if a and b else None
    return {"duration_ratio": ratio("duration_s"), "articulation_rate_ratio": ratio("articulation_rate_syl_s"),
            "pause_fraction_ref_part": [r.get("pause_fraction"), p.get("pause_fraction")],
            "pitch_spread_ratio": ratio("pitch_spread_st"), "energy_range_ratio": ratio("energy_range_db")}


def summarize(part, res):
    pred = defaultdict(list)
    for e in res["events"]:
        if e["scored"] and e["source"] == "participant":
            pred[e["category"]].append((e["start_s"], e["end_s"]))
    truth = defaultdict(list)
    for l in part.get("flaw_labels", []):
        truth[l["category"]].append((float(l["start_s"]), float(l["end_s"])))
    return {"id": part["id"], "speaker": part.get("speaker_id"), "passage": part.get("passage_id"), "cross_speaker": bool(part.get("cross_speaker")),
            "pred": dict(pred), "truth": dict(truth), "overall": res["scores"]["overall"], "duration_s": res["participant"]["profile"]["duration_s"],
            "alignment": res["alignment_method"], "flags": {k: len(v) for k, v in pred.items()},
            "partial": [l for l in part.get("flaw_labels", []) if l.get("annotation_method") != "instructed_whole_recording"],
            "categories": {k: v["score"] for k, v in res["scores"]["categories"].items()},
            "diag": _diag(res)}


def aggregate(files):
    out = {"n_files": len(files), "per_category": {}}
    for c in EVAL_CATS:
        out["per_category"][c] = {"segment_level": segment_level(files, c), "file_level": file_level(files, c)}
    partial = [f for f in files if f["partial"]]
    if partial:
        out["event_level_partial_labels"] = {}
        for c in EVAL_CATS:
            m = [event_matches(f["pred"].get(c, []), f["truth"].get(c, [])) for f in partial if c in f["truth"] or f["pred"].get(c)]
            if m:
                ious = [x["mean_iou"] for x in m if x["mean_iou"] is not None]
                out["event_level_partial_labels"][c] = {"tp": sum(x["tp"] for x in m), "fp": sum(x["fp"] for x in m), "fn": sum(x["fn"] for x in m),
                                                        "mean_iou": float(np.mean(ious)) if ious else None}
    clean = [f for f in files if not f["truth"]]
    if clean:
        mins = sum(f["duration_s"] for f in clean) / 60
        allc = defaultdict(int)
        for f in clean:
            for k, v in f["flags"].items():
                allc[k] += v
        out["clean_controls"] = {"n_files": len(clean), "files_with_any_flag": sum(1 for f in clean if any(f["flags"].values())),
                                 "flags_per_minute": {k: round(v / mins, 2) for k, v in allc.items()}, "flags": dict(allc),
                                 "mean_score": float(np.mean([f["overall"] for f in clean if f["overall"] is not None])),
                                 "cross_speaker_files": sum(f["cross_speaker"] for f in clean),
                                 "cross_speaker_files_with_flag": sum(1 for f in clean if f["cross_speaker"] and any(f["flags"].values()))}
    groups = defaultdict(lambda: {"clean": None, "flawed": []})
    for f in files:
        if f["overall"] is None:
            continue
        g = groups[(f["speaker"], f["passage"])]
        (g["flawed"].append(f["overall"]) if f["truth"] else g.__setitem__("clean", f["overall"]))
    out["ranking_flawed_vs_clean"] = pairwise_ranking({f"{k[0]}_{k[1]}": v for k, v in groups.items()})
    out["mean_score_flawed"] = float(np.mean([f["overall"] for f in files if f["truth"] and f["overall"] is not None])) if any(f["truth"] for f in files) else None
    return out


def evaluate(pairs, cfg, asr, stress=False):
    files = []
    for ref, part in pairs:
        t0 = time.time()
        res = run_pair(ref, part, cfg, asr)
        files.append(summarize(part, res))
        d = files[-1]["diag"]
        print(f"  {part['id']:<22} score={res['scores']['overall']} events={files[-1]['flags']} ({time.time() - t0:.0f}s)", flush=True)
        print(f"      vs ref: duration x{d['duration_ratio']}  speech-rate x{d['articulation_rate_ratio']}  pause-frac {d['pause_fraction_ref_part']}  pitch-spread x{d['pitch_spread_ratio']}  energy-range x{d['energy_range_ratio']}", flush=True)
    out = aggregate(files)
    out["files"] = files
    if pairs:  # repeatability: same input twice -> identical score
        again = summarize(pairs[0][1], run_pair(pairs[0][0], pairs[0][1], cfg, asr))
        out["repeat_max_abs_score_diff"] = abs((again["overall"] or 0) - (files[0]["overall"] or 0))
    if stress:
        rng = np.random.default_rng(0)
        out["stress"] = []
        for ref, part in [p for p in pairs if not p[1].get("flaw_labels")]:
            base = next(f for f in files if f["id"] == part["id"])
            def gain(y): return (y * 10 ** (-12 / 20)).astype(np.float32)
            def noise(y):
                p = float(np.mean(y ** 2)); return (y + rng.normal(0, np.sqrt(p / 10 ** 1.5), len(y))).astype(np.float32)  # SNR 15 dB
            for name, fn in (("gain -12 dB", gain), ("white noise SNR 15 dB", noise)):
                s = summarize(part, run_pair(ref, part, cfg, asr, mutate=fn))
                out["stress"].append({"id": part["id"], "condition": name, "flags": s["flags"], "score": s["overall"], "score_change": None if None in (s["overall"], base["overall"]) else round(s["overall"] - base["overall"], 1)})
    return out


def sweep(pairs, cfg_over, asr):
    grids = {"detection.rushed.ratio": ([1.15, 1.2, 1.3, 1.4, 1.5], "rushed_pacing"), "detection.slow.ratio": ([0.6, 0.7, 0.8, 0.85], "slow_pacing"),
             "detection.pauses.frequent.min_fraction": ([0.15, 0.2, 0.25, 0.3], "excessive_pauses"), "detection.monotone.ref_ratio": ([0.4, 0.5, 0.6, 0.7], "monotone_delivery"),
             "detection_stage2.aligned_pacing.rushed_speed_ratio": ([1.15, 1.2, 1.3, 1.4], "rushed_pacing"), "detection_stage2.aligned_pacing.slow_speed_ratio": ([0.6, 0.7, 0.8, 0.85], "slow_pacing"),
             "detection_stage2.aligned_pacing.min_event_s": ([1.5, 3.0, 5.0], "rushed_pacing"), "alignment.chunk_max_words": ([10, 16, 24], "rushed_pacing"),
             "detection.rushed.ref_guard_quantile": ([1.0, 0.9, 0.75, 0.5], "rushed_pacing"), "detection.slow.ref_guard_quantile": ([1.0, 0.9, 0.75, 0.5], "slow_pacing"),
             "detection.pauses.frequent.ref_guard_quantile": ([1.0, 0.9, 0.75, 0.5], "excessive_pauses"), "detection.monotone.ref_guard_quantile": ([1.0, 0.9, 0.75, 0.5], "monotone_delivery")}
    rows = []
    for key, (vals, cat) in grids.items():
        for v in vals:
            d = {}
            cur = d
            parts = key.split(".")
            for p in parts[:-1]:
                cur[p] = {}; cur = cur[p]
            cur[parts[-1]] = v
            cfg = load_config(overrides=_merge(cfg_over, d))
            files = [summarize(p, run_pair(r, p, cfg, asr)) for r, p in pairs]
            seg = segment_level(files, cat); fl = file_level(files, cat)
            rows.append({"param": key, "value": v, "category": cat, "segment_f1": seg["f1"], "segment_precision": seg["precision"], "segment_recall": seg["recall"], "file_f1": fl["f1"], "fp_files": fl["fp"]})
            print(f"  {key}={v}: seg F1={_f(seg['f1'])} file F1={_f(fl['f1'])} FP files={fl['fp']}", flush=True)
    return rows


def _merge(a, b):
    out = json.loads(json.dumps(a))
    def rec(x, y):
        for k, v in y.items():
            if isinstance(v, dict): rec(x.setdefault(k, {}), v)
            else: x[k] = v
    rec(out, b)
    return out


def _f(x): return "n/a" if x is None else f"{x:.2f}"


def report_md(results, split):
    L = [f"# EchoJudge evaluation report (split: {split})", "",
         "Labels are instructed flaws over each recording's speech extent (not listener-verified). Time-weighted metrics compare detected spans with labelled spans; file-level metrics ask whether a category fired at all.", ""]
    for mode, r in results.items():
        L += [f"## Mode: {mode} ({r['n_files']} participant recordings)", "", "| Category | Seg P | Seg R | Seg F1 | File P | File R | File F1 | File TP/FP/FN/TN |", "|---|---|---|---|---|---|---|---|"]
        for c, m in r["per_category"].items():
            s, f = m["segment_level"], m["file_level"]
            L.append(f"| {c} | {_f(s['precision'])} | {_f(s['recall'])} | {_f(s['f1'])} | {_f(f['precision'])} | {_f(f['recall'])} | {_f(f['f1'])} | {f['tp']}/{f['fp']}/{f['fn']}/{f['tn']} |")
        cc = r.get("clean_controls")
        if cc:
            L += ["", f"Clean controls: {cc['files_with_any_flag']}/{cc['n_files']} files had at least one scored flag; flags per minute {cc['flags_per_minute']}; mean score {cc['mean_score']:.1f}; cross-speaker clean files with a flag: {cc['cross_speaker_files_with_flag']}/{cc['cross_speaker_files']}."]
        rk = r["ranking_flawed_vs_clean"]
        L += ["", f"Score ranking: flawed recording scored below its clean control in {rk['flawed_below_clean']}/{rk['pairs']} pairs; mean flawed score {_f(r['mean_score_flawed'])}.", f"Repeatability: max |score diff| on a repeated run = {r.get('repeat_max_abs_score_diff')}."]
        if r.get("event_level_partial_labels"):
            L += ["", f"Event-level (partial-span labels only): {json.dumps(r['event_level_partial_labels'])}"]
        if r.get("stress"):
            L += ["", "Stress tests on clean controls:", "", "| File | Condition | Flags | Score change |", "|---|---|---|---|"] + [f"| {s['id']} | {s['condition']} | {s['flags']} | {s['score_change']} |" for s in r["stress"]]
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--asr", default="none", choices=["none", "auto", "both"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--include-synthetic", action="store_true")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if a.sweep and a.split != "dev":
        sys.exit("Refusing to sweep thresholds on anything but --split dev (the test split must stay untouched).")
    ov = {"asr": {k: v for k, v in (("model_size", a.model), ("device", a.device)) if v}}
    cfg = load_config(overrides=ov)
    man = load_manifest(ROOT / "data")
    pairs = load_pairs(man, a.split, a.include_synthetic)
    if not pairs:
        sys.exit("No labelled participant recordings found. Run scripts/build_manifest.py (see docs/RECORDING_GUIDE.md).")
    print(f"{len(pairs)} participant recordings (split={a.split}); config {cfg['version']}")
    OUT.mkdir(exist_ok=True)
    modes = ["none", "auto"] if a.asr == "both" else [a.asr]
    results = {}
    for mode in modes:
        asr = None
        if mode == "auto":
            from backend.transcription.asr import load_asr
            asr = CachedASR(load_asr(cfg), f"{cfg['asr']['model_size']}|{cfg['asr']['language']}")
            asr.log = lambda m: print(m, flush=True)
        print(f"== mode {mode}")
        if a.sweep:
            rows = sweep(pairs, ov, asr)
            (OUT / f"sweep_{mode}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            continue
        results[mode] = evaluate(pairs, cfg, asr, stress=a.stress)
        (OUT / f"eval_{a.split}_{mode}.json").write_text(json.dumps(results[mode], indent=2), encoding="utf-8")
    if results:
        md = report_md(results, a.split)
        (OUT / "report.md").write_text(md, encoding="utf-8")
        print("\n" + md)


if __name__ == "__main__":
    main()
