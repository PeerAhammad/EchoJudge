"""FastAPI app. Heavy work runs in a single background worker; the API only validates, stores and reports.

Privacy: uploads are written under runtime.upload_dir/<random id>/ with server-chosen names (client filenames
are never used), kept only for runtime.retention_minutes, and removable immediately with DELETE /api/results/{id}.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.api.dataset import ID_RE, load_manifest, public_view, resolve_audio
from backend.api.jobs import JobStore
from backend.audio.io import SUPPORTED_EXT
from backend.config import load_config
from backend.pipeline import analyze_files

ROOT = Path(__file__).resolve().parent.parent.parent
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
MODELS = {"tiny", "base", "small", "medium"}
MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".flac": "audio/flac"}


class SampleRequest(BaseModel):
    participant_id: str
    asr: str | None = None
    model: str | None = None


def create_app(data_dir: str | Path | None = None, asr_factory: Callable | None = None,
               config_overrides: dict | None = None) -> FastAPI:
    base_cfg = load_config(overrides=config_overrides)
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    rt = base_cfg["runtime"]
    upload_dir = Path(rt["upload_dir"])
    store = JobStore(upload_dir if upload_dir.is_absolute() else ROOT / upload_dir,
                     rt["retention_minutes"], rt["max_processing_s"])
    max_mb = base_cfg["audio"]["max_file_mb"]
    asr_cache: dict = {}
    asr_lock = threading.Lock()

    def get_asr(cfg):
        key = (cfg["asr"]["model_size"], cfg["asr"]["device"])
        with asr_lock:
            if key not in asr_cache:
                if asr_factory:
                    asr_cache[key] = asr_factory(cfg)
                else:
                    from backend.transcription.asr import load_asr
                    asr_cache[key] = load_asr(cfg)
            return asr_cache[key]

    app = FastAPI(title="EchoJudge", version="0.3",
                  description="Contrastive speech evaluation: timestamped, evidence-backed delivery feedback.")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > (2 * max_mb + 2) * 1_000_000:
            return JSONResponse({"detail": "Request too large."}, status_code=413)
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # ------------------------------------------------------------------ helpers
    def runner(ref_path: Path, part_path: Path, text: str | None, asr_mode: str, model: str | None, truth=None):
        def fn(job):
            ov = {"asr": {"model_size": model}} if model else None
            cfg = load_config(overrides=ov)
            asr = None
            if asr_mode == "auto":
                try:
                    asr = get_asr(cfg)
                    asr.log = job.say
                except Exception as e:
                    job.say(f"ASR unavailable ({type(e).__name__}); running signal-only.")
            else:
                job.say("ASR disabled: signal-only analysis.")
            job.say("Analyzing audio...")
            res = analyze_files(str(ref_path), str(part_path), cfg, asr=asr, ref_text=text)
            res["analysis_id"] = job.id
            if truth is not None:
                res["ground_truth"] = truth
            job.say("Done.")
            return res
        return fn

    def check_opts(asr_mode, model):
        asr_mode = asr_mode or rt["default_asr"]
        if asr_mode not in ("auto", "none"):
            raise HTTPException(400, "asr must be 'auto' or 'none'.")
        if model and model not in MODELS:
            raise HTTPException(400, f"model must be one of {sorted(MODELS)}.")
        return asr_mode, model

    async def save_upload(up: UploadFile, dest_dir: Path, name: str) -> Path:
        ext = Path(up.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXT:
            raise HTTPException(400, f"{name}: unsupported file type. Use WAV or MP3.")
        dest, size = dest_dir / f"{name}{ext}", 0
        with open(dest, "wb") as f:
            while chunk := await up.read(1 << 20):
                size += len(chunk)
                if size > max_mb * 1_000_000:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"{name}: file exceeds {max_mb} MB.")
                f.write(chunk)
        if size == 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, f"{name}: file is empty.")
        return dest

    def job_or_404(jid: str):
        if not JOB_ID.match(jid):
            raise HTTPException(404, "Unknown analysis.")
        job = store.get(jid)
        if job is None:
            raise HTTPException(404, "Unknown or expired analysis.")
        return job

    # ------------------------------------------------------------------ endpoints
    @app.get("/health")
    def health():
        try:
            import faster_whisper  # noqa: F401
            asr_ok = True
        except Exception:
            asr_ok = False
        return {"status": "ok", "version": "0.3", "asr_available": asr_ok, "config_version": base_cfg["version"],
                "retention_minutes": rt["retention_minutes"], "max_file_mb": max_mb}

    @app.post("/api/analyze", status_code=202)
    async def analyze(reference: UploadFile = File(...), participant: UploadFile = File(...),
                      ref_transcript: str | None = Form(None), asr: str | None = Form(None),
                      model: str | None = Form(None)):
        asr_mode, model = check_opts(asr, model)
        if ref_transcript and len(ref_transcript) > 20000:
            raise HTTPException(400, "Transcript too long (max 20000 characters).")
        job = store.new()
        try:
            rp = await save_upload(reference, job.dir, "reference")
            pp = await save_upload(participant, job.dir, "participant")
        except HTTPException:
            store.delete(job.id)
            raise
        job.files = {"reference": rp, "participant": pp}
        job.say("Files received.")
        store.submit(job, runner(rp, pp, (ref_transcript or "").strip() or None, asr_mode, model))
        return {"analysis_id": job.id, "status": job.status}

    @app.get("/api/results/{analysis_id}")
    def results(analysis_id: str):
        job = job_or_404(analysis_id)
        body = {"analysis_id": job.id, "status": job.status, "log": job.log, "error": job.error}
        if job.status == "done":
            body["result"] = job.result
        return body

    @app.delete("/api/results/{analysis_id}")
    def delete_result(analysis_id: str):
        job_or_404(analysis_id)
        store.delete(analysis_id)
        return {"deleted": True}

    @app.get("/api/audio/{analysis_id}/{which}")
    def audio(analysis_id: str, which: str):
        job = job_or_404(analysis_id)
        p = job.files.get(which)
        if p is None or not Path(p).is_file():
            raise HTTPException(404, "Audio not available.")
        return FileResponse(p, media_type=MIME.get(Path(p).suffix.lower(), "application/octet-stream"))

    # ------------------------------------------------------------------ dataset
    @app.get("/api/dataset")
    def dataset():
        man = load_manifest(data_dir)
        items = [{"id": r["id"], "passage_id": r.get("passage_id"), "speaker_id": r.get("speaker_id"),
                  "role": r.get("role"), "reference_id": r.get("reference_id"), "synthetic": bool(r.get("synthetic")),
                  "n_labels": len(r.get("flaw_labels", [])), "split": r.get("split"), "license": r.get("license"),
                  "description": r.get("description")} for r in man.values()]
        return {"count": len(items), "items": items,
                "note": None if items else "No dataset manifest found. See docs/dataset.md (Stage 4)."}

    @app.get("/api/dataset/{sample_id}")
    def dataset_item(sample_id: str):
        if not ID_RE.match(sample_id):
            raise HTTPException(404, "Unknown sample.")
        rec = load_manifest(data_dir).get(sample_id)
        if rec is None:
            raise HTTPException(404, "Unknown sample.")
        return public_view(rec)

    @app.get("/api/dataset/{sample_id}/audio")
    def dataset_audio(sample_id: str):
        rec = load_manifest(data_dir).get(sample_id) if ID_RE.match(sample_id) else None
        p = resolve_audio(data_dir, rec) if rec else None
        if p is None:
            raise HTTPException(404, "Audio not available.")
        return FileResponse(p, media_type=MIME.get(p.suffix.lower(), "application/octet-stream"))

    @app.post("/api/analyze_sample", status_code=202)
    def analyze_sample(req: SampleRequest):
        asr_mode, model = check_opts(req.asr, req.model)
        man = load_manifest(data_dir)
        part = man.get(req.participant_id) if ID_RE.match(req.participant_id) else None
        if part is None or part.get("role") != "participant" or part.get("reference_id") not in man:
            raise HTTPException(404, "Participant sample (with a reference) not found.")
        ref = man[part["reference_id"]]
        rp, pp = resolve_audio(data_dir, ref), resolve_audio(data_dir, part)
        if rp is None or pp is None:
            raise HTTPException(404, "Sample audio files are missing.")
        job = store.new()
        job.files = {"reference": rp, "participant": pp}
        job.say(f"Loaded dataset sample {part['id']} vs {ref['id']}.")
        truth = {"labels": part.get("flaw_labels", []), "synthetic": bool(part.get("synthetic")),
                 "description": part.get("description"), "sample_id": part["id"]}
        store.submit(job, runner(rp, pp, ref.get("transcript"), asr_mode, model, truth))
        return {"analysis_id": job.id, "status": job.status}

    @app.post("/api/evaluate", status_code=501)
    def evaluate():
        return JSONResponse({"detail": "Batch evaluation is not implemented yet (Stage 4). "
                                       "Run scripts/evaluate.py when it is available."}, status_code=501)

    # ------------------------------------------------------------------ frontend
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "frontend" / "index.html", media_type="text/html")

    return app
