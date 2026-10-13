"""In-memory job store + single-worker executor (analysis is CPU-heavy: one job at a time)."""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    id: str
    dir: Path
    status: str = "queued"          # queued | running | done | error | timeout
    created: float = field(default_factory=time.time)
    started: float | None = None
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    files: dict[str, Path] = field(default_factory=dict)   # {"reference": path, "participant": path}

    def say(self, msg: str):
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")


class JobStore:
    def __init__(self, root: Path, retention_minutes: float, max_processing_s: float):
        self.root, self.retention, self.max_s = Path(root), retention_minutes * 60, max_processing_s
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.cleanup()

    def new(self) -> Job:
        self.cleanup()
        jid = uuid.uuid4().hex
        d = self.root / jid
        d.mkdir(parents=True)
        job = Job(jid, d)
        with self.lock:
            self.jobs[jid] = job
        return job

    def get(self, jid: str) -> Job | None:
        job = self.jobs.get(jid)
        if job and job.status == "running" and job.started and time.time() - job.started > self.max_s:
            job.status, job.error = "timeout", "Processing took too long and was abandoned."
        return job

    def submit(self, job: Job, fn):
        def run():
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(job)
                if job.status == "running":
                    job.status = "done"
            except Exception as e:  # message must be safe: only AudioError text is passed through
                from backend.audio.io import AudioError
                job.error = str(e) if isinstance(e, AudioError) else "Analysis failed (internal error)."
                job.status = "error"
                if not isinstance(e, AudioError):
                    import logging
                    logging.getLogger("echojudge").exception("analysis failed")
        self.pool.submit(run)

    def delete(self, jid: str) -> bool:
        with self.lock:
            job = self.jobs.pop(jid, None)
        shutil.rmtree(self.root / jid, ignore_errors=True)
        return job is not None

    def cleanup(self):
        now = time.time()
        for d in self.root.iterdir() if self.root.exists() else []:
            try:
                if d.is_dir() and now - d.stat().st_mtime > self.retention:
                    shutil.rmtree(d, ignore_errors=True)
                    self.jobs.pop(d.name, None)
            except OSError:
                pass
