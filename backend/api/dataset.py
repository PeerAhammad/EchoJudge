"""Dataset manifest access (data/manifest.jsonl). Schema documented in docs/dataset.md (Stage 4 expands it).

One JSON object per line, e.g.:
{"id": "p01_ref", "passage_id": "p01", "speaker_id": "spk1", "role": "reference"|"participant",
 "reference_id": "p01_ref" (participants only), "audio_path": "reference/p01_ref.wav", "transcript": "...",
 "flaw_labels": [{"category": "...", "start_s": 0, "end_s": 1, "severity": 1, "annotation_method":
                  "injected_synthetic"|"human"|"self_reported"}], "synthetic": false,
 "license": "...", "provenance": "...", "split": "train|val|test"}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def load_manifest(data_dir: Path) -> dict[str, dict]:
    p = Path(data_dir) / "manifest.jsonl"
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and ID_RE.match(str(rec.get("id", ""))):
            out[rec["id"]] = rec
    return out


def resolve_audio(data_dir: Path, rec: dict) -> Path | None:
    """Audio path inside data_dir only (no path traversal)."""
    root = Path(data_dir).resolve()
    p = (root / str(rec.get("audio_path", ""))).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    return p if p.is_file() else None


def public_view(rec: dict) -> dict:
    """Metadata safe for clients (no server paths)."""
    return {k: v for k, v in rec.items() if k != "audio_path"}
