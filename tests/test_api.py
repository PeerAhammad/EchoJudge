import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.transcription.types import Transcript
from synth import SR, baseline, fake_words, synth_speech


class SeqASR:
    """Returns the queued word lists in call order (reference first, then participant)."""
    def __init__(self, lists):
        self.lists, self.i, self.log = lists, 0, lambda m: None

    def transcribe(self, y, sr, prompt=None):
        w = self.lists[self.i % len(self.lists)]
        self.i += 1
        return Transcript(list(w), "en", "fake", "fake", "cpu", [])


def wait(client, aid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/results/{aid}").json()
        if r["status"] in ("done", "error", "timeout"):
            return r
        time.sleep(0.2)
    raise AssertionError("job did not finish")


@pytest.fixture()
def env(tmp_path):
    ref, tr = synth_speech([(18, 4.5)] * 4, [0.5] * 3, seed=1)
    part, tp = synth_speech([(18, 4.5), (18, 7.2), (18, 7.2), (18, 4.5)], [0.5] * 3, seed=3)
    d = tmp_path / "data"
    (d / "reference").mkdir(parents=True)
    (d / "participant").mkdir()
    sf.write(d / "reference" / "r1.wav", ref, SR)
    sf.write(d / "participant" / "p1.wav", part, SR)
    man = [
        {"id": "r1", "passage_id": "x", "speaker_id": "s", "role": "reference", "audio_path": "reference/r1.wav",
         "transcript": "w0 w1", "synthetic": True},
        {"id": "p1", "passage_id": "x", "speaker_id": "s", "role": "participant", "reference_id": "r1",
         "audio_path": "participant/p1.wav", "synthetic": True, "description": "test",
         "flaw_labels": [{"category": "rushed_pacing", "start_s": 5.1, "end_s": 10.56, "severity": 2,
                          "annotation_method": "injected_synthetic"}]},
        {"id": "evil", "role": "reference", "audio_path": "../../etc/passwd"},
    ]
    (d / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in man), encoding="utf-8")
    app = create_app(data_dir=d, asr_factory=lambda cfg: SeqASR([fake_words(tr), fake_words(tp)]),
                     config_overrides={"runtime": {"upload_dir": str(tmp_path / "up")}})
    return TestClient(app), d, (ref, part), tmp_path


def upload(client, ref_path, part_path, **form):
    with open(ref_path, "rb") as a, open(part_path, "rb") as b:
        return client.post("/api/analyze", files={"reference": ("my private name.wav", a, "audio/wav"),
                                                  "participant": ("x.wav", b, "audio/wav")}, data=form)


def test_health_and_index(env):
    c, *_ = env
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/").status_code == 200


def test_analyze_signal_only_end_to_end(env):
    c, d, _, tmp = env
    r = upload(c, d / "reference" / "r1.wav", d / "participant" / "p1.wav", asr="none")
    assert r.status_code == 202
    aid = r.json()["analysis_id"]
    res = wait(c, aid)
    assert res["status"] == "done", res
    assert res["result"]["scores"]["overall"] is not None and res["result"]["events"] is not None
    stored = [p.name for p in (tmp / "up" / aid).iterdir()]
    assert all("private" not in n for n in stored)          # client filenames never used
    assert c.get(f"/api/audio/{aid}/participant").status_code == 200
    assert c.delete(f"/api/results/{aid}").json()["deleted"]
    assert c.get(f"/api/results/{aid}").status_code == 404
    assert not (tmp / "up" / aid).exists()


def test_analyze_with_asr_uses_word_alignment(env):
    c, d, *_ = env
    aid = upload(c, d / "reference" / "r1.wav", d / "participant" / "p1.wav", asr="auto").json()["analysis_id"]
    res = wait(c, aid)
    assert res["status"] == "done" and res["result"]["alignment_method"] == "asr_words+text_diff"
    assert any(e["category"] == "rushed_pacing" for e in res["result"]["events"])
    assert res["result"]["participant"]["rate_series"]


def test_bad_inputs_rejected_cleanly(env, tmp_path):
    c, d, *_ = env
    bad = tmp_path / "bad.txt"
    bad.write_text("hi")
    with open(bad, "rb") as f:
        r = c.post("/api/analyze", files={"reference": ("a.txt", f, "text/plain"), "participant": ("b.wav", b"x", "audio/wav")})
    assert r.status_code == 400
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"this is not audio" * 50)
    aid = upload(c, junk, junk, asr="none").json()["analysis_id"]
    res = wait(c, aid)
    assert res["status"] == "error" and str(tmp_path) not in json.dumps(res) and "Traceback" not in json.dumps(res)
    assert upload(c, junk, junk, asr="bogus").status_code == 400
    assert c.get("/api/results/not-an-id").status_code == 404


def test_dataset_endpoints_and_sample_analysis(env):
    c, d, *_ = env
    items = c.get("/api/dataset").json()["items"]
    assert {i["id"] for i in items} >= {"r1", "p1"}
    assert "audio_path" not in json.dumps(c.get("/api/dataset/p1").json())
    assert c.get("/api/dataset/..%2F..%2Fx").status_code in (404, 422)
    assert c.get("/api/dataset/evil/audio").status_code == 404   # traversal blocked
    r = c.post("/api/analyze_sample", json={"participant_id": "p1", "asr": "auto"})
    assert r.status_code == 202
    res = wait(c, r.json()["analysis_id"])
    assert res["status"] == "done" and res["result"]["ground_truth"]["labels"][0]["category"] == "rushed_pacing"
    assert c.post("/api/analyze_sample", json={"participant_id": "r1"}).status_code == 404


def test_evaluate_is_honestly_unimplemented(env):
    c, *_ = env
    assert c.post("/api/evaluate").status_code == 501
