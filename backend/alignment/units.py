"""Comparable units between reference and participant: (ref interval, participant interval)."""
from __future__ import annotations

from dataclasses import dataclass

from backend.alignment.dtw import DTWMap
from backend.features.extract import FeatureSet
from backend.features.stats import pause_time_in
from backend.transcription.types import Word


@dataclass
class Unit:
    ref_start: float
    ref_end: float
    part_start: float
    part_end: float
    label: str
    n_words: int
    source: str  # "words" | "dtw"
    ref_pause: float = 0.0   # silent-pause seconds inside the reference interval
    part_pause: float = 0.0  # silent-pause seconds inside the participant interval

    @property
    def ref_speech(self) -> float:
        return max((self.ref_end - self.ref_start) - self.ref_pause, 1e-3)

    @property
    def part_speech(self) -> float:
        return max((self.part_end - self.part_start) - self.part_pause, 1e-3)

    @property
    def speed_ratio(self) -> float:
        """>1: participant covered the same content faster than the reference.
        Pauses are excluded on both sides so pacing and pause flaws are not double-counted."""
        return self.ref_speech / self.part_speech


def _pauses(fs, a, b) -> float:
    return pause_time_in(fs, a, b)[0] if fs is not None else 0.0


def units_from_words(pairs, ref_words: list[Word], part_words: list[Word], cfg: dict,
                     ref_fs: FeatureSet | None = None, part_fs: FeatureSet | None = None) -> list[Unit]:
    a = cfg["alignment"]
    pm = dict(pairs)
    chunks: list[list[int]] = []
    cur: list[int] = []
    for i, w in enumerate(ref_words):
        if cur and (w.start - ref_words[i - 1].end >= a["chunk_gap_s"] or len(cur) >= a["chunk_max_words"]):
            chunks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)
    units = []
    for ch in chunks:
        m = [i for i in ch if i in pm]
        # Split at insertions/deletions: consecutive matched words must advance equally in reference and
        # participant (substitutions advance both equally). Otherwise the participant's extra/missing words
        # would inflate or shrink the duration being compared.
        runs, cur = [], []
        for i in m:
            if cur and (i - cur[-1]) != (pm[i] - pm[cur[-1]]):
                runs.append(cur)
                cur = []
            cur.append(i)
        if cur:
            runs.append(cur)
        for run in runs:
            span = run[-1] - run[0] + 1
            if len(run) < a["chunk_min_matched"] or len(run) / span < a["chunk_min_matched_fraction"]:
                continue
            r0, r1 = ref_words[run[0]].start, ref_words[run[-1]].end
            p0, p1 = part_words[pm[run[0]]].start, part_words[pm[run[-1]]].end
            if r1 - r0 < a["unit_min_ref_s"] or p1 <= p0:
                continue
            label = f"{ref_words[run[0]].text} ... {ref_words[run[-1]].text}"
            units.append(Unit(r0, r1, p0, p1, label, len(run), "words", _pauses(ref_fs, r0, r1), _pauses(part_fs, p0, p1)))
    return units


def units_from_dtw(ref_fs: FeatureSet, dmap: DTWMap, cfg: dict, part_fs: FeatureSet | None = None) -> list[Unit]:
    mu = cfg["alignment"]["dtw"]["min_unit_s"]
    out = []
    for s, e in ref_fs.segments:
        if e - s < mu:
            continue
        p0, p1 = dmap.map(s), dmap.map(e)
        if p1 > p0:
            out.append(Unit(s, e, p0, p1, f"reference segment {s:.1f}-{e:.1f}s", 0, "dtw", _pauses(ref_fs, s, e), _pauses(part_fs, p0, p1)))
    return out
