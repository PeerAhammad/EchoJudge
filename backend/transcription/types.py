from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    norm: str
    start: float
    end: float
    prob: float | None = None

    def to_dict(self) -> dict:
        return {"text": self.text, "norm": self.norm, "start_s": round(self.start, 3),
                "end_s": round(self.end, 3), "prob": None if self.prob is None else round(self.prob, 3)}


@dataclass
class Transcript:
    words: list[Word]
    language: str | None = None
    backend: str = "unknown"
    model: str | None = None
    device: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)
