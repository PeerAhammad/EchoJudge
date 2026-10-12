"""Typed result schemas shared by detection, scoring, API and evaluation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

CATEGORIES = (
    "rushed_pacing", "slow_pacing", "excessive_pauses", "monotone_delivery",
    "weak_emphasis", "fluency_interruption", "signal_quality",
)


def fmt_time(t: float) -> str:
    """Seconds -> MM:SS.s"""
    m, s = divmod(max(t, 0.0), 60.0)
    return f"{int(m):02d}:{s:04.1f}"


class FlawEvent(BaseModel):
    category: str
    source: str = "participant"            # "participant" | "reference" (quality events only)
    start_s: float
    end_s: float
    severity: int = Field(ge=1, le=4)      # 1 (mild) .. 4 (severe); band edges in config
    detection_method: str
    participant_value: float | None = None
    reference_value: float | None = None
    threshold: float | None = None
    unit: str = ""
    ratio: float | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    explanation: str
    recommendation: str
    # Rule-based detections are NOT calibrated probabilities; kept explicit on purpose.
    calibrated_confidence: float | None = None
    basis: str = "rule-based threshold (uncalibrated)"
    scored: bool = True                    # signal-quality events are reported, never scored
    experimental: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def span_text(self) -> str:
        return f"{fmt_time(self.start_s)}-{fmt_time(self.end_s)}"
