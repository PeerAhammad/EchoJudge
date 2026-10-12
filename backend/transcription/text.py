from __future__ import annotations

import re


def normalize_token(w: str) -> str:
    w = w.lower().replace("\u2019", "'")
    w = re.sub(r"[^\w']+", "", w, flags=re.UNICODE)
    return w.strip("'")


def tokenize(text: str) -> list[str]:
    return [t for t in (normalize_token(x) for x in text.split()) if t]
