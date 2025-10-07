import re
from typing import List

_punct = re.compile(r"\s+|[\t\n\r]+")

def clean_text(text: str) -> str:
    # Minimal normalization; extend with phonemizer later
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t

def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p]