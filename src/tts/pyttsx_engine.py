from pathlib import Path
from typing import Optional
import pyttsx3
from .base import TTSEngine

class PyttsxEngine(TTSEngine):
    def __init__(self):
        self.engine = pyttsx3.init()

    def synth_to_wav(self, text: str, out_path: Path, speed: float = 1.0) -> Path:
        # pyttsx3 rate control (approx)
        base_rate = self.engine.getProperty('rate') or 200
        self.engine.setProperty('rate', int(base_rate * speed))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine.save_to_file(text, str(out_path))
        self.engine.runAndWait()
        return out_path