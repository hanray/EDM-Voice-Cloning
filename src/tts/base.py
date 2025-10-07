from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class TTSEngine(ABC):
    @abstractmethod
    def synth_to_wav(self, text: str, out_path: Path, speed: float = 1.0) -> Path:
        ...