"""
Diff-SVC integration scaffold.
Currently a placeholder that passes audio through; hook in a real Diff-SVC model here.
"""

from pathlib import Path
from typing import Optional, Tuple
import logging
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

class DiffSVCConverter:
    """Placeholder Diff-SVC converter.

    Replace `convert` with actual Diff-SVC inference when the model and dependencies
    are available. This class currently normalizes and passes audio through.
    """

    def __init__(self, model_path: Optional[Path] = None, device: str = "auto"):
        self.model_path = Path(model_path) if model_path else None
        self.device = device

    def is_ready(self) -> bool:
        """Return True if a model path is provided; extend with real checks later."""
        return self.model_path is not None and self.model_path.exists()

    def convert(
        self,
        source_audio_path: Path,
        output_path: Path,
        speaker: Optional[str] = None,
        diffusion_steps: int = 30,
        denoise: float = 0.6,
    ) -> Tuple[Path, int]:
        """Pass-through conversion; replace with Diff-SVC inference.

        Args:
            source_audio_path: path to source vocal (wav)
            output_path: where to save converted audio
            speaker: optional speaker ID/embedding name
            diffusion_steps: planned hook for quality/speed tradeoff
            denoise: planned hook for denoising strength
        """
        logger.warning("Diff-SVC model not wired yet; returning original audio.")
        audio, sr = sf.read(source_audio_path)

        # Simple safety normalization to avoid clipping
        peak = np.max(np.abs(audio)) or 1.0
        audio = audio / max(peak, 1e-6)
        sf.write(output_path, audio, sr)
        return output_path, sr
