# src/tts/coqui_engine.py
from pathlib import Path
from TTS.api import TTS
import numpy as np
import soundfile as sf

class CoquiEngine:
    def __init__(self):
        self.xtts = None  # Lazy load
        self.device = "cpu"  # Change to "cuda" if you have GPU
        
    def _load_xtts(self):
        """Load XTTS model for voice cloning"""
        if self.xtts is None:
            print("[Coqui] Loading XTTS v2 for voice cloning (this may take a minute)...")
            self.xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            print("[Coqui] XTTS loaded successfully")
        return self.xtts
    
    def synth_with_clone(self, text: str, out_path: Path, reference_wav: Path, speed: float = 1.0):
        """Generate speech using voice cloning"""
        model = self._load_xtts()
        
        # XTTS does actual voice cloning - generates text in the reference voice
        model.tts_to_file(
            text=text,
            file_path=str(out_path),
            speaker_wav=str(reference_wav),
            language="en",
            split_sentences=True  # Better for longer texts
        )
        
        # Apply speed adjustment if needed
        if abs(speed - 1.0) > 0.01:
            import librosa
            y, sr = sf.read(str(out_path))
            y_stretched = librosa.effects.time_stretch(y, rate=speed)
            sf.write(str(out_path), y_stretched, sr)
            
        return out_path
    
    def synth_to_wav(self, text: str, out_path: Path, speed: float = 1.0):
        """Standard TTS without voice cloning"""
        model = self._load_xtts()
        
        # Use default XTTS voice
        model.tts_to_file(
            text=text,
            file_path=str(out_path),
            language="en"
        )
        
        if abs(speed - 1.0) > 0.01:
            import librosa
            y, sr = sf.read(str(out_path))
            y_stretched = librosa.effects.time_stretch(y, rate=speed)
            sf.write(str(out_path), y_stretched, sr)
            
        return out_path