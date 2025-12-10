# src/tts/manager.py
from pathlib import Path
from typing import Dict
import threading
import uuid
import time
from .pyttsx_engine import PyttsxEngine
from .edge_engine import EdgeEngine
from ..config import ROBOTIC_ID

VoiceKey = str

class VoiceManager:
    def __init__(self):
        self.synthesis_lock = threading.Lock()
        self.active_jobs: Dict[str, dict] = {}

        # Add voices here. Keys are what the UI shows; values are Edge voice IDs.
        self.engine_factories = {
            # Baseline
            ROBOTIC_ID: lambda: PyttsxEngine(),

            # English (US/GB/AU)
            "Realistic Male":   lambda: EdgeEngine("en-US-GuyNeural"),
            "Realistic Female": lambda: EdgeEngine("en-US-AriaNeural"),
            "British Male":     lambda: EdgeEngine("en-GB-RyanNeural"),
            "British Female":   lambda: EdgeEngine("en-GB-SoniaNeural"),
            "Australian Male":  lambda: EdgeEngine("en-AU-WilliamNeural"),
            "Australian Female":lambda: EdgeEngine("en-AU-NatashaNeural"),

            # --- Spanish (requested) ---
            "Spanish (ES) Male":    lambda: EdgeEngine("es-ES-AlvaroNeural"),
            "Spanish (ES) Female":  lambda: EdgeEngine("es-ES-ElviraNeural"),
            "Spanish (MX) Male":    lambda: EdgeEngine("es-MX-JorgeNeural"),
            "Spanish (MX) Female":  lambda: EdgeEngine("es-MX-DaliaNeural"),
            "Spanish (US) Male":    lambda: EdgeEngine("es-US-AlonsoNeural"),
            "Spanish (US) Female":  lambda: EdgeEngine("es-US-PalomaNeural"),

            # A few other high-quality free picks
            "French Male":      lambda: EdgeEngine("fr-FR-HenriNeural"),
            "French Female":    lambda: EdgeEngine("fr-FR-DeniseNeural"),
            "German Male":      lambda: EdgeEngine("de-DE-ConradNeural"),
            "German Female":    lambda: EdgeEngine("de-DE-KatjaNeural"),
            "Italian Male":     lambda: EdgeEngine("it-IT-DiegoNeural"),
            "Italian Female":   lambda: EdgeEngine("it-IT-IsabellaNeural"),
            "Portuguese (BR) Male":   lambda: EdgeEngine("pt-BR-AntonioNeural"),
            "Portuguese (BR) Female": lambda: EdgeEngine("pt-BR-FranciscaNeural"),
            "Japanese Male":    lambda: EdgeEngine("ja-JP-KeitaNeural"),
            "Japanese Female":  lambda: EdgeEngine("ja-JP-NanamiNeural"),
        }

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())[:8]
        self.active_jobs[job_id] = {"status": "pending", "start_time": time.time(), "progress": 0}
        return job_id

    def update_job_status(self, job_id: str, status: str, progress: int = None):
        if job_id in self.active_jobs:
            self.active_jobs[job_id]["status"] = status
            if progress is not None:
                self.active_jobs[job_id]["progress"] = progress

    def cleanup_job(self, job_id: str):
        self.active_jobs.pop(job_id, None)

    def get_job_status(self, job_id: str) -> dict:
        return self.active_jobs.get(job_id, {"status": "not_found", "progress": 0})

    def get_available_voices(self) -> list:
        return list(self.engine_factories.keys())

    def synth(self, voice: VoiceKey, text: str, out_path: Path, speed: float = 1.0,
              job_id: str = None, performance_mode: str = "normal") -> Path:
        if voice not in self.engine_factories:
            raise ValueError(f"Voice '{voice}' not available. Available: {self.get_available_voices()}")

        if job_id:
            self.update_job_status(job_id, "starting", 10)

        with self.synthesis_lock:
            try:
                if job_id:
                    self.update_job_status(job_id, "synthesizing", 30)

                engine = self.engine_factories[voice]()

                if job_id:
                    self.update_job_status(job_id, "processing", 70)

                if hasattr(engine, "synth_with_performance_mode"):
                    result = engine.synth_with_performance_mode(
                        text, out_path, speed=speed, performance_mode=performance_mode
                    )
                else:
                    result = engine.synth_to_wav(text, out_path, speed=speed)

                if job_id:
                    self.update_job_status(job_id, "completed", 100)
                return result
            except Exception:
                if job_id:
                    self.update_job_status(job_id, "failed", 0)
                raise
            
def synth_with_voice_clone(self, text: str, out_path: Path, reference_wav: Path, 
                          speed: float = 1.0, job_id: str = None) -> Path:
    """Synthesize using voice cloning"""
    if job_id:
        self.update_job_status(job_id, "loading_voice_model", 20)
    
    with self.synthesis_lock:
        try:
            from .coqui_engine import CoquiEngine
            engine = CoquiEngine()
            
            if job_id:
                self.update_job_status(job_id, "cloning_voice", 40)
            
            result = engine.synth_with_clone(text, out_path, reference_wav, speed)
            
            if job_id:
                self.update_job_status(job_id, "voice_cloned", 80)
            
            return result
        except Exception as e:
            print(f"[VoiceManager] Voice cloning failed: {e}")
            if job_id:
                self.update_job_status(job_id, "clone_failed", 0)
            raise