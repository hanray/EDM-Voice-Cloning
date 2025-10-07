# src/api/server.py
from pathlib import Path
import time
import threading
import io
import uuid
from typing import Literal, Optional, Dict, Any, Tuple

import numpy as np
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Project-level config & engines
from ..config import OUTPUT_DIR

# Try to import voice cloning components
try:
    from ..audio.chatterbox_voice_cloning import (
        convert_voice_to_reference,
        clone_voice_from_text,
        VoiceCloningSettings,
        is_voice_cloning_available
    )
    VOICE_CLONING_AVAILABLE = True
    print("✅ Neural voice cloning available")
except ImportError as e:
    print(f"⚠️  Voice cloning not available: {e}")
    VOICE_CLONING_AVAILABLE = False
    
    def convert_voice_to_reference(*args, **kwargs):
        return None
    
    def clone_voice_from_text(*args, **kwargs):
        return None
        
    def is_voice_cloning_available():
        return False

# Simplified audio processing
try:
    from ..audio.export import to_mp3
except ImportError:
    def to_mp3(input_path, output_path):
        """Fallback - just copy the file"""
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path


# -------------------------
# App + CORS
# -------------------------
app = FastAPI(title="EDM Vocal Generator API - Neural Voice Cloning")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reference cache for uploaded reference clips - store in memory to save disk space
# id -> {"audio_data": np.ndarray, "sr": int, "ts": float, "profile": dict}
REF_CACHE: Dict[str, Dict[str, Any]] = {}


# -------------------------
# API router (prefix /api)
# -------------------------
api = APIRouter(prefix="/api")


# -------------------------
# Models
# -------------------------
class SynthesisRequest(BaseModel):
    text: str
    voice: str
    speed: float = 1.0
    pitch: float = 0.0

    performance_mode: Literal["normal", "chant", "rap", "ballad", "aggressive"] = "normal"

    # Producer features
    singing_mode: bool = False
    autotune: bool = False
    doubles: bool = False
    bpm: int = 120

    # FX
    reverb_mix: float = 0.0
    reverb_size: float = 0.5
    delay_mix: float = 0.0
    delay_sync: bool = True

    # Character
    warmth: float = 0.0
    presence: float = 0.0
    width: float = 0.5

    # Quality
    sample_rate: int = 48000

    # Humanization (style/prosody)
    humanize: bool = True
    humanization_intensity: float = 0.7
    reference_profile: Optional[Dict[str, Any]] = None  # from /analyze_reference
    reference_adapt: bool = False                       # adapt prosody/tone to reference

    # Voice conversion (Seed-VC)
    timbre_clone: bool = False
    clone_strength: float = 0.8                         # 0..1 blend
    clone_quality: str = "balanced"                     # "fast", "balanced", "high"
    reference_id: Optional[str] = None                  # id returned by /analyze_reference

    format: Literal["wav", "mp3"] = "wav"


# -------------------------
# Utility endpoints
# -------------------------
@api.get("/health")
def health():
    seedvc_status = is_seedvc_available()
    return {
        "ok": True, 
        "seedvc_available": seedvc_status,
        "voice_conversion": "enabled" if seedvc_status else "disabled"
    }


@api.get("/voices")
def voices():
    available_voices = list(vm.engine_factories.keys())
    return {"voices": available_voices, "speakers": COQUI_SPEAKERS}


# -------------------------
# Reference analysis
# -------------------------
@api.post("/analyze_reference")
async def analyze_reference_endpoint(file: UploadFile = File(...)):
    """
    Accept an audio file, store it in memory, and return a compact analysis profile + reference_id.
    Response: { success: true, profile: {...}, ref_id: "..." }
    """
    try:
        data = await file.read()

        # Try libsndfile (wav/flac/aiff/etc.)
        try:
            y, sr = sf.read(io.BytesIO(data), always_2d=False)
            if isinstance(y, np.ndarray) and y.ndim > 1:
                y = y.mean(axis=1)
        except Exception:
            # Fallback for mp3/ogg/etc.
            y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)

        # Generate reference ID
        ref_id = str(uuid.uuid4())[:8]
        
        # Ensure we have valid audio data
        if len(y) == 0:
            raise Exception("Audio file appears to be empty or corrupted")

        # Analyze the reference audio
        analyzer = ReferenceAnalyzer()
        profile = analyzer.analyze_reference(np.asarray(y, dtype=np.float32), sr)

        # Store in memory cache (more efficient than disk storage)
        REF_CACHE[ref_id] = {
            "audio_data": y.astype(np.float32), 
            "sr": sr, 
            "ts": time.time(), 
            "profile": profile
        }
        
        return {"success": True, "profile": profile, "ref_id": ref_id}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


# -------------------------
# Performance Mode helpers
# -------------------------
def apply_performance_mode(text: str, mode: str, bpm: int) -> Tuple[str, Dict[str, Any]]:
    settings: Dict[str, Any] = {}
    if mode == "chant":
        words = text.split()
        text = " ".join([f"{w}." if i % 2 == 0 else w for i, w in enumerate(words)])
        settings = {"speed_modifier": 0.9, "consonant_boost": 0.3, "rhythm_emphasis": True}
    elif mode == "rap":
        words = text.split()
        text = " ".join([f"{w}," if len(w) > 3 else w for w in words])
        settings = {"speed_modifier": 1.2, "consonant_boost": 0.5, "attack_sharpening": 0.4}
    elif mode == "ballad":
        text = text.replace(",", "...").replace(".", "...")
        settings = {"speed_modifier": 0.8, "legato_emphasis": 0.4, "pitch_variation": 0.8}
    elif mode == "aggressive":
        words = text.split()
        text = " ".join([f"{w}!" if len(w) > 4 else w for w in words])
        settings = {"speed_modifier": 1.1, "consonant_boost": 0.6, "attack_sharpening": 0.6}
    return text, settings


# -------------------------
# Main synthesis
# -------------------------
@api.post("/synthesize")
def synth(req: SynthesisRequest):
    # 1) Validate
    if not req.text.strip():
        return JSONResponse({"error": "Empty text"}, status_code=400)
    if req.voice not in vm.engine_factories:
        available = list(vm.engine_factories.keys())
        return JSONResponse(
            {"error": f"Voice '{req.voice}' not available. Available voices: {available}"},
            status_code=400,
        )

    # Create job
    job_id = vm.create_job()

    try:
        from datetime import datetime
        base = OUTPUT_DIR / f"vocal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id}"
        raw_wav = base.with_suffix(".raw.wav")     # TTS output (dry)
        stage_wav = base.with_suffix(".stage.wav") # after timbre clone (if any)
        fx_wav = base.with_suffix(".wav")          # after effects/humanization

        # 2) Performance preprocessing
        vm.update_job_status(job_id, "preprocessing", 15)
        processed_text, perf = apply_performance_mode(req.text, req.performance_mode, req.bpm)

        # 3) TTS (dry)
        vm.update_job_status(job_id, "synthesizing", 30)
        vm.synth(
            req.voice,
            processed_text,
            raw_wav,
            speed=req.speed * perf.get("speed_modifier", 1.0),
            job_id=job_id,
            performance_mode=req.performance_mode,
        )

        # 4) Voice conversion (Seed-VC) BEFORE effects
        vm.update_job_status(job_id, "converting_voice", 55)
        in_for_fx = raw_wav
        
        if req.timbre_clone and req.reference_id:
            ref_entry = REF_CACHE.get(req.reference_id)
            if ref_entry and "audio_data" in ref_entry:
                try:
                    # Create temporary reference file for voice conversion
                    temp_ref_path = OUTPUT_DIR / f"temp_ref_{req.reference_id}.wav"
                    sf.write(str(temp_ref_path), ref_entry["audio_data"], ref_entry["sr"])
                    
                    # Attempt Seed-VC voice conversion
                    conversion_success = apply_seedvc_conversion(
                        source_wav=str(raw_wav),
                        reference_wav=str(temp_ref_path),
                        output_wav=str(stage_wav),
                        strength=float(np.clip(req.clone_strength, 0.0, 1.0)),
                        quality=req.clone_quality or DEFAULT_SEEDVC_QUALITY
                    )
                    
                    # Clean up temporary file
                    try:
                        temp_ref_path.unlink(missing_ok=True)
                    except:
                        pass  # Ignore cleanup errors
                    
                    if conversion_success:
                        in_for_fx = stage_wav
                        vm.update_job_status(job_id, "voice_converted", 65)
                    else:
                        # Fallback: copy raw file if Seed-VC fails
                        copy_file_as_fallback(str(raw_wav), str(stage_wav))
                        in_for_fx = stage_wav
                        vm.update_job_status(job_id, "conversion_fallback", 65)
                        print(f"[server] Voice conversion failed, using original audio")
                        
                except Exception as conv_err:
                    # Fallback: copy raw file if error occurs
                    copy_file_as_fallback(str(raw_wav), str(stage_wav))
                    in_for_fx = stage_wav
                    print(f"[server] Voice conversion error: {conv_err}, using original audio")
            else:
                # No valid reference, proceed with original
                vm.update_job_status(job_id, "no_reference", 60)

        # 5) FX + Humanization (prosody/style)
        vm.update_job_status(job_id, "applying_effects", 75)
        enhanced_cfg = EnhancedEffectSettings(
            pitch=req.pitch,
            speed=1.0 if req.voice != "Robotic" else req.speed,
            reverb_mix=req.reverb_mix,
            delay_mix=req.delay_mix,
            compress=True,
            # Producer
            singing_mode=req.singing_mode,
            autotune=req.autotune,
            doubles=req.doubles,
            bpm=req.bpm,
            reverb_size=req.reverb_size,
            delay_sync=req.delay_sync,
            warmth=req.warmth,
            presence=req.presence,
            width=req.width,
            sample_rate=req.sample_rate,
            # Performance
            performance_mode=req.performance_mode,
            performance_settings=perf,
            # Humanization (style)
            humanize=req.humanize,
            humanization_intensity=req.humanization_intensity,
            reference_profile=req.reference_profile,
            reference_adapt=req.reference_adapt,
        )
        process_vocal_wav(in_for_fx, fx_wav, enhanced_cfg)

        # 6) Optional MP3
        final_path = fx_wav
        if req.format == "mp3":
            vm.update_job_status(job_id, "converting_mp3", 90)
            mp3_path = base.with_suffix(".mp3")
            to_mp3(fx_wav, mp3_path)
            final_path = mp3_path

        vm.update_job_status(job_id, "completed", 100)
        media_type = "audio/mpeg" if req.format == "mp3" else "audio/wav"
        return FileResponse(final_path, media_type=media_type, filename=final_path.name)

    except Exception as e:
        vm.update_job_status(job_id, "failed", 0)
        return JSONResponse({"error": str(e), "job_id": job_id}, status_code=500)
    finally:
        def delayed_cleanup():
            time.sleep(2)
            vm.cleanup_job(job_id)
        threading.Thread(target=delayed_cleanup, daemon=True).start()


# -------------------------
# Job tracking passthrough
# -------------------------
@api.get("/jobs/{job_id}/status")
def get_job_status(job_id: str):
    return vm.get_job_status(job_id)


@api.get("/jobs/active")
def get_active_jobs():
    return {"active_jobs": list(vm.active_jobs.keys()), "count": len(vm.active_jobs)}


# -------------------------
# Seed-VC Status endpoint
# -------------------------
@api.get("/seedvc/status")
def seedvc_status():
    """Check Seed-VC availability and provide installation info"""
    available = is_seedvc_available()
    return {
        "available": available,
        "status": "ready" if available else "not_installed",
        "message": "Seed-VC is ready for voice conversion" if available else 
                  "Seed-VC not found. Install with: pip install -r requirements_seedvc.txt",
        "features": {
            "voice_conversion": available,
            "zero_shot": available,
            "real_time": available
        }
    }


# Register router BEFORE static
app.include_router(api)


# -------------------------
# Serve static web UI
# -------------------------
ROOT = Path(__file__).resolve().parents[2]
UI_DIR = ROOT / "ui-web"
UI_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
