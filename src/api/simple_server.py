# src/api/simple_server.py
"""
Simplified FastAPI server focused on neural voice cloning with chatterbox-tts
"""

from pathlib import Path
import time
import io
import uuid
import tempfile
import logging
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Voice cloning imports
try:
    from ..audio.chatterbox_voice_cloning import (
        ChatterboxVoiceCloner,
        get_voice_cloner,
        clone_voice_simple,
        test_voice_cloning
    )
    VOICE_CLONING_AVAILABLE = True
    logging.info("✅ Neural voice cloning available")
    
    # Initialize voice cloner
    voice_cloner = get_voice_cloner()
    if voice_cloner and voice_cloner.is_available():
        logging.info("✅ Voice cloner initialized")
    else:
        logging.warning("⚠️ Voice cloner initialization failed")
        VOICE_CLONING_AVAILABLE = False
        voice_cloner = None
    
except ImportError as e:
    logging.warning(f"⚠️  Voice cloning not available: {e}")
    VOICE_CLONING_AVAILABLE = False
    voice_cloner = None

# -------------------------
# App Setup
# -------------------------
app = FastAPI(title="Neural Voice Cloning API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for the web UI)
try:
    app.mount("/static", StaticFiles(directory="ui-web"), name="static")
except Exception:
    pass  # Static files optional

# -------------------------
# Data Models
# -------------------------

class VoiceCloneRequest(BaseModel):
    text: str
    reference_id: Optional[str] = None
    
    # Neural voice cloning settings
    quality: str = "balanced"  # "fast", "balanced", "high"
    similarity_boost: float = 0.8
    diversity: float = 0.3
    optimize_short_text: bool = True
    
    # BPM and timing settings for EDM vocals
    bpm: Optional[float] = None  # Target BPM for vocal timing
    sync_to_beat: bool = False   # Align vocal phrases to beat grid
    quantize_timing: bool = False  # Quantize vocal timing to musical grid
    
    # Audio processing settings
    add_reverb: bool = False
    reverb_amount: float = 0.3
    add_delay: bool = False
    delay_time: float = 0.125  # Delay time in seconds (1/8 note at 120 BPM)

class VoiceConversionRequest(BaseModel):
    source_audio_id: str
    reference_id: str
    quality: str = "balanced"
    similarity_boost: float = 0.8
    
    # BPM settings for conversion
    bpm: Optional[float] = None
    sync_to_beat: bool = False

# -------------------------
# Storage
# -------------------------

# In-memory storage for uploaded audio files
# Format: {id: {"audio": np.ndarray, "sr": int, "timestamp": float}}
audio_storage: Dict[str, Dict[str, Any]] = {}

def cleanup_old_files():
    """Remove files older than 1 hour"""
    current_time = time.time()
    to_remove = []
    for file_id, data in audio_storage.items():
        if current_time - data["timestamp"] > 3600:  # 1 hour
            to_remove.append(file_id)
    
    for file_id in to_remove:
        del audio_storage[file_id]
        logging.info(f"Cleaned up old file: {file_id}")

# -------------------------
# Endpoints
# -------------------------

@app.get("/")
async def root():
    """Serve the main web interface"""
    return FileResponse("ui-web/index.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    cleanup_old_files()
    
    return {
        "status": "healthy",
        "voice_cloning_available": VOICE_CLONING_AVAILABLE,
        "stored_files": len(audio_storage),
        "timestamp": time.time()
    }

@app.post("/api/upload_reference")
async def upload_reference(file: UploadFile = File(...)):
    """Upload a reference audio file for voice cloning"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Read file data
        data = await file.read()
        
        # Load audio
        audio, sr = sf.read(io.BytesIO(data))
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Generate unique ID and file path
        file_id = str(uuid.uuid4())
        temp_path = Path(tempfile.mktemp(suffix=f"_{file.filename}"))
        
        # Save file to disk for Gradio client
        with open(temp_path, "wb") as f:
            f.write(data)
        
        # Store in memory with file path
        audio_storage[file_id] = {
            "audio": audio,
            "sr": sr,
            "timestamp": time.time(),
            "filename": file.filename,
            "file_path": str(temp_path)
        }
        
        cleanup_old_files()
        
        logging.info(f"Uploaded reference audio: {file.filename} -> {file_id}")
        
        return {
            "success": True,
            "reference_id": file_id,
            "filename": file.filename,
            "duration": len(audio) / sr,
            "sample_rate": sr
        }
        
    except Exception as e:
        logging.error(f"Failed to upload reference: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload an audio file (alias for upload_reference for UI compatibility)"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Read file data
        data = await file.read()
        
        # Load audio
        audio, sr = sf.read(io.BytesIO(data))
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Generate unique ID and file path
        file_id = str(uuid.uuid4())
        temp_path = Path(tempfile.mktemp(suffix=f"_{file.filename}"))
        
        # Save file to disk for Gradio client
        with open(temp_path, "wb") as f:
            f.write(data)
        
        # Store in memory with file path
        audio_storage[file_id] = {
            "audio": audio,
            "sr": sr,
            "timestamp": time.time(),
            "filename": file.filename,
            "file_path": str(temp_path)
        }
        
        cleanup_old_files()
        
        logging.info(f"Uploaded audio: {file.filename} -> {file_id}")
        
        return {
            "success": True,
            "id": file_id,  # UI expects 'id' not 'reference_id'
            "filename": file.filename,
            "duration": len(audio) / sr,
            "sample_rate": sr
        }
        
    except Exception as e:
        logging.error(f"Failed to upload audio: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/get_audio/{audio_id}")
async def get_audio(audio_id: str):
    """Get uploaded audio file by ID"""
    if audio_id not in audio_storage:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    try:
        audio_data = audio_storage[audio_id]
        audio = audio_data["audio"]
        sr = audio_data["sr"]
        
        # Convert to WAV in memory
        output = io.BytesIO()
        sf.write(output, audio, sr, format='WAV')
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=audio_{audio_id}.wav"}
        )
        
    except Exception as e:
        logging.error(f"Failed to get audio {audio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get audio: {str(e)}")

@app.post("/api/clone_voice")
async def clone_voice(request: VoiceCloneRequest):
    """Clone a voice to speak the given text"""
    if not VOICE_CLONING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Voice cloning not available")
    
    if not request.reference_id or request.reference_id not in audio_storage:
        raise HTTPException(status_code=400, detail="Invalid reference ID")
    
    try:
        # Get reference audio file path
        ref_data = audio_storage[request.reference_id]
        ref_audio_path = ref_data["file_path"]
        
        # Perform voice cloning with BPM processing
        logging.info(f"Cloning voice for text: '{request.text[:50]}...'")
        
        # Use actual voice cloning with parameters from request
        # Map quality setting to neural voice parameters
        temperature_adjustment = {
            "fast": 0.2,      # Lower temperature for faster, more predictable results
            "balanced": 0.5,  # Balanced temperature
            "high": 0.8       # Higher temperature for more diverse results
        }.get(request.quality, 0.5)
        
        # Calculate final temperature (combine quality and diversity settings)
        final_temperature = min(1.0, temperature_adjustment + request.diversity * 0.5)
        
        cloned_audio, sample_rate = voice_cloner.clone_voice(
            text=request.text,
            reference_audio_path=ref_audio_path,
            # Map UI parameters to Chatterbox API parameters
            exaggeration=request.similarity_boost,  # Map similarity_boost to exaggeration
            temperature=final_temperature,          # Combine quality and diversity
            seed=0,                                # Keep random for now
            cfg_weight=0.5,                        # Default CFG weight
            optimize_short_text=request.optimize_short_text,  # Text optimization toggle
            # BPM parameters from request
            bpm=request.bpm,
            # Additional BPM processing parameters
            sync_to_beat=request.sync_to_beat,
            quantize_timing=request.quantize_timing,
            add_reverb=request.add_reverb,
            reverb_amount=request.reverb_amount,
            add_delay=request.add_delay,
            delay_time=request.delay_time
        )
        
        if cloned_audio is None:
            raise HTTPException(status_code=500, detail="Voice cloning failed")
        
        # Save to temporary file
        output_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(output_path, cloned_audio, sample_rate)
        
        logging.info(f"Voice cloning completed: {output_path}")
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"cloned_voice_{int(time.time())}.wav"
        )
        
    except Exception as e:
        logging.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cloning failed: {str(e)}")

@app.post("/api/convert_voice")
async def convert_voice(request: VoiceConversionRequest):
    """Convert one voice to sound like another"""
    if not VOICE_CLONING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Voice conversion not available")
    
    if request.source_audio_id not in audio_storage:
        raise HTTPException(status_code=400, detail="Invalid source audio ID")
        
    if request.reference_id not in audio_storage:
        raise HTTPException(status_code=400, detail="Invalid reference ID")
    
    try:
        # Get reference audio file path for voice conversion
        ref_data = audio_storage[request.reference_id]
        ref_audio_path = ref_data["file_path"]
        
        logging.info("Converting voice characteristics...")
        
        # Use voice cloning with default text for voice conversion
        # This is a simplified approach - could be enhanced later
        conversion_text = "This is a voice conversion test."
        
        cloned_audio, sample_rate = voice_cloner.clone_voice(
            text=conversion_text,
            reference_audio_path=ref_audio_path,
            exaggeration=0.5,
            temperature=0.8,
            seed=0,
            cfg_weight=0.5
        )
        
        # Save to temporary file
        output_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(output_path, cloned_audio, sample_rate)
        
        logging.info(f"Voice conversion completed: {output_path}")
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"converted_voice_{int(time.time())}.wav"
        )
        
    except Exception as e:
        logging.error(f"Voice conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

@app.get("/api/stored_files")
async def list_stored_files():
    """List all stored audio files"""
    cleanup_old_files()
    
    files = {}
    for file_id, data in audio_storage.items():
        files[file_id] = {
            "filename": data.get("filename", "unknown"),
            "duration": len(data["audio"]) / data["sr"],
            "sample_rate": data["sr"],
            "timestamp": data["timestamp"]
        }
    
    return {"files": files}

@app.delete("/api/stored_files/{file_id}")
async def delete_stored_file(file_id: str):
    """Delete a stored audio file"""
    if file_id not in audio_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    del audio_storage[file_id]
    logging.info(f"Deleted file: {file_id}")
    
    return {"success": True, "message": f"File {file_id} deleted"}

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    import uvicorn
    import sys
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Neural Voice Cloning API...")
    
    if VOICE_CLONING_AVAILABLE:
        logging.info("✅ Voice cloning is available")
    else:
        logging.warning("⚠️  Voice cloning is NOT available - running in demo mode")
    
    # Check for port argument
    port = 8004
    if len(sys.argv) > 1 and sys.argv[1] == "--port" and len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            port = 8004
    
    uvicorn.run(app, host="0.0.0.0", port=port)