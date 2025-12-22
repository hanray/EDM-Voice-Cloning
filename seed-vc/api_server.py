import os
import tempfile
import asyncio
from typing import Generator, Optional

import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from hydra.utils import instantiate
from omegaconf import DictConfig
import edge_tts
import librosa
import soundfile as sf

from seed_vc_wrapper import SeedVCWrapper

# Global model holders
vc_wrapper_v1: Optional[SeedVCWrapper] = None
vc_wrapper_v2 = None

# Device / dtype
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

app = FastAPI(title="Seed-VC API", version="0.1")


async def _generate_tts(text: str, voice: str = "en-US-GuyNeural") -> str:
    """Generate speech from text using edge-tts and return a temp file path."""
    if not text or text.strip() == "":
        raise ValueError("Text is required for TTS")
    communicate = edge_tts.Communicate(text, voice or "en-US-GuyNeural")
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    await communicate.save(path)
    return path


def load_v2_models():
    global vc_wrapper_v2
    if vc_wrapper_v2 is not None:
        return vc_wrapper_v2

    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper_v2 = instantiate(cfg)
    vc_wrapper_v2.load_checkpoints()
    vc_wrapper_v2.to(device)
    vc_wrapper_v2.eval()
    vc_wrapper_v2.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    return vc_wrapper_v2


def load_v1_wrapper():
    global vc_wrapper_v1
    if vc_wrapper_v1 is not None:
        return vc_wrapper_v1
    vc_wrapper_v1 = SeedVCWrapper(device=device)
    return vc_wrapper_v1


def _write_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "" )[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


def _apply_bpm_adjustment(audio_path: str, target_bpm: Optional[float]) -> str:
    """Optionally time-stretch the audio to a target BPM in-place."""
    if target_bpm is None:
        return audio_path
    try:
        target_bpm_val = float(target_bpm)
    except (TypeError, ValueError):
        return audio_path
    if target_bpm_val <= 0:
        return audio_path

    updated_path = audio_path
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        source_bpm, _ = librosa.beat.beat_track(y=audio, sr=sr)
        if source_bpm and source_bpm > 0:
            stretch_ratio = source_bpm / target_bpm_val
            stretched = librosa.effects.time_stretch(audio, rate=stretch_ratio)
            try:
                sf.write(audio_path, stretched, sr)
            except Exception:
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                sf.write(temp_path, stretched, sr)
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
                updated_path = temp_path
            print(f"[bpm] adjusted {source_bpm:.2f}->{target_bpm_val:.2f} (x{stretch_ratio:.2f})")
    except Exception as exc:
        print(f"[bpm] skip adjustment: {exc}")

    return updated_path


@app.get("/health")
def health():
    return {"status": "ok"}


def _stream_v1(source_path: str, target_path: str, **kwargs) -> Generator[bytes, None, None]:
    wrapper = load_v1_wrapper()
    try:
        for mp3_bytes, _ in wrapper.convert_voice(source=source_path, target=target_path, stream_output=True, **kwargs):
            if mp3_bytes:
                yield mp3_bytes
    finally:
        for p in (source_path, target_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _stream_v2(source_path: str, target_path: str, **kwargs) -> Generator[bytes, None, None]:
    wrapper = load_v2_models()
    try:
        for mp3_bytes, _ in wrapper.convert_voice_with_streaming(
            source_audio_path=source_path,
            target_audio_path=target_path,
            stream_output=True,
            device=device,
            dtype=dtype,
            **kwargs,
        ):
            if mp3_bytes:
                yield mp3_bytes
    finally:
        for p in (source_path, target_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@app.post("/v1/convert")
async def convert_v1(
    source_audio: UploadFile = File(...),
    target_audio: UploadFile = File(...),
    diffusion_steps: int = Form(10),
    length_adjust: float = Form(1.0),
    inference_cfg_rate: float = Form(0.7),
    f0_condition: bool = Form(False),
    auto_f0_adjust: bool = Form(True),
    pitch_shift: int = Form(0),
    target_bpm: Optional[float] = Form(None),
):
    source_path = _write_upload_to_temp(source_audio)
    target_path = _write_upload_to_temp(target_audio)

    source_path = _apply_bpm_adjustment(source_path, target_bpm)

    stream = _stream_v1(
        source_path=source_path,
        target_path=target_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift,
    )

    return StreamingResponse(stream, media_type="audio/mpeg")


@app.post("/v1/convert_text")
async def convert_v1_text(
    text: str = Form(...),
    target_audio: UploadFile = File(...),
    tts_voice: str = Form("en-US-GuyNeural"),
    diffusion_steps: int = Form(10),
    length_adjust: float = Form(1.0),
    inference_cfg_rate: float = Form(0.7),
    f0_condition: bool = Form(False),
    auto_f0_adjust: bool = Form(True),
    pitch_shift: int = Form(0),
    target_bpm: Optional[float] = Form(None),
):
    # Synthesize text to speech, then run the same v1 pipeline using TTS as source
    try:
        source_path = await _generate_tts(text, tts_voice)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"TTS failed: {e}"})

    source_path = _apply_bpm_adjustment(source_path, target_bpm)
    target_path = _write_upload_to_temp(target_audio)

    stream = _stream_v1(
        source_path=source_path,
        target_path=target_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift,
    )

    return StreamingResponse(stream, media_type="audio/mpeg")


@app.post("/v2/convert")
async def convert_v2(
    source_audio: UploadFile = File(...),
    target_audio: UploadFile = File(...),
    diffusion_steps: int = Form(30),
    length_adjust: float = Form(1.0),
    intelligibility_cfg_rate: float = Form(0.7, alias="intelligebility_cfg_rate"),
    similarity_cfg_rate: float = Form(0.7),
    top_p: float = Form(0.7),
    temperature: float = Form(0.7),
    repetition_penalty: float = Form(1.5),
    convert_style: bool = Form(False),
    anonymization_only: bool = Form(False),
    target_bpm: Optional[float] = Form(None),
):
    source_path = _write_upload_to_temp(source_audio)
    target_path = _write_upload_to_temp(target_audio)

    source_path = _apply_bpm_adjustment(source_path, target_bpm)

    stream = _stream_v2(
        source_path=source_path,
        target_path=target_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        intelligibility_cfg_rate=intelligibility_cfg_rate,
        similarity_cfg_rate=similarity_cfg_rate,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        convert_style=convert_style,
        anonymization_only=anonymization_only,
    )

    return StreamingResponse(stream, media_type="audio/mpeg")


@app.post("/v2/convert_text")
async def convert_v2_text(
    text: str = Form(...),
    target_audio: UploadFile = File(...),
    tts_voice: str = Form("en-US-GuyNeural"),
    diffusion_steps: int = Form(30),
    length_adjust: float = Form(1.0),
    intelligibility_cfg_rate: float = Form(0.7, alias="intelligebility_cfg_rate"),
    similarity_cfg_rate: float = Form(0.7),
    top_p: float = Form(0.7),
    temperature: float = Form(0.7),
    repetition_penalty: float = Form(1.5),
    convert_style: bool = Form(False),
    anonymization_only: bool = Form(False),
    target_bpm: Optional[float] = Form(None),
):
    try:
        source_path = await _generate_tts(text, tts_voice)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"TTS failed: {e}"})

    source_path = _apply_bpm_adjustment(source_path, target_bpm)
    target_path = _write_upload_to_temp(target_audio)

    stream = _stream_v2(
        source_path=source_path,
        target_path=target_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        intelligibility_cfg_rate=intelligibility_cfg_rate,
        similarity_cfg_rate=similarity_cfg_rate,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        convert_style=convert_style,
        anonymization_only=anonymization_only,
    )

    return StreamingResponse(stream, media_type="audio/mpeg")


if __name__ == "__main__":
    # Run with: python api_server.py --host 0.0.0.0 --port 7860
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    uvicorn.run("api_server:app", host=args.host, port=args.port, reload=False)
