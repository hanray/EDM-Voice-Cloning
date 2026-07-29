import io
import json
import os
import re
import tempfile
import zipfile
import asyncio
from typing import Generator, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, Response
import edge_tts
import librosa
import numpy as np
import soundfile as sf

from seed_vc_wrapper import SeedVCWrapper
import audio_utils
import chatterbox_provider
import fx_chain
import job_status
import vocal_cadence

# Global model holder
vc_wrapper_v1: Optional[SeedVCWrapper] = None

# Device / dtype
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

app = FastAPI(title="Seed-VC API", version="0.2")


async def _generate_tts(
    text: str,
    voice: str = "en-US-GuyNeural",
    engine: str = "edge",
    prompt_path: Optional[str] = None,
) -> str:
    """Generate speech from text and return a temp WAV path.

    engine="edge" uses edge-tts (cloud, fast, flat delivery).
    engine="chatterbox" uses the local Chatterbox model, cloning the voice
    in prompt_path (we pass the conversion reference) — expressive input
    means Seed-VC starts much closer to the target.
    """
    if not text or text.strip() == "":
        raise ValueError("Text is required for TTS")
    job_status.set_stage("tts")

    if engine == "chatterbox":
        def _run():
            job_status.adjust_waiting(+1)
            with job_status.gpu_lock:
                job_status.adjust_waiting(-1)
                return chatterbox_provider.generate(
                    text, device=device.type, voice_prompt_path=prompt_path
                )

        try:
            audio, sr = await asyncio.to_thread(_run)
        except RuntimeError as e:
            raise ValueError(str(e))
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        audio_utils.save_audio(audio, path, sr)
        return path

    communicate = edge_tts.Communicate(text, voice or "en-US-GuyNeural")
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    await communicate.save(path)
    # Convert to wav immediately for downstream processing
    return _ensure_wav(path)


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
    return _ensure_wav(path)


def _ensure_wav(path: str) -> str:
    """If an mp3 was uploaded, convert once to wav and use that path downstream."""
    if not path or path.lower().endswith(".wav"):
        return path

    new_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(new_fd)
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
        sf.write(wav_path, audio, sr)
        try:
            os.remove(path)
        except OSError:
            pass
        return wav_path
    except Exception:
        # If conversion fails, fall back to original path
        try:
            os.remove(wav_path)
        except OSError:
            pass
        return path


def _prepare_reference(upload: UploadFile) -> str:
    """Write, validate, and preprocess the reference (target voice) audio.

    Preprocessing (DC offset removal, edge-silence trim, peak capping) is
    voicebox's fix for "reference sounds fine but conversion comes out bad" —
    raggedy references were a likely contributor to the old quality problems.
    Raises ValueError with a user-facing message when the file is unusable.
    """
    raw_path = _write_upload_to_temp(upload)
    ok, err, audio, sr = audio_utils.validate_and_load_reference_audio(raw_path)
    if not ok:
        try:
            os.remove(raw_path)
        except OSError:
            pass
        raise ValueError(err)
    audio_utils.save_audio(audio, raw_path, sr)
    return raw_path


def _apply_source_musicality(
    source_path: str,
    cadence_mode: str,
    music_key: str,
    retune_ms: float,
    vibrato_cents: float,
    drift_cents: float,
    octave_shift: int,
    midi_path: Optional[str],
    target_bpm: Optional[float],
    bpm_stretch: bool,
    grid_quantize: bool,
    grid_subdivision: int,
    grid_strength: float,
) -> str:
    """Impose musical pitch (cadence) and timing (grid) on the source audio
    BEFORE conversion, so the Seed-VC singing model follows a musical f0
    instead of flat speech. Order: pitch rewrite -> tempo stretch -> grid snap
    (grid positions only make sense at the final tempo)."""
    needs_cadence = cadence_mode not in (None, "", "none")
    needs_grid = bool(grid_quantize and target_bpm)
    needs_stretch = bool(bpm_stretch and target_bpm)
    if not (needs_cadence or needs_grid or needs_stretch):
        return source_path

    job_status.set_stage("cadence")
    audio, sr = librosa.load(source_path, sr=None, mono=True)

    if needs_cadence:
        audio = vocal_cadence.apply_cadence(
            audio,
            sr,
            mode=cadence_mode,
            key=music_key,
            retune_ms=retune_ms,
            vibrato_cents=vibrato_cents,
            drift_cents=drift_cents,
            octave_shift=octave_shift,
            midi_path=midi_path,
        )

    if needs_stretch:
        try:
            source_bpm, _ = librosa.beat.beat_track(y=audio, sr=sr)
            source_bpm = float(source_bpm)
            if source_bpm > 0:
                audio = librosa.effects.time_stretch(audio, rate=source_bpm / float(target_bpm))
                print(f"[bpm] stretched {source_bpm:.2f}->{float(target_bpm):.2f}")
        except Exception as exc:
            print(f"[bpm] skip stretch: {exc}")

    if needs_grid:
        audio = vocal_cadence.quantize_to_grid(
            audio,
            sr,
            bpm=float(target_bpm),
            subdivision=grid_subdivision,
            strength=grid_strength,
        )

    audio_utils.save_audio(audio.astype(np.float32), source_path, sr)
    return source_path


def _resolve_fx(fx_preset: str, fx_chain_json: str, target_bpm: Optional[float]):
    """Resolve the requested FX into a validated chain (or None)."""
    chain = None
    if fx_chain_json:
        try:
            chain = json.loads(fx_chain_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"fx_chain is not valid JSON: {e}")
    elif fx_preset and fx_preset != "none":
        chain = fx_chain.get_preset(fx_preset, bpm=target_bpm)
        if chain is None:
            raise ValueError(
                f"Unknown fx_preset '{fx_preset}'. Available: {list(fx_chain.PRESETS.keys())}"
            )
    if chain:
        err = fx_chain.validate_effects_chain(chain)
        if err:
            raise ValueError(err)
    return chain


def _decode_wav_chunks(chunks: list) -> tuple:
    """Decode a list of standalone WAV byte blobs and concatenate them."""
    parts = []
    sr = None
    for blob in chunks:
        data, chunk_sr = sf.read(io.BytesIO(blob), dtype="float32")
        sr = sr or chunk_sr
        parts.append(data)
    return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), sr


def _postprocess_full(
    audio: np.ndarray,
    sr: int,
    effects: Optional[list],
    trim_output: bool,
    normalize_output: bool,
) -> bytes:
    """Voicebox-style output hygiene + FX, returning final WAV bytes."""
    job_status.set_stage("postprocessing")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if trim_output:
        if audio_utils.has_tts_runaway(audio, sr):
            print("[post] runaway tail detected; trimming at internal silence gap")
        audio = audio_utils.trim_tts_output(audio, sr)
    if effects:
        audio = fx_chain.apply_effects(audio, sr, effects)
    if normalize_output:
        audio = audio_utils.normalize_audio(audio, target_db=-16.0, peak_limit=0.95)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def _normalize_lyrics(text: str) -> str:
    """Strip mid-phrase punctuation that makes TTS pause/slur (user-verified:
    removing commas improves enunciation). Sentence enders are kept."""
    text = re.sub(r"[,;:]+", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _zip_stems(stems: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, data in stems.items():
            z.writestr(name, data)
    return buf.getvalue()


async def _respond_debug(chunk_gen, stems: dict, effects, trim_output, normalize_output):
    """Debug-stems path: buffer the conversion, add raw + post stages, zip.
    Lets the user hear exactly which pipeline station degrades enunciation."""
    def _build():
        chunks = list(chunk_gen)
        audio, sr = _decode_wav_chunks(chunks)
        raw = io.BytesIO()
        sf.write(raw, audio, sr, subtype="PCM_16", format="WAV")
        stems["03_converted_raw.wav"] = raw.getvalue()
        try:
            stems["04_final.wav"] = _postprocess_full(
                audio, sr, effects, trim_output, normalize_output
            )
        finally:
            job_status.set_stage(None)
        return _zip_stems(stems)

    data = await asyncio.to_thread(_build)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=vocal-stems.zip"},
    )


def _cleanup_paths(*paths: Optional[str]) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _run_v1(source_path: str, target_path: str, **kwargs):
    """Yield WAV chunks from the V1 pipeline, serialized behind the GPU lock."""
    wrapper = load_v1_wrapper()
    job_status.adjust_waiting(+1)
    with job_status.gpu_lock:
        job_status.adjust_waiting(-1)
        job_status.set_stage("converting", job="v1")
        try:
            for wav_bytes, _ in wrapper.convert_voice(
                source=source_path, target=target_path, stream_output=True, **kwargs
            ):
                if wav_bytes:
                    yield wav_bytes
        finally:
            job_status.set_stage(None)
            _cleanup_paths(source_path, target_path)


def _respond(chunk_gen, effects, trim_output, normalize_output):
    """Stream raw chunks when no post-processing is needed; otherwise buffer
    the full conversion, apply output hygiene + FX, and return one WAV."""
    if not (effects or trim_output or normalize_output):
        return StreamingResponse(chunk_gen, media_type="audio/wav")

    def _buffered():
        chunks = list(chunk_gen)
        audio, sr = _decode_wav_chunks(chunks)
        try:
            return _postprocess_full(audio, sr, effects, trim_output, normalize_output)
        finally:
            job_status.set_stage(None)

    return _buffered


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), **job_status.snapshot()}


@app.get("/events/status")
async def events_status():
    """SSE stream of pipeline status for UI progress display."""
    return StreamingResponse(
        job_status.sse_status_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/fx/presets")
def fx_presets():
    return {"presets": list(fx_chain.PRESETS.keys()), "effects": list(fx_chain.EFFECT_REGISTRY.keys())}


async def _prepare_common(
    target_audio: UploadFile,
    source_path: str,
    cadence_mode: str,
    music_key: str,
    retune_ms: float,
    vibrato_cents: float,
    drift_cents: float,
    octave_shift: int,
    melody_midi: Optional[UploadFile],
    target_bpm: Optional[float],
    bpm_stretch: bool,
    grid_quantize: bool,
    grid_subdivision: int,
    grid_strength: float,
    pre_target_path: Optional[str] = None,
):
    """Shared source-musicality + reference-preparation steps.

    pre_target_path: reference already prepared by the caller (the
    chatterbox path preps it early to use as the TTS voice prompt).
    """
    job_status.set_stage("preprocessing")
    midi_path = None
    if melody_midi is not None and (melody_midi.filename or ""):
        fd, midi_path = tempfile.mkstemp(suffix=".mid")
        with os.fdopen(fd, "wb") as f:
            f.write(melody_midi.file.read())

    try:
        source_path = await asyncio.to_thread(
            _apply_source_musicality,
            source_path,
            cadence_mode,
            music_key,
            retune_ms,
            vibrato_cents,
            drift_cents,
            octave_shift,
            midi_path,
            target_bpm,
            bpm_stretch,
            grid_quantize,
            grid_subdivision,
            grid_strength,
        )
    finally:
        _cleanup_paths(midi_path)

    if pre_target_path is not None:
        return source_path, pre_target_path
    target_path = await asyncio.to_thread(_prepare_reference, target_audio)
    return source_path, target_path


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
    bpm_stretch: bool = Form(True),
    cadence_mode: str = Form("none"),
    music_key: str = Form("A minor"),
    retune_ms: float = Form(0.0),
    vibrato_cents: float = Form(0.0),
    drift_cents: float = Form(8.0),
    octave_shift: int = Form(0),
    melody_midi: Optional[UploadFile] = File(None),
    grid_quantize: bool = Form(False),
    grid_subdivision: int = Form(4),
    grid_strength: float = Form(1.0),
    fx_preset: str = Form("none"),
    fx_chain: str = Form(""),
    trim_output: bool = Form(True),
    normalize_output: bool = Form(False),
    debug_stems: bool = Form(False),
):
    stems: dict = {}
    source_path = _write_upload_to_temp(source_audio)
    try:
        effects = _resolve_fx(fx_preset, fx_chain, target_bpm)
        if debug_stems:
            stems["01_source_raw.wav"] = _read_bytes(source_path)
        source_path, target_path = await _prepare_common(
            target_audio, source_path, cadence_mode, music_key, retune_ms,
            vibrato_cents, drift_cents, octave_shift, melody_midi,
            target_bpm, bpm_stretch, grid_quantize, grid_subdivision, grid_strength,
        )
        if debug_stems:
            stems["02_source_musical.wav"] = _read_bytes(source_path)
    except ValueError as e:
        job_status.set_stage(None)
        _cleanup_paths(source_path)
        return JSONResponse(status_code=400, content={"error": str(e)})

    chunk_gen = _run_v1(
        source_path=source_path,
        target_path=target_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift,
    )

    if debug_stems:
        return await _respond_debug(chunk_gen, stems, effects, trim_output, normalize_output)

    result = _respond(chunk_gen, effects, trim_output, normalize_output)
    if isinstance(result, StreamingResponse):
        return result
    wav = await asyncio.to_thread(result)
    return Response(content=wav, media_type="audio/wav")


@app.post("/v1/convert_text")
async def convert_v1_text(
    text: str = Form(...),
    target_audio: UploadFile = File(...),
    tts_voice: str = Form("en-US-GuyNeural"),
    tts_engine: str = Form("chatterbox"),
    diffusion_steps: int = Form(10),
    length_adjust: float = Form(1.0),
    inference_cfg_rate: float = Form(0.7),
    f0_condition: bool = Form(False),
    auto_f0_adjust: bool = Form(True),
    pitch_shift: int = Form(0),
    target_bpm: Optional[float] = Form(None),
    bpm_stretch: bool = Form(True),
    cadence_mode: str = Form("none"),
    music_key: str = Form("A minor"),
    retune_ms: float = Form(0.0),
    vibrato_cents: float = Form(0.0),
    drift_cents: float = Form(8.0),
    octave_shift: int = Form(0),
    melody_midi: Optional[UploadFile] = File(None),
    grid_quantize: bool = Form(False),
    grid_subdivision: int = Form(4),
    grid_strength: float = Form(1.0),
    fx_preset: str = Form("none"),
    fx_chain: str = Form(""),
    trim_output: bool = Form(True),
    normalize_output: bool = Form(False),
    debug_stems: bool = Form(False),
    smooth_punctuation: bool = Form(True),
):
    stems: dict = {}
    pre_target_path = None
    if smooth_punctuation:
        text = _normalize_lyrics(text)
    try:
        effects = _resolve_fx(fx_preset, fx_chain, target_bpm)
        if tts_engine == "chatterbox":
            # Prep the reference early so Chatterbox can clone it as the
            # TTS voice — the source enters Seed-VC already near the target.
            pre_target_path = await asyncio.to_thread(_prepare_reference, target_audio)
        source_path = await _generate_tts(
            text, tts_voice, engine=tts_engine, prompt_path=pre_target_path
        )
    except ValueError as e:
        job_status.set_stage(None)
        _cleanup_paths(pre_target_path)
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        job_status.set_stage(None)
        _cleanup_paths(pre_target_path)
        return JSONResponse(status_code=400, content={"error": f"TTS failed: {e}"})

    if debug_stems:
        stems["01_source_raw.wav"] = _read_bytes(source_path)

    try:
        source_path, target_path = await _prepare_common(
            target_audio, source_path, cadence_mode, music_key, retune_ms,
            vibrato_cents, drift_cents, octave_shift, melody_midi,
            target_bpm, bpm_stretch, grid_quantize, grid_subdivision, grid_strength,
            pre_target_path=pre_target_path,
        )
    except ValueError as e:
        job_status.set_stage(None)
        _cleanup_paths(source_path, pre_target_path)
        return JSONResponse(status_code=400, content={"error": str(e)})

    if debug_stems:
        stems["02_source_musical.wav"] = _read_bytes(source_path)

    chunk_gen = _run_v1(
        source_path=source_path,
        target_path=target_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift,
    )

    if debug_stems:
        return await _respond_debug(chunk_gen, stems, effects, trim_output, normalize_output)

    result = _respond(chunk_gen, effects, trim_output, normalize_output)
    if isinstance(result, StreamingResponse):
        return result
    wav = await asyncio.to_thread(result)
    return Response(content=wav, media_type="audio/wav")


if __name__ == "__main__":
    # Run with: python api_server.py --host 0.0.0.0 --port 7860
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    uvicorn.run("api_server:app", host=args.host, port=args.port, reload=False)
