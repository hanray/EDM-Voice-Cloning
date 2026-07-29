"""Audio hygiene utilities.

Ported/adapted from jamiepine/voicebox (MIT) backend/utils/audio.py:
reference-audio preprocessing and validation, TTS output trimming,
runaway detection, normalization, and atomic WAV writes.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0,
    peak_limit: float = 0.95,
) -> np.ndarray:
    """RMS-normalize toward target_db, clip-safe.

    If the RMS gain would push the peak past peak_limit, the gain is reduced
    so the peak lands exactly at peak_limit — never hard-clipped. Distortion
    belongs in the DAW, on purpose, not here by accident.
    """
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio**2))
    if rms <= 0:
        return audio
    gain = (10 ** (target_db / 20)) / rms
    peak = float(np.abs(audio).max())
    if peak * gain > peak_limit:
        gain = peak_limit / peak
    return audio * gain


def save_audio(audio: np.ndarray, path: str, sample_rate: int) -> None:
    """Atomic WAV write: temp file + os.replace so crashes never leave a
    corrupt/partial file at the destination."""
    temp_path = f"{path}.tmp"
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(temp_path, audio, sample_rate, format="WAV")
        os.replace(temp_path, path)
    except Exception as e:
        try:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
        except Exception:
            pass
        raise OSError(f"Failed to save audio to {path}: {e}") from e


def has_tts_runaway(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    max_internal_silence_ms: int = 2000,
) -> bool:
    """Detect speech followed by a long silence and then more output — the
    shape of a TTS/VC model that missed EOS and resumed with hallucinated
    audio. Leading/trailing silence doesn't count."""
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return False

    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)
    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    seen_speech = False
    consecutive_silence = 0

    for i in range(n_frames):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        is_speech = np.sqrt(np.mean(frame**2)) >= threshold_linear
        if is_speech:
            if seen_speech and consecutive_silence >= max_silence_frames:
                return True
            seen_speech = True
            consecutive_silence = 0
        elif seen_speech:
            consecutive_silence += 1

    return False


def trim_tts_output(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    min_silence_ms: int = 200,
    max_internal_silence_ms: int = 1000,
    fade_ms: int = 30,
) -> np.ndarray:
    """Trim trailing silence and post-silence hallucination from generated
    audio: cut at the first internal silence gap longer than
    max_internal_silence_ms, trim trailing silence, cosine fade-out."""
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return audio

    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)

    rms = np.array(
        [
            np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ]
    )
    is_speech = rms >= threshold_linear

    first_speech = 0
    for i, s in enumerate(is_speech):
        if s:
            first_speech = max(0, i - 1)
            break

    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    consecutive_silence = 0
    cut_frame = n_frames

    for i in range(first_speech, n_frames):
        if is_speech[i]:
            consecutive_silence = 0
        else:
            consecutive_silence += 1
            if consecutive_silence >= max_silence_frames:
                cut_frame = i - consecutive_silence + 1
                break

    min_silence_frames = int(min_silence_ms / frame_ms)
    end_frame = cut_frame
    while end_frame > first_speech and not is_speech[end_frame - 1]:
        end_frame -= 1
    end_frame = min(end_frame + min_silence_frames, cut_frame)

    start_sample = first_speech * frame_len
    end_sample = min(end_frame * frame_len, len(audio))
    trimmed = audio[start_sample:end_sample].copy()

    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and len(trimmed) > fade_samples:
        fade = np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        trimmed[-fade_samples:] *= fade

    return trimmed


def preprocess_reference_audio(
    audio: np.ndarray,
    sample_rate: int,
    peak_target: float = 0.95,
    trim_top_db: float = 40.0,
    edge_padding_ms: int = 100,
) -> np.ndarray:
    """Clean a reference sample: remove DC offset, trim edge silence
    (40 dB — below speech dynamic range so soft syllables survive), re-pad
    a short anchor silence, cap slightly-hot peaks instead of rejecting."""
    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio

    audio = audio - float(np.mean(audio))

    trimmed, _ = librosa.effects.trim(audio, top_db=trim_top_db)
    if 0 < trimmed.size < audio.size:
        pad_each = int(sample_rate * edge_padding_ms / 1000)
        headroom = (audio.size - trimmed.size) // 2
        pad = min(pad_each, max(headroom, 0))
        if pad > 0:
            trimmed = np.pad(trimmed, (pad, pad), mode="constant")
        audio = trimmed

    peak = float(np.abs(audio).max())
    if peak > peak_target and peak > 0:
        audio = audio * (peak_target / peak)

    return audio


def validate_and_load_reference_audio(
    audio_path: str,
    min_duration: float = 1.0,
    max_duration: float = 60.0,
    min_rms: float = 0.005,
) -> Tuple[bool, Optional[str], Optional[np.ndarray], Optional[int]]:
    """Load + preprocess a reference file, then gate on duration and RMS.

    Returns (is_valid, error_message, audio, sample_rate). Bounds are looser
    than voicebox's TTS defaults because Seed-VC references can usefully be
    longer clips of sung material.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        audio = preprocess_reference_audio(audio, sr)
        duration = len(audio) / sr

        if duration < min_duration:
            return False, f"Reference audio too short (minimum {min_duration}s)", None, None
        if duration > max_duration:
            return False, f"Reference audio too long (maximum {max_duration}s)", None, None

        rms = np.sqrt(np.mean(audio**2))
        if rms < min_rms:
            return False, "Reference audio is too quiet or silent", None, None

        return True, None, audio, sr
    except Exception as e:
        detail = str(e) or type(e).__name__
        return False, f"Error validating reference audio: {detail}", None, None
