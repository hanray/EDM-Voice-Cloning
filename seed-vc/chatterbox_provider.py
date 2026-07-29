"""Local Chatterbox TTS provider — the expressive alternative to edge-tts.

Why: edge-tts produces flat robotic narration; feeding that into Seed-VC
caps how alive the vocal can sound. Chatterbox (ResembleAI, 0.5B, local)
does zero-shot voice cloning from a reference clip, so we prompt it with
the SAME reference audio used for conversion — the TTS output starts out
close to the target voice and Seed-VC has far less work to do.

Integration quirks ported from jamiepine/voicebox (MIT) backends/base.py +
chatterbox_backend.py, which is chatterbox-tts run in production:
  - installed with --no-deps (upstream pins numpy<1.26 / torch==2.6 that
    would wreck this venv; the code runs fine on our torch 2.5.1+cu121)
  - float64→float32 patches for S3Tokenizer.log_mel_spectrogram and
    VoiceEncoder.forward (librosa hands float64 to float32 weights)
  - eager attention on the T3 transformer (sdpa breaks output_attentions)
  - torch.load forced to map_location="cpu" when running CPU-only
"""

from __future__ import annotations

import threading
import types
from typing import Optional

import numpy as np

_model = None
_model_lock = threading.Lock()
_load_patch_lock = threading.Lock()

# Voicebox's tuned defaults for the English model
DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5
DEFAULT_TEMPERATURE = 0.8


def is_available() -> bool:
    try:
        import chatterbox  # noqa: F401
        return True
    except ImportError:
        return False


def _patch_f32(model) -> None:
    """Patch float64 -> float32 dtype mismatches in upstream chatterbox."""
    import torch

    _tokzr = model.s3gen.tokenizer
    _orig_log_mel = _tokzr.log_mel_spectrogram.__func__

    def _f32_log_mel(self_tokzr, audio, padding=0):
        if torch.is_tensor(audio):
            audio = audio.float()
        return _orig_log_mel(self_tokzr, audio, padding)

    _tokzr.log_mel_spectrogram = types.MethodType(_f32_log_mel, _tokzr)

    _ve = model.ve
    _orig_ve_forward = _ve.forward.__func__

    def _f32_ve_forward(self_ve, mels):
        return _orig_ve_forward(self_ve, mels.float())

    _ve.forward = types.MethodType(_f32_ve_forward, _ve)


def _load_model(device: str):
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model

        import torch
        from chatterbox.tts import ChatterboxTTS

        if device == "cpu":
            _orig_torch_load = torch.load

            def _patched_load(*args, **kwargs):
                kwargs.setdefault("map_location", "cpu")
                return _orig_torch_load(*args, **kwargs)

            with _load_patch_lock:
                torch.load = _patched_load
                try:
                    model = ChatterboxTTS.from_pretrained(device=device)
                finally:
                    torch.load = _orig_torch_load
        else:
            model = ChatterboxTTS.from_pretrained(device=device)

        # sdpa attention breaks output_attentions in the T3 transformer
        t3_tfmr = model.t3.tfmr
        if hasattr(t3_tfmr, "config") and hasattr(t3_tfmr.config, "_attn_implementation"):
            t3_tfmr.config._attn_implementation = "eager"
            for layer in getattr(t3_tfmr, "layers", []):
                if hasattr(layer, "self_attn"):
                    layer.self_attn._attn_implementation = "eager"

        _patch_f32(model)
        _model = model
        return _model


def generate(
    text: str,
    device: str = "cuda",
    voice_prompt_path: Optional[str] = None,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[np.ndarray, int]:
    """Synthesize text, optionally cloning the voice in voice_prompt_path.

    Returns (mono float32 audio, sample_rate). Raises RuntimeError with an
    actionable message when chatterbox isn't installed.
    """
    if not is_available():
        raise RuntimeError(
            "chatterbox-tts is not installed in this environment "
            "(pip install --no-deps chatterbox-tts, plus conformer diffusers "
            "resemble-perth s3tokenizer pyloudnorm)"
        )

    model = _load_model(device)
    kwargs = dict(
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    if voice_prompt_path:
        kwargs["audio_prompt_path"] = voice_prompt_path

    wav = model.generate(text, **kwargs)
    audio = wav.squeeze().cpu().numpy().astype(np.float32)
    return audio, model.sr
