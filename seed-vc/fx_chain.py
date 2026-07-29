"""Post-processing FX chains via Spotify pedalboard.

Ported/adapted from jamiepine/voicebox (MIT) backend/utils/effects.py.
Chains are JSON-serializable lists of {type, enabled, params} dicts so
they can travel over the API. Adds club-oriented presets with BPM-synced
delay times on top of voicebox's originals.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
from pedalboard import (
    Chorus,
    Compressor,
    Delay,
    Gain,
    HighpassFilter,
    LowpassFilter,
    Pedalboard,
    PitchShift,
    Reverb,
)

EFFECT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "chorus": {
        "cls": Chorus,
        "label": "Chorus / Flanger",
        "params": {
            "rate_hz": {"default": 1.0, "min": 0.01, "max": 20.0},
            "depth": {"default": 0.5, "min": 0.0, "max": 1.0},
            "feedback": {"default": 0.0, "min": 0.0, "max": 0.95},
            "centre_delay_ms": {"default": 7.0, "min": 0.5, "max": 50.0},
            "mix": {"default": 0.5, "min": 0.0, "max": 1.0},
        },
    },
    "reverb": {
        "cls": Reverb,
        "label": "Reverb",
        "params": {
            "room_size": {"default": 0.5, "min": 0.0, "max": 1.0},
            "damping": {"default": 0.5, "min": 0.0, "max": 1.0},
            "wet_level": {"default": 0.33, "min": 0.0, "max": 1.0},
            "dry_level": {"default": 0.4, "min": 0.0, "max": 1.0},
            "width": {"default": 1.0, "min": 0.0, "max": 1.0},
        },
    },
    "delay": {
        "cls": Delay,
        "label": "Delay",
        "params": {
            "delay_seconds": {"default": 0.3, "min": 0.01, "max": 2.0},
            "feedback": {"default": 0.3, "min": 0.0, "max": 0.95},
            "mix": {"default": 0.3, "min": 0.0, "max": 1.0},
        },
    },
    "compressor": {
        "cls": Compressor,
        "label": "Compressor",
        "params": {
            "threshold_db": {"default": -20.0, "min": -60.0, "max": 0.0},
            "ratio": {"default": 4.0, "min": 1.0, "max": 20.0},
            "attack_ms": {"default": 10.0, "min": 0.1, "max": 100.0},
            "release_ms": {"default": 100.0, "min": 10.0, "max": 1000.0},
        },
    },
    "gain": {
        "cls": Gain,
        "label": "Gain",
        "params": {"gain_db": {"default": 0.0, "min": -40.0, "max": 40.0}},
    },
    "highpass": {
        "cls": HighpassFilter,
        "label": "High-Pass Filter",
        "params": {"cutoff_frequency_hz": {"default": 80.0, "min": 20.0, "max": 8000.0}},
    },
    "lowpass": {
        "cls": LowpassFilter,
        "label": "Low-Pass Filter",
        "params": {"cutoff_frequency_hz": {"default": 8000.0, "min": 200.0, "max": 20000.0}},
    },
    "pitch_shift": {
        "cls": PitchShift,
        "label": "Pitch Shift",
        "params": {"semitones": {"default": 0.0, "min": -12.0, "max": 12.0}},
    },
}


def _eighth(bpm: float) -> float:
    return 60.0 / bpm / 2.0


def _dotted_eighth(bpm: float) -> float:
    return 60.0 / bpm * 0.75


PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "techno_chant": [
        {"type": "highpass", "params": {"cutoff_frequency_hz": 120.0}},
        {"type": "compressor", "params": {"threshold_db": -18.0, "ratio": 6.0, "attack_ms": 3.0, "release_ms": 80.0}},
        {"type": "delay", "params": {"delay_seconds": 0.23, "feedback": 0.35, "mix": 0.18}},
        {"type": "reverb", "params": {"room_size": 0.4, "damping": 0.6, "wet_level": 0.15, "dry_level": 0.85, "width": 1.0}},
    ],
    "tech_house_vox": [
        {"type": "highpass", "params": {"cutoff_frequency_hz": 150.0}},
        {"type": "lowpass", "params": {"cutoff_frequency_hz": 9000.0}},
        {"type": "compressor", "params": {"threshold_db": -16.0, "ratio": 8.0, "attack_ms": 2.0, "release_ms": 60.0}},
        {"type": "delay", "params": {"delay_seconds": 0.35, "feedback": 0.25, "mix": 0.15}},
    ],
    "big_room_wash": [
        {"type": "highpass", "params": {"cutoff_frequency_hz": 200.0}},
        {"type": "reverb", "params": {"room_size": 0.9, "damping": 0.25, "wet_level": 0.5, "dry_level": 0.5, "width": 1.0}},
        {"type": "delay", "params": {"delay_seconds": 0.45, "feedback": 0.4, "mix": 0.25}},
    ],
    "robotic": [
        {"type": "chorus", "params": {"rate_hz": 0.2, "depth": 1.0, "feedback": 0.35, "centre_delay_ms": 7.0, "mix": 0.5}},
    ],
    "radio": [
        {"type": "highpass", "params": {"cutoff_frequency_hz": 300.0}},
        {"type": "lowpass", "params": {"cutoff_frequency_hz": 3500.0}},
        {"type": "compressor", "params": {"threshold_db": -15.0, "ratio": 6.0, "attack_ms": 5.0, "release_ms": 50.0}},
        {"type": "gain", "params": {"gain_db": 6.0}},
    ],
    "echo_chamber": [
        {"type": "reverb", "params": {"room_size": 0.85, "damping": 0.3, "wet_level": 0.45, "dry_level": 0.55, "width": 1.0}},
        {"type": "delay", "params": {"delay_seconds": 0.25, "feedback": 0.3, "mix": 0.2}},
    ],
    "deep_voice": [
        {"type": "pitch_shift", "params": {"semitones": -3.0}},
        {"type": "lowpass", "params": {"cutoff_frequency_hz": 6000.0}},
        {"type": "compressor", "params": {"threshold_db": -18.0, "ratio": 3.0, "attack_ms": 10.0, "release_ms": 150.0}},
    ],
}


def get_preset(name: str, bpm: Optional[float] = None) -> Optional[List[Dict[str, Any]]]:
    """Fetch a preset chain; when bpm is given, delay times are re-synced
    (techno_chant/big_room_wash → 1/8, tech_house_vox → dotted 1/8)."""
    chain = PRESETS.get(name)
    if chain is None:
        return None
    chain = copy.deepcopy(chain)
    if bpm and bpm > 0:
        sync = _dotted_eighth(bpm) if name == "tech_house_vox" else _eighth(bpm)
        for effect in chain:
            if effect["type"] == "delay":
                effect["params"]["delay_seconds"] = float(np.clip(sync, 0.01, 2.0))
    return chain


def validate_effects_chain(effects_chain: List[Dict[str, Any]]) -> Optional[str]:
    """Return None if valid, else an error message."""
    if not isinstance(effects_chain, list):
        return "effects_chain must be a list"
    for i, effect in enumerate(effects_chain):
        if not isinstance(effect, dict):
            return f"Effect at index {i} must be a dict"
        effect_type = effect.get("type")
        if effect_type not in EFFECT_REGISTRY:
            return f"Unknown effect type '{effect_type}' at index {i}. Available: {list(EFFECT_REGISTRY.keys())}"
        params = effect.get("params", {})
        if not isinstance(params, dict):
            return f"Effect '{effect_type}' at index {i}: params must be a dict"
        registry = EFFECT_REGISTRY[effect_type]
        for param_name, value in params.items():
            if param_name not in registry["params"]:
                return f"Effect '{effect_type}' at index {i}: unknown param '{param_name}'"
            pdef = registry["params"][param_name]
            if not isinstance(value, (int, float)):
                return f"Effect '{effect_type}' at index {i}: param '{param_name}' must be a number"
            if value < pdef["min"] or value > pdef["max"]:
                return (
                    f"Effect '{effect_type}' at index {i}: param '{param_name}' "
                    f"must be between {pdef['min']} and {pdef['max']} (got {value})"
                )
    return None


def apply_effects(audio: np.ndarray, sample_rate: int, effects_chain: List[Dict[str, Any]]) -> np.ndarray:
    """Apply a chain to mono or (channels, samples) audio."""
    if not effects_chain:
        return audio

    plugins = []
    for effect in effects_chain:
        if not effect.get("enabled", True):
            continue
        registry = EFFECT_REGISTRY[effect["type"]]
        params = {
            pname: effect.get("params", {}).get(pname, pdef["default"])
            for pname, pdef in registry["params"].items()
        }
        plugins.append(registry["cls"](**params))

    board = Pedalboard(plugins)
    audio_2d = audio[np.newaxis, :] if audio.ndim == 1 else audio
    processed = board(audio_2d.astype(np.float32), sample_rate)
    return processed[0] if audio.ndim == 1 else processed
