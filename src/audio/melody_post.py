"""
Lightweight post-processing to nudge TTS output toward a more musical/singing feel.
Heuristics only: pitch-shift to a fixed key center and add subtle vibrato.
"""

from typing import Tuple
import numpy as np
import librosa

# Fixed reference: A4 = 440 Hz
A4 = 440.0

# Simple major scale semitone offsets from tonic
MAJOR_OFFSETS = np.array([0, 2, 4, 5, 7, 9, 11, 12])


def hz_to_semitones(freq_hz: float) -> float:
    """Convert Hz to semitone distance from A4."""
    return 12.0 * np.log2(freq_hz / A4)


def semitones_to_hz(semitones: float) -> float:
    """Convert semitones from A4 back to Hz."""
    return A4 * (2.0 ** (semitones / 12.0))


def quantize_to_major(freq_hz: float, tonic_hz: float) -> float:
    """Snap frequency to nearest note in major scale relative to tonic."""
    if freq_hz <= 0:
        return freq_hz
    # Map to semitone space
    semis = hz_to_semitones(freq_hz)
    tonic_semi = hz_to_semitones(tonic_hz)
    rel = semis - tonic_semi
    # Wrap to 0-12
    rel_mod = np.mod(rel, 12.0)
    nearest = MAJOR_OFFSETS[np.argmin(np.abs(MAJOR_OFFSETS - rel_mod))]
    snapped = tonic_semi + nearest + 12.0 * np.floor((rel) / 12.0)
    return semitones_to_hz(snapped)


def estimate_pitch_track(audio: np.ndarray, sr: int) -> np.ndarray:
    """Estimate f0 using librosa.yin; returns Hz with NaNs for unvoiced."""
    return librosa.yin(audio, fmin=60.0, fmax=800.0, sr=sr)


def apply_vibrato(audio: np.ndarray, sr: int, depth_semitones: float = 0.2, rate_hz: float = 5.0) -> np.ndarray:
    """Apply simple pitch LFO via phase-vocoder resampling."""
    if depth_semitones <= 0:
        return audio
    t = np.arange(len(audio)) / sr
    lfo = depth_semitones * np.sin(2 * np.pi * rate_hz * t)
    # Convert semitone LFO to resampling curve
    rate_curve = 2.0 ** (lfo / 12.0)
    # Cumulative time map
    cum = np.cumsum(rate_curve)
    cum = cum / cum[-1]
    target_len = len(audio)
    sample_pos = cum * (len(audio) - 1)
    return np.interp(np.arange(target_len), np.arange(target_len), audio[sample_pos.astype(np.int32)])


def pitch_align_to_key(audio: np.ndarray, sr: int, tonic_hz: float = 261.63) -> Tuple[np.ndarray, float]:
    """Estimate pitch and nudge toward a major scale around tonic_hz."""
    f0 = estimate_pitch_track(audio, sr)
    f0_clean = np.where(np.isnan(f0), 0.0, f0)
    # Compute average voiced pitch
    voiced = f0_clean[f0_clean > 0]
    if len(voiced) == 0:
        return audio, 0.0
    mean_f0 = np.median(voiced)
    snapped = quantize_to_major(mean_f0, tonic_hz)
    shift_semitones = hz_to_semitones(snapped) - hz_to_semitones(mean_f0)
    # Apply uniform pitch shift
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift_semitones)
    return shifted, shift_semitones


def make_more_singing(audio: np.ndarray, sr: int, tonic_hz: float = 261.63) -> np.ndarray:
    """Main entry: pitch-align to key and add subtle vibrato."""
    aligned, _ = pitch_align_to_key(audio, sr, tonic_hz=tonic_hz)
    with_vibrato = apply_vibrato(aligned, sr, depth_semitones=0.2, rate_hz=5.0)
    # Normalize to prevent clipping
    peak = np.max(np.abs(with_vibrato)) or 1.0
    return with_vibrato / max(peak, 1e-6)
