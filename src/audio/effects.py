# src/audio/effects.py

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import re
import random
import unicodedata
import numpy as np
import soundfile as sf
import librosa
from pedalboard import Pedalboard, Reverb, Delay, Compressor, Chorus, Gain, HighpassFilter, LowpassFilter
from scipy.ndimage import gaussian_filter1d

from .humanize import (
    humanize_vocal,
    ReferenceAnalyzer,
    HumanizationParams,
)

# Import neural voice cloning with fallback chain
try:
    from .chatterbox_voice_cloning import (
        apply_seedvc_conversion, 
        is_seedvc_available,
        convert_voice_to_reference,
        clone_voice_from_text,
        VoiceCloningSettings
    )
    SEEDVC_AVAILABLE = True
    print("[Effects] ✅ Using Chatterbox neural voice cloning")
except ImportError:
    try:
        from .rvc_converter import convert_voice_file, is_rvc_available as is_seedvc_available
        SEEDVC_AVAILABLE = True
        print("[Effects] Using RVC for voice conversion")
    except ImportError:
        try:
            from .seedvc_converter import convert_voice_file, is_seedvc_available
            SEEDVC_AVAILABLE = True
            print("[Effects] Using Seed-VC fallback for voice conversion")
        except ImportError:
            print("[Effects] No voice conversion available - falling back to basic audio processing")
            SEEDVC_AVAILABLE = False
            
            def apply_seedvc_conversion(*args, **kwargs):
                return None
                
            def is_seedvc_available():
                return False


# =========================
# BACKWARD COMPATIBILITY
# =========================

class EffectSettings:
    """Original settings class for backward compatibility"""
    def __init__(
        self,
        pitch_semitones: float = 0.0,
        speed: float = 1.0,
        reverb_mix: float = 0.0,
        delay_mix: float = 0.0,
        compress: bool = True,
    ):
        self.pitch_semitones = pitch_semitones
        self.speed = speed
        self.reverb_mix = reverb_mix
        self.delay_mix = delay_mix
        self.compress = compress


class EnhancedEffectSettings:
    """Enhanced settings that combine old and new approaches"""
    def __init__(self, **kwargs):
        # Classic FX
        self.pitch_semitones = kwargs.get("pitch", 0.0)
        self.speed = kwargs.get("speed", 1.0)
        self.reverb_mix = kwargs.get("reverb_mix", 0.0)
        self.delay_mix = kwargs.get("delay_mix", 0.0)
        self.compress = kwargs.get("compress", True)

        # Performance
        self.performance_mode = kwargs.get("performance_mode", "normal")
        self.bpm = kwargs.get("bpm", 120)
        self.sample_rate = kwargs.get("sample_rate", 48000)

        # Humanization / reference
        self.humanize = kwargs.get("humanize", True)
        self.humanization_intensity = kwargs.get("humanization_intensity", 0.7)
        self.reference_profile = kwargs.get("reference_profile", None)
        self.reference_adapt = kwargs.get("reference_adapt", False)

        # Voice conversion (RVC/Seed-VC)
        self.voice_conversion = kwargs.get("voice_conversion", False)
        self.conversion_strength = kwargs.get("conversion_strength", 0.8)
        self.conversion_quality = kwargs.get("conversion_quality", "balanced")

        # passthrough
        self.hints = kwargs


# =========================
# SIMPLE PIPE (fallback)
# =========================

def process_wav(in_wav: Path, out_wav: Path, cfg: EffectSettings) -> Path:
    """Original mechanical processing"""
    y, sr = sf.read(in_wav)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    if cfg.speed and abs(cfg.speed - 1.0) > 1e-3:
        y = librosa.effects.time_stretch(y, rate=cfg.speed)
    if cfg.pitch_semitones and abs(cfg.pitch_semitones) > 1e-3:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=cfg.pitch_semitones)

    board = Pedalboard([])
    if cfg.compress:
        board.append(Compressor())
    if cfg.reverb_mix > 0:
        board.append(Reverb(room_size=0.3, wet_level=cfg.reverb_mix))
    if cfg.delay_mix > 0:
        board.append(Delay(delay_seconds=0.28, mix=cfg.delay_mix))

    if len(board) > 0:
        y = board(y, sr)

    sf.write(out_wav, y, sr)
    return out_wav


# =========================
# ANALYSIS STRUCTURE
# =========================

@dataclass
class VocalAnalysis:
    energy_profile: np.ndarray
    onset_strength: float
    spectral_centroid: float
    consonant_clarity: float
    dynamics_range: float
    tempo: float
    pitch_variance: float
    harmonic_ratio: float


class PerformanceProcessor:
    """
    Adaptive performance processing per mode.
    """

    def __init__(self, mode: str = "normal", bpm: int = 120, target_sr: int = 48000):
        self.mode = mode
        self.bpm = bpm
        self.sr = target_sr

    def analyze_vocal(self, y: np.ndarray, sr: int) -> VocalAnalysis:
        hop_length = 512
        energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength = float(np.mean(onset_env))
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = float(np.mean(spectral_centroids))
        y_har, y_perc = librosa.effects.hpss(y)
        consonant_clarity = float(np.sum(np.abs(y_perc)) / (np.sum(np.abs(y)) + 1e-6))
        dynamics_range = float(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-6))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pv = []
        for t in range(pitches.shape[1]):
            idx = mags[:, t].argmax()
            f = pitches[idx, t]
            if f > 0:
                pv.append(f)
        pitch_variance = float(np.std(pv) if pv else 0.0)
        harmonic_ratio = float(np.sum(np.abs(y_har)) / (np.sum(np.abs(y)) + 1e-6))
        return VocalAnalysis(
            energy,
            onset_strength,
            avg_centroid,
            consonant_clarity,
            dynamics_range,
            float(tempo),
            pitch_variance,
            harmonic_ratio,
        )

    def process_chant_mode(self, y: np.ndarray, sr: int, a: VocalAnalysis) -> np.ndarray:
        out = y.copy()
        
        # Rhythmic gating - more pronounced
        gate_freq = self.bpm / 60
        t = np.arange(len(out)) / sr
        gate = 0.6 + 0.4 * (np.sin(2 * np.pi * gate_freq * 2 * t) > 0).astype(float)
        out *= gate
        
        # Boost consonants for clarity
        if a.consonant_clarity < 0.5:
            y_h, y_p = librosa.effects.hpss(out)
            out = y_h * 0.7 + y_p * 1.8
        
        # Tight compression for consistent level
        out = Pedalboard([
            Compressor(threshold_db=-15, ratio=6, attack_ms=2, release_ms=50),
            HighpassFilter(cutoff_frequency_hz=80)
        ])(out, sr)
        
        # Gentle limiting
        out = np.tanh(out * 0.85) * 1.15
        
        # Hard limiting with character
        out = np.tanh(out * 2.2) / 1.8
        
        return out

    def process_rap_mode(self, y: np.ndarray, sr: int, a: VocalAnalysis) -> np.ndarray:
        out = y.copy()
        
        # Aggressive transient shaping for punch
        env = np.abs(out)
        env_smooth = gaussian_filter1d(env, sigma=sr//1000)
        transients = env - env_smooth
        transient_boost = np.clip(transients * 2.0, 0, 0.3)
        out = out + out * transient_boost
        
        # Boost high-frequency consonants drastically for rap clarity
        y_h, y_p = librosa.effects.hpss(out, margin=2.0)
        out = y_h * 0.5 + y_p * 3.0
        
        # High-ratio compression for tight sound
        out = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=100),
            Compressor(threshold_db=-10, ratio=12, attack_ms=0.3, release_ms=15),
            Gain(gain_db=2)
        ])(out, sr)
        
        # Short, tight delay for rhythmic emphasis
        delay_time = 60.0 / self.bpm / 8
        out = Pedalboard([
            Delay(delay_seconds=delay_time, mix=0.12, feedback=0.05)
        ])(out, sr)
        
        # Slight saturation for edge
        out = np.tanh(out * 1.1) / 1.05
        
        return out

    def process_ballad_mode(self, y: np.ndarray, sr: int, a: VocalAnalysis) -> np.ndarray:
        out = y.copy()
        
        # Smooth out harsh transients
        if a.onset_strength > 15:
            env = np.abs(out)
            smooth = gaussian_filter1d(env, sigma=200)
            out *= (smooth / (env + 1e-6)) ** 0.4
        
        # Add warmth with subtle chorus
        if a.harmonic_ratio < 0.8:
            out = Pedalboard([Chorus(rate_hz=0.2, depth=0.15, mix=0.15)])(out, sr)
        
        # Gentle compression for smooth dynamics
        out = Pedalboard([
            Compressor(threshold_db=-20, ratio=2.0, attack_ms=20, release_ms=200)
        ])(out, sr)
        
        return out

    def process_aggressive_mode(self, y: np.ndarray, sr: int, a: VocalAnalysis) -> np.ndarray:
        out = y.copy()
        
        # Harsh saturation for grit
        if a.harmonic_ratio > 0.4:
            out = np.sign(out) * np.minimum(np.abs(out) * 2.0, np.abs(out) ** 0.6)
        
        # Boost percussive elements heavily
        y_h, y_p = librosa.effects.hpss(out, margin=1.5)
        out = y_h * 0.6 + y_p * 3.5
        
        # Aggressive compression and EQ
        out = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=120),
            Compressor(threshold_db=-8, ratio=15, attack_ms=0.1, release_ms=10),
            Gain(gain_db=4)
        ])(out, sr)
        
        return out

    def process_normal_mode(self, y: np.ndarray, sr: int, a: VocalAnalysis) -> np.ndarray:
        out = y.copy()
        
        # Light consonant enhancement if needed
        if a.consonant_clarity < 0.3:
            y_h, y_p = librosa.effects.hpss(out)
            out = y_h + y_p * 1.2
        
        # Gentle compression for consistency
        if a.dynamics_range > 8:
            out = Pedalboard([
                Compressor(threshold_db=-15, ratio=3, attack_ms=8, release_ms=80)
            ])(out, sr)
        
        # Subtle brightness if too dark
        if a.spectral_centroid < 1200:
            out = out + np.diff(out, prepend=out[0]) * 0.08
        
        return out

    def process(
        self,
        in_path: str,
        out_path: str,
        speed: float = 1.0,
        pitch: float = 0.0,
        cfg: Optional[EnhancedEffectSettings] = None,
    ) -> str:
        # Load
        y, sr = sf.read(in_path)
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1)

        # Resample
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
            sr = self.sr

        # Early (lightweight) reference adaptation (prosody/tone) BEFORE creative FX
        if cfg and cfg.reference_adapt and cfg.reference_profile:
            y = adapt_vocal_to_reference(y, sr, cfg.reference_profile, cfg.humanization_intensity)

        # User speed/pitch
        if abs(speed - 1.0) > 1e-3:
            y = librosa.effects.time_stretch(y, rate=speed)
        if abs(pitch) > 1e-3:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)

        # Analysis & mode processing
        a = self.analyze_vocal(y, sr)
        if self.mode == "chant":
            y = self.process_chant_mode(y, sr, a)
        elif self.mode == "rap":
            y = self.process_rap_mode(y, sr, a)
        elif self.mode == "ballad":
            y = self.process_ballad_mode(y, sr, a)
        elif self.mode == "aggressive":
            y = self.process_aggressive_mode(y, sr, a)
        else:
            y = self.process_normal_mode(y, sr, a)

        # Final micro-humanization (after FX)
        if cfg and cfg.humanize:
            y = humanize_vocal(
                y,
                sr,
                intensity=cfg.humanization_intensity,
                reference_profile=cfg.reference_profile,
                adapt_to_reference=cfg.reference_adapt,
            )

        # Limiting
        mx = float(np.max(np.abs(y)) + 1e-12)
        if mx > 0.95:
            y = y * (0.95 / mx)

        sf.write(out_path, y, sr)
        return out_path


# =========================
# LIGHTWEIGHT REFERENCE ADAPTATION (prosody/tone)
# =========================

def adapt_vocal_to_reference(
    y: np.ndarray, sr: int, ref_profile: Dict[str, Any], intensity: float = 0.7
) -> np.ndarray:
    """
    Very light spectral tilt toward reference brightness using spectral centroid proxy.
    This is NOT voice cloning — it just nudges tone.
    """
    try:
        target_centroid = float(ref_profile.get("spectral_centroid", 0.0))
        if target_centroid <= 0:
            return y

        # Current centroid
        cur_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        if cur_centroid <= 0:
            return y

        # Compute tilt exponent to move toward target
        ratio = np.clip(target_centroid / cur_centroid, 0.5, 2.0)
        alpha = (ratio - 1.0) * 0.5 * float(np.clip(intensity, 0.0, 1.0))

        # STFT, apply frequency-dependent tilt, iSTFT
        n_fft = 2048
        hop = n_fft // 4
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
        mag, phase = np.abs(S), np.angle(S)
        freqs = np.linspace(0.0, sr / 2.0, mag.shape[0])
        norm_f = (freqs / (sr / 2.0 + 1e-9)) ** alpha
        mag2 = mag * norm_f[:, None]
        Sout = mag2 * np.exp(1j * phase)
        y2 = librosa.istft(Sout, hop_length=hop, window="hann", length=len(y))

        # Gentle blend
        out = (0.7 * y + 0.3 * y2).astype(np.float32)
        mx = np.max(np.abs(out)) + 1e-12
        if mx > 0.99:
            out = out * (0.99 / mx)
        return out
    except Exception:
        return y


# =========================
# VOICE CONVERSION (RVC/SEED-VC)
# =========================

def apply_seedvc_conversion(
    source_wav: str, 
    reference_wav: str, 
    output_wav: str, 
    strength: float = 0.8,
    quality: str = "balanced"
) -> bool:
    """
    Apply voice conversion (RVC or Seed-VC) with fallback
    
    Args:
        source_wav: Path to source audio
        reference_wav: Path to reference audio
        output_wav: Path to output audio
        strength: Conversion strength (0.0-1.0)
        quality: Conversion quality ("fast", "balanced", "high")
        
    Returns:
        True if conversion successful, False if failed (caller should handle fallback)
    """
    if not SEEDVC_AVAILABLE:
        print("[Effects] No voice conversion available")
        return False
    
    try:
        success = convert_voice_file(
            source_wav=source_wav,
            reference_wav=reference_wav,
            output_wav=output_wav,
            quality=quality,
            strength=strength
        )
        if success:
            print(f"[Effects] Voice conversion successful: {output_wav}")
        else:
            print("[Effects] Voice conversion failed")
        return success
    except Exception as e:
        print(f"[Effects] Voice conversion error: {e}")
        return False


def copy_file_as_fallback(source_path: str, output_path: str) -> bool:
    """Copy source file to output as fallback when conversion fails"""
    try:
        import shutil
        shutil.copy2(source_path, output_path)
        return True
    except Exception as e:
        print(f"[Effects] Fallback copy failed: {e}")
        return False


# =========================
# CHANT TEXT HELPERS (pre-TTS)
# =========================

def _normalize_words(text: str, lowercase: bool = True) -> List[str]:
    """
    Normalize punctuation/quotes and split into tokens.
    Keeps internal apostrophes (e.g., don't).
    """
    text = text.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    text = unicodedata.normalize("NFKD", text)
    if lowercase:
        text = text.lower()
    return re.findall(r"[A-Za-z0-9]+'[A-Za-z0-9]+|[A-Za-z0-9]+", text)

def chant_pairs(words: List[str], *, skip_first: bool = False) -> List[str]:
    """
    Build 2-word pairs over a circular word list.
    If skip_first=True, start from words[1] (wrap-around).
    For odd lengths, last pair wraps.
    """
    if not words:
        return []
    offset = 1 if skip_first else 0
    starts = list(range(0, len(words), 2))
    out = []
    for s in starts:
        i = (s + offset) % len(words)
        j = (i + 1) % len(words)
        out.append(f"{words[i]} {words[j]}")
    return out

def chant_twos(
    text: str,
    *,
    cycles: int = 1,
    skip_first_chance: float = 0.33,
    sep: str = ", ",
    lowercase: bool = True,
    force_skip_first: bool | None = None,
    rng: random.Random | None = None,
    return_pairs: bool = False,
):
    """
    Main entry. Returns a chant line of 2-word chunks, sometimes rotated by one word.
    """
    words = _normalize_words(text, lowercase=lowercase)
    if not words:
        return [] if return_pairs else ""

    rng = rng or random
    do_skip = (rng.random() < float(skip_first_chance)) if force_skip_first is None else bool(force_skip_first)
    pairs = chant_pairs(words, skip_first=do_skip)

    if return_pairs:
        return pairs

    line = sep.join(pairs)
    return sep.join([line] * max(1, int(cycles)))

def prepare_text_for_tts(
    text: str,
    cfg: Optional["EnhancedEffectSettings"] = None,
) -> str:
    """
    Convenience wrapper that respects cfg.performance_mode == 'chant' and reads options
    from cfg.hints (falls back to sensible defaults). Safe no-op for other modes.
    """
    try:
        if cfg and getattr(cfg, "performance_mode", None) == "chant":
            hints = getattr(cfg, "hints", {}) or {}
            return chant_twos(
                text,
                cycles=int(hints.get("chant_cycles", 1)),
                skip_first_chance=float(hints.get("chant_skip_first_chance", 0.33)),
                sep=str(hints.get("chant_sep", ", ")),
                lowercase=bool(hints.get("chant_lowercase", True)),
                force_skip_first=hints.get("chant_force_skip_first", None),
            )
        return text
    except Exception:
        return text


# =========================
# UNIFIED ENTRY
# =========================

def process_vocal_wav(in_wav: Path, out_wav: Path, cfg: EnhancedEffectSettings) -> Path:
    """
    Entry point used by server. Includes early reference adaptation
    and final micro humanization. Voice conversion (RVC/Seed-VC) is invoked
    from the server step *before* this function if enabled.
    """
    try:
        processor = PerformanceProcessor(
            mode=cfg.performance_mode,
            bpm=cfg.bpm,
            target_sr=cfg.sample_rate or 48000,
        )
        return Path(
            processor.process(
                str(in_wav),
                str(out_wav),
                speed=cfg.speed,
                pitch=cfg.pitch_semitones,
                cfg=cfg,
            )
        )
    except Exception as e:
        print(f"[Effects] Processing error: {e} → fallback")
        basic_cfg = EffectSettings(
            pitch_semitones=cfg.pitch_semitones,
            speed=cfg.speed,
            reverb_mix=cfg.reverb_mix,
            delay_mix=cfg.delay_mix,
            compress=cfg.compress,
        )
        return process_wav(in_wav, out_wav, basic_cfg)