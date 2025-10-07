# src/audio/humanize.py

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import librosa
from scipy import signal


# ---------------------------
# Public dataclass (imported by effects/server)
# ---------------------------

@dataclass
class HumanizationParams:
    """
    Controls for humanization. All units documented below.
    """
    intensity: float = 0.7                 # 0..1 global mix
    # Pitch movement
    vibrato_rate_hz: float = 5.5           # Hz
    vibrato_depth_cents: float = 20.0      # cents (�)
    pitch_drift_amount: float = 0.15       # semitones (slow wander amplitude)
    # Timing / phrasing
    jitter_ms: float = 5.0                 # micro-timing (� ms)
    # Breathing / artifacts
    breathiness: float = 0.003             # noise amount
    enable_micro_timing: bool = True
    enable_breath: bool = True
    enable_artifacts: bool = True


# ---------------------------
# Reference analyzer (used by /api/analyze_reference)
# ---------------------------

class ReferenceAnalyzer:
    """
    Extract comprehensive voice characteristics for cloning.
    Analyzes both prosodic and timbral features needed for voice conversion.
    """

    def analyze_reference(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32, copy=False)

        duration_seconds = float(len(y) / max(1, sr))
        rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
        peak = float(np.max(np.abs(y)) + 1e-12)
        crest = float(peak / max(rms, 1e-12))

        # 1. Basic spectral characteristics
        f, psd = signal.welch(y, sr, nperseg=min(4096, len(y)))
        psd_norm = psd / (np.max(psd) + 1e-12) if psd.size else psd
        spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        if psd_norm.size:
            bright_band = (f >= 5000) & (f <= 10000)
            sibilance_ratio = float(psd_norm[bright_band].mean() / (psd_norm.mean() + 1e-12))
        else:
            sibilance_ratio = 1.0

        # 2. Pitch characteristics for voice identity
        f0_mean = 0.0
        f0_std = 0.0
        vibrato_rate_hz = 0.0
        vibrato_depth_cents = 0.0
        voice_pitch_range = 0.0
        
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
            )
            f0_clean = f0[~np.isnan(f0)]
            if f0_clean.size > 10:
                f0_mean = float(np.median(f0_clean))  # More robust than mean
                f0_std = float(np.std(f0_clean))
                voice_pitch_range = float(np.percentile(f0_clean, 95) - np.percentile(f0_clean, 5))
                
                # Vibrato analysis
                if f0_clean.size > 100:
                    cents = 1200.0 * np.log2(f0_clean / np.median(f0_clean))
                    cents = cents - np.mean(cents)
                    S = np.abs(np.fft.rfft(cents))
                    freq = np.fft.rfftfreq(len(cents), d=1 / 100.0)
                    if S.size > 1:
                        idx = int(np.argmax(S[1:]) + 1)
                        vibrato_rate_hz = float(freq[idx])
                    vibrato_depth_cents = float(np.std(cents))
        except Exception:
            pass

        # 3. Timbral characteristics (for voice cloning)
        timbre_features = {}
        try:
            # MFCC features (spectral envelope)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            timbre_features.update({
                'mfcc_mean': mfcc.mean(axis=1).tolist(),
                'mfcc_std': mfcc.std(axis=1).tolist(),
            })
            
            # Spectral features
            spectral_rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
            spectral_bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
            zero_crossing_rate = float(librosa.feature.zero_crossing_rate(y).mean())
            
            timbre_features.update({
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'zero_crossing_rate': zero_crossing_rate,
            })
            
            # Formant estimates (simplified)
            hop_length = 512
            S = librosa.stft(y, hop_length=hop_length)
            magnitude = np.abs(S)
            
            # Estimate first 3 formants using spectral peaks
            freqs = librosa.fft_frequencies(sr=sr, n_fft=magnitude.shape[0]*2-1)
            avg_spectrum = np.mean(magnitude, axis=1)
            
            # Find peaks in the spectrum
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(avg_spectrum, height=np.max(avg_spectrum) * 0.1)
            peak_freqs = freqs[peaks]
            
            # Take first 3 peaks as formant estimates
            formants = peak_freqs[:3] if len(peak_freqs) >= 3 else list(peak_freqs) + [0] * (3 - len(peak_freqs))
            timbre_features['formants'] = formants.tolist()
            
        except Exception as e:
            print(f"[ReferenceAnalyzer] Timbre analysis failed: {e}")
            timbre_features = {'error': 'timbre_analysis_failed'}

        # 4. Voice quality indicators
        voice_quality = {}
        try:
            # Harmonic-to-noise ratio estimate
            harmonic = librosa.effects.harmonic(y)
            percussive = librosa.effects.percussive(y)
            hnr = float(np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-12))
            
            # Breathiness estimate (high frequency noise)
            high_freq_energy = float(np.mean(y[1:] - y[:-1])**2)  # Simple high-freq estimate
            
            voice_quality.update({
                'harmonic_to_noise_ratio': hnr,
                'breathiness': high_freq_energy,
                'voiced_percentage': float(np.mean(voiced_flag)) if 'voiced_flag' in locals() else 0.5,
            })
        except Exception:
            voice_quality = {'error': 'voice_quality_analysis_failed'}

        return {
            # Basic audio characteristics
            "duration_seconds": duration_seconds,
            "rms": rms,
            "crest": crest,
            "spectral_centroid": spectral_centroid,
            "sibilance_ratio": sibilance_ratio,
            
            # Pitch/prosody (for compatibility)
            "vibrato_rate_hz": vibrato_rate_hz,
            "vibrato_depth_cents": vibrato_depth_cents,
            
            # Voice identity features (NEW - for cloning)
            "fundamental_frequency": f0_mean,
            "pitch_variability": f0_std,
            "pitch_range": voice_pitch_range,
            "timbre_features": timbre_features,
            "voice_quality": voice_quality,
            
            # Cloning readiness flag
            "cloning_ready": timbre_features.get('error') is None and voice_quality.get('error') is None,
        }


# ---------------------------
# Humanizer
# ---------------------------

class VocalHumanizer:
    def __init__(self, sr: int):
        self.sr = sr

    # ---- micro timing ----
    def add_micro_timing(self, y: np.ndarray, p: HumanizationParams) -> np.ndarray:
        if not p.enable_micro_timing or p.jitter_ms <= 0:
            return y
        onset_frames = librosa.onset.onset_detect(y=y, sr=self.sr, backtrack=True)
        if onset_frames.size < 2:
            return y

        hop = 512
        out = y.copy()
        max_shift = int((p.jitter_ms / 1000.0) * self.sr)

        for i in range(len(onset_frames) - 1):
            start = int(onset_frames[i] * hop)
            end = int(onset_frames[i + 1] * hop)
            if start >= len(out) or end <= start:
                continue
            seg = out[start:end]

            shift = int(np.clip(np.random.normal(0, max_shift / 2), -max_shift, max_shift))
            if shift == 0:
                continue

            if shift > 0:
                # delay: insert zeros at start of segment
                pad = np.zeros(min(shift, len(seg) // 2), dtype=out.dtype)
                seg_shifted = np.concatenate([pad, seg[: len(seg) - len(pad)]])
            else:
                # rush: drop a few samples at start
                k = min(-shift, len(seg) // 2)
                seg_shifted = seg[k:]
                seg_shifted = np.pad(seg_shifted, (0, len(seg) - len(seg_shifted)))

            out[start:end] = seg_shifted
        return out

    # ---- pitch drift / vibrato ----
    def add_pitch_drift(self, y: np.ndarray, p: HumanizationParams) -> np.ndarray:
        """
        Add natural pitch variations including vibrato and random drift.
        Uses short frames and the midpoint shift so vibrato doesn't cancel.
        """
        if p.intensity <= 0:
            return y

        n = len(y)
        t = np.arange(n) / self.sr

        # Slow wander (semitones)
        drift = p.pitch_drift_amount * (
            np.sin(2 * np.pi * 0.7 * t) + 0.5 * np.sin(2 * np.pi * 1.19 * t)
        )

        # Vibrato (convert cents -> semitones)
        vib_depth_semitones = float(p.vibrato_depth_cents) / 100.0
        vibrato = vib_depth_semitones * np.sin(
            2 * np.pi * max(0.1, p.vibrato_rate_hz) * t
        )

        # Slight random walk, smoothed
        rand = np.cumsum(np.random.normal(0, 0.001 * p.intensity, n))
        if n > 9:
            rand = signal.savgol_filter(rand, max(9, (n // 100) * 2 + 1), 3)

        total = (drift + vibrato + rand) * p.intensity  # semitone curve

        # Overlap-add with midpoint-based shift per frame
        frame = 2048
        hop = frame // 2
        win = np.hanning(frame)
        out = np.zeros(n, dtype=np.float32)
        acc = np.zeros(n, dtype=np.float32)

        i = 0
        while i < n:
            start = i
            end = min(i + frame, n)
            chunk = y[start:end]

            # pad last frame
            if len(chunk) < frame:
                pad = frame - len(chunk)
                chunk = np.pad(chunk, (0, pad))

            mid = min(start + frame // 2, n - 1)
            shift = float(total[mid])  # instantaneous value at frame center

            if abs(shift) > 1e-4:
                proc = librosa.effects.pitch_shift(
                    chunk, sr=self.sr, n_steps=shift, bins_per_octave=48
                )
            else:
                proc = chunk

            proc = proc[:frame] * win
            out[start:start + frame] += proc[: max(0, n - start)]
            acc[start:start + frame] += win[: max(0, n - start)]

            i += hop

        acc[acc == 0] = 1.0
        return (out / acc)[:n]

    # ---- breathing / dynamics ----
    def add_breath_dynamics(self, y: np.ndarray, p: HumanizationParams) -> np.ndarray:
        if not p.enable_breath:
            return y

        energy = librosa.feature.rms(y=y)[0]
        thr = np.percentile(energy, 20)
        is_sil = energy < thr
        diff = np.diff(is_sil.astype(np.int8), prepend=0)
        starts = np.where(diff == -1)[0]
        ends = np.where(diff == 1)[0]

        out = y.copy()
        hop = 512

        # breath attack/release shaping
        for i in range(min(len(starts), len(ends))):
            s = int(starts[i] * hop)
            e = int(min(ends[i] * hop, len(out)))
            if e - s < 32:
                continue

            attack = int(0.02 * self.sr)
            if s + attack < e:
                env = np.exp(np.linspace(-3, 0, attack))
                out[s:s + attack] *= env

            release = int(0.05 * self.sr * p.intensity)
            if e - release > s:
                env = np.exp(np.linspace(0, -1 * p.intensity, release))
                out[e - release:e] *= env

        # breath noise (HF band-pass noise modulated by envelope)
        if p.breathiness > 0:
            noise = np.random.normal(0, p.breathiness, len(y)).astype(np.float32)
            sos = signal.butter(4, [800, 4000], btype="band", fs=self.sr, output="sos")
            breath = signal.sosfilt(sos, noise)
            env = np.abs(signal.hilbert(y))
            breath *= env / (np.max(env) + 1e-6)
            out = out + breath

        return out

    # ---- occasional artifacts ----
    def add_performance_artifacts(self, y: np.ndarray, p: HumanizationParams) -> np.ndarray:
        if not p.enable_artifacts:
            return y

        out = y.copy()
        rng = np.random.default_rng()

        # rare crack on high-energy spot
        if rng.random() < 0.25 * p.intensity:
            energy = librosa.feature.rms(y=y)[0]
            hi = np.where(energy > np.percentile(energy, 85))[0]
            if hi.size:
                f = int(rng.choice(hi))
                s = f * 512
                L = int(0.03 * self.sr)
                if s + L < len(out):
                    seg = out[s:s + L]
                    cracked = librosa.effects.pitch_shift(seg, sr=self.sr, n_steps=12)
                    out[s:s + L] = seg * 0.3 + cracked * 0.7

        # light vocal fry at phrase ends
        if p.intensity > 0.5:
            energy = librosa.feature.rms(y=y)[0]
            ends, _ = signal.find_peaks(-energy, distance=max(1, self.sr // 512))
            for f in ends[-min(3, len(ends)):] if len(ends) else []:
                if rng.random() < 0.4:
                    e = int(f * 512)
                    L = int(0.1 * self.sr)
                    if e - L > 0:
                        chunk = out[e - L:e]
                        t = np.arange(L) / self.sr
                        patt = signal.square(2 * np.pi * 70 * t + rng.normal(0, 0.5, L))
                        patt = (patt + 1) / 2 * 0.7 + 0.3
                        out[e - L:e] = chunk * patt.astype(chunk.dtype)

        return out

    # ---- full chain ----
    def apply(self, y: np.ndarray, params: HumanizationParams) -> np.ndarray:
        y = self.add_micro_timing(y, params)
        y = self.add_pitch_drift(y, params)
        y = self.add_breath_dynamics(y, params)
        y = self.add_performance_artifacts(y, params)
        return y


# ---------------------------
# Helper to adapt params from a reference profile
# ---------------------------

def _params_from_reference(intensity: float,
                           ref: Optional[Dict[str, Any]]) -> HumanizationParams:
    p = HumanizationParams(intensity=float(np.clip(intensity, 0.0, 1.0)))
    if not ref:
        return p

    # Vibrato: clamp to musical range
    if isinstance(ref.get("vibrato_rate_hz", None), (int, float)):
        p.vibrato_rate_hz = float(np.clip(ref["vibrato_rate_hz"], 4.0, 8.0))
    if isinstance(ref.get("vibrato_depth_cents", None), (int, float)):
        p.vibrato_depth_cents = float(np.clip(ref["vibrato_depth_cents"], 5.0, 80.0))

    # Tone proxy ? adjust breathiness slightly
    sib = ref.get("sibilance_ratio", 1.0)
    if sib > 1.2:
        p.breathiness = 0.002  # already bright; keep noise subtle
    elif sib < 0.8:
        p.breathiness = 0.004  # a hair more air

    # Cleaner singers ? less random drift
    if p.vibrato_depth_cents < 15.0:
        p.pitch_drift_amount = 0.08
    else:
        p.pitch_drift_amount = 0.15

    return p


# ---------------------------
# Simple public function used by effects.py
# ---------------------------

def humanize_vocal(
    y: np.ndarray,
    sr: int,
    intensity: float = 0.7,
    reference_profile: Optional[Dict[str, Any]] = None,
    adapt_reference: Optional[bool] = None,      # preferred
    adapt_to_reference: Optional[bool] = None,   # legacy alias
) -> np.ndarray:
    """
    Entry point used by the effects pipeline.
    `adapt_reference` is preferred; `adapt_to_reference` kept for compatibility.
    """
    if y is None or len(y) == 0:
        return y

    # Resolve either spelling
    adapt = bool(adapt_reference) if adapt_reference is not None else bool(adapt_to_reference)

    params = _params_from_reference(intensity, reference_profile) if adapt \
        else HumanizationParams(intensity=float(np.clip(intensity, 0.0, 1.0)))

    humanizer = VocalHumanizer(sr=sr)
    return humanizer.apply(y.astype(np.float32, copy=False), params)
