"""Vocal cadence engine: turn static TTS speech into musical material.

The gap this closes: TTS output is spoken — no melody, no grid. Seed-VC
converts timbre but follows the source's pitch/rhythm, so musicality has
to be imposed on the source BEFORE conversion. This module does that with
WORLD-vocoder analysis/resynthesis (pyworld): the spectral envelope (sp)
and aperiodicity (ap) are kept, only f0 is rewritten — formants and
intelligibility survive, pitch becomes whatever we ask for.

Modes
  chant     — flatten all voiced frames to the key's root note (classic
              techno/tech-house monotone vocal), octave chosen nearest the
              speaker's natural median pitch.
  autotune  — quantize each voiced frame to the nearest scale note with a
              retune-speed glide (T-Pain/hard-tune at retune_ms=0).
  melody    — drive f0 from a MIDI file's note track (mido), stretched or
              looped to fit the vocal's duration.

Rhythm
  quantize_to_grid() snaps detected syllable onsets to the nearest
  subdivision of a BPM grid by time-shifting segments (no stretching, so
  syllables keep their natural length) with short crossfades.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import librosa
import numpy as np
import pyworld as pw

FRAME_PERIOD_MS = 5.0

# Scale intervals in semitones from the root
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "chromatic": list(range(12)),
}

NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_key(key: str) -> Tuple[int, List[int]]:
    """Parse "A minor", "F# major", "c dorian", "Bb" → (root pitch class, scale).

    Defaults to minor when no scale name is given (techno rarely disagrees).
    """
    m = re.match(r"^\s*([A-Ga-g])\s*([#b]?)\s*[_ ]*(\w*)\s*$", key or "A minor")
    if not m:
        return 9, SCALES["minor"]  # A minor
    root = NOTE_OFFSETS[m.group(1).upper()]
    if m.group(2) == "#":
        root += 1
    elif m.group(2) == "b":
        root -= 1
    root %= 12
    scale_name = (m.group(3) or "minor").lower()
    scale = SCALES.get(scale_name, SCALES["minor"])
    return root, scale


def midi_to_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0)


def world_analyze(audio: np.ndarray, sr: int):
    x = audio.astype(np.float64)
    f0, t = pw.harvest(x, sr, frame_period=FRAME_PERIOD_MS)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return f0, sp, ap


def world_synthesize(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray, sr: int) -> np.ndarray:
    y = pw.synthesize(
        np.ascontiguousarray(f0, dtype=np.float64),
        np.ascontiguousarray(sp, dtype=np.float64),
        np.ascontiguousarray(ap, dtype=np.float64),
        sr,
        frame_period=FRAME_PERIOD_MS,
    )
    return y.astype(np.float32)


def _glide(target_midi: np.ndarray, voiced: np.ndarray, retune_ms: float) -> np.ndarray:
    """One-pole smoothing toward the target pitch — the 'retune speed' knob.
    retune_ms=0 gives a hard snap; larger values give portamento glides.
    Smoothing state resets across unvoiced gaps so consonants don't smear."""
    if retune_ms <= 0:
        return target_midi
    alpha = float(np.exp(-FRAME_PERIOD_MS / retune_ms))
    out = target_midi.copy()
    state = None
    for i in range(len(out)):
        if not voiced[i]:
            state = None
            continue
        if state is None:
            state = out[i]
        else:
            state = alpha * state + (1 - alpha) * out[i]
        out[i] = state
    return out


def _humanize(midi: np.ndarray, voiced: np.ndarray, vibrato_cents: float,
              vibrato_hz: float, drift_cents: float) -> np.ndarray:
    """Optional dead-robot antidote: slow vibrato plus band-limited drift."""
    n = len(midi)
    if n == 0:
        return midi
    t = np.arange(n) * (FRAME_PERIOD_MS / 1000.0)
    out = midi.copy()
    if vibrato_cents > 0:
        out = out + voiced * (vibrato_cents / 100.0) * np.sin(2 * np.pi * vibrato_hz * t)
    if drift_cents > 0:
        noise = np.random.default_rng(1234).standard_normal(n)
        # crude low-pass: cumulative smoothing keeps drift slow
        kernel = int(200 / FRAME_PERIOD_MS)  # ~200 ms
        if kernel > 1:
            noise = np.convolve(noise, np.ones(kernel) / kernel, mode="same")
        noise = noise / (np.abs(noise).max() + 1e-9)
        out = out + voiced * (drift_cents / 100.0) * noise
    return out


def _quantize_to_scale(midi: np.ndarray, root: int, scale: List[int]) -> np.ndarray:
    """Snap each MIDI value to the nearest pitch in the scale."""
    out = midi.copy()
    for i, m in enumerate(midi):
        if m <= 0:
            continue
        octave = int(np.floor(m / 12.0))
        candidates = []
        for oct_ in (octave - 1, octave, octave + 1):
            for interval in scale:
                candidates.append(oct_ * 12 + root + interval)
        candidates = np.array(candidates, dtype=np.float64)
        out[i] = candidates[np.argmin(np.abs(candidates - m))]
    return out


def _midi_notes_from_file(midi_path: str) -> List[Tuple[float, float, int]]:
    """Read (start_s, end_s, note) from the densest track of a MIDI file."""
    import mido

    mid = mido.MidiFile(midi_path)
    tempo = 500000  # default 120 BPM
    tpq = mid.ticks_per_beat or 480
    events: List[Tuple[float, float, int]] = []
    for track in mid.tracks:
        now_ticks = 0
        active = {}
        track_events = []
        local_tempo = tempo
        for msg in track:
            now_ticks += msg.time
            now_s = mido.tick2second(now_ticks, tpq, local_tempo)
            if msg.type == "set_tempo":
                local_tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = now_s
            elif msg.type in ("note_off", "note_on"):
                start = active.pop(msg.note, None)
                if start is not None and now_s > start:
                    track_events.append((start, now_s, msg.note))
        if len(track_events) > len(events):
            events = track_events
    events.sort()
    return events


def apply_cadence(
    audio: np.ndarray,
    sr: int,
    mode: str = "none",
    key: str = "A minor",
    retune_ms: float = 0.0,
    vibrato_cents: float = 0.0,
    vibrato_hz: float = 5.5,
    drift_cents: float = 8.0,
    midi_path: Optional[str] = None,
    melody_fit: str = "stretch",
    octave_shift: int = 0,
) -> np.ndarray:
    """Rewrite the pitch contour of speech according to the chosen mode.

    Returns audio at the same sample rate. mode="none" is a no-op.
    """
    if mode in (None, "", "none"):
        return audio

    f0, sp, ap = world_analyze(audio, sr)
    voiced = f0 > 0
    if not voiced.any():
        return audio

    midi = hz_to_midi(f0)
    root, scale = parse_key(key)

    if mode == "chant":
        # Root note in the octave nearest the speaker's median voiced pitch
        median_midi = float(np.median(midi[voiced]))
        candidates = np.array([o * 12 + root for o in range(2, 8)], dtype=np.float64)
        target_note = candidates[np.argmin(np.abs(candidates - median_midi))]
        target = np.where(voiced, target_note, 0.0)

    elif mode == "autotune":
        target = np.where(voiced, _quantize_to_scale(midi, root, scale), 0.0)

    elif mode == "melody":
        if not midi_path:
            raise ValueError("melody mode requires a MIDI file")
        notes = _midi_notes_from_file(midi_path)
        if not notes:
            raise ValueError("no notes found in MIDI file")
        duration = len(f0) * FRAME_PERIOD_MS / 1000.0
        melody_end = max(end for _, end, _ in notes)
        scale_t = (duration / melody_end) if (melody_fit == "stretch" and melody_end > 0) else 1.0

        frame_t = np.arange(len(f0)) * (FRAME_PERIOD_MS / 1000.0)
        target = np.zeros_like(f0)
        note_starts = np.array([s * scale_t for s, _, _ in notes])
        for i, ft in enumerate(frame_t):
            t_mel = ft if melody_fit == "stretch" else (ft % melody_end)
            # active note, else most recent note (held through gaps so every
            # voiced frame lands on melody rather than falling back to speech)
            idx = int(np.searchsorted(note_starts, t_mel, side="right")) - 1
            if idx >= 0:
                target[i] = float(notes[idx][2])
        target = np.where(voiced & (target > 0), target, 0.0)

    else:
        raise ValueError(f"unknown cadence mode: {mode}")

    target = target + 12.0 * octave_shift
    v = voiced & (target > 0)
    target = _glide(target, v, retune_ms)
    target = _humanize(target, v.astype(np.float64), vibrato_cents, vibrato_hz, drift_cents)

    new_f0 = np.where(v, midi_to_hz(target), 0.0)
    out = world_synthesize(new_f0, sp, ap, sr)
    # WORLD resynthesis can overshoot 0 dBFS; match the source's peak so the
    # converted signal enters Seed-VC at a sane level.
    out_peak = float(np.abs(out).max())
    src_peak = float(np.abs(audio).max())
    if out_peak > 0 and src_peak > 0 and out_peak > src_peak:
        out = out * (src_peak / out_peak)
    return out


def quantize_to_grid(
    audio: np.ndarray,
    sr: int,
    bpm: float,
    subdivision: int = 4,
    strength: float = 1.0,
    fade_ms: float = 8.0,
) -> np.ndarray:
    """Snap syllable onsets to a BPM grid by time-shifting segments.

    subdivision: grid steps per beat (4 = 16th notes, 2 = 8ths).
    strength: 0..1, how far each onset moves toward its grid slot.
    Segments keep their natural length; overlaps crossfade, gaps stay silent.
    """
    if bpm <= 0 or len(audio) == 0:
        return audio

    step = 60.0 / bpm / subdivision
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr, backtrack=True, units="samples"
    )
    if len(onset_frames) == 0:
        return audio

    onsets = np.concatenate([[0], onset_frames, [len(audio)]])
    onsets = np.unique(onsets)

    out_len = int(len(audio) + sr * step)  # headroom for forward shifts
    out = np.zeros(out_len, dtype=np.float32)
    fade_n = max(1, int(sr * fade_ms / 1000.0))

    for i in range(len(onsets) - 1):
        seg = audio[onsets[i] : onsets[i + 1]].astype(np.float32)
        if len(seg) == 0:
            continue
        t_on = onsets[i] / sr
        t_grid = round(t_on / step) * step
        t_new = t_on + strength * (t_grid - t_on)
        pos = max(0, int(t_new * sr))
        end = min(pos + len(seg), out_len)
        seg = seg[: end - pos]
        if len(seg) == 0:
            continue
        # short fades on segment edges to avoid clicks at seams
        n = min(fade_n, len(seg) // 2)
        if n > 0:
            env = np.ones(len(seg), dtype=np.float32)
            env[:n] = np.linspace(0.0, 1.0, n)
            env[-n:] = np.linspace(1.0, 0.0, n)
            seg = seg * env
        out[pos:end] += seg

    peak = float(np.abs(out).max())
    src_peak = float(np.abs(audio).max())
    if peak > 0 and src_peak > 0 and peak > src_peak:
        out = out * (src_peak / peak)

    # trim headroom tail silence
    nz = np.nonzero(np.abs(out) > 1e-5)[0]
    if len(nz):
        out = out[: nz[-1] + 1]
    return out
