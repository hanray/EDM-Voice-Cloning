# Musicality Layer — Making TTS Speech Land in a Track

The core problem this solves: TTS output is static spoken speech, so Seed-VC
(which follows the source's pitch and rhythm) could never produce singing or
rap. This layer imposes musical pitch and timing on the source **before**
conversion; the 44.1 kHz f0-conditioned Seed-VC singing model then follows the
imposed contour and heals DSP artifacts while converting timbre.

Pipeline: `text → edge-TTS → [cadence: WORLD f0 rewrite] → [tempo stretch] →
[grid quantize] → Seed-VC → [trim/runaway cleanup] → [FX chain] → [normalize]`

## Modules

- `vocal_cadence.py` — WORLD-vocoder (pyworld) analysis/resynthesis. Only f0
  is rewritten; spectral envelope and aperiodicity are untouched, so formants
  and intelligibility survive. Modes:
  - **chant** — flatten voiced frames to the key's root note (octave picked
    nearest the speaker's median pitch). The classic techno monotone vocal.
  - **autotune** — snap the natural speech contour to the nearest scale note;
    `retune_ms` 0 = hard tune, higher = portamento glide.
  - **melody** — drive pitch from a MIDI file (mido), stretched or looped to
    fit the vocal length.
  - `quantize_to_grid()` — snap syllable onsets (librosa onset detection) to
    1/8 or 1/16 grid at the target BPM by segment time-shifting with
    crossfades; `strength` blends between natural and fully quantized.
- `audio_utils.py` — ported from voicebox: reference preprocessing (DC offset,
  edge trim, peak capping) + validation, output trimming (cuts post-silence
  hallucination tails), runaway detection, RMS normalize, atomic WAV writes.
- `fx_chain.py` — ported from voicebox: pedalboard effect chains. Club presets
  (`techno_chant`, `tech_house_vox`, `big_room_wash`) auto-sync delay times to
  the target BPM (1/8 or dotted 1/8). Custom chains accepted as JSON via the
  `fx_chain` form field.
- `job_status.py` — process-wide GPU lock (serializes concurrent conversions
  instead of OOMing) + `/events/status` SSE stream the UI uses for stage
  progress (`preprocessing → tts → cadence → converting → postprocessing`).

## New API form fields (all four convert endpoints)

| Field | Default | Meaning |
|---|---|---|
| `cadence_mode` | `none` | `none` / `chant` / `autotune` / `melody` |
| `music_key` | `A minor` | root + scale, e.g. `F minor`, `C# phrygian` |
| `retune_ms` | `0` | pitch glide time constant (0 = hard snap) |
| `vibrato_cents` / `drift_cents` | `0` / `8` | humanization |
| `octave_shift` | `0` | shift imposed pitch by octaves |
| `melody_midi` | — | MIDI file upload (melody mode) |
| `target_bpm` | — | grid/stretch/delay-sync tempo |
| `bpm_stretch` | `true`* | legacy whole-clip tempo stretch (*UI sends false by default) |
| `grid_quantize` | `false` | snap onsets to grid (needs `target_bpm`) |
| `grid_subdivision` | `4` | 4 = 1/16 notes, 2 = 1/8 notes |
| `grid_strength` | `1.0` | 0..1 quantize amount |
| `fx_preset` | `none` | see `GET /fx/presets` |
| `fx_chain` | — | JSON chain, overrides preset |
| `trim_output` | `true` | voicebox output hygiene (error removal only) |
| `normalize_output` | `false` | off by default — output is a dry stem for the DAW; when enabled: RMS −16 dB, clip-safe |

V1 note: the UI's Model Mode selector now genuinely wires to the backend —
"Singing (44 kHz)" sends `f0_condition=true`. Use it with any cadence mode.

## Recommended starting recipes (tech house / techno)

- **Chant hook**: lyrics mode → cadence `chant`, key = your track's key,
  `target_bpm` = track BPM, grid quantize ON at 1/16, model mode Singing,
  FX `techno_chant`.
- **Melodic hook**: cadence `melody` + a simple 1-bar MIDI phrase,
  `retune_ms` 20–40, vibrato 10–15 cents.
- **Rap-ish flow**: cadence `none` or `autotune` with `chromatic` scale,
  grid quantize ON, strength 0.7–0.9 (full 1.0 sounds robotic — sometimes
  that's the point).

## Next layer (researched July 2026, not yet implemented)

DSP can chant and autotune, but real sung phrasing needs a singing model.
The ranked path (see research):

1. **ACE-Step 1.5** (MIT, github.com/ace-step/ACE-Step-1.5) — local
   lyrics→song generation in <10 s on consumer GPUs (6–16 GB tiers). Use as a
   *reference vocal generator*: generate a vocal in the right style, extract
   the acapella (built-in extraction or Demucs `htdemucs`), then feed through
   Seed-VC for timbre control. Best next integration: a sidecar service +
   one new endpoint (`/v1/generate_vocal`).
2. **YuE** (Apache-2.0) — slower but outputs native separate vocal stems;
   strongest lyric adherence for rap.
3. **YingMusic-Singer** (2026) — zero-shot lyrics+MIDI→sung vocal; watch its
   license/VRAM story.

Also worth adding to the DSP layer later: WhisperX forced alignment for
word-accurate grid mapping (current onset detection is heuristic), and
pyrubberband R3 for stretch-based (rather than shift-based) quantization.
