import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AudioField, AudioValue } from './components/AudioField';

type InputMode = 'audio' | 'text';
type CadenceMode = 'none' | 'chant' | 'autotune' | 'melody';

interface VoiceParams {
  diffusion_steps: number;
  length_adjust: number;
  inference_cfg_rate: number;
  auto_f0_adjust: boolean;
  pitch_shift: number;
  model_mode: 'voice' | 'singing';
  tts_voice: string;
  edge_fallback: boolean; // when on, use Edge instead of local Chatterbox
}

interface MusicalityParams {
  cadence_mode: CadenceMode;
  key_root: string;
  key_scale: string;
  retune_ms: number;
  vibrato_cents: number;
  drift_cents: number;
  octave_shift: number;
  target_bpm: number | null;
  bpm_stretch: boolean;
  grid_quantize: boolean;
  grid_subdivision: number;
  grid_strength: number;
  fx_preset: string;
  trim_output: boolean;
  normalize_output: boolean;
}

const defaultVoice: VoiceParams = {
  diffusion_steps: 10,
  length_adjust: 1.0,
  inference_cfg_rate: 0.7,
  auto_f0_adjust: false,
  pitch_shift: 0,
  model_mode: 'singing',
  tts_voice: 'en-US-GuyNeural',
  edge_fallback: false,
};

const defaultMusicality: MusicalityParams = {
  // Chant by default: cloning copies the voice, not the melody — without a
  // cadence mode, lyrics come out as plain speech.
  cadence_mode: 'chant',
  key_root: 'A',
  key_scale: 'minor',
  retune_ms: 0,
  vibrato_cents: 0,
  drift_cents: 8,
  octave_shift: 0,
  target_bpm: null,
  bpm_stretch: false,
  grid_quantize: false,
  grid_subdivision: 4,
  grid_strength: 0.85,
  fx_preset: 'none',
  trim_output: true,
  normalize_output: false, // dry stems — level decisions belong in the DAW
};

const FX_PRESETS = [
  { value: 'none', label: 'None (dry)' },
  { value: 'techno_chant', label: 'Techno Chant' },
  { value: 'tech_house_vox', label: 'Tech House Vox' },
  { value: 'big_room_wash', label: 'Big Room Wash' },
  { value: 'robotic', label: 'Robotic' },
  { value: 'radio', label: 'Radio' },
  { value: 'echo_chamber', label: 'Echo Chamber' },
  { value: 'deep_voice', label: 'Deep Voice' },
];

const CADENCE_MODES: { value: CadenceMode; label: string }[] = [
  { value: 'chant', label: 'Chant' },
  { value: 'autotune', label: 'Autotune' },
  { value: 'melody', label: 'Melody' },
  { value: 'none', label: 'Natural' },
];

const STATIONS = [
  { key: 'source', label: 'Source', stages: ['preprocessing', 'tts'] },
  { key: 'cadence', label: 'Cadence + Grid', stages: ['cadence'] },
  { key: 'convert', label: 'Neural Convert', stages: ['converting'] },
  { key: 'post', label: 'Post', stages: ['postprocessing'] },
];

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (data?.error) return data.error;
  } catch {
    /* not JSON */
  }
  return `Conversion failed (HTTP ${res.status})`;
}

async function streamToUrl(res: Response, onStatus?: (s: string) => void): Promise<string> {
  const contentType = res.headers.get('content-type') || 'audio/wav';
  const reader = res.body?.getReader();
  if (!reader) {
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  }
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      received += value.length;
      onStatus?.(`Receiving audio — ${(received / 1024).toFixed(0)} KB`);
    }
  }
  const blob = new Blob(chunks as BlobPart[], { type: contentType });
  return URL.createObjectURL(blob);
}

function appendMusicality(form: FormData, m: MusicalityParams, midi: File | null) {
  form.append('cadence_mode', m.cadence_mode);
  form.append('music_key', `${m.key_root} ${m.key_scale}`);
  form.append('retune_ms', String(m.retune_ms));
  form.append('vibrato_cents', String(m.vibrato_cents));
  form.append('drift_cents', String(m.drift_cents));
  form.append('octave_shift', String(m.octave_shift));
  if (m.target_bpm !== null && m.target_bpm !== undefined) {
    form.append('target_bpm', String(m.target_bpm));
  }
  form.append('bpm_stretch', String(m.bpm_stretch));
  form.append('grid_quantize', String(m.grid_quantize));
  form.append('grid_subdivision', String(m.grid_subdivision));
  form.append('grid_strength', String(m.grid_strength));
  form.append('fx_preset', m.fx_preset);
  form.append('trim_output', String(m.trim_output));
  form.append('normalize_output', String(m.normalize_output));
  if (m.cadence_mode === 'melody' && midi) {
    form.append('melody_midi', midi);
  }
}

async function convert(
  source: File | null,
  target: File,
  voice: VoiceParams,
  musicality: MusicalityParams,
  midi: File | null,
  inputMode: InputMode,
  inputText: string,
  onStatus?: (s: string) => void,
): Promise<string> {
  const form = new FormData();
  form.append('target_audio', target);
  form.append('diffusion_steps', String(voice.diffusion_steps));
  form.append('length_adjust', String(voice.length_adjust));
  form.append('inference_cfg_rate', String(voice.inference_cfg_rate));
  form.append('f0_condition', String(voice.model_mode === 'singing'));
  form.append('auto_f0_adjust', String(voice.auto_f0_adjust));
  form.append('pitch_shift', String(voice.pitch_shift));
  appendMusicality(form, musicality, midi);

  let endpoint = '/api/v1/convert';
  if (inputMode === 'text') {
    endpoint = '/api/v1/convert_text';
    form.append('text', inputText);
    form.append('tts_engine', voice.edge_fallback ? 'edge' : 'chatterbox');
    form.append('tts_voice', voice.tts_voice);
  } else {
    if (!source) throw new Error('Source audio missing');
    form.append('source_audio', source);
  }

  onStatus?.('Uploading…');
  const res = await fetch(endpoint, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await readError(res));
  onStatus?.('Processing…');
  return streamToUrl(res, onStatus);
}

const STAGE_LABELS: Record<string, string> = {
  preprocessing: 'Preparing audio',
  tts: 'Synthesizing voice',
  cadence: 'Applying cadence',
  converting: 'Neural conversion',
  postprocessing: 'Finishing',
};

function SliderRow({ label, value, min, max, step, unit, onChange }: {
  label: string; value: number; min: number; max: number; step: number; unit?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="slider-row">
      <div className="label">{label}</div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} />
      <div className="value">{value}{unit || ''}</div>
    </div>
  );
}

function Switch({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <input type="checkbox" className="switch" checked={checked} disabled={disabled}
      onChange={(e) => onChange(e.target.checked)} />
  );
}

export default function App() {
  const [source, setSource] = useState<AudioValue | null>(null);
  const [reference, setReference] = useState<AudioValue | null>(null);
  const [voice, setVoice] = useState<VoiceParams>(defaultVoice);
  const [musicality, setMusicality] = useState<MusicalityParams>(defaultMusicality);
  const [melodyMidi, setMelodyMidi] = useState<File | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('Ready');
  const [stage, setStage] = useState<string | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [inputText, setInputText] = useState<string>('');
  const [serverUp, setServerUp] = useState<boolean | null>(null);
  const sseRef = useRef<EventSource | null>(null);

  useEffect(() => {
    fetch('/api/health').then((r) => setServerUp(r.ok)).catch(() => setServerUp(false));
  }, []);

  useEffect(() => {
    if (!loading) {
      sseRef.current?.close();
      sseRef.current = null;
      setStage(null);
      return;
    }
    const es = new EventSource('/api/events/status');
    es.onmessage = (ev) => {
      try {
        const snap = JSON.parse(ev.data);
        if (snap.state === 'busy' && snap.stage) {
          setStage(snap.stage);
          const queued = snap.waiting > 0 ? ` · ${snap.waiting} queued` : '';
          setStatus(`${STAGE_LABELS[snap.stage] || snap.stage}${queued}`);
        }
      } catch { /* ignore */ }
    };
    sseRef.current = es;
    return () => { es.close(); sseRef.current = null; };
  }, [loading]);

  const canGenerate = useMemo(() => {
    if (!reference) return false;
    if (inputMode === 'text') return inputText.trim().length > 0;
    return !!source;
  }, [inputMode, inputText, reference, source]);

  const m = musicality;
  const setM = (patch: Partial<MusicalityParams>) => setMusicality({ ...m, ...patch });
  const setV = (patch: Partial<VoiceParams>) => setVoice({ ...voice, ...patch });

  const submit = async () => {
    if (!reference) return;
    if (m.cadence_mode === 'melody' && !melodyMidi) {
      setError('Melody mode needs a MIDI file');
      return;
    }
    setError(null);
    setStatus('Starting…');
    setLoading(true);
    try {
      const url = await convert(
        source?.file || null, reference.file, voice, m, melodyMidi,
        inputMode, inputText, setStatus,
      );
      setOutputUrl(url);
      setStatus('Done');
    } catch (e: any) {
      setError(e?.message || 'Failed to convert');
      setStatus('Failed');
    } finally {
      setLoading(false);
    }
  };

  const stationState = (stages: string[]) => {
    if (!loading) return '';
    if (stage && stages.includes(stage)) return ' station--active';
    const order = ['preprocessing', 'tts', 'cadence', 'converting', 'postprocessing'];
    if (stage && Math.min(...stages.map((s) => order.indexOf(s))) < order.indexOf(stage)) {
      return ' station--done';
    }
    return '';
  };

  return (
    <div className="app-shell">
      <div className="topbar">
        <div className="brand">
          <h1>Neural Vocal Studio</h1>
          <span className="brand__sub">Seed-VC · Chatterbox · WORLD</span>
        </div>
        <span className={`led ${loading ? 'led--busy' : serverUp === false ? 'led--err' : serverUp ? 'led--ok' : ''}`}>
          {loading ? status : serverUp === false ? 'Engine offline' : serverUp ? 'Engine ready' : 'Checking…'}
        </span>
      </div>

      <div className="chain">
        {STATIONS.map((s, i) => (
          <React.Fragment key={s.key}>
            {i > 0 && <span className="chain__arrow">→</span>}
            <span className={`station${stationState(s.stages)}`}>{s.label}</span>
          </React.Fragment>
        ))}
      </div>

      <div className="grid-row">
        <div className="panel">
          <div className="panel__header">
            <span>Source</span>
            <div className="seg">
              <button className={`seg__opt${inputMode === 'text' ? ' seg__opt--on' : ''}`}
                onClick={() => setInputMode('text')} disabled={loading}>Lyrics</button>
              <button className={`seg__opt${inputMode === 'audio' ? ' seg__opt--on' : ''}`}
                onClick={() => setInputMode('audio')} disabled={loading}>Audio</button>
            </div>
          </div>
          <div className="panel__body">
            {inputMode === 'text' ? (
              <>
                <textarea
                  className="input"
                  rows={5}
                  placeholder="Type the lyrics or hook…"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={loading}
                  style={{ width: '100%', resize: 'vertical' }}
                />
                <div className="hint">
                  Voice is synthesized locally by Chatterbox, cloning your reference clip.
                </div>
                <details className="adv">
                  <summary>TTS options</summary>
                  <div className="adv__body">
                    <div className="control-row">
                      <label className="label">Edge fallback</label>
                      <Switch checked={voice.edge_fallback} onChange={(v) => setV({ edge_fallback: v })} />
                      <span className="hint">Microsoft Edge TTS: instant but flat robotic delivery.</span>
                    </div>
                    {voice.edge_fallback && (
                      <div className="control-row">
                        <label className="label">Edge voice</label>
                        <select className="select" value={voice.tts_voice}
                          onChange={(e) => setV({ tts_voice: e.target.value })} disabled={loading}>
                          <option value="en-US-GuyNeural">English US Male</option>
                          <option value="en-US-AriaNeural">English US Female</option>
                          <option value="en-GB-RyanNeural">English UK Male</option>
                          <option value="en-GB-SoniaNeural">English UK Female</option>
                          <option value="ja-JP-KeitaNeural">Japanese Male</option>
                          <option value="ja-JP-NanamiNeural">Japanese Female</option>
                          <option value="zh-CN-YunxiNeural">Chinese Male</option>
                          <option value="zh-CN-XiaoxiaoNeural">Chinese Female</option>
                        </select>
                      </div>
                    )}
                  </div>
                </details>
              </>
            ) : (
              <>
                <AudioField label="Source Audio" onChange={setSource} onError={setError} />
                <div className="hint">
                  Upload or record a vocal — sung input converts best. Leave cadence on
                  Natural for material that is already musical.
                </div>
              </>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel__header"><span>Reference Voice</span></div>
          <div className="panel__body">
            <AudioField label="Reference Audio" onChange={setReference} onError={setError} />
            <div className="hint">
              The voice the output will sound like — also used as the cloning prompt
              for lyrics mode. 5–30s of clean solo vocal works best.
            </div>
          </div>
        </div>
      </div>

      <div className="grid-row">
        <div className="panel">
          <div className="panel__header"><span>Cadence &amp; Pitch</span></div>
          <div className="panel__body">
            <div className="control-row">
              <div className="seg">
                {CADENCE_MODES.map((c) => (
                  <button key={c.value}
                    className={`seg__opt${m.cadence_mode === c.value ? ' seg__opt--on' : ''}`}
                    onClick={() => setM({ cadence_mode: c.value })} disabled={loading}>
                    {c.label}
                  </button>
                ))}
              </div>
            </div>
            {m.cadence_mode === 'none' && (
              <div className="hint">No pitch processing — output follows the source's natural speech contour.</div>
            )}
            {m.cadence_mode !== 'none' && (
              <>
                <div className="control-row">
                  <label className="label">Key</label>
                  <select className="select" value={m.key_root}
                    onChange={(e) => setM({ key_root: e.target.value })} disabled={loading}>
                    {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'].map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                  <select className="select" value={m.key_scale}
                    onChange={(e) => setM({ key_scale: e.target.value })} disabled={loading}>
                    <option value="minor">Minor</option>
                    <option value="major">Major</option>
                    <option value="minor_pentatonic">Minor Pentatonic</option>
                    <option value="phrygian">Phrygian</option>
                    <option value="dorian">Dorian</option>
                    <option value="harmonic_minor">Harmonic Minor</option>
                    <option value="chromatic">Chromatic</option>
                  </select>
                </div>
                {m.cadence_mode === 'melody' && (
                  <div className="control-row">
                    <label className="label">Melody MIDI</label>
                    <input className="input" type="file" accept=".mid,.midi"
                      onChange={(e) => setMelodyMidi(e.target.files?.[0] ?? null)} disabled={loading} />
                    {melodyMidi && <span className="badge">{melodyMidi.name}</span>}
                  </div>
                )}
                <details className="adv">
                  <summary>Fine tune</summary>
                  <div className="adv__body">
                    <SliderRow label="Retune speed" value={m.retune_ms} min={0} max={200} step={5} unit="ms"
                      onChange={(v) => setM({ retune_ms: v })} />
                    <SliderRow label="Vibrato" value={m.vibrato_cents} min={0} max={50} step={1} unit="¢"
                      onChange={(v) => setM({ vibrato_cents: v })} />
                    <SliderRow label="Humanize drift" value={m.drift_cents} min={0} max={30} step={1} unit="¢"
                      onChange={(v) => setM({ drift_cents: v })} />
                    <SliderRow label="Octave shift" value={m.octave_shift} min={-2} max={2} step={1}
                      onChange={(v) => setM({ octave_shift: v })} />
                  </div>
                </details>
              </>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel__header"><span>Rhythm</span></div>
          <div className="panel__body">
            <div className="control-row">
              <label className="label">Track BPM</label>
              <input className="input input--num" type="number" placeholder="128"
                value={m.target_bpm ?? ''}
                onChange={(e) => setM({ target_bpm: e.target.value === '' ? null : Number(e.target.value) })}
                disabled={loading} />
            </div>
            <div className="control-row">
              <label className="label">Grid quantize</label>
              <Switch checked={m.grid_quantize} onChange={(v) => setM({ grid_quantize: v })}
                disabled={loading || !m.target_bpm} />
              <select className="select" style={{ width: 110 }} value={m.grid_subdivision}
                onChange={(e) => setM({ grid_subdivision: Number(e.target.value) })}
                disabled={loading || !m.grid_quantize}>
                <option value={2}>1/8 notes</option>
                <option value={4}>1/16 notes</option>
              </select>
            </div>
            {m.grid_quantize && (
              <SliderRow label="Strength" value={m.grid_strength} min={0} max={1} step={0.05}
                onChange={(v) => setM({ grid_strength: v })} />
            )}
            <div className="control-row">
              <label className="label">Tempo stretch</label>
              <Switch checked={m.bpm_stretch} onChange={(v) => setM({ bpm_stretch: v })}
                disabled={loading || !m.target_bpm} />
              <span className="hint">Time-stretch the whole take toward Track BPM.</span>
            </div>
            <div className="hint">
              Quantize is taste, not a rule — A/B it. 0.7–0.9 strength keeps some human pocket;
              never use it on already-sung sources.
            </div>
          </div>
        </div>
      </div>

      <div className="grid-row">
        <div className="panel">
          <div className="panel__header"><span>Output</span></div>
          <div className="panel__body">
            <div className="control-row" style={{ justifyContent: 'space-between' }}>
              <div className="control-row">
                <button className="transport" onClick={submit} disabled={!canGenerate || loading}>
                  {loading ? 'Working…' : 'Generate'}
                </button>
                {outputUrl && (
                  <a className="button" href={outputUrl} download="vocal-stem.wav">Download WAV</a>
                )}
                {outputUrl && (
                  <button className="button" onClick={() => setOutputUrl(null)}>Clear</button>
                )}
              </div>
              <span className="badge">{status}</span>
            </div>
            {error && <div className="error-text">{error}</div>}
            {outputUrl && <audio controls src={outputUrl} />}

            <details className="adv">
              <summary>Post — dry by default</summary>
              <div className="adv__body">
                <div className="control-row">
                  <label className="label">FX preset</label>
                  <select className="select" value={m.fx_preset}
                    onChange={(e) => setM({ fx_preset: e.target.value })} disabled={loading}>
                    {FX_PRESETS.map((p) => (
                      <option key={p.value} value={p.value}>{p.label}</option>
                    ))}
                  </select>
                  <span className="hint">Prefer doing this in the DAW — presets are for quick previews.</span>
                </div>
                <div className="control-row">
                  <label className="label">Trim output</label>
                  <Switch checked={m.trim_output} onChange={(v) => setM({ trim_output: v })} />
                  <span className="hint">Removes AI hallucination tails. Error cleanup, not sound shaping.</span>
                </div>
                <div className="control-row">
                  <label className="label">Normalize</label>
                  <Switch checked={m.normalize_output} onChange={(v) => setM({ normalize_output: v })} />
                  <span className="hint">Preview loudness only; clip-safe. Off = untouched dry stem.</span>
                </div>
              </div>
            </details>

            <details className="adv">
              <summary>Voice model</summary>
              <div className="adv__body">
                <div className="control-row">
                  <label className="label">Model</label>
                  <div className="seg">
                    <button className={`seg__opt${voice.model_mode === 'singing' ? ' seg__opt--on' : ''}`}
                      onClick={() => setV({ model_mode: 'singing' })} disabled={loading}>
                      Singing 44k
                    </button>
                    <button className={`seg__opt${voice.model_mode === 'voice' ? ' seg__opt--on' : ''}`}
                      onClick={() => setV({ model_mode: 'voice' })} disabled={loading}>
                      Speech 22k
                    </button>
                  </div>
                  <span className="hint">Singing follows pitch (needed for cadence modes).</span>
                </div>
                <div className="control-row">
                  <label className="label">Auto F0 adjust</label>
                  <Switch checked={voice.auto_f0_adjust} onChange={(v) => setV({ auto_f0_adjust: v })} />
                  <span className="hint">Shift pitch register toward the reference automatically.</span>
                </div>
                <SliderRow label="Pitch shift" value={voice.pitch_shift} min={-24} max={24} step={1} unit="st"
                  onChange={(v) => setV({ pitch_shift: v })} />
                <SliderRow label="Diffusion steps" value={voice.diffusion_steps} min={1} max={100} step={1}
                  onChange={(v) => setV({ diffusion_steps: v })} />
                <SliderRow label="Length adjust" value={voice.length_adjust} min={0.5} max={2.0} step={0.1} unit="×"
                  onChange={(v) => setV({ length_adjust: v })} />
                <SliderRow label="CFG rate" value={voice.inference_cfg_rate} min={0} max={1} step={0.1}
                  onChange={(v) => setV({ inference_cfg_rate: v })} />
              </div>
            </details>
          </div>
        </div>
      </div>
    </div>
  );
}
