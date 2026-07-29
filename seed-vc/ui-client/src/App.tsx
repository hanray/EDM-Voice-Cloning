import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AudioField, AudioValue } from './components/AudioField';

type Engine = 'v1' | 'v2';
type InputMode = 'audio' | 'text';
type CadenceMode = 'none' | 'chant' | 'autotune' | 'melody';

interface ConversionParamsV1 {
  diffusion_steps: number;
  length_adjust: number;
  inference_cfg_rate: number;
  auto_f0_adjust: boolean;
  pitch_shift: number;
  model_mode: 'voice' | 'singing';
  tts_voice?: string;
}

interface ConversionParamsV2 {
  diffusion_steps: number;
  length_adjust: number;
  intelligibility_cfg_rate: number;
  similarity_cfg_rate: number;
  top_p: number;
  temperature: number;
  repetition_penalty: number;
  convert_style: boolean;
  anonymization_only: boolean;
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
  grid_subdivision: number; // steps per beat: 2 = 1/8, 4 = 1/16
  grid_strength: number;
  fx_preset: string;
  trim_output: boolean;
  normalize_output: boolean;
}

const defaultV1: ConversionParamsV1 = {
  diffusion_steps: 10,
  length_adjust: 1.0,
  inference_cfg_rate: 0.7,
  auto_f0_adjust: false,
  pitch_shift: 0,
  model_mode: 'singing',
  tts_voice: 'en-US-GuyNeural',
};

const defaultV2: ConversionParamsV2 = {
  diffusion_steps: 30,
  length_adjust: 1.0,
  intelligibility_cfg_rate: 0.7,
  similarity_cfg_rate: 0.7,
  top_p: 0.7,
  temperature: 0.7,
  repetition_penalty: 1.5,
  convert_style: false,
  anonymization_only: false,
};

const defaultMusicality: MusicalityParams = {
  cadence_mode: 'none',
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
  grid_strength: 1.0,
  fx_preset: 'none',
  trim_output: true,
  normalize_output: true,
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
      if (received > 0) onStatus?.(`Receiving audio... ${(received / 1024).toFixed(1)} KB`);
    }
  }
  onStatus?.('Finalizing output...');
  const blob = new Blob(chunks as BlobPart[], { type: contentType });
  return URL.createObjectURL(blob);
}

async function convertV1(
  source: File | null,
  target: File,
  params: ConversionParamsV1,
  musicality: MusicalityParams,
  midi: File | null,
  inputMode: InputMode,
  inputText: string,
  onStatus?: (s: string) => void,
): Promise<string> {
  const form = new FormData();
  form.append('target_audio', target);
  form.append('diffusion_steps', String(params.diffusion_steps));
  form.append('length_adjust', String(params.length_adjust));
  form.append('inference_cfg_rate', String(params.inference_cfg_rate));
  // Singing mode IS the f0-conditioned 44.1 kHz model — one selector, wired for real now.
  form.append('f0_condition', String(params.model_mode === 'singing'));
  form.append('auto_f0_adjust', String(params.auto_f0_adjust));
  form.append('pitch_shift', String(params.pitch_shift));
  appendMusicality(form, musicality, midi);

  let endpoint = '/api/v1/convert';
  if (inputMode === 'text') {
    endpoint = '/api/v1/convert_text';
    form.append('text', inputText);
    form.append('tts_voice', params.tts_voice || 'en-US-GuyNeural');
  } else {
    if (!source) throw new Error('Source audio missing');
    form.append('source_audio', source);
  }

  onStatus?.('Uploading audio...');
  const res = await fetch(endpoint, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await readError(res));

  onStatus?.('Converting...');
  return streamToUrl(res, onStatus);
}

async function convertV2(
  source: File | null,
  target: File,
  params: ConversionParamsV2,
  musicality: MusicalityParams,
  midi: File | null,
  inputMode: InputMode,
  inputText: string,
  ttsVoice: string,
  onStatus?: (s: string) => void,
): Promise<string> {
  const form = new FormData();
  form.append('target_audio', target);
  form.append('diffusion_steps', String(params.diffusion_steps));
  form.append('length_adjust', String(params.length_adjust));
  form.append('intelligibility_cfg_rate', String(params.intelligibility_cfg_rate));
  form.append('similarity_cfg_rate', String(params.similarity_cfg_rate));
  form.append('top_p', String(params.top_p));
  form.append('temperature', String(params.temperature));
  form.append('repetition_penalty', String(params.repetition_penalty));
  form.append('convert_style', String(params.convert_style));
  form.append('anonymization_only', String(params.anonymization_only));
  appendMusicality(form, musicality, midi);

  let endpoint = '/api/v2/convert';
  if (inputMode === 'text') {
    endpoint = '/api/v2/convert_text';
    form.append('text', inputText);
    form.append('tts_voice', ttsVoice || 'en-US-GuyNeural');
  } else {
    if (!source) throw new Error('Source audio missing');
    form.append('source_audio', source);
  }

  onStatus?.('Uploading audio...');
  const res = await fetch(endpoint, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await readError(res));

  onStatus?.('Converting...');
  return streamToUrl(res, onStatus);
}

const STAGE_LABELS: Record<string, string> = {
  preprocessing: 'Preparing audio...',
  tts: 'Synthesizing speech...',
  cadence: 'Applying cadence/melody...',
  converting: 'Neural voice conversion...',
  postprocessing: 'Trimming + FX...',
};

function SliderRow({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; }) {
  return (
    <div className="slider-row">
      <div className="label" style={{ width: 180 }}>{label}</div>
      <input className="input" type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
      <div className="badge">{value}</div>
    </div>
  );
}

export default function App() {
  const [engine, setEngine] = useState<Engine>('v1');
  const [source, setSource] = useState<AudioValue | null>(null);
  const [reference, setReference] = useState<AudioValue | null>(null);
  const [paramsV1, setParamsV1] = useState<ConversionParamsV1>(defaultV1);
  const [paramsV2, setParamsV2] = useState<ConversionParamsV2>(defaultV2);
  const [musicality, setMusicality] = useState<MusicalityParams>(defaultMusicality);
  const [melodyMidi, setMelodyMidi] = useState<File | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('Idle');
  const [inputMode, setInputMode] = useState<InputMode>('audio');
  const [inputText, setInputText] = useState<string>('');
  const sseRef = useRef<EventSource | null>(null);

  // Live pipeline status from the server while a conversion runs
  useEffect(() => {
    if (!loading) {
      sseRef.current?.close();
      sseRef.current = null;
      return;
    }
    const es = new EventSource('/api/events/status');
    es.onmessage = (ev) => {
      try {
        const snap = JSON.parse(ev.data);
        if (snap.state === 'busy' && snap.stage) {
          const label = STAGE_LABELS[snap.stage] || snap.stage;
          const queued = snap.waiting > 0 ? ` (${snap.waiting} queued)` : '';
          setStatus(`${label}${queued}`);
        }
      } catch {
        /* ignore malformed frames */
      }
    };
    es.onerror = () => {
      /* server may not be reachable mid-restart; fetch errors surface elsewhere */
    };
    sseRef.current = es;
    return () => {
      es.close();
      sseRef.current = null;
    };
  }, [loading]);

  const canConvert = useMemo(() => {
    if (!reference) return false;
    if (inputMode === 'text') return inputText.trim().length > 0;
    return !!source;
  }, [inputMode, inputText, reference, source]);

  const submit = async () => {
    if (!reference) return;
    if (inputMode === 'audio' && !source) {
      setError('Source audio is required');
      return;
    }
    if (inputMode === 'text' && inputText.trim().length === 0) {
      setError('Please enter lyrics/text');
      return;
    }
    if (musicality.cadence_mode === 'melody' && !melodyMidi) {
      setError('Melody mode needs a MIDI file');
      return;
    }
    setError(null);
    setStatus('Preparing request...');
    setLoading(true);
    try {
      const url = engine === 'v2'
        ? await convertV2(source?.file || null, reference.file, paramsV2, musicality, melodyMidi, inputMode, inputText, paramsV1.tts_voice || 'en-US-GuyNeural', setStatus)
        : await convertV1(source?.file || null, reference.file, paramsV1, musicality, melodyMidi, inputMode, inputText, setStatus);
      setOutputUrl(url);
      setStatus('Completed');
    } catch (e: any) {
      setError(e?.message || 'Failed to convert');
      setStatus('Failed');
    } finally {
      setLoading(false);
    }
  };

  const m = musicality;
  const setM = (patch: Partial<MusicalityParams>) => setMusicality({ ...m, ...patch });

  return (
    <div className="app-shell">
      <h1 style={{ marginBottom: '1rem', letterSpacing: '0.04em', textTransform: 'uppercase' }}>EDM Neural Vocal Studio</h1>

      <div className="control-row" style={{ alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
        <span className="label" style={{ width: 120 }}>Engine</span>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className="button"
            style={{ opacity: engine === 'v1' ? 1 : 0.7 }}
            onClick={() => setEngine('v1')}
            disabled={loading}
          >
            V1 (DiT)
          </button>
          <button
            className="button"
            style={{ opacity: engine === 'v2' ? 1 : 0.7 }}
            onClick={() => setEngine('v2')}
            disabled={loading}
          >
            V2 (Streaming)
          </button>
        </div>
      </div>

      <div className="grid-row">
        <div className="panel">
          <div className="panel__header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span>Source Input</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <button
                className="button"
                style={{ padding: '0.35rem 0.6rem', fontSize: '0.9rem' }}
                onClick={() => {
                  setInputMode(inputMode === 'audio' ? 'text' : 'audio');
                  setStatus('Idle');
                  setError(null);
                }}
                disabled={loading}
                title={inputMode === 'audio' ? 'Switch to lyrics/text input' : 'Switch to audio upload'}
              >
                {inputMode === 'audio' ? '✎ Lyrics' : '🎙️ Audio'}
              </button>
            </div>
          </div>
          <div className="panel__body">
            {inputMode === 'audio' ? (
              <AudioField label="Source Audio" onChange={setSource} onError={setError} />
            ) : (
              <div className="control-row" style={{ flexDirection: 'column', gap: '0.5rem' }}>
                <label className="label">Lyrics / Text</label>
                <textarea
                  className="input"
                  rows={4}
                  placeholder="Enter lyrics or text to synthesize"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={loading}
                  style={{ width: '100%' }}
                />
                <div className="control-row" style={{ gap: '0.5rem', alignItems: 'center' }}>
                  <label className="label">TTS Voice</label>
                  <select
                    className="select"
                    value={paramsV1.tts_voice}
                    onChange={(e) => setParamsV1({ ...paramsV1, tts_voice: e.target.value })}
                    disabled={loading}
                  >
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
              </div>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel__header">Reference Voice</div>
          <div className="panel__body">
            <AudioField label="Reference Audio" onChange={setReference} onError={setError} />
          </div>
        </div>
      </div>

      <div className="grid-row">
        <div className="panel">
          <div className="panel__header">Musicality — Cadence &amp; Pitch</div>
          <div className="panel__body">
            <div className="control-row">
              <label className="label">Cadence Mode</label>
              <select className="select" value={m.cadence_mode} onChange={(e) => setM({ cadence_mode: e.target.value as CadenceMode })} disabled={loading}>
                <option value="none">Natural (speech)</option>
                <option value="chant">Chant (root-note monotone)</option>
                <option value="autotune">Autotune (scale quantize)</option>
                <option value="melody">Melody (MIDI file)</option>
              </select>
            </div>
            {m.cadence_mode !== 'none' && (
              <>
                <div className="control-row">
                  <label className="label">Key</label>
                  <select className="select" value={m.key_root} onChange={(e) => setM({ key_root: e.target.value })} disabled={loading}>
                    {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'].map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                  <select className="select" value={m.key_scale} onChange={(e) => setM({ key_scale: e.target.value })} disabled={loading}>
                    <option value="minor">Minor</option>
                    <option value="major">Major</option>
                    <option value="minor_pentatonic">Minor Pentatonic</option>
                    <option value="phrygian">Phrygian</option>
                    <option value="dorian">Dorian</option>
                    <option value="harmonic_minor">Harmonic Minor</option>
                    <option value="chromatic">Chromatic</option>
                  </select>
                </div>
                <SliderRow label="Retune Speed (ms)" value={m.retune_ms} min={0} max={200} step={5} onChange={(v) => setM({ retune_ms: v })} />
                <SliderRow label="Vibrato (cents)" value={m.vibrato_cents} min={0} max={50} step={1} onChange={(v) => setM({ vibrato_cents: v })} />
                <SliderRow label="Drift / Humanize (cents)" value={m.drift_cents} min={0} max={30} step={1} onChange={(v) => setM({ drift_cents: v })} />
                <SliderRow label="Octave Shift" value={m.octave_shift} min={-2} max={2} step={1} onChange={(v) => setM({ octave_shift: v })} />
                {m.cadence_mode === 'melody' && (
                  <div className="control-row">
                    <label className="label">Melody MIDI</label>
                    <input
                      className="input"
                      type="file"
                      accept=".mid,.midi"
                      onChange={(e) => setMelodyMidi(e.target.files?.[0] ?? null)}
                      disabled={loading}
                    />
                    {melodyMidi && <span className="badge">{melodyMidi.name}</span>}
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel__header">Musicality — Rhythm &amp; FX</div>
          <div className="panel__body">
            <div className="control-row">
              <label className="label">Target BPM</label>
              <input className="input" type="number" value={m.target_bpm ?? ''} placeholder="e.g. 128"
                onChange={(e) => setM({ target_bpm: e.target.value === '' ? null : Number(e.target.value) })} disabled={loading} />
              <label className="label">Tempo Stretch</label>
              <input type="checkbox" checked={m.bpm_stretch} onChange={(e) => setM({ bpm_stretch: e.target.checked })} disabled={loading} />
            </div>
            <div className="control-row">
              <label className="label">Grid Quantize</label>
              <input type="checkbox" checked={m.grid_quantize} onChange={(e) => setM({ grid_quantize: e.target.checked })} disabled={loading || !m.target_bpm} />
              <select className="select" value={m.grid_subdivision} onChange={(e) => setM({ grid_subdivision: Number(e.target.value) })} disabled={loading || !m.grid_quantize}>
                <option value={2}>1/8 notes</option>
                <option value={4}>1/16 notes</option>
              </select>
            </div>
            {m.grid_quantize && (
              <SliderRow label="Quantize Strength" value={m.grid_strength} min={0} max={1} step={0.05} onChange={(v) => setM({ grid_strength: v })} />
            )}
            <div className="control-row">
              <label className="label">FX Preset</label>
              <select className="select" value={m.fx_preset} onChange={(e) => setM({ fx_preset: e.target.value })} disabled={loading}>
                {FX_PRESETS.map((p) => (
                  <option key={p.value} value={p.value}>{p.label}</option>
                ))}
              </select>
            </div>
            <div className="control-row">
              <label className="label">Trim Output</label>
              <input type="checkbox" checked={m.trim_output} onChange={(e) => setM({ trim_output: e.target.checked })} disabled={loading} />
              <label className="label">Normalize</label>
              <input type="checkbox" checked={m.normalize_output} onChange={(e) => setM({ normalize_output: e.target.checked })} disabled={loading} />
            </div>
          </div>
        </div>
      </div>

      {engine === 'v1' ? (
        <div className="grid-row">
          <div className="panel">
            <div className="panel__header">Neural Voice Settings (V1)</div>
            <div className="panel__body">
              <SliderRow label="Diffusion Steps" value={paramsV1.diffusion_steps} min={1} max={200} step={1} onChange={(v) => setParamsV1({ ...paramsV1, diffusion_steps: v })} />
              <SliderRow label="Length Adjust" value={paramsV1.length_adjust} min={0.5} max={2.0} step={0.1} onChange={(v) => setParamsV1({ ...paramsV1, length_adjust: v })} />
              <SliderRow label="CFG Rate" value={paramsV1.inference_cfg_rate} min={0} max={1} step={0.1} onChange={(v) => setParamsV1({ ...paramsV1, inference_cfg_rate: v })} />
            </div>
          </div>

          <div className="panel">
            <div className="panel__header">Singing / F0 (V1)</div>
            <div className="panel__body">
              <div className="control-row">
                <label className="label">Model Mode</label>
                <select className="select" value={paramsV1.model_mode} onChange={(e) => setParamsV1({ ...paramsV1, model_mode: e.target.value as 'voice' | 'singing' })} disabled={loading}>
                  <option value="voice">Voice (22kHz)</option>
                  <option value="singing">Singing (44kHz, follows pitch)</option>
                </select>
              </div>
              <div className="control-row">
                <label className="label">Auto F0 Adjust</label>
                <input type="checkbox" checked={paramsV1.auto_f0_adjust} onChange={(e) => setParamsV1({ ...paramsV1, auto_f0_adjust: e.target.checked })} />
              </div>
              <SliderRow label="Pitch Shift (semitones)" value={paramsV1.pitch_shift} min={-24} max={24} step={1} onChange={(v) => setParamsV1({ ...paramsV1, pitch_shift: v })} />
            </div>
          </div>
        </div>
      ) : (
        <div className="grid-row">
          <div className="panel">
            <div className="panel__header">Generation Settings (V2)</div>
            <div className="panel__body">
              <SliderRow label="Diffusion Steps" value={paramsV2.diffusion_steps} min={1} max={200} step={1} onChange={(v) => setParamsV2({ ...paramsV2, diffusion_steps: v })} />
              <SliderRow label="Length Adjust" value={paramsV2.length_adjust} min={0.5} max={2.0} step={0.1} onChange={(v) => setParamsV2({ ...paramsV2, length_adjust: v })} />
              <SliderRow label="Intelligibility CFG" value={paramsV2.intelligibility_cfg_rate} min={0} max={1} step={0.05} onChange={(v) => setParamsV2({ ...paramsV2, intelligibility_cfg_rate: v })} />
              <SliderRow label="Similarity CFG" value={paramsV2.similarity_cfg_rate} min={0} max={1} step={0.05} onChange={(v) => setParamsV2({ ...paramsV2, similarity_cfg_rate: v })} />
              <SliderRow label="Top-p" value={paramsV2.top_p} min={0.1} max={1} step={0.05} onChange={(v) => setParamsV2({ ...paramsV2, top_p: v })} />
              <SliderRow label="Temperature" value={paramsV2.temperature} min={0.1} max={2.0} step={0.1} onChange={(v) => setParamsV2({ ...paramsV2, temperature: v })} />
              <SliderRow label="Repetition Penalty" value={paramsV2.repetition_penalty} min={0.8} max={3.0} step={0.1} onChange={(v) => setParamsV2({ ...paramsV2, repetition_penalty: v })} />
              <div className="control-row" style={{ gap: '0.75rem', alignItems: 'center' }}>
                <label className="label">Convert Style</label>
                <input type="checkbox" checked={paramsV2.convert_style} onChange={(e) => setParamsV2({ ...paramsV2, convert_style: e.target.checked })} />
                <label className="label">Anonymize Only</label>
                <input type="checkbox" checked={paramsV2.anonymization_only} onChange={(e) => setParamsV2({ ...paramsV2, anonymization_only: e.target.checked })} />
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid-row">
        <div className="panel">
          <div className="panel__header">Output</div>
          <div className="panel__body">
            <div className="control-row">
              <button className="button" onClick={submit} disabled={!canConvert || loading}>{loading ? 'Converting...' : 'Generate Neural Vocal'}</button>
              <button className="button" onClick={() => setOutputUrl(null)} disabled={!outputUrl}>Clear Output</button>
            </div>
            <div className="badge" style={{ marginTop: '0.35rem' }}>Status: {status}</div>
            {error && <div style={{ color: '#fca5a5' }}>{error}</div>}
            {outputUrl ? (
              <audio controls src={outputUrl} style={{ width: '100%' }} />
            ) : (
              <div className="badge">No output yet</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
