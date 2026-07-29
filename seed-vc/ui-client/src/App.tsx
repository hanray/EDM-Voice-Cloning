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
  edge_fallback: boolean;
  smooth_punctuation: boolean;
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
  debug_stems: boolean;
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
  smooth_punctuation: true,
};

const defaultMusicality: MusicalityParams = {
  // Natural by default while enunciation is being dialed in — the user
  // A/Bs cadence per take. Chant is one chip-click away.
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
  grid_strength: 0.85,
  fx_preset: 'none',
  trim_output: true,
  normalize_output: false, // dry stems — level decisions belong in the DAW
  debug_stems: true, // auto-save all pipeline stages while dialing in quality
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

const SCALE_LABELS: Record<string, string> = {
  minor: 'min',
  major: 'maj',
  minor_pentatonic: 'min pent',
  phrygian: 'phry',
  dorian: 'dor',
  harmonic_minor: 'harm min',
  chromatic: 'chrom',
};

const CADENCE_LABELS: Record<CadenceMode, string> = {
  chant: 'Chant',
  autotune: 'Autotune',
  melody: 'Melody',
  none: 'Natural',
};

const ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const STAGE_ORDER = ['preprocessing', 'tts', 'cadence', 'converting', 'postprocessing'];
const STAGE_STEPS = [
  { label: 'SRC', stages: ['preprocessing', 'tts'] },
  { label: 'CAD', stages: ['cadence'] },
  { label: 'CONV', stages: ['converting'] },
  { label: 'POST', stages: ['postprocessing'] },
];

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (data?.error) return data.error;
  } catch { /* not JSON */ }
  return `Conversion failed (HTTP ${res.status})`;
}

interface ConvertResult {
  url: string;
  savedTo: string | null;
}

async function streamToUrl(res: Response, onStatus?: (s: string) => void): Promise<ConvertResult> {
  const contentType = res.headers.get('content-type') || 'audio/wav';
  const savedTo = res.headers.get('X-Saved-To');
  const reader = res.body?.getReader();
  if (!reader) {
    const blob = await res.blob();
    return { url: URL.createObjectURL(blob), savedTo };
  }
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      received += value.length;
      onStatus?.(`Receiving — ${(received / 1024).toFixed(0)} KB`);
    }
  }
  const blob = new Blob(chunks as BlobPart[], { type: contentType });
  return { url: URL.createObjectURL(blob), savedTo };
}

async function convert(
  source: File | null,
  target: File,
  voice: VoiceParams,
  m: MusicalityParams,
  midi: File | null,
  inputMode: InputMode,
  inputText: string,
  onStatus?: (s: string) => void,
): Promise<ConvertResult> {
  const form = new FormData();
  form.append('target_audio', target);
  form.append('diffusion_steps', String(voice.diffusion_steps));
  form.append('length_adjust', String(voice.length_adjust));
  form.append('inference_cfg_rate', String(voice.inference_cfg_rate));
  form.append('f0_condition', String(voice.model_mode === 'singing'));
  form.append('auto_f0_adjust', String(voice.auto_f0_adjust));
  form.append('pitch_shift', String(voice.pitch_shift));
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
  form.append('debug_stems', String(m.debug_stems));
  if (m.cadence_mode === 'melody' && midi) form.append('melody_midi', midi);

  let endpoint = '/api/v1/convert';
  if (inputMode === 'text') {
    endpoint = '/api/v1/convert_text';
    form.append('text', inputText);
    form.append('tts_engine', voice.edge_fallback ? 'edge' : 'chatterbox');
    form.append('tts_voice', voice.tts_voice);
    form.append('smooth_punctuation', String(voice.smooth_punctuation));
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

function Switch({ checked, onChange, disabled }: {
  checked: boolean; onChange: (v: boolean) => void; disabled?: boolean;
}) {
  return (
    <input type="checkbox" className="switch" checked={checked} disabled={disabled}
      onChange={(e) => onChange(e.target.checked)} />
  );
}

function Chip({ id, k, v, set, alert, open, onToggle, right, children }: {
  id: string; k: string; v: string; set?: boolean; alert?: boolean;
  open: string | null; onToggle: (id: string | null) => void;
  right?: boolean; children: React.ReactNode;
}) {
  const isOpen = open === id;
  return (
    <div className={`chip-wrap${right ? ' chip-wrap--right' : ''}`}>
      <button
        className={`chip${isOpen ? ' chip--open' : ''}${alert ? ' chip--alert' : ''}`}
        onClick={() => onToggle(isOpen ? null : id)}
      >
        <span className="chip__k">{k}</span>
        <span className={`chip__v${set ? ' chip__v--set' : ''}`}>{v}</span>
      </button>
      {isOpen && <div className="pop">{children}</div>}
    </div>
  );
}

export default function App() {
  const [source, setSource] = useState<AudioValue | null>(null);
  const [reference, setReference] = useState<AudioValue | null>(null);
  const [voice, setVoice] = useState<VoiceParams>(defaultVoice);
  const [m, setMusicality] = useState<MusicalityParams>(defaultMusicality);
  const [melodyMidi, setMelodyMidi] = useState<File | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [savedTo, setSavedTo] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('Ready');
  const [stage, setStage] = useState<string | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [inputText, setInputText] = useState<string>('');
  const [serverUp, setServerUp] = useState<boolean | null>(null);
  const [openChip, setOpenChip] = useState<string | null>(null);
  const sseRef = useRef<EventSource | null>(null);

  const setM = (patch: Partial<MusicalityParams>) => setMusicality({ ...m, ...patch });
  const setV = (patch: Partial<VoiceParams>) => setVoice({ ...voice, ...patch });

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
          setStatus(snap.waiting > 0 ? `${snap.stage} · ${snap.waiting} queued` : snap.stage);
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

  const submit = async () => {
    if (!reference) { setError('Add a reference voice first'); setOpenChip('ref'); return; }
    if (m.cadence_mode === 'melody' && !melodyMidi) {
      setError('Melody mode needs a MIDI file');
      setOpenChip('cadence');
      return;
    }
    setOpenChip(null);
    setError(null);
    setStatus('Starting…');
    setLoading(true);
    try {
      const result = await convert(
        source?.file || null, reference.file, voice, m, melodyMidi,
        inputMode, inputText, setStatus,
      );
      setOutputUrl(result.url);
      setSavedTo(result.savedTo);
      setStatus('Done');
    } catch (e: any) {
      setError(e?.message || 'Failed to convert');
      setStatus('Failed');
    } finally {
      setLoading(false);
    }
  };

  const stepClass = (stages: string[]) => {
    if (!loading || !stage) return '';
    if (stages.includes(stage)) return ' stagebar__step--on';
    if (Math.min(...stages.map((s) => STAGE_ORDER.indexOf(s))) < STAGE_ORDER.indexOf(stage)) {
      return ' stagebar__step--done';
    }
    return '';
  };

  const postSummary = useMemo(() => {
    const parts: string[] = [];
    if (m.fx_preset !== 'none') parts.push(FX_PRESETS.find((p) => p.value === m.fx_preset)?.label || m.fx_preset);
    if (m.normalize_output) parts.push('norm');
    return parts.length ? parts.join(' · ') : 'Dry';
  }, [m.fx_preset, m.normalize_output]);

  const rhythmSummary = useMemo(() => {
    if (!m.target_bpm) return 'Free';
    const bits = [`${m.target_bpm}`];
    if (m.grid_quantize) bits.push(m.grid_subdivision === 4 ? '1/16' : '1/8');
    if (m.bpm_stretch) bits.push('stretch');
    return bits.join(' · ');
  }, [m.target_bpm, m.grid_quantize, m.grid_subdivision, m.bpm_stretch]);

  return (
    <div className="app-shell">
      {openChip && <div className="pop-backdrop" onClick={() => setOpenChip(null)} />}

      <div className="topbar">
        <div className="brand"><h1>Neural Vocal Studio</h1></div>
        <span className={`led ${loading ? 'led--busy' : serverUp === false ? 'led--err' : serverUp ? 'led--ok' : ''}`}>
          {loading ? status : serverUp === false ? 'Engine offline' : serverUp ? 'Engine ready' : 'Checking…'}
        </span>
      </div>

      <div className="hero">
        <div className="hero__inner">
          <div className="hero__row">
            <div className="hero__main">
              <div className="hero__tabs">
                <button className={`hero__tab${inputMode === 'text' ? ' hero__tab--on' : ''}`}
                  onClick={() => setInputMode('text')} disabled={loading}>Lyrics</button>
                <button className={`hero__tab${inputMode === 'audio' ? ' hero__tab--on' : ''}`}
                  onClick={() => setInputMode('audio')} disabled={loading}>Audio in</button>
              </div>
              {inputMode === 'text' ? (
                <textarea
                  placeholder="Type the hook. It comes back in your reference voice, in key, on the grid."
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  disabled={loading}
                />
              ) : (
                <AudioField label="Source Audio" onChange={setSource} onError={setError} />
              )}
            </div>

            <div className="orb-stage">
              <div
                className={[
                  'orb',
                  `orb--${m.cadence_mode === 'none' ? 'natural' : m.cadence_mode}`,
                  reference ? 'orb--armed' : '',
                  m.grid_quantize ? 'orb--grid' : '',
                  loading ? 'orb--busy' : '',
                  outputUrl && !loading ? 'orb--done' : '',
                ].filter(Boolean).join(' ')}
                style={{
                  ['--orb-hue' as any]: `${ROOTS.indexOf(m.key_root) * 30}deg`,
                  ['--orb-speed' as any]: m.target_bpm ? `${(240 / m.target_bpm).toFixed(2)}s` : '4s',
                }}
              >
                <div className="orb__halo" />
                <div className="orb__layer orb__layer--a" />
                <div className="orb__layer orb__layer--b" />
                <div className="orb__layer orb__layer--c" />
                <div className="orb__core" />
                <div className="orb__ring" />
              </div>
              <div className={`orb-caption${loading ? ' orb-caption--live' : outputUrl ? ' orb-caption--done' : ''}`}>
                {loading
                  ? (stage || 'working')
                  : outputUrl
                    ? 'stem ready'
                    : `${m.key_root} ${SCALE_LABELS[m.key_scale] || m.key_scale} · ${CADENCE_LABELS[m.cadence_mode].toLowerCase()}${m.target_bpm ? ` · ${m.target_bpm}` : ''}`}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="rail">
        <Chip id="ref" k="Ref" open={openChip} onToggle={setOpenChip}
          v={reference ? (reference.file.name.length > 18 ? reference.file.name.slice(0, 16) + '…' : reference.file.name) : 'none'}
          set={!!reference} alert={!reference}>
          <p className="pop__title">Reference voice</p>
          <AudioField label="Reference Audio" onChange={setReference} onError={setError} />
          <p className="hint">The output voice. Also the cloning prompt in lyrics mode. 5–30s of clean solo vocal.</p>
        </Chip>

        <Chip id="cadence" k="Cadence" open={openChip} onToggle={setOpenChip}
          v={CADENCE_LABELS[m.cadence_mode]} set={m.cadence_mode !== 'none'}>
          <p className="pop__title">Cadence</p>
          <div className="seg">
            {(Object.keys(CADENCE_LABELS) as CadenceMode[]).map((c) => (
              <button key={c} className={`seg__opt${m.cadence_mode === c ? ' seg__opt--on' : ''}`}
                onClick={() => setM({ cadence_mode: c })}>{CADENCE_LABELS[c]}</button>
            ))}
          </div>
          {m.cadence_mode === 'melody' && (
            <div className="control-row">
              <label className="label">MIDI</label>
              <input className="input" type="file" accept=".mid,.midi"
                onChange={(e) => setMelodyMidi(e.target.files?.[0] ?? null)} />
              {melodyMidi && <span className="badge">{melodyMidi.name}</span>}
            </div>
          )}
          {m.cadence_mode !== 'none' && (
            <>
              <SliderRow label="Retune" value={m.retune_ms} min={0} max={200} step={5} unit="ms"
                onChange={(v) => setM({ retune_ms: v })} />
              <SliderRow label="Vibrato" value={m.vibrato_cents} min={0} max={50} step={1} unit="¢"
                onChange={(v) => setM({ vibrato_cents: v })} />
              <SliderRow label="Drift" value={m.drift_cents} min={0} max={30} step={1} unit="¢"
                onChange={(v) => setM({ drift_cents: v })} />
              <SliderRow label="Octave" value={m.octave_shift} min={-2} max={2} step={1}
                onChange={(v) => setM({ octave_shift: v })} />
            </>
          )}
          {m.cadence_mode === 'none' && (
            <p className="hint">No pitch processing — output follows the source's natural contour.</p>
          )}
        </Chip>

        {m.cadence_mode !== 'none' && (
          <Chip id="key" k="Key" open={openChip} onToggle={setOpenChip}
            v={`${m.key_root} ${SCALE_LABELS[m.key_scale] || m.key_scale}`} set>
            <p className="pop__title">Key</p>
            <div className="control-row">
              <select className="select" value={m.key_root} onChange={(e) => setM({ key_root: e.target.value })}>
                {ROOTS.map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
              <select className="select" value={m.key_scale} onChange={(e) => setM({ key_scale: e.target.value })}>
                <option value="minor">Minor</option>
                <option value="major">Major</option>
                <option value="minor_pentatonic">Minor Pentatonic</option>
                <option value="phrygian">Phrygian</option>
                <option value="dorian">Dorian</option>
                <option value="harmonic_minor">Harmonic Minor</option>
                <option value="chromatic">Chromatic</option>
              </select>
            </div>
          </Chip>
        )}

        <Chip id="rhythm" k="BPM" open={openChip} onToggle={setOpenChip}
          v={rhythmSummary} set={!!m.target_bpm}>
          <p className="pop__title">Rhythm</p>
          <div className="control-row">
            <label className="label">Track BPM</label>
            <input className="input input--num" type="number" placeholder="128"
              value={m.target_bpm ?? ''}
              onChange={(e) => setM({ target_bpm: e.target.value === '' ? null : Number(e.target.value) })} />
          </div>
          <div className="control-row">
            <label className="label">Grid snap</label>
            <Switch checked={m.grid_quantize} onChange={(v) => setM({ grid_quantize: v })} disabled={!m.target_bpm} />
            <select className="select" style={{ width: 100 }} value={m.grid_subdivision}
              onChange={(e) => setM({ grid_subdivision: Number(e.target.value) })} disabled={!m.grid_quantize}>
              <option value={2}>1/8</option>
              <option value={4}>1/16</option>
            </select>
          </div>
          {m.grid_quantize && (
            <SliderRow label="Strength" value={m.grid_strength} min={0} max={1} step={0.05}
              onChange={(v) => setM({ grid_strength: v })} />
          )}
          <div className="control-row">
            <label className="label">Stretch</label>
            <Switch checked={m.bpm_stretch} onChange={(v) => setM({ bpm_stretch: v })} disabled={!m.target_bpm} />
            <span className="hint">Time-stretch the whole take to BPM.</span>
          </div>
          <p className="hint">Snap is taste — A/B it. 0.7–0.9 keeps pocket. Never on sung sources.</p>
        </Chip>

        {inputMode === 'text' && (
          <Chip id="tts" k="TTS" open={openChip} onToggle={setOpenChip}
            v={voice.edge_fallback ? 'Edge' : 'Chatterbox'} set={!voice.edge_fallback}>
            <p className="pop__title">Text-to-speech</p>
            <div className="control-row">
              <label className="label">Edge fallback</label>
              <Switch checked={voice.edge_fallback} onChange={(v) => setV({ edge_fallback: v })} />
            </div>
            <div className="control-row">
              <label className="label">Smooth punct.</label>
              <Switch checked={voice.smooth_punctuation} onChange={(v) => setV({ smooth_punctuation: v })} />
              <span className="hint">Strips commas before TTS — cleaner enunciation.</span>
            </div>
            {voice.edge_fallback ? (
              <div className="control-row">
                <label className="label">Voice</label>
                <select className="select" value={voice.tts_voice} onChange={(e) => setV({ tts_voice: e.target.value })}>
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
            ) : (
              <p className="hint">Chatterbox runs locally and clones your reference clip — expressive input, better conversions. Edge is instant but flat.</p>
            )}
          </Chip>
        )}

        <Chip id="model" k="Model" open={openChip} onToggle={setOpenChip}
          v={voice.model_mode === 'singing' ? 'Singing 44k' : 'Speech 22k'} set={voice.model_mode === 'singing'}>
          <p className="pop__title">Voice model</p>
          <div className="seg">
            <button className={`seg__opt${voice.model_mode === 'singing' ? ' seg__opt--on' : ''}`}
              onClick={() => setV({ model_mode: 'singing' })}>Singing 44k</button>
            <button className={`seg__opt${voice.model_mode === 'voice' ? ' seg__opt--on' : ''}`}
              onClick={() => setV({ model_mode: 'voice' })}>Speech 22k</button>
          </div>
          <p className="hint">Singing follows pitch — required for cadence modes.</p>
          <div className="control-row">
            <label className="label">Auto F0</label>
            <Switch checked={voice.auto_f0_adjust} onChange={(v) => setV({ auto_f0_adjust: v })} />
            <span className="hint">Match pitch register to reference.</span>
          </div>
          <SliderRow label="Pitch" value={voice.pitch_shift} min={-24} max={24} step={1} unit="st"
            onChange={(v) => setV({ pitch_shift: v })} />
          <SliderRow label="Steps" value={voice.diffusion_steps} min={1} max={100} step={1}
            onChange={(v) => setV({ diffusion_steps: v })} />
          <SliderRow label="Length" value={voice.length_adjust} min={0.5} max={2.0} step={0.1} unit="×"
            onChange={(v) => setV({ length_adjust: v })} />
          <SliderRow label="CFG" value={voice.inference_cfg_rate} min={0} max={1} step={0.1}
            onChange={(v) => setV({ inference_cfg_rate: v })} />
        </Chip>

        <Chip id="post" k="Post" open={openChip} onToggle={setOpenChip}
          v={postSummary} set={postSummary !== 'Dry'} right>
          <p className="pop__title">Post — dry by default</p>
          <div className="control-row">
            <label className="label">FX preset</label>
            <select className="select" value={m.fx_preset} onChange={(e) => setM({ fx_preset: e.target.value })}>
              {FX_PRESETS.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
            </select>
          </div>
          <div className="control-row">
            <label className="label">Trim tails</label>
            <Switch checked={m.trim_output} onChange={(v) => setM({ trim_output: v })} />
            <span className="hint">Removes AI hallucination tails only.</span>
          </div>
          <div className="control-row">
            <label className="label">Normalize</label>
            <Switch checked={m.normalize_output} onChange={(v) => setM({ normalize_output: v })} />
            <span className="hint">Preview loudness, clip-safe. Off = untouched stem.</span>
          </div>
          <div className="control-row">
            <label className="label">Debug stems</label>
            <Switch checked={m.debug_stems} onChange={(v) => setM({ debug_stems: v })} />
            <span className="hint">Auto-saves every stage (raw TTS → cadence → converted → final) as WAVs in the outputs folder.</span>
          </div>
          <p className="hint">Mixing belongs in the DAW — presets are for quick previews.</p>
        </Chip>
      </div>

      <div className="deck">
        <button className={`transport${loading ? ' transport--busy' : ''}`}
          onClick={submit} disabled={!canGenerate || loading}>
          {loading ? 'Working' : 'Generate'}
        </button>
        {loading && (
          <span className="stagebar">
            {STAGE_STEPS.map((s, i) => (
              <React.Fragment key={s.label}>
                {i > 0 && <span>—</span>}
                <span className={`stagebar__step${stepClass(s.stages)}`}>{s.label}</span>
              </React.Fragment>
            ))}
          </span>
        )}
        {!loading && !canGenerate && (
          <span className="hint">
            {!reference ? 'Add a reference voice to start.' : inputMode === 'text' ? 'Type some lyrics.' : 'Add source audio.'}
          </span>
        )}
      </div>

      {error && <p className="error-text" style={{ marginTop: '0.9rem' }}>{error}</p>}

      {outputUrl && (
        <div className="outbar">
          <div className="outbar__row">
            <span className="badge">{status}</span>
            <a className="button" href={outputUrl} download="vocal-stem.wav">Download WAV</a>
            <button className="button" onClick={() => { setOutputUrl(null); setSavedTo(null); }}>Clear</button>
          </div>
          <audio controls src={outputUrl} />
          {savedTo && (
            <p className="hint">
              Saved to <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)', userSelect: 'all' }}>{savedTo}</span>
              {m.debug_stems && ' — stages 01–03 + final.wav inside. The stage where quality drops is your culprit.'}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
