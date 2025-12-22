import React, { useEffect, useMemo, useState } from 'react';
import { AudioField, AudioValue } from './components/AudioField';

type Engine = 'v1' | 'v2';
type InputMode = 'audio' | 'text';

interface ConversionParamsV1 {
  diffusion_steps: number;
  length_adjust: number;
  inference_cfg_rate: number;
  f0_condition: boolean;
  auto_f0_adjust: boolean;
  pitch_shift: number;
  model_mode: 'voice' | 'singing';
  target_bpm?: number | null;
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
  target_bpm?: number | null;
}

const defaultV1: ConversionParamsV1 = {
  diffusion_steps: 10,
  length_adjust: 1.0,
  inference_cfg_rate: 0.7,
  f0_condition: false,
  auto_f0_adjust: false,
  pitch_shift: 0,
  model_mode: 'voice',
  target_bpm: null,
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
  target_bpm: null,
};

async function streamToUrl(res: Response, onStatus?: (s: string) => void): Promise<string> {
  const contentType = res.headers.get('content-type') || 'audio/mpeg';
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
  const blob = new Blob(chunks, { type: contentType });
  return URL.createObjectURL(blob);
}

async function convertV1(
  source: File | null,
  target: File,
  params: ConversionParamsV1,
  inputMode: InputMode,
  inputText: string,
  onStatus?: (s: string) => void,
): Promise<string> {
  const form = new FormData();
  form.append('target_audio', target);
  form.append('diffusion_steps', String(params.diffusion_steps));
  form.append('length_adjust', String(params.length_adjust));
  form.append('inference_cfg_rate', String(params.inference_cfg_rate));
  form.append('f0_condition', String(params.f0_condition));
  form.append('auto_f0_adjust', String(params.auto_f0_adjust));
  form.append('pitch_shift', String(params.pitch_shift));
  if (params.target_bpm !== null && params.target_bpm !== undefined) {
    form.append('target_bpm', String(params.target_bpm));
  }

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
  if (!res.ok) throw new Error('Conversion failed');

  onStatus?.('Converting (streaming)...');
  return streamToUrl(res, onStatus);
}

async function convertV2(
  source: File | null,
  target: File,
  params: ConversionParamsV2,
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
  if (params.target_bpm !== null && params.target_bpm !== undefined) {
    form.append('target_bpm', String(params.target_bpm));
  }

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
  if (!res.ok) throw new Error('Conversion failed');

  onStatus?.('Converting (streaming)...');
  return streamToUrl(res, onStatus);
}

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
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('Idle');
  const [inputMode, setInputMode] = useState<InputMode>('audio');
  const [inputText, setInputText] = useState<string>('');

  const canConvert = useMemo(() => {
    if (!reference) return false;
    if (engine === 'v2') {
      if (inputMode === 'text') return inputText.trim().length > 0;
      return !!source;
    }
    if (inputMode === 'audio') return !!source;
    return inputText.trim().length > 0;
  }, [engine, inputMode, inputText, reference, source]);

  const submit = async () => {
    if (!reference) return;
    if (engine === 'v2' && inputMode === 'audio' && !source) {
      setError('Source audio is required for V2');
      return;
    }
    if (engine === 'v1' && inputMode === 'text' && inputText.trim().length === 0) {
        setError('Please enter lyrics/text');
        return;
    }
    if (engine === 'v2' && inputMode === 'text' && inputText.trim().length === 0) {
      setError('Please enter lyrics/text');
      return;
    }
    setError(null);
    setStatus('Preparing request...');
    setLoading(true);
    try {
      const url = engine === 'v2'
        ? await convertV2(source?.file || null, reference.file, paramsV2, inputMode, inputText, paramsV1.tts_voice || 'en-US-GuyNeural', setStatus)
        : await convertV1(source?.file || null, reference.file, paramsV1, inputMode, inputText, setStatus);
      setOutputUrl(url);
      setStatus('Completed');
    } catch (e: any) {
      setError(e?.message || 'Failed to convert');
      setStatus('Failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <h1 style={{ marginBottom: '1rem', letterSpacing: '0.04em', textTransform: 'uppercase' }}>Seed-VC UI (Custom)</h1>

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
        <span className="badge">Lyrics/TTS is V1-only</span>
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
                <select className="select" value={paramsV1.model_mode} onChange={(e) => setParamsV1({ ...paramsV1, model_mode: e.target.value as 'voice' | 'singing' })}>
                  <option value="voice">Voice (22kHz)</option>
                  <option value="singing">Singing (44kHz)</option>
                </select>
              </div>
              <div className="control-row">
                <label className="label">Enable F0</label>
                <input type="checkbox" checked={paramsV1.f0_condition} onChange={(e) => setParamsV1({ ...paramsV1, f0_condition: e.target.checked })} />
                <label className="label">Auto F0 Adjust</label>
                <input type="checkbox" checked={paramsV1.auto_f0_adjust} onChange={(e) => setParamsV1({ ...paramsV1, auto_f0_adjust: e.target.checked })} />
              </div>
              <SliderRow label="Pitch Shift (semitones)" value={paramsV1.pitch_shift} min={-24} max={24} step={1} onChange={(v) => setParamsV1({ ...paramsV1, pitch_shift: v })} />
              <div className="control-row">
                <label className="label">Target BPM (optional)</label>
                <input className="input" type="number" value={paramsV1.target_bpm ?? ''} onChange={(e) => setParamsV1({ ...paramsV1, target_bpm: e.target.value === '' ? null : Number(e.target.value) })} />
              </div>
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
              <div className="control-row">
                <label className="label">Target BPM (optional)</label>
                <input className="input" type="number" value={paramsV2.target_bpm ?? ''} onChange={(e) => setParamsV2({ ...paramsV2, target_bpm: e.target.value === '' ? null : Number(e.target.value) })} />
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
