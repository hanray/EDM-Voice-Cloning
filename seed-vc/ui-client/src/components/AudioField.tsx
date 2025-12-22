import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import { useDropzone } from 'react-dropzone';
import { useMediaRecorder } from '../lib/useMediaRecorder';
import { audioBufferToWav } from '../lib/wav';
import classNames from 'classnames';

export type AudioValue = {
  file: File;
  url: string;
};

export interface AudioFieldProps {
  label: string;
  initialUrl?: string;
  onChange?: (value: AudioValue | null) => void;
  onError?: (msg: string) => void;
  height?: number;
}

export const AudioField: React.FC<AudioFieldProps> = ({ label, initialUrl, onChange, onError, height = 160 }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<ReturnType<typeof RegionsPlugin.create> | null>(null);
  const [duration, setDuration] = useState(0);
  const [loading, setLoading] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [volume, setVolume] = useState(1);
  const [regionId, setRegionId] = useState<string | null>(null);
  const { state: recState, start: recStart, stop: recStop, pause: recPause, resume: recResume, mediaBlob, error: recError, clear: recClear } = useMediaRecorder({ mimeType: 'audio/webm' });

  const notifyError = useCallback((msg: string) => { onError?.(msg); }, [onError]);

  const resetRegion = useCallback(() => {
    if (regionsRef.current) {
      regionsRef.current.getRegions().forEach((r) => r.remove());
    }
    setRegionId(null);
  }, []);

  const applyRegionStyles = useCallback((selectedId: string | null) => {
    const regions = regionsRef.current?.getRegions() || [];
    regions.forEach((r) => {
      const el = (r as any).element as HTMLElement | undefined;
      if (!el) return;
      // Default orange outline; highlight selected in yellow.
      el.style.border = '2px solid rgba(255,165,0,0.85)';
      el.style.boxSizing = 'border-box';
      el.style.borderRadius = '6px';
      el.style.background = 'rgba(255,165,0,0.18)';
      if (selectedId && r.id === selectedId) {
        el.style.border = '2px solid rgba(255,215,0,0.95)';
        el.style.background = 'rgba(255,215,0,0.22)';
      }
    });
  }, []);

  const loadAudio = useCallback(async (file: File) => {
    setLoading(true);
    resetRegion();
    try {
      const url = URL.createObjectURL(file);
      if (!wavesurferRef.current) return;
      await wavesurferRef.current.load(url);
      onChange?.({ file, url });
    } catch (e: any) {
      notifyError(e?.message || 'Failed to load audio');
    } finally {
      setLoading(false);
    }
  }, [notifyError, onChange, resetRegion]);

  useEffect(() => {
    const ws = WaveSurfer.create({
      container: containerRef.current!,
      waveColor: '#7dd3fc',
      progressColor: '#22d3ee',
      height,
      barWidth: 2,
      barGap: 2,
      cursorColor: '#22d3ee',
      responsive: true,
    });
    const regions = ws.registerPlugin(RegionsPlugin.create({
      dragSelection: {
        slop: 5,
        color: 'rgba(255,165,0,0.18)',
      }
    }));
    wavesurferRef.current = ws;
    regionsRef.current = regions;

    ws.on('ready', () => {
      const dur = ws.getDuration();
      setDuration(dur);
      // Auto-create a full-length region so trim is usable immediately.
      if (regionsRef.current && regionsRef.current.getRegions().length === 0 && dur > 0) {
        const r = regionsRef.current.addRegion({
          start: 0,
          end: dur,
          drag: true,
          resize: true,
          color: 'rgba(255,165,0,0.18)'
        });
        setRegionId(r.id);
        applyRegionStyles(r.id);
      }
    });
    ws.on('error', (msg) => notifyError(msg));
    ws.on('region-created', (r) => {
      setRegionId(r.id);
      applyRegionStyles(r.id);
    });
    ws.on('region-updated', (r) => {
      setRegionId(r.id);
      applyRegionStyles(r.id);
    });
    ws.on('region-click', (r, e) => {
      e?.stopPropagation?.();
      setRegionId(r.id);
      applyRegionStyles(r.id);
    });

    return () => {
      ws.destroy();
    };
  }, [applyRegionStyles, height, notifyError]);

  useEffect(() => {
    if (initialUrl && wavesurferRef.current) {
      (async () => {
        try {
          await wavesurferRef.current!.load(initialUrl);
        } catch (e: any) {
          notifyError(e?.message || 'Failed to load initial audio');
        }
      })();
    }
  }, [initialUrl, notifyError]);

  useEffect(() => {
    if (!mediaBlob) return;
    const file = new File([mediaBlob], 'recording.webm', { type: mediaBlob.type });
    loadAudio(file);
  }, [mediaBlob, loadAudio]);

  const onDrop = useCallback((accepted: File[]) => {
    const file = accepted[0];
    if (file) loadAudio(file);
  }, [loadAudio]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'audio/*': [] }, multiple: false });

  const playPause = useCallback(() => wavesurferRef.current?.playPause(), []);
  const stop = useCallback(() => { wavesurferRef.current?.stop(); }, []);
  const skip = useCallback((sec: number) => wavesurferRef.current?.skip(sec), []);
  const changeRate = useCallback((rate: number) => { setPlaybackRate(rate); wavesurferRef.current?.setPlaybackRate(rate); }, []);
  const changeVolume = useCallback((v: number) => { setVolume(v); wavesurferRef.current?.setVolume(v); }, []);

  const trimToRegion = useCallback(async () => {
    const ws = wavesurferRef.current;
    if (!ws || !regionId) return;
    const region = regionsRef.current?.getRegions().find((r) => r.id === regionId);
    if (!region) return;
    const buffer = ws.getDecodedData();
    const sampleRate = buffer.sampleRate;
    const start = Math.floor(region.start * sampleRate);
    const end = Math.floor(region.end * sampleRate);
    const trimmed = buffer.getChannelData(0).slice(start, end);
    const outBuffer = new AudioBuffer({ length: trimmed.length, sampleRate, numberOfChannels: 1 });
    outBuffer.copyToChannel(trimmed, 0, 0);
    const wav = audioBufferToWav(outBuffer);
    const blob = new Blob([wav], { type: 'audio/wav' });
    const file = new File([blob], 'trimmed.wav', { type: 'audio/wav' });
    await loadAudio(file);
  }, [loadAudio, regionId]);

  const clear = useCallback(() => {
    resetRegion();
    wavesurferRef.current?.empty();
    onChange?.(null);
  }, [onChange, resetRegion]);

  return (
    <div className="audio-shell">
      <div className="badge" style={{ marginBottom: '0.5rem' }}>♪ {label}</div>
      <div {...getRootProps({ className: 'dropzone', style: { border: '1px dashed rgba(34,211,238,0.3)', borderRadius: '10px', padding: '0.75rem', cursor: 'pointer', background: isDragActive ? 'rgba(34,211,238,0.08)' : 'transparent' } })}>
        <input {...getInputProps()} />
        <div className="wave-container" style={{ minHeight: height }}>
          <div ref={containerRef} />
        </div>
      </div>
      <div className="audio-actions">
        <button className="button" onClick={() => skip(-5)} disabled={loading}>⏪</button>
        <button className="button" onClick={playPause} disabled={loading}>▶</button>
        <button className="button" onClick={() => skip(5)} disabled={loading}>⏩</button>
        <button className="button" onClick={() => changeRate(playbackRate === 1 ? 1.25 : 1)} disabled={loading}>{playbackRate === 1 ? '1X' : '1.25X'}</button>
        <button className="button" onClick={() => changeVolume(volume > 0 ? 0 : 1)} disabled={loading}>{volume > 0 ? '🔊' : '🔇'}</button>
        <button className="button" onClick={trimToRegion} disabled={!regionId || loading}>✂ Trim</button>
        <button className="button" onClick={clear} disabled={loading}>✕ Clear</button>
      </div>
      <div className="audio-actions">
        <button className={classNames('button', { disabled: recState === 'recording' })} onClick={recStart} disabled={recState === 'recording'}>● Record</button>
        <button className="button" onClick={recPause} disabled={recState !== 'recording'}>Pause</button>
        <button className="button" onClick={recResume} disabled={recState !== 'paused'}>Resume</button>
        <button className="button" onClick={recStop} disabled={recState === 'idle'}>Stop Rec</button>
      </div>
      {recError && <div style={{ color: '#fca5a5', fontSize: '0.85rem' }}>{recError}</div>}
      {duration > 0 && <div className="badge">Length: {duration.toFixed(2)}s</div>}
    </div>
  );
};
