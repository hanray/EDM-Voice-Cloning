import { useCallback, useEffect, useRef, useState } from 'react';

export type RecordingState = 'idle' | 'recording' | 'paused';

export interface UseMediaRecorderOptions {
  mimeType?: string;
}

export function useMediaRecorder(options: UseMediaRecorderOptions = {}) {
  const [state, setState] = useState<RecordingState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [mediaBlob, setMediaBlob] = useState<Blob | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream, { mimeType: options.mimeType });
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: options.mimeType || 'audio/webm' });
        setMediaBlob(blob);
        chunksRef.current = [];
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setState('recording');
    } catch (e: any) {
      setError(e?.message || 'Failed to start recording');
      setState('idle');
    }
  }, [options.mimeType]);

  const stop = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (mr && (mr.state === 'recording' || mr.state === 'paused')) {
      mr.stop();
      mr.stream.getTracks().forEach((t) => t.stop());
    }
    setState('idle');
  }, []);

  const pause = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (mr && mr.state === 'recording') {
      mr.pause();
      setState('paused');
    }
  }, []);

  const resume = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (mr && mr.state === 'paused') {
      mr.resume();
      setState('recording');
    }
  }, []);

  useEffect(() => {
    return () => {
      const mr = mediaRecorderRef.current;
      if (mr) {
        try { mr.stop(); } catch {}
        mr.stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return { state, error, mediaBlob, start, stop, pause, resume, clear: () => setMediaBlob(null) };
}
