# src/tts/edge_engine.py
import asyncio
from pathlib import Path
import edge_tts
from .base import TTSEngine
from pydub import AudioSegment  # pip install pydub (requires ffmpeg)

class EdgeEngine(TTSEngine):
    def __init__(self, voice: str = 'en-US-AriaNeural'):
        self.voice = voice

    def synth_to_wav(self, text: str, out_path: Path, speed: float = 1.0) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_tmp = out_path.with_suffix(".edge.mp3")

        async def generate():
            pct = int((speed - 1.0) * 100)  # e.g., 1.2 -> +20%
            rate = f"{pct:+d}%"
            comm = edge_tts.Communicate(text, self.voice, rate=rate)
            await comm.save(str(mp3_tmp))

        asyncio.run(generate())

        # Convert MP3 -> WAV for downstream FX
        audio = AudioSegment.from_file(mp3_tmp, format="mp3")
        # (optional) enforce your engine sample rate/bit depth:
        audio = audio.set_frame_rate(22050).set_sample_width(2).set_channels(1)
        audio.export(out_path, format="wav")
        return out_path
