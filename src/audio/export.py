from pathlib import Path
import soundfile as sf
from pydub import AudioSegment


def ensure_wav(path: Path) -> Path:
    # No-op helper (we always render to WAV already)
    return path


def to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> Path:
    audio = AudioSegment.from_wav(wav_path)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(mp3_path, format="mp3", bitrate=bitrate)
    return mp3_path