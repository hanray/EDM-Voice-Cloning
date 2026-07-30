"""ACE-Step 1.5 sidecar client — native singing as an extra output.

ACE-Step generates *sung* vocals from lyrics (melody, phrasing, breath —
things the TTS→convert chain approximates). It runs as its own local REST
service (start_ace.bat → http://127.0.0.1:8001) in its own uv-managed venv,
in a separate process so its VRAM is only held while it's running.

When the UI's "ACE sung vocal" toggle is on, we ask it for an a-cappella
take of the same lyrics and drop it into the output folder as
05_ace_vocal.wav — an extra blob to audition against the main pipeline,
and raw material to feed BACK through Seed-VC (audio mode) for timbre.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Optional, Tuple

ACE_BASE = "http://127.0.0.1:8001"
DEFAULT_MODEL = "acestep-v15-turbo"

ACAPPELLA_PROMPT = (
    "a cappella, solo {gender} vocal, dry studio recording, no instruments, "
    "no reverb, clean, {style}"
)


def _post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        ACE_BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def is_up(timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(ACE_BASE + "/health")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def estimate_duration(lyrics: str) -> int:
    """Rough sung-duration guess from word count, clamped to ACE's sweet spot."""
    words = len(lyrics.split())
    return int(max(10, min(60, words * 0.45 + 4)))


def generate_vocal(
    lyrics: str,
    bpm: Optional[float] = None,
    key: Optional[str] = None,
    style: str = "electronic, tech house",
    gender: str = "female",
    duration: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    poll_timeout_s: float = 420.0,
) -> Tuple[Optional[bytes], Optional[str]]:
    """Generate an a-cappella vocal. Returns (wav_or_mp3_bytes, error)."""
    if not is_up():
        return None, (
            "ACE-Step sidecar is not running — start it with start_ace.bat "
            "(first run downloads models)"
        )

    payload = {
        "prompt": ACAPPELLA_PROMPT.format(gender=gender, style=style),
        "lyrics": lyrics,
        "audio_duration": duration or estimate_duration(lyrics),
        "model": model,
        "vocal_language": "en",
    }
    if bpm:
        payload["bpm"] = int(bpm)
    if key:
        # ACE expects e.g. "F Minor"
        parts = key.split()
        payload["key_scale"] = " ".join(p.capitalize() for p in parts[:2])

    try:
        sub = _post("/release_task", payload)
    except Exception as e:
        return None, f"ACE submit failed: {e}"

    task_id = (sub.get("data") or {}).get("task_id") or sub.get("task_id")
    if not task_id:
        return None, f"ACE submit returned no task_id: {sub}"

    deadline = time.time() + poll_timeout_s
    while time.time() < deadline:
        time.sleep(3)
        try:
            res = _post("/query_result", {"task_id_list": [task_id]})
        except Exception:
            continue
        items = (res.get("data") or {})
        # data may be a dict keyed by id or a list
        if isinstance(items, dict):
            items = items.get(task_id) or items.get("results") or items
        entry = None
        if isinstance(items, list) and items:
            entry = items[0]
        elif isinstance(items, dict):
            entry = items
        if not entry:
            continue
        status = entry.get("status")
        result = entry.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = None
        if status == 2:
            return None, f"ACE generation failed: {entry.get('error') or entry}"
        if status == 1 or (result and isinstance(result, list) and result):
            file_url = None
            if isinstance(result, list) and result:
                file_url = result[0].get("file")
            if not file_url:
                return None, f"ACE finished but returned no file: {entry}"
            if file_url.startswith("/"):
                file_url = ACE_BASE + file_url
            try:
                with urllib.request.urlopen(file_url, timeout=60) as r:
                    return r.read(), None
            except Exception as e:
                return None, f"ACE audio download failed: {e}"

    return None, "ACE generation timed out"
