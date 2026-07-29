"""GPU serialization + live status for the API server.

Adapted from voicebox's serial task-queue idea, simplified for this
server's direct-response model: a process-wide GPU lock makes concurrent
conversions queue instead of fighting for VRAM (the old symptom: two
requests → CUDA OOM or a zombie port), and a tiny status registry feeds
an SSE endpoint so the UI can show what stage a generation is in.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Optional

# Only one model inference at a time — Seed-VC easily fills consumer VRAM.
gpu_lock = threading.Lock()

_state_lock = threading.Lock()
_state = {
    "state": "idle",       # idle | busy
    "stage": None,          # preprocessing | tts | cadence | converting | postprocessing
    "job": None,            # short label of the running job
    "waiting": 0,           # requests queued behind the GPU lock
    "updated_at": time.time(),
}


def set_stage(stage: Optional[str], job: Optional[str] = None) -> None:
    with _state_lock:
        _state["stage"] = stage
        _state["state"] = "idle" if stage is None else "busy"
        if job is not None or stage is None:
            _state["job"] = job
        _state["updated_at"] = time.time()


def adjust_waiting(delta: int) -> None:
    with _state_lock:
        _state["waiting"] = max(0, _state["waiting"] + delta)
        _state["updated_at"] = time.time()


def snapshot() -> dict:
    with _state_lock:
        return dict(_state)


async def sse_status_stream(poll_s: float = 0.5, heartbeat_s: float = 15.0):
    """Async generator for an SSE response: emits on every status change,
    plus a heartbeat comment so proxies don't kill the connection."""
    last_sent = None
    last_beat = time.time()
    while True:
        snap = snapshot()
        key = (snap["state"], snap["stage"], snap["job"], snap["waiting"])
        if key != last_sent:
            last_sent = key
            yield f"data: {json.dumps(snap)}\n\n"
        elif time.time() - last_beat > heartbeat_s:
            last_beat = time.time()
            yield ": ping\n\n"
        await asyncio.sleep(poll_s)
