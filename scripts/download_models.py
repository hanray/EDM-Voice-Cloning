from pathlib import Path
import json
from TTS.api import TTS

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models" / "coqui"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "tts_models/en/vctk/vits"

print(f"Downloading model: {MODEL_NAME} (first run only)...")
tts = TTS(MODEL_NAME)

info = {"model_name": MODEL_NAME}
(MODELS_DIR / "download_info.json").write_text(json.dumps(info, indent=2))
print("Done. Model cached. You can now run the app offline.")
