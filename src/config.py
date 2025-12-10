from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = ASSETS_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
BIT_DEPTH = 16

# Voice conversion defaults
DEFAULT_SEEDVC_QUALITY = "balanced"
DEFAULT_SEEDVC_STRENGTH = 0.8

# What manager.py expects
ROBOTIC_ID = "Robotic"

# Coqui model + two speaker presets (works offline after first download)
COQUI_MODEL = "tts_models/en/vctk/vits"
COQUI_SPEAKERS = {
    "Realistic Male": "p243",   # deeper male-ish
    "Realistic Female": "p231", # brighter female-ish
}

# Seed-VC Configuration
SEEDVC_CACHE_DIR = ASSETS_DIR / "seedvc_cache"
SEEDVC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Diff-SVC (singing voice conversion) configuration
DIFFSVC_MODELS_DIR = ASSETS_DIR / "diffsvc_models"
DIFFSVC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Default Diff-SVC model placeholder; set to actual checkpoint path when available
DIFFSVC_DEFAULT_MODEL = DIFFSVC_MODELS_DIR / "diffsvc_default.pth"

# Voice conversion quality presets
SEEDVC_QUALITY_PRESETS = {
    "fast": {
        "diffusion_steps": 10,
        "inference_cfg_rate": 0.0,
        "description": "Fast conversion (~10s), lower quality"
    },
    "balanced": {
        "diffusion_steps": 25, 
        "inference_cfg_rate": 0.7,
        "description": "Balanced conversion (~20s), good quality"
    },
    "high": {
        "diffusion_steps": 50,
        "inference_cfg_rate": 0.7,
        "description": "High quality conversion (~40s), best results"
    }
}

# Default Seed-VC settings
DEFAULT_SEEDVC_QUALITY = "balanced"
DEFAULT_SEEDVC_STRENGTH = 0.8  # 0.0 = no conversion, 1.0 = full conversion