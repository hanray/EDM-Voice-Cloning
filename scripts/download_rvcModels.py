#!/usr/bin/env python3
"""
Download and setup RVC models for EDM Vocal Generator
Run this once after installation to set up voice conversion models.
"""

import os
import json
import urllib.request
import zipfile
import hashlib
from pathlib import Path
import sys

# Get project root (assumes script is in scripts/ folder)
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "assets" / "rvc_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Curated selection of high-quality pretrained RVC models
MODEL_MANIFEST = {
    "female_generic_egirl": {
        "url": "https://huggingface.co/ZokaxDesu/e-girl/resolve/main/e-girl.zip?download=true",
        "sha256": None,
        "description": "Clean female timbre (48k, RMVPE)",
        "gender": "female", "style": "pop", "sample_rate": 48000
    },
    "femaleV_cyberpunk": {
        "url": "https://huggingface.co/Astral-P/FemaleV48k/resolve/main/FemaleV48k.zip?download=true",
        "sha256": None,
        "description": "Powerful female lead (48k)",
        "gender": "female", "style": "pop", "sample_rate": 48000
    },
    "male_bg3_narrator": {
        "url": "https://huggingface.co/GarrGarr/RVC_Models/resolve/main/BG3Narrator.zip?download=true",
        "sha256": None,
        "description": "Deep neutral male narrator (48k)",
        "gender": "male", "style": "speech", "sample_rate": 48000
    },
    "male_arby_baritone": {
        "url": "https://huggingface.co/OwlCity/OwlCityRVC/resolve/main/Arby%27s%20Narrator.zip?download=true",
        "sha256": None,
        "description": "Baritone narrator (48k)",
        "gender": "male", "style": "speech", "sample_rate": 48000
    },
    "male_hl2_m07": {
        "url": "https://huggingface.co/QuickWick/Music-AI-Voices/resolve/main/Half%20Life%202%20(Male%2007)%20(RVC)%201K%20Epoch%2028K%20Steps/Half%20Life%202%20(Male%2007)%20(RVC)%201K%20Epoch%2028K%20Steps.zip?download=true",
        "sha256": None,
        "description": "Neutral NA male (long train, 48k)",
        "gender": "male", "style": "speech", "sample_rate": 48000
    },
    "male_stanley_narrator": {
        "url": "https://huggingface.co/GarrGarr/RVC_Models/resolve/main/Stanley%20Parable%20%5BThe%20Narrator%5D%20(RVC)%20150%20Epoch.zip?download=true",
        "sha256": None,
        "description": "Neutral storyteller male (48k)",
        "gender": "male", "style": "speech", "sample_rate": 48000
    },
    "male_cartoon_torchman": {
        "url": "https://huggingface.co/PhoenixStormJr/Megaman-NT-Warrior-Torch-Man-RVC/resolve/main/TorchMan.zip?download=true",
        "sha256": None,
        "description": "Expressive stylized male (48k)",
        "gender": "male", "style": "rock", "sample_rate": 48000
    },
    "female_nyanners_narrator": {
        "url": "https://huggingface.co/autobots/Nyanners-Narrator-RVC/resolve/main/Nyanners_Narrator.zip?download=true",
        "sha256": None,
        "description": "Soft female narrator (48k)",
        "gender": "female", "style": "speech", "sample_rate": 48000
    },
    "male_french_narrator": {
        "url": "https://huggingface.co/quichequill/RVC-V2_Models/resolve/main/French%20Narrator%20(SpongeBob%20SquarePants).zip?download=true",
        "sha256": None,
        "description": "Male narrator w/ French accent (48k)",
        "gender": "male", "style": "speech", "sample_rate": 48000
    }
}


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def download_file(url, dest_path, description=""):
    """Download a file with progress indication"""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  ERROR: Failed to download {description}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f"  Extracting to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"  ERROR: Failed to extract {zip_path}: {e}")
        return False

def verify_model_files(model_dir):
    """Verify that essential RVC model files exist"""
    required_files = [
        "model.pth",  # Main model weights
        "config.json" # Model configuration
    ]
    
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            print(f"  WARNING: Missing required file: {file_name}")
            return False
    
    print(f"  ✓ Model files verified")
    return True

def setup_models():
    """Download and setup all RVC models"""
    print("=" * 60)
    print("EDM Vocal Generator - RVC Models Setup")
    print("=" * 60)
    print(f"Installing models to: {MODELS_DIR}")
    print()
    
    manifest_data = {
        "version": "1.0",
        "models": {},
        "setup_date": "",
        "total_models": len(MODEL_MANIFEST)
    }
    
    success_count = 0
    
    for model_name, model_info in MODEL_MANIFEST.items():
        print(f"Setting up model: {model_name}")
        print(f"  Description: {model_info['description']}")
        
        model_dir = MODELS_DIR / model_name
        model_dir.mkdir(exist_ok=True)
        
        zip_path = MODELS_DIR / f"{model_name}.zip"
        
        # Skip if model already exists and is valid
        if verify_model_files(model_dir):
            print(f"  ✓ Model {model_name} already exists and is valid")
            manifest_data["models"][model_name] = model_info
            success_count += 1
            print()
            continue
        
        # Download model
        if not download_file(model_info["url"], zip_path, model_name):
            print(f"  ✗ Failed to download {model_name}")
            print()
            continue
        
        # Verify hash (skip for now with placeholder hashes)
        # if model_info["sha256"] != "placeholder_hash":
        #     actual_hash = calculate_sha256(zip_path)
        #     if actual_hash != model_info["sha256"]:
        #         print(f"  ✗ Hash mismatch for {model_name}")
        #         zip_path.unlink()  # Delete corrupted file
        #         continue
        
        # Extract model
        if not extract_zip(zip_path, model_dir):
            print(f"  ✗ Failed to extract {model_name}")
            print()
            continue
        
        # Clean up zip file
        zip_path.unlink()
        
        # Verify extraction
        if verify_model_files(model_dir):
            manifest_data["models"][model_name] = model_info
            success_count += 1
            print(f"  ✓ Successfully installed {model_name}")
        else:
            print(f"  ✗ Model {model_name} installation incomplete")
        
        print()
    
    # Save manifest
    manifest_path = MODELS_DIR / "manifest.json"
    from datetime import datetime
    manifest_data["setup_date"] = datetime.now().isoformat()
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    print("=" * 60)
    print("Setup Complete!")
    print(f"Successfully installed: {success_count}/{len(MODEL_MANIFEST)} models")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Manifest saved to: {manifest_path}")
    
    if success_count == 0:
        print("\nWARNING: No models were successfully installed!")
        print("Please check your internet connection and try again.")
        return False
    elif success_count < len(MODEL_MANIFEST):
        print(f"\nNOTE: Only {success_count} models installed successfully.")
        print("The app will work with available models.")
    
    print("\nYou can now use the RVC voice conversion feature in the app!")
    return True

if __name__ == "__main__":
    try:
        success = setup_models()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)