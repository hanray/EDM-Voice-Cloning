#!/usr/bin/env python
"""
Download models for EDM Vocal Generator
Run this after installing requirements
"""

import os
import sys
from pathlib import Path

def download_models():
    """Download required TTS models"""
    
    print("=" * 50)
    print("EDM Vocal Generator - Model Downloader")
    print("=" * 50)
    
    # Create models directory
    models_dir = Path("models/coqui")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from TTS.api import TTS
        
        # List available models
        print("\nAvailable models:")
        print(TTS.list_models())
        
        # Download recommended model for vocals
        model_name = "tts_models/en/vctk/vits"
        print(f"\nDownloading model: {model_name}")
        print("This may take a few minutes on first run...")
        
        # Initialize TTS (this downloads the model)
        tts = TTS(model_name)
        
        # Save model info
        info = {
            "model_name": model_name,
            "model_path": str(tts.model_path) if hasattr(tts, 'model_path') else "downloaded",
            "status": "ready"
        }
        
        info_file = models_dir / "model_info.json"
        import json
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"\n? Model downloaded successfully!")
        print(f"? Model info saved to: {info_file}")
        
    except ImportError:
        print("ERROR: TTS not installed. Please run:")
        print("  pip install TTS")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try upgrading TTS: pip install --upgrade TTS")
        print("3. Check available disk space (models are ~500MB)")
        sys.exit(1)

if __name__ == "__main__":
    download_models()
