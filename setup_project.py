# setup_project.py
# Run this from your project root: EDM-Vocal-Generator

import os
import sys
import json
from pathlib import Path

def create_project_structure():
    """Create all necessary directories and files"""
    
    # Define project structure
    directories = [
        "models/coqui",
        "scripts",
        "src/api",
        "src/core",
        "src/ui",
        "output",
        "presets"
    ]
    
    # Create directories
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"? Created directory: {dir_path}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "src/api/__init__.py",
        "src/core/__init__.py",
        "src/ui/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"? Created: {init_file}")

def create_requirements_file():
    """Create requirements.txt with working dependencies"""
    
    requirements = """# TTS Engines
TTS==0.22.0
pyttsx3==2.90

# Audio Processing
librosa==0.10.1
soundfile==0.12.1
pedalboard==0.8.0
pydub==0.25.1

# Web backend (optional for API)
fastapi==0.111.0
uvicorn[standard]==0.30.1

# Core utilities
numpy==1.24.4
scipy==1.11.4
torch>=2.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("? Created requirements.txt")

def create_download_models_script():
    """Create the model download script"""
    
    download_script = '''#!/usr/bin/env python
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
        print("\\nAvailable models:")
        print(TTS.list_models())
        
        # Download recommended model for vocals
        model_name = "tts_models/en/vctk/vits"
        print(f"\\nDownloading model: {model_name}")
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
        
        print(f"\\n? Model downloaded successfully!")
        print(f"? Model info saved to: {info_file}")
        
    except ImportError:
        print("ERROR: TTS not installed. Please run:")
        print("  pip install TTS")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        print("\\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try upgrading TTS: pip install --upgrade TTS")
        print("3. Check available disk space (models are ~500MB)")
        sys.exit(1)

if __name__ == "__main__":
    download_models()
'''
    
    # Create scripts directory if it doesn't exist
    Path("scripts").mkdir(exist_ok=True)
    
    # Write the download script
    script_path = Path("scripts/download_models.py")
    with open(script_path, "w") as f:
        f.write(download_script)
    
    print(f"? Created: {script_path}")

def create_simple_test_script():
    """Create a simple test script to verify everything works"""
    
    test_script = '''#!/usr/bin/env python
"""
Simple test to verify TTS is working
"""

import pyttsx3
from pathlib import Path

def test_basic_tts():
    """Test basic TTS functionality"""
    
    print("Testing basic TTS...")
    
    # Test pyttsx3 (robotic voice)
    engine = pyttsx3.init()
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate test audio
    test_text = "Hello, this is a test of the EDM vocal generator"
    output_file = output_dir / "test_robotic.wav"
    
    engine.save_to_file(test_text, str(output_file))
    engine.runAndWait()
    
    if output_file.exists():
        print(f"? Basic TTS working! Audio saved to: {output_file}")
    else:
        print("? Basic TTS failed")
    
    # Test advanced TTS if available
    try:
        from TTS.api import TTS
        print("\\nTesting advanced TTS...")
        
        # Use the downloaded model
        model_name = "tts_models/en/vctk/vits"
        tts = TTS(model_name)
        
        # Generate with a specific speaker
        output_file = output_dir / "test_realistic.wav"
        tts.tts_to_file(
            text=test_text,
            speaker="p225",  # Female voice
            file_path=str(output_file)
        )
        
        if output_file.exists():
            print(f"? Advanced TTS working! Audio saved to: {output_file}")
        else:
            print("? Advanced TTS failed")
            
    except ImportError:
        print("\\nAdvanced TTS not available. Install with: pip install TTS")
    except Exception as e:
        print(f"\\nAdvanced TTS error: {e}")

if __name__ == "__main__":
    test_basic_tts()
'''
    
    test_path = Path("test_tts.py")
    with open(test_path, "w") as f:
        f.write(test_script)
    
    print(f"? Created: {test_path}")

def main():
    """Main setup function"""
    
    print("=" * 50)
    print("EDM Vocal Generator - Project Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"\nCurrent directory: {current_dir}")
    
    if current_dir.name != "EDM-Vocal-Generator":
        print("\nWARNING: You should run this from the 'EDM-Vocal-Generator' directory")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\nSetting up project structure...")
    
    # Create everything
    create_project_structure()
    create_requirements_file()
    create_download_models_script()
    create_simple_test_script()
    
    print("\n" + "=" * 50)
    print("Setup complete! Next steps:")
    print("=" * 50)
    print("\n1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Download models:")
    print("   python scripts/download_models.py")
    print("\n3. Test the installation:")
    print("   python test_tts.py")
    print("\nIf you get errors, try:")
    print("   pip install --upgrade pip")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")

if __name__ == "__main__":
    main()