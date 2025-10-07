#!/usr/bin/env python
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
        print("\nTesting advanced TTS...")
        
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
        print("\nAdvanced TTS not available. Install with: pip install TTS")
    except Exception as e:
        print(f"\nAdvanced TTS error: {e}")

if __name__ == "__main__":
    test_basic_tts()
