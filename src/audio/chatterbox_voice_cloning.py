# src/audio/chatterbox_voice_cloning.py

"""
Neural voice cloning using Gradio client for ResembleAI/Chatterbox.
Connects to the Hugging Face space for true neural voice            # Apply BPM processing if requested
            if bpm and BPM_PROCESSING_AVAILABLE:
                logger.info("Applying BPM-aware audio processing...")
                audio = apply_bpm_effects(
                    audio, 
                    sample_rate, 
                    bpm=bpm,
                    **kwargs
                )
                logger.info("✅ BPM processing completed")
            
            # Post-process single words for better quality
            if optimize_short_text and len(text.split()) == 1:
                audio = self._post_process_single_word(audio, sample_rate)
            
            logger.info("✅ Voice cloning completed successfully")
            return audio, sample_rate"""

import os
import io
import tempfile
import logging
import shutil
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from gradio_client import Client, handle_file
    GRADIO_CLIENT_AVAILABLE = True
    logger.info("✅ Gradio client imported successfully")
except ImportError as e:
    GRADIO_CLIENT_AVAILABLE = False
    logger.error(f"❌ Gradio client not available: {e}")

# Import BPM processing
try:
    from .bpm_processing import apply_bpm_effects
    BPM_PROCESSING_AVAILABLE = True
    logger.info("✅ BPM processing module available")
except ImportError as e:
    BPM_PROCESSING_AVAILABLE = False
    logger.warning(f"⚠️ BPM processing not available: {e}")
    
    # Fallback function if BPM processing not available
    def apply_bpm_effects(audio, sample_rate, **kwargs):
        return audio

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("⚠️ soundfile not available, using numpy for audio I/O")

class ChatterboxVoiceCloner:
    """
    Voice cloning using Gradio client for ResembleAI/Chatterbox.
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize the voice cloner with Gradio client."""
        self.device = device
        self.client = None
        self.sample_rate = 22050  # Standard rate for most TTS systems
        
        logger.info(f"Initializing Chatterbox Gradio client")
        
        if not GRADIO_CLIENT_AVAILABLE:
            raise RuntimeError("gradio_client is required but not available")
            
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the Gradio client connection."""
        try:
            # Connect to the ResembleAI/Chatterbox Hugging Face space
            self.client = Client("ResembleAI/Chatterbox")
            logger.info("✅ Connected to ResembleAI/Chatterbox space")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Chatterbox space: {e}")
            raise RuntimeError(f"Failed to initialize Chatterbox client: {e}")
    
    def _optimize_text_for_tts(self, text: str) -> str:
        """
        Optimize text for better neural TTS synthesis without adding words.
        Focus on parameter adjustments instead of text modification.
        
        Args:
            text: Input text
            
        Returns:
            Original text (we'll optimize via parameters instead)
        """
        text = text.strip()
        
        # Don't modify the text - just return it as-is
        # We'll handle single word optimization through parameter tuning
        
        # Add punctuation if missing for better prosody (minimal change)
        if not text.endswith(('.', '!', '?', ',', ';', ':')):
            # For single words, add period for better ending
            if len(text.split()) == 1:
                text += "."
        
        return text
    
    def _post_process_single_word(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Post-process single word audio to improve naturalness.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Enhanced audio array
        """
        try:
            # Trim silence more aggressively for single words
            # Find start and end of speech
            energy = np.abs(audio)
            energy_smooth = np.convolve(energy, np.ones(int(sample_rate * 0.01)), mode='same')
            threshold = np.max(energy_smooth) * 0.1
            
            speech_indices = np.where(energy_smooth > threshold)[0]
            if len(speech_indices) > 0:
                start_idx = max(0, speech_indices[0] - int(sample_rate * 0.05))  # 50ms padding
                end_idx = min(len(audio), speech_indices[-1] + int(sample_rate * 0.1))  # 100ms padding
                audio = audio[start_idx:end_idx]
            
            # Apply gentle fade in/out to avoid clicks
            fade_samples = int(sample_rate * 0.01)  # 10ms fade
            if len(audio) > fade_samples * 2:
                # Fade in
                audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # Fade out
                audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            logger.info("✅ Single word post-processing completed")
            return audio
            
        except Exception as e:
            logger.warning(f"Single word post-processing failed: {e}, returning original audio")
            return audio
    
    def clone_voice(
        self, 
        text: str, 
        reference_audio_path: str,
        bpm: Optional[float] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        seed: int = 0,
        cfg_weight: float = 0.5,
        optimize_short_text: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Clone voice using the Gradio API.
        
        Args:
            text: Text to synthesize (max 300 chars)
            reference_audio_path: Path to reference audio file
            bpm: BPM for audio processing (optional)
            exaggeration: Exaggeration level (0.0-1.0, neutral=0.5)
            temperature: Temperature for generation (0.0-1.0)
            seed: Random seed (0 for random)
            cfg_weight: CFG/Pace weight (0.0-1.0)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        logger.info(f"Cloning voice for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        if not self.client:
            raise RuntimeError("Chatterbox client not initialized")
        
        # Optimize parameters for single words instead of modifying text
        if optimize_short_text:
            word_count = len(text.split())
            if word_count == 1:
                # For single words, adjust parameters for better synthesis
                exaggeration = min(1.0, exaggeration + 0.2)  # Boost exaggeration
                temperature = max(0.5, temperature - 0.1)    # Reduce temperature for stability
                logger.info(f"Single word detected: boosted exaggeration to {exaggeration}, reduced temperature to {temperature}")
            
            text = self._optimize_text_for_tts(text)
            
        # Limit text to 300 characters as per API requirement
        if len(text) > 300:
            text = text[:297] + "..."
            logger.warning("Text truncated to 300 characters")
            
        try:
            # Call the Gradio API
            result = self.client.predict(
                text_input=text,
                audio_prompt_path_input=handle_file(reference_audio_path),
                exaggeration_input=exaggeration,
                temperature_input=temperature,
                seed_num_input=seed,
                cfgw_input=cfg_weight,
                api_name="/generate_tts_audio"
            )
            
            logger.info("✅ Voice synthesis completed via Chatterbox API")
            
            # The result should be a file path to the generated audio
            output_audio_path = result
            
            # Load the audio file
            if SOUNDFILE_AVAILABLE:
                audio, sample_rate = sf.read(output_audio_path)
            else:
                # Fallback to basic numpy loading
                # This is a simplified fallback - in practice you'd want proper audio loading
                raise RuntimeError("soundfile required for audio loading")
                
            # Apply BPM processing if requested
            if bpm and BPM_PROCESSING_AVAILABLE:
                logger.info("Applying BPM-aware audio processing...")
                audio = apply_bpm_effects(
                    audio, 
                    sample_rate, 
                    bpm=bpm,
                    **kwargs
                )
                logger.info("✅ BPM processing completed")
            
            logger.info("✅ Voice cloning completed successfully")
            return audio, sample_rate
            
        except Exception as e:
            logger.error(f"❌ Voice cloning failed: {e}")
            raise RuntimeError(f"Voice cloning failed: {e}")
    
    def is_available(self) -> bool:
        """Check if the voice cloner is available and functional."""
        return GRADIO_CLIENT_AVAILABLE and self.client is not None


def get_voice_cloner(device: str = "auto") -> Optional[ChatterboxVoiceCloner]:
    """
    Get a voice cloner instance.
    
    Args:
        device: Device to use ("auto", "cpu", "cuda")
        
    Returns:
        Voice cloner instance or None if not available
    """
    try:
        return ChatterboxVoiceCloner(device=device)
    except Exception as e:
        logger.error(f"Failed to create voice cloner: {e}")
        return None


def clone_voice_simple(
    text: str, 
    reference_audio_path: str, 
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Simple interface for voice cloning.
    
    Args:
        text: Text to synthesize
        reference_audio_path: Path to reference audio
        output_path: Output file path (auto-generated if None)
        **kwargs: Additional parameters for cloning
        
    Returns:
        Path to generated audio file
    """
    cloner = get_voice_cloner()
    if not cloner:
        raise RuntimeError("Voice cloner not available")
    
    # Clone the voice
    audio, sample_rate = cloner.clone_voice(text, reference_audio_path, **kwargs)
    
    # Generate output path if not provided
    if output_path is None:
        import uuid
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        output_path = f"vocal_{timestamp}_{unique_id}.wav"
    
    # Save the audio
    if SOUNDFILE_AVAILABLE:
        sf.write(output_path, audio, sample_rate)
    else:
        raise RuntimeError("soundfile required for audio saving")
    
    return output_path


# Test function
def test_voice_cloning():
    """Test the voice cloning functionality."""
    logger.info("Testing Chatterbox voice cloning...")
    
    try:
        cloner = get_voice_cloner()
        if cloner and cloner.is_available():
            logger.info("✅ Voice cloner is available and ready")
            return True
        else:
            logger.error("❌ Voice cloner not available")
            return False
    except Exception as e:
        logger.error(f"❌ Voice cloning test failed: {e}")
        return False


if __name__ == "__main__":
    # Test when run directly
    test_voice_cloning()