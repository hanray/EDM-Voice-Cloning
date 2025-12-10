# Changelog

All notable changes to the EDM Neural Voice Generator project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-10-06

### 🎉 Complete Project Overhaul - Neural Voice Cloning Focus

This version represents a fundamental redesign, focusing on real-time neural voice cloning with an EDM producer workflow.

### Added
- **Neural Voice Cloning**: Real-time voice cloning using Gradio/HuggingFace Chatterbox models
- **Modern Cyberpunk Web Interface**: 
  - Full-screen grid layout with subtle tech grid background
  - Scanline effects that trigger during voice generation
  - Custom-styled audio players with EDM theme
  - Drag & drop file upload with visual feedback
- **EDM Producer Tools**:
  - BPM presets: 70 Trap, 128 House, 140 Dubstep, 174 D&B, BPM Free
  - Musical timing controls (sync to beat, quantize timing)
  - Audio effects (reverb, delay) for vocal processing
  - Quality modes: Fast preview, balanced, high quality
- **One-Click Deployment**:
  - `start_edm_generator_auto.bat` - Automatic setup and browser opening
  - `start_edm_generator.bat` - Manual server startup
  - `stop_edm_generator.bat` - Clean server shutdown
- **Smart Processing**:
  - Automatic parameter optimization for single words vs. sentences
  - Adaptive wave animations based on processing time
  - Real-time status indicators with visual feedback
- **Security & Deployment**:
  - Comprehensive .gitignore protecting sensitive files
  - Public GitHub repository with proper documentation
  - Environment variable protection

### Technical Implementation
- **FastAPI Backend**: Modern Python web framework replacing complex TTS pipeline
- **Chatterbox Integration**: Direct connection to ResembleAI neural voice models
- **Simplified Architecture**: Streamlined from multi-engine system to focused voice cloning
- **Cloud Processing**: Eliminated local model dependencies
- **Professional Audio Handling**: soundfile-based audio processing with proper format support

### Changed
- **Complete Technology Stack Overhaul**: From local TTS engines to cloud neural processing
- **User Experience**: Simplified from complex multi-step workflow to single-page application
- **Project Focus**: Shifted from general TTS to specialized voice cloning for music production
- **Performance**: Faster startup, lower resource usage, cloud processing
- **Interface**: Web-only (removed desktop GUI), modern responsive design

### Removed
- **Local TTS Engines**: Coqui VITS, pyttsx3, edge-tts implementations
- **RVC Voice Conversion**: Complex local voice conversion pipeline
- **Desktop GUI**: PyQt6 interface (web interface only)
- **Model Downloads**: No longer required - uses cloud processing
- **Complex Audio Effects**: Simplified to core reverb/delay for EDM production
- **Multiple Voice Profiles**: Focused on user-uploaded reference voices

### Breaking Changes
- Configuration files from previous versions are incompatible
- API endpoints completely changed (new /api/ prefix structure)
- Different parameter names and ranges
- New authentication/file handling system

### Migration Notes
- This is effectively a new application - no direct migration path from v2.0.0
- Users should expect completely different capabilities and workflow
- Previous local models and configuration not needed

---

## [2.0.0] - 2025-09-20 (Legacy)

### Historical Note
Previous version focused on local TTS engines and RVC voice conversion. This approach proved complex and resource-intensive, leading to the complete redesign in v2.1.0.

### Legacy Features (No Longer Supported)
- Local RVC voice conversion with PyTorch
- Multiple TTS engines (Coqui, pyttsx3, edge-tts)
- Desktop PyQt6 GUI
- Local model management and downloads
- Complex audio processing pipeline
  - Voice cloning toggle and strength slider
  - Quality selection (fast/balanced/high)
  - Reference audio upload and management
- **Quality Settings**: Configurable processing modes for speed vs quality

### Enhanced
- **Web Interface**: Modern UI with voice cloning controls
- **Audio Pipeline**: Multi-stage processing (TTS → Voice Conversion → Effects)
- **Performance Modes**: Chant, rap, ballad, aggressive vocal styles
- **Producer Controls**: Autotune, doubles, singing mode, BPM sync
- **Effects System**: Professional reverb, delay, compression
- **Humanization**: Advanced prosody and timing variations

### Technical Improvements
- **PyTorch Integration**: GPU acceleration support
- **Memory Optimization**: Efficient tensor operations
- **Error Handling**: Robust fallback systems
- **Modular Architecture**: Separate voice conversion pipeline
- **Real-time Processing**: Optimized for music production workflow

### Dependencies Updated
- PyTorch >= 2.0.0 (with TorchAudio)
- Transformers >= 4.55.0
- NumPy >= 2.0.0
- SciPy >= 1.16.0
- Enhanced audio processing stack

## [1.0.0] - 2025-08-10

### Initial Release
- **Core TTS**: Coqui VITS voice synthesis
- **Web Interface**: FastAPI server with browser UI
- **Desktop GUI**: PyQt6 application
- **Basic Effects**: Reverb, delay, pitch/speed control
- **Export Options**: WAV and MP3 output
- **Multi-platform**: Windows, macOS, Linux support

### Features
- Multiple voice engines (Coqui, pyttsx3, Edge TTS)
- Real-time audio preview
- Customizable effects and settings
- Batch processing capabilities
- Professional audio output quality