# EDM Vocal Generator with Voice Cloning
*Professional AI voice synthesis and conversion for music production*

## 🎤 Features

### Core Voice Synthesis
- **High-Quality TTS**: Coqui VITS models for natural speech synthesis
- **Multiple Voices**: Built-in speaker profiles for different vocal styles
- **Performance Modes**: Chant, Rap, Ballad, Aggressive vocal styles
- **Producer Controls**: Autotune, doubles, singing mode, BPM sync

### Voice Conversion & Cloning
- **RVC Voice Cloning**: Custom implementation using PyTorch
- **Reference Matching**: Upload audio samples for voice characteristics
- **Advanced Algorithms**: F0 extraction, formant analysis, spectral matching
- **Quality Controls**: Adjustable clone strength and processing quality

### Audio Processing
- **Professional Effects**: Reverb, delay, compression, EQ
- **Humanization**: Natural prosody and timing variations  
- **Character Controls**: Warmth, presence, stereo width
- **Export Options**: WAV, MP3 with configurable sample rates

### Interface Options
- **Web UI**: Modern browser-based interface with real-time controls
- **REST API**: Full programmatic access for automation
- **Desktop GUI**: Qt-based application (PyQt6)

## 🚀 Quick Start

### Installation
```bash
# Windows (PowerShell)
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS/Linux  
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Download Models (First Run)
```bash
python scripts/download_models.py
```

### Start the Server
```bash
python -m src.main
# Opens web interface at http://localhost:8000
```

## 🎯 Voice Cloning Usage

### Web Interface
1. Upload reference audio via "Analyze Reference" 
2. Enable "Voice Cloning" toggle
3. Adjust clone strength (0-100%)
4. Generate vocals with your custom voice

### API Usage
```python
# Analyze reference audio
POST /api/analyze_reference
# Upload audio file, returns reference_id

# Synthesize with voice cloning  
POST /api/synthesize
{
  "text": "Your lyrics here",
  "voice": "coqui_female", 
  "timbre_clone": true,
  "reference_id": "ref_12345678",
  "clone_strength": 0.8,
  "clone_quality": "balanced"
}
```

## 🔧 Technical Implementation

### Voice Conversion Pipeline
- **F0 Extraction**: Autocorrelation-based pitch detection
- **Formant Analysis**: LPC (Linear Predictive Coding) 
- **Spectral Matching**: Mel spectrogram envelope transfer
- **Phase Vocoder**: Professional pitch shifting
- **Real-time Processing**: Optimized for music production workflow

### Audio Processing Chain
1. **Text-to-Speech**: Coqui VITS synthesis
2. **Voice Conversion**: RVC-style cloning (optional)
3. **Effects Processing**: Humanization, FX, character
4. **Export**: High-quality audio rendering

## 📁 Project Structure
```
├── src/
│   ├── api/          # FastAPI server & endpoints
│   ├── audio/        # Voice conversion & effects
│   ├── tts/          # Text-to-speech engines  
│   ├── ui/           # Desktop GUI (PyQt6)
│   └── main.py       # Application entry point
├── ui-web/           # Web interface (HTML/JS)
├── scripts/          # Setup & utility scripts
├── models/           # Downloaded AI models
└── assets/output/    # Generated audio files
```

## 🎵 Perfect for EDM Production
- **Vocal Chops**: Generate custom vocal samples
- **Drop Vocals**: Create powerful vocal hooks
- **Vocal Layers**: Stack multiple voices and effects
- **Reference Matching**: Clone specific vocal timbres
- **Professional Output**: Studio-ready audio quality

## 💻 System Requirements
- **Python**: 3.9-3.12 (3.12 recommended)
- **RAM**: 4GB minimum, 8GB+ recommended for voice cloning
- **Storage**: 2GB for models and dependencies
- **GPU**: Optional - CUDA-compatible GPU for faster processing
- **Audio**: FFmpeg for MP3 export (optional)

## 🔧 Troubleshooting

### Installation Issues
```bash
# PyTorch CPU-only (if CUDA issues)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Missing audio codecs
pip install soundfile librosa

# FFmpeg for MP3 export
# Windows: choco install ffmpeg
# macOS: brew install ffmpeg  
# Linux: sudo apt install ffmpeg
```

### Common Issues
- **Slow processing**: Enable GPU acceleration or reduce sample rate
- **Memory errors**: Reduce clone_strength or use "fast" quality mode
- **Audio artifacts**: Adjust humanization_intensity or effects settings
- **Import errors**: Ensure virtual environment is activated

### Performance Optimization
- **GPU**: Install CUDA-compatible PyTorch for 3-5x speedup
- **Quality vs Speed**: Use "fast" mode for drafts, "high" for final
- **Batch Processing**: Process multiple vocals via API automation

## 📝 License & Credits
- **Coqui TTS**: Mozilla Public License 2.0
- **Custom RVC**: Original implementation using PyTorch
- **Audio Effects**: Pedalboard (Spotify) + custom DSP
- **Web UI**: FastAPI + modern JavaScript

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support
- **Issues**: GitHub Issues tracker
- **Docs**: See `/docs` folder for detailed API documentation  
- **Examples**: Check `/examples` for usage patterns