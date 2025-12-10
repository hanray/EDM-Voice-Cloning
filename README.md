# 🎵 EDM Neural Voice Generator

**Real-time neural voice cloning with BPM-aware processing for electronic music production**

[![GitHub](https://img.shields.io/badge/GitHub-hanray%2FEDM--Voice--Cloning-blue?logo=github)](https://github.com/hanray/EDM-Voice-Cloning)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal?logo=fastapi)](https://fastapi.tiangolo.com)

## ✨ What This Actually Does

This is a **neural voice cloning application** that lets you:

1. **Upload a reference voice** (any audio file with speech)
2. **Type any text** you want that voice to say
3. **Generate realistic speech** that sounds like the reference voice
4. **Optimize for EDM production** with BPM-aware timing and effects

Perfect for creating custom vocal samples, drops, and vocal chops for your electronic music tracks!

## 🚀 Quick Start (Windows)

### Method 1: One-Click Startup (Recommended)
1. **Download** or clone this repository
2. **Double-click** `start_edm_generator_auto.bat`
3. **Wait** for automatic setup and browser opening
4. **Upload** a reference voice and start generating!

### Method 2: Manual Setup
```powershell
# Clone the repository
git clone https://github.com/hanray/EDM-Voice-Cloning.git
cd EDM-Voice-Cloning

# Create virtual environment
python -m venv env311
.\env311\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m src.api.simple_server --port 8005
```

Then open: **http://localhost:8005**

## �️ Features

### 🎙️ Neural Voice Cloning
- **Upload any voice sample** (MP3, WAV, M4A, FLAC)
- **Type any text** for the cloned voice to speak
- **Adjustable similarity** and diversity controls
- **Auto-optimization** for short text/single words

### 🎵 EDM Producer Tools
- **BPM presets**: 70 Trap, 128 House, 140 Dubstep, 174 D&B
- **Musical timing**: Sync to beat grid, quantize timing
- **Audio effects**: Reverb, delay for vocal processing
- **Quality modes**: Fast preview, balanced, high quality

### 🌐 Modern Web Interface
- **Cyberpunk aesthetic** with grid lines and scanline effects
- **Drag & drop** file uploads
- **Real-time controls** with live parameter adjustment
- **Audio playback** for reference and generated vocals
- **Processing animations** with BPM-aware wave visualizations

### ⚡ Technical Specs
- **Powered by Gradio/HuggingFace** neural voice models
- **FastAPI backend** for reliable API performance
- **Real-time processing** optimized for music production
- **Automatic parameter tuning** for different text lengths

## 🎯 Perfect For

- **EDM Producers** creating custom vocal samples
- **Content Creators** needing specific voice styles
- **Music Producers** generating vocal chops and drops
- **Audio Engineers** experimenting with voice synthesis
- **Anyone** wanting to clone voices for creative projects

## 📋 Requirements

- **Windows 10/11** (primary support)
- **Python 3.9+** (3.11 recommended)
- **4GB RAM** minimum (8GB+ recommended)
- **2GB storage** for dependencies
- **Internet connection** for neural model access

## �️ Project Structure

```
📁 EDM-Voice-Cloning/
├── 🚀 start_edm_generator_auto.bat    # One-click startup
├── 🎨 ui-web/                         # Web interface
│   ├── index.html                     # Main app
│   └── styles.css                     # Cyberpunk styling
├── ⚙️ src/                            # Backend code
│   ├── api/simple_server.py           # FastAPI server
│   └── audio/chatterbox_voice_cloning.py  # Voice cloning
├── 📄 requirements.txt                # Python dependencies
└── 📖 README.md                       # This file
```

## 🎵 BPM Presets Explained

| Preset | BPM | Genre | Use Case |
|--------|-----|-------|----------|
| **BPM Free** | Natural | Any | Realistic speech timing |
| **70 Trap** | 70 | Trap/Hip-Hop | Slow, heavy vocals |
| **128 House** | 128 | House/Electro | Standard EDM tempo |
| **140 Dubstep** | 140 | Dubstep/Bass | High-energy drops |
| **174 D&B** | 174 | Drum & Bass | Fast breakbeats |

## 🔧 Troubleshooting

### Server Won't Start
```powershell
# Check Python version
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Manual server start
cd "your\project\path"
python -m src.api.simple_server --port 8005
```

### Voice Cloning Issues
- **Upload fails**: Check file format (MP3, WAV, M4A, FLAC supported)
- **Poor quality**: Try "High Quality" mode and increase similarity
- **Slow processing**: Use "Fast" mode for testing

### Browser Issues
- **Page won't load**: Check if server is running on port 8005
- **Styling broken**: Hard refresh with Ctrl+F5
- **Upload stuck**: Try smaller audio files (<50MB)

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your improvements
4. **Test** thoroughly
5. **Submit** a pull request

## 📝 License

This project uses various open-source components:
- **Neural models**: Accessed via Gradio/HuggingFace
- **FastAPI**: Modern Python web framework
- **Audio processing**: Soundfile, NumPy

## ⚠️ Disclaimer

- **Ethical use only**: Don't clone voices without permission
- **Respect privacy**: Don't upload personal/private audio
- **Follow laws**: Voice cloning may have legal restrictions
- **Be responsible**: Use for creative/educational purposes

## 🌟 Star This Repo

If this project helps your music production, give it a ⭐!

---

**Made for the EDM community** 🎧💫