# API Documentation - EDM Neural Voice Generator

## Overview
The EDM Neural Voice Generator provides a FastAPI-based REST API for neural voice cloning with BPM-aware processing. This API is optimized for electronic music production workflows.

## Base URL
```
http://localhost:8005/api
```

## Authentication
No authentication required for local usage.

## Health Check

### GET /api/health
Check if the server is running and cleanup old files.

**Response:**
```json
{
    "status": "healthy",
    "message": "EDM Neural Voice Generator API",
    "timestamp": 1696636800.123,
    "cleanup_performed": true
}
```

## Voice Cloning Workflow

### 1. Upload Reference Audio

Upload an audio file containing speech to use as a voice reference.

**Endpoint:** `POST /api/upload_reference`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): Audio file containing speech
  - **Supported formats**: WAV, MP3, FLAC, OGG
  - **Recommended**: Clear speech, minimal background noise
  - **Duration**: 30 seconds to 5 minutes optimal

**Response:**
```json
{
    "success": true,
    "reference_id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "reference_voice.wav",
    "duration": 45.67,
    "sample_rate": 44100
}
```

**Error Response:**
```json
{
    "detail": "Unsupported audio format"
}
```

### 2. Generate Voice Clone

Generate speech using the uploaded reference voice.

**Endpoint:** `POST /api/clone_voice`

**Content-Type:** `application/json`

**Request Body:**
```json
{
    "text": "Turn up the energy tonight!",
    "reference_id": "550e8400-e29b-41d4-a716-446655440000",
    "quality": "balanced",
    "similarity_boost": 0.8,
    "diversity": 0.3,
    "optimize_short_text": true,
    "bpm": 128,
    "sync_to_beat": false,
    "quantize_timing": false,
    "add_reverb": false,
    "reverb_amount": 0.3,
    "add_delay": false,
    "delay_time": 0.125
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to be spoken by the cloned voice |
| `reference_id` | string | required | ID from upload_reference response |
| `quality` | string | "balanced" | Processing quality: "fast", "balanced", "high" |
| `similarity_boost` | float | 0.8 | Voice similarity (0.1-1.0, higher = more similar) |
| `diversity` | float | 0.3 | Voice variation (0.0-1.0, higher = more diverse) |
| `optimize_short_text` | boolean | true | Auto-optimize for single words/short phrases |
| `bpm` | integer | 128 | Target BPM for vocal timing (60-200) |
| `sync_to_beat` | boolean | false | Align vocal phrases to beat grid |
| `quantize_timing` | boolean | false | Quantize vocal timing to musical grid |
| `add_reverb` | boolean | false | Apply reverb effect |
| `reverb_amount` | float | 0.3 | Reverb intensity (0.0-1.0) |
| `add_delay` | boolean | false | Apply delay effect |
| `delay_time` | float | 0.125 | Delay time in seconds |

**Response:**
- **Content-Type**: `audio/wav`
- **Body**: Binary audio data (WAV format)

**Error Response:**
```json
{
    "detail": "Invalid reference ID"
}
```

## EDM Producer Presets

### BPM Presets
Common BPM values for electronic music genres:

| Genre | BPM | Use Case |
|-------|-----|----------|
| BPM Free | 0 | Natural speech timing |
| Trap | 70 | Slow, heavy vocals |
| House | 128 | Standard EDM tempo |
| Dubstep | 140 | High-energy drops |
| Drum & Bass | 174 | Fast breakbeats |

### Quality Modes

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| "fast" | ⚡ Fast | 📊 Basic | Quick previews, testing |
| "balanced" | ⚖️ Medium | 📈 Good | Production work |
| "high" | 🐌 Slow | 🎯 Best | Final masters |

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid parameters or missing data |
| 404 | Not Found | Reference ID not found |
| 500 | Server Error | Internal processing error |
| 503 | Service Unavailable | Voice cloning service not available |

### Common Error Messages

```json
{
    "detail": "Please upload a reference audio file first"
}
```

```json
{
    "detail": "Voice cloning not available"
}
```

```json
{
    "detail": "Cloning failed: Processing timeout"
}
```

## Usage Examples

### Python Example

```python
import requests
import json

# 1. Upload reference audio
with open('reference_voice.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8005/api/upload_reference',
        files={'file': f}
    )
reference_data = response.json()
reference_id = reference_data['reference_id']

# 2. Generate voice clone
payload = {
    "text": "This is a test of voice cloning",
    "reference_id": reference_id,
    "quality": "balanced",
    "similarity_boost": 0.8,
    "bpm": 128,
    "add_reverb": True
}

response = requests.post(
    'http://localhost:8005/api/clone_voice',
    headers={'Content-Type': 'application/json'},
    data=json.dumps(payload)
)

# Save generated audio
with open('generated_voice.wav', 'wb') as f:
    f.write(response.content)
```

### JavaScript Example

```javascript
// Upload reference audio
const fileInput = document.getElementById('audioFile');
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch('/api/upload_reference', {
    method: 'POST',
    body: formData
});
const uploadData = await uploadResponse.json();

// Generate voice clone
const generateResponse = await fetch('/api/clone_voice', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        text: 'Generate this text with the uploaded voice',
        reference_id: uploadData.reference_id,
        quality: 'balanced',
        similarity_boost: 0.8,
        bpm: 128
    })
});

// Create audio URL
const audioBlob = await generateResponse.blob();
const audioUrl = URL.createObjectURL(audioBlob);
document.getElementById('audioPlayer').src = audioUrl;
```

## Rate Limiting

- No rate limiting implemented for local usage
- Processing time depends on text length and quality settings
- Large texts (>500 characters) may take longer to process

## File Management

- Uploaded files are temporarily stored and cleaned up automatically
- Generated audio is served directly without permanent storage
- Old files are cleaned up on server restart and health checks

## Technical Notes

- Neural processing is handled by Gradio/HuggingFace infrastructure
- Audio processing uses soundfile for professional quality
- BPM processing is integrated into the voice generation pipeline
- All audio responses are in WAV format with appropriate headers
    "pitch_stats": {"mean": 220.5, "std": 15.2},
    "formants": [800, 1200, 2400, 3200],
    "spectral_centroid": 1500.0,
    "voice_quality": {"breathiness": 0.3, "roughness": 0.1}
  },
  "ref_id": "a1b2c3d4"
}
```

### 2. Synthesize with Voice Cloning
Generate speech using the reference voice characteristics.

**Endpoint:** `POST /api/synthesize`

**Content-Type:** `application/json`

**Parameters:**
```json
{
  "text": "Your lyrics or text here",
  "voice": "coqui_female",
  "timbre_clone": true,
  "reference_id": "a1b2c3d4",
  "clone_strength": 0.8,
  "clone_quality": "balanced",
  "pitch": 0.0,
  "speed": 1.0,
  "performance_mode": "normal",
  "singing_mode": false,
  "autotune": false,
  "reverb_mix": 0.2,
  "format": "wav"
}
```

**Voice Cloning Parameters:**
- `timbre_clone` (boolean): Enable voice conversion
- `reference_id` (string): ID from analyze_reference response
- `clone_strength` (float): Blend ratio 0.0-1.0 (0=original, 1=full clone)
- `clone_quality` (string): Processing quality ("fast", "balanced", "high")

**Response:** Audio file download

## System Status

### Check Voice Conversion Availability
**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "ok": true,
  "seedvc_available": true,
  "voice_conversion": "enabled"
}
```

### Voice Conversion System Status
**Endpoint:** `GET /api/seedvc/status`

**Response:**
```json
{
  "available": true,
  "status": "ready",
  "message": "Voice conversion is ready",
  "features": {
    "voice_conversion": true,
    "zero_shot": true,
    "real_time": true
  }
}
```

## Available Voices
**Endpoint:** `GET /api/voices`

**Response:**
```json
{
  "voices": ["coqui_female", "coqui_male", "edge_female", "edge_male"],
  "speakers": ["female_1", "female_2", "male_1", "male_2"]
}
```

## Performance Modes
- `normal`: Standard speech synthesis
- `chant`: Rhythmic, ceremonial style
- `rap`: Fast-paced with emphasis
- `ballad`: Slow, emotional delivery
- `aggressive`: High-energy, intense

## Quality Settings
- `fast`: Minimal processing, fastest speed
- `balanced`: Good quality/speed balance (recommended)
- `high`: Maximum quality, slower processing

## Error Handling
All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `500`: Server error

Error responses include details:
```json
{
  "error": "Description of the error",
  "job_id": "error_context_id"
}
```

## Rate Limiting
No rate limiting for local usage. For production deployment, implement appropriate rate limiting based on your needs.

## Examples

### Python Example
```python
import requests

# Upload reference
with open("reference_voice.wav", "rb") as f:
    response = requests.post("http://localhost:8000/api/analyze_reference", files={"file": f})
    ref_data = response.json()

# Generate with cloning
payload = {
    "text": "Hello world, this is a cloned voice",
    "voice": "coqui_female",
    "timbre_clone": True,
    "reference_id": ref_data["ref_id"],
    "clone_strength": 0.8,
    "clone_quality": "balanced"
}

response = requests.post("http://localhost:8000/api/synthesize", json=payload)
with open("output.wav", "wb") as f:
    f.write(response.content)
```

### JavaScript Example
```javascript
// Upload reference
const formData = new FormData();
formData.append('file', audioFile);

const refResponse = await fetch('/api/analyze_reference', {
    method: 'POST',
    body: formData
});
const refData = await refResponse.json();

// Generate with cloning
const synthResponse = await fetch('/api/synthesize', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        text: "Hello world, this is a cloned voice",
        voice: "coqui_female",
        timbre_clone: true,
        reference_id: refData.ref_id,
        clone_strength: 0.8,
        clone_quality: "balanced"
    })
});

const audioBlob = await synthResponse.blob();
```