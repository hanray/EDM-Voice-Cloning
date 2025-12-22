import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import gradio as gr
import torch
import torchaudio
import librosa
from modules.commons import build_model, load_checkpoint, recursive_munch, str2bool
import yaml
from hf_utils import load_custom_model_from_hf
import numpy as np
from pydub import AudioSegment
import argparse
import edge_tts
import asyncio
import tempfile

# Model cache for lazy loading
model_cache = {
    "loaded": False,
    "model": None,
    "semantic_fn": None,
    "vocoder_fn": None,
    "campplus_model": None,
    "to_mel": None,
    "mel_fn_args": None
}

# Load model and configuration
fp16 = False
device = None
def load_models(args):
    global sr, hop_length, fp16, device
    fp16 = args.fp16
    print(f"Using device: {device}")
    print(f"Using fp16: {fp16}")
    if args.checkpoint is None or args.checkpoint == "":
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                                         "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                                                                         "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    else:
        dit_checkpoint_path = args.checkpoint
        dit_config_path = args.config
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu", weights_only=True))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                         load_only_params=True, ignore_modules=[], is_distributed=False)
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(sum(p.numel() for p in vocos[key].parameters() if p.requires_grad) for key in vocos.keys())
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == 'cnhubert':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )
        hubert_model_name = config['model_params']['speech_tokenizer']['name']
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  padding=True,
                                                  sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    elif speech_tokenizer_type == 'xlsr':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   padding=True,
                                                   sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )
def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

bitrate = "320k"

model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args = None, None, None, None, None, None
overlap_wave_len = None
max_context_window = None
sr = None
hop_length = None
overlap_frame_len = 16

# Popular edge-tts voices
TTS_VOICES = {
    "English-US-Male-1": "en-US-GuyNeural",
    "English-US-Female-1": "en-US-JennyNeural",
    "English-US-Male-2": "en-US-ChristopherNeural",
    "English-US-Female-2": "en-US-AriaNeural",
    "English-UK-Male": "en-GB-RyanNeural",
    "English-UK-Female": "en-GB-SoniaNeural",
    "Chinese-Female": "zh-CN-XiaoxiaoNeural",
    "Chinese-Male": "zh-CN-YunxiNeural",
    "Japanese-Female": "ja-JP-NanamiNeural",
    "Japanese-Male": "ja-JP-KeitaNeural",
    "Spanish-Female": "es-ES-ElviraNeural",
    "Spanish-Male": "es-ES-AlvaroNeural",
    "French-Female": "fr-FR-DeniseNeural",
    "French-Male": "fr-FR-HenriNeural",
}

async def generate_tts_async(text, voice):
    """Generate speech from text using edge-tts"""
    voice_name = TTS_VOICES.get(voice, "en-US-GuyNeural")
    communicate = edge_tts.Communicate(text, voice_name)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
    
    await communicate.save(tmp_path)
    return tmp_path

def generate_tts(text, voice):
    """Synchronous wrapper for TTS generation"""
    if not text or text.strip() == "":
        return None
    return asyncio.run(generate_tts_async(text, voice))

def detect_bpm(audio, sr):
    """Detect BPM from audio using librosa"""
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        return float(tempo)
    except:
        return None

def apply_bpm_adjustment(audio, sr, source_bpm, target_bpm):
    """Apply time-stretching to match target BPM"""
    if source_bpm is None or target_bpm is None or source_bpm <= 0 or target_bpm <= 0:
        return audio
    
    stretch_ratio = source_bpm / target_bpm  # Note: inverted for librosa time_stretch
    stretched_audio = librosa.effects.time_stretch(audio, rate=stretch_ratio)
    return stretched_audio

def ensure_models_loaded(progress=None):
    """Lazy load models on first use"""
    if model_cache["loaded"]:
        return
    
    if progress is not None:
        progress(0.0, desc="Loading models...")
    
    print("\n🚀 Loading models for first use...")
    
    # Get args from global scope
    global model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args
    global sr, hop_length
    
    # Create minimal args
    class Args:
        def __init__(self):
            self.checkpoint = None
            self.config = None
            self.fp16 = fp16
    
    args = Args()
    
    if progress is not None:
        progress(0.3, desc="Loading main model...")
    
    # Load all models
    model, semantic_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args = load_models(args)
    
    if progress is not None:
        progress(0.9, desc="Finalizing...")
    
    # Cache models
    model_cache["loaded"] = True
    model_cache["model"] = model
    model_cache["semantic_fn"] = semantic_fn
    model_cache["vocoder_fn"] = vocoder_fn
    model_cache["campplus_model"] = campplus_model
    model_cache["to_mel"] = to_mel
    model_cache["mel_fn_args"] = mel_fn_args
    
    if progress is not None:
        progress(1.0, desc="Ready!")
    
    print("✅ Models loaded successfully!\n")

@torch.no_grad()
@torch.inference_mode()
def voice_conversion(source, target, diffusion_steps, length_adjust, inference_cfg_rate, input_mode="audio", input_text="", tts_voice="English-US-Male-1", model_mode="Voice (22kHz)", f0_condition=False, pitch_shift=0, auto_f0_adjust=False, target_bpm=None, progress=gr.Progress()):
    # Ensure models are loaded
    ensure_models_loaded(progress)
    
    # Switch to singing model if needed
    use_singing_model = (model_mode == "Singing (44kHz)" or f0_condition)
    if use_singing_model:
        progress(0.05, desc="Loading singing voice model...")
        # TODO: Load 44kHz model - for now use existing model
        pass
    
    inference_module = model_cache["model"]
    mel_fn = model_cache["to_mel"]
    
    # Handle text input mode
    if input_mode == "text":
        if not input_text or input_text.strip() == "":
            raise ValueError("Please provide text input")
        # Generate TTS audio
        progress(0.1, desc="Generating speech from text...")
        tts_audio_path = generate_tts(input_text, tts_voice)
        if tts_audio_path is None:
            raise ValueError("Failed to generate TTS audio")
        source = tts_audio_path
    elif source is None:
        raise ValueError("Please provide source audio or text input")
    
    # Load audio
    progress(0.15, desc="Loading audio...")
    source_audio = librosa.load(source, sr=sr)[0]
    
    # BPM processing - auto-detect source, apply if target provided
    use_bpm = target_bpm is not None and target_bpm > 0
    if use_bpm:
        progress(0.2, desc="Processing BPM...")
        # Auto-detect source BPM
        detected_bpm = detect_bpm(source_audio, sr)
        if detected_bpm:
            source_bpm = detected_bpm
            print(f"🎵 Detected source BPM: {source_bpm:.1f}")
            ratio = target_bpm / source_bpm
            print(f"🎵 Adjusting tempo: {source_bpm:.1f} BPM → {target_bpm:.1f} BPM (ratio: {ratio:.2f}x)")
            source_audio = apply_bpm_adjustment(source_audio, sr, source_bpm, target_bpm)
        else:
            print("⚠️ Could not detect source BPM, skipping tempo adjustment")
    
    progress(0.25, desc="Processing voice conversion...")
    ref_audio = librosa.load(target, sr=sr)[0]

    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    # Get cached models
    semantic_fn = model_cache["semantic_fn"]
    vocoder_fn = model_cache["vocoder_fn"]
    campplus_model = model_cache["campplus_model"]

    # Resample
    ref_waves_16k = torchaudio.functional.resample(ref_audio, orig_freq=sr, new_freq=16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, orig_freq=sr, new_freq=16000)
    # if source audio less than 30 seconds, whisper can handle in one forward
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:  # first chunk
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
            S_alt = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, orig_freq=sr, new_freq=16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    F0_ori = None
    F0_alt = None
    shifted_f0_alt = None

    # Length regulation
    cond, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    prompt_condition, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

    max_source_window = max_context_window - mel2.size(2)
    # split source condition (cond) into chunks
    processed_frames = 0
    generated_wave_chunks = []
    # generate chunk by chunk and stream the output
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            # Voice Conversion
            vc_target = inference_module.cfm.inference(cat_condition,
                                                       torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                       mel2, style2, None, diffusion_steps,
                                                       inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target.float())[0]
        if vc_wave.ndim == 1:
            vc_wave = vc_wave.unsqueeze(0)
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy().astype(np.float32)
                generated_wave_chunks.append(output_wave)
                yield (sr, np.concatenate(generated_wave_chunks).astype(np.float32))
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy().astype(np.float32)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len).astype(np.float32)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            yield (sr, np.concatenate(generated_wave_chunks).astype(np.float32))
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len).astype(np.float32)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len


def main(args):
    global sr, hop_length, overlap_wave_len, max_context_window
    
    # Set SR and hop_length from config (lazy loaded models will use these)
    if args.checkpoint is None or args.checkpoint == "":
        from hf_utils import load_custom_model_from_hf
        _, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                       "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                                                       "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    else:
        dit_config_path = args.config
    config = yaml.safe_load(open(dit_config_path, "r"))
    sr = config["preprocess_params"]["sr"]
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    
    # streaming and chunk processing related params
    max_context_window = sr // hop_length * 30
    overlap_wave_len = overlap_frame_len * hop_length
    
    description = ("Zero-shot voice conversion with in-context learning. For local deployment please check "
                   "[GitHub repository](https://github.com/Plachtaa/seed-vc) for details and updates.")
    
    # Load CSS from script directory
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme.css")
    custom_css = open(css_path, "r", encoding="utf-8").read() if os.path.exists(css_path) else ""
    
    # JavaScript for wave animation and scanline control
    custom_js = """
    <script>
    let processingStartTime = null;
    let animationTimeoutId = null;
    
    function createWaveAnimation(intensity = 'fast') {
        const bars = Array.from({length: 15}, () => '<div class="wave-bar"></div>').join('');
        return `
            <div class="wave-animation">
                <div class="wave-bars ${intensity}">
                    ${bars}
                </div>
                <div class="processing-text">Processing Neural Voice...</div>
            </div>
        `;
    }
    
    function startProcessingAnimation() {
        processingStartTime = Date.now();
        const outputViz = document.getElementById('outputViz');
        if (!outputViz) return;
        
        // Trigger scanline effect
        const scanline = document.querySelector('.scanline');
        if (scanline) {
            scanline.classList.remove('active');
            scanline.offsetHeight;
            scanline.classList.add('active');
            setTimeout(() => scanline.classList.remove('active'), 8000);
        }
        
        // Start with fast animation
        outputViz.innerHTML = createWaveAnimation('fast');
        
        // Upgrade to medium after 2 seconds
        if (animationTimeoutId) clearTimeout(animationTimeoutId);
        animationTimeoutId = setTimeout(() => {
            if (processingStartTime && outputViz) {
                outputViz.innerHTML = createWaveAnimation('medium');
                
                // Upgrade to long after 5 more seconds
                animationTimeoutId = setTimeout(() => {
                    if (processingStartTime && outputViz) {
                        outputViz.innerHTML = createWaveAnimation('long');
                    }
                }, 5000);
            }
        }, 2000);
    }
    
    function stopProcessingAnimation() {
        processingStartTime = null;
        if (animationTimeoutId) {
            clearTimeout(animationTimeoutId);
            animationTimeoutId = null;
        }
        const outputViz = document.getElementById('outputViz');
        if (outputViz) {
            outputViz.innerHTML = `
                <div class="wave-animation">
                    <div style="color: var(--green-500); font-size: 1rem; font-weight: 600;">
                        🎉 Voice Generated Successfully!
                    </div>
                    <div style="color: var(--slate-400); font-size: 0.875rem; margin-top: 0.5rem;">
                        Check the audio player below
                    </div>
                </div>
            `;
        }
    }
    
    // Attach to generate button
    document.addEventListener('DOMContentLoaded', function() {
        const observer = new MutationObserver(function() {
            const generateBtn = document.querySelector('.btn-generate');
            if (generateBtn && !generateBtn.dataset.enhanced) {
                generateBtn.dataset.enhanced = 'true';
                generateBtn.addEventListener('click', startProcessingAnimation);
            }
        });
        observer.observe(document.body, {childList: true, subtree: true});
    });
    </script>
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Base(), head=custom_js) as demo:
        # Scanline effect overlay
        gr.HTML('<div class="scanline"></div><div class="tech-grid"></div>')
        
        gr.Markdown(f"# 🎙️ SEED-VC NEURAL VOICE CONVERTER")
        
        # Description with examples link on same row
        gr.HTML(f'''
        <div style="display: flex; align-items: center; justify-content: space-between; margin: 1rem 0; flex-wrap: wrap; gap: 1rem;">
            <p style="color: var(--slate-300); margin: 0;">{description}</p>
            <a href="#" class="examples-link" onclick="document.getElementById('examplesModal').classList.add('active'); return false;">📝 View Examples</a>
        </div>
        ''')
        
        # Examples modal
        gr.HTML('''
        
        <div id="examplesModal" class="examples-modal" onclick="if(event.target.id === 'examplesModal') this.classList.remove('active');">
            <div class="modal-content" onclick="event.stopPropagation();">
                <button class="modal-close" onclick="document.getElementById('examplesModal').classList.remove('active');">×</button>
                <h2>🎵 USAGE EXAMPLES</h2>
                <table class="examples-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Example 1 (Voice)</th>
                            <th>Example 2 (Singing)</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td><strong>Input Mode</strong></td><td>audio</td><td>audio</td></tr>
                        <tr><td><strong>Source Audio</strong></td><td>yae_0.wav</td><td>jay_0.wav</td></tr>
                        <tr><td><strong>Text Input</strong></td><td>(empty)</td><td>(empty)</td></tr>
                        <tr><td><strong>TTS Voice</strong></td><td>English-US-Male-1</td><td>English-US-Male-1</td></tr>
                        <tr><td><strong>Reference Audio</strong></td><td>dingzhen_0.wav</td><td>azuma_0.wav</td></tr>
                        <tr><td><strong>Diffusion Steps</strong></td><td>25</td><td>50</td></tr>
                        <tr><td><strong>Length Adjust</strong></td><td>1.0</td><td>1.0</td></tr>
                        <tr><td><strong>CFG Rate</strong></td><td>0.7</td><td>0.7</td></tr>
                        <tr><td><strong>Model Mode</strong></td><td>Voice (22kHz)</td><td>Singing (44kHz)</td></tr>
                        <tr><td><strong>Enable F0</strong></td><td>false</td><td>true</td></tr>
                        <tr><td><strong>Pitch Shift</strong></td><td>0</td><td>0</td></tr>
                        <tr><td><strong>Auto F0 Adjust</strong></td><td>false</td><td>false</td></tr>
                        <tr><td><strong>Target BPM</strong></td><td>(blank)</td><td>140</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        ''')
        
        # TOP ROW: Input Source + Reference Voice
        with gr.Row(equal_height=True, elem_classes="grid-row"):
            # PANEL 1: Input Mode + Source
            with gr.Column(scale=1, elem_classes="grid-panel", elem_id="panel-source-input"):
                gr.Markdown("### 📁 SOURCE INPUT")
                input_mode = gr.Radio(
                    ["audio", "text"], 
                    value="audio", 
                    label="Input Mode", 
                    info="Audio file or Text-to-Speech",
                    elem_id="input-mode"
                )
                source_audio = gr.Audio(
                    type="filepath", 
                    label="Source Audio", 
                    visible=True,
                    elem_id="source-audio"
                )
                input_text = gr.Textbox(
                    label="Text Input", 
                    placeholder="Enter text to synthesize...", 
                    lines=4, 
                    visible=False,
                    elem_id="input-text"
                )
                tts_voice = gr.Dropdown(
                    choices=list(TTS_VOICES.keys()), 
                    value="English-US-Male-1", 
                    label="TTS Voice", 
                    visible=False,
                    elem_id="tts-voice"
                )
            
            # PANEL 2: Reference Voice
            with gr.Column(scale=1, elem_classes="grid-panel", elem_id="panel-reference-voice"):
                gr.Markdown("### 🎯 REFERENCE VOICE")
                target_audio = gr.Audio(
                    type="filepath", 
                    label="Reference Audio (up to 25s)",
                    elem_id="reference-audio"
                )
        
        # MIDDLE ROW: Neural Settings + Singing Controls
        with gr.Row(equal_height=True, elem_classes="grid-row"):
            # PANEL 3: Neural Settings
            with gr.Column(scale=1, elem_classes="grid-panel", elem_id="panel-neural-settings"):
                gr.Markdown("### ⚙️ NEURAL VOICE SETTINGS")
                diffusion_steps = gr.Slider(
                    minimum=1, 
                    maximum=200, 
                    value=10, 
                    step=1, 
                    label="Diffusion Steps", 
                    info="10=fast, 50-100=best quality",
                    elem_id="diffusion-steps"
                )
                inference_cfg_rate = gr.Slider(
                    minimum=0.0, 
                    maximum=1.0, 
                    step=0.1, 
                    value=0.7, 
                    label="CFG Rate", 
                    info="Inference guidance",
                    elem_id="cfg-rate"
                )
                length_adjust = gr.Slider(
                    minimum=0.5, 
                    maximum=2.0, 
                    step=0.1, 
                    value=1.0, 
                    label="Length Adjust", 
                    info="<1.0=faster, >1.0=slower",
                    elem_id="length-adjust"
                )
            
            # PANEL 4: Singing/F0 Controls
            with gr.Column(scale=1, elem_classes="grid-panel", elem_id="panel-singing-controls"):
                gr.Markdown("### 🎵 SINGING VOICE CONTROLS")
                model_mode = gr.Radio(
                    ["Voice (22kHz)", "Singing (44kHz)"],
                    value="Voice (22kHz)",
                    label="Model Mode",
                    info="Singing for melody/rap",
                    elem_id="model-mode"
                )
                f0_condition = gr.Checkbox(
                    label="Enable F0 (Pitch) Conditioning",
                    value=False,
                    info="Essential for singing",
                    elem_id="f0-condition"
                )
                pitch_shift = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    value=0,
                    label="Pitch Shift (Semitones)",
                    info="Transpose for key matching",
                    elem_id="pitch-shift"
                )
                auto_f0_adjust = gr.Checkbox(
                    label="Auto F0 Adjust",
                    value=False,
                    info="Auto-match pitch to reference",
                    elem_id="auto-f0-adjust"
                )
        
        # Toggle visibility based on input mode
        def update_visibility(mode):
            if mode == "audio":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
        
        input_mode.change(
            fn=update_visibility, 
            inputs=[input_mode], 
            outputs=[source_audio, input_text, tts_voice]
        )
        
        # BOTTOM ROW: BPM Settings + Output
        with gr.Row(equal_height=True, elem_classes="grid-row"):
            # PANEL 5: BPM/Tempo Controls
            with gr.Column(scale=1, elem_classes="grid-panel", elem_id="panel-bpm-settings"):
                gr.Markdown("### 🎵 EDM BPM SETTINGS")
                target_bpm = gr.Number(
                    label="Target BPM (Auto if blank)", 
                    value=None,
                    precision=0,
                    info="Leave blank for original tempo",
                    elem_id="target-bpm"
                )
                gr.Markdown("💡 **Tip:** Enable F0 + Singing mode for melody. Use pitch shift for key matching.")
            
            # PANEL 6: Output Visualization + Audio
            with gr.Column(scale=1, elem_classes="grid-panel", elem_id="panel-output"):
                gr.Markdown("### 🚀 OUTPUT")
                submit_btn = gr.Button(
                    "GENERATE NEURAL VOCAL", 
                    variant="primary", 
                    size="lg",
                    elem_classes="btn-generate",
                    elem_id="generate-button"
                )
                
                output_full = gr.Audio(
                    label="💾 Generated Vocal", 
                    streaming=False, 
                    format='wav',
                    elem_id="output-audio",
                    show_download_button=False,
                    show_share_button=False
                )
                
                # Wave animation container (hidden, controlled by JS)
                output_viz = gr.HTML(
                    '<div id="outputViz" style="display: none;"></div>',
                    elem_classes="wave-container"
                )
        
        submit_btn.click(
            fn=voice_conversion,
            inputs=[source_audio, target_audio, diffusion_steps, length_adjust, inference_cfg_rate, input_mode, input_text, tts_voice, model_mode, f0_condition, pitch_shift, auto_f0_adjust, target_bpm],
            outputs=[output_full]
        )
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-C", "--config", type=str, default=None)
    parser.add_argument("-fp16", "--fp16", type=str2bool, default=True)
    parser.add_argument("--dev", action="store_true", help="Enable hot-reload for development")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Seed-VC starting with lazy loading...")
    print(f"💻 Device: {device}")
    print(f"⚡ Models will load on first inference request")
    print(f"🎵 Features: TTS + BPM Control + Fast Startup\n")
    
    # Create UI (instant startup, no model loading yet!)
    demo = main(args)
    if args.dev:
        print("🔥 Hot-reload enabled! Changes to .py and .css files will auto-restart the server.")
    demo.launch(inbrowser=True)


