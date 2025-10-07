# src/ui/app.py
from pathlib import Path
from datetime import datetime
from PyQt6 import QtWidgets

from ..config import OUTPUT_DIR, ROBOTIC_ID
from ..pipeline.preprocess import clean_text
from ..tts.manager import VoiceManager
from ..audio.effects import process_wav, EffectSettings
from ..audio.export import to_mp3


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EDM Vocal Generator - MVP")
        self.vm = VoiceManager()

        # Widgets
        self.text = QtWidgets.QPlainTextEdit()

        self.voice = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("Refresh voices")
        self.refresh_btn.clicked.connect(self.populate_voices)

        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(0.5, 1.5)
        self.speed.setSingleStep(0.05)
        self.speed.setValue(1.0)

        self.pitch = QtWidgets.QDoubleSpinBox()
        self.pitch.setRange(-12, 12)
        self.pitch.setSingleStep(0.5)
        self.pitch.setValue(0.0)

        self.rev = QtWidgets.QDoubleSpinBox()
        self.rev.setRange(0.0, 1.0)
        self.rev.setSingleStep(0.05)
        self.rev.setValue(0.15)

        self.dly = QtWidgets.QDoubleSpinBox()
        self.dly.setRange(0.0, 1.0)
        self.dly.setSingleStep(0.05)
        self.dly.setValue(0.10)

        self.render_btn = QtWidgets.QPushButton("Generate and Export")
        self.render_btn.clicked.connect(self.on_render)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Text:", self.text)

        voice_row = QtWidgets.QHBoxLayout()
        voice_row.addWidget(self.voice)
        voice_row.addWidget(self.refresh_btn)
        form.addRow("Voice:", voice_row)

        form.addRow("Speed:", self.speed)
        form.addRow("Pitch (semitones):", self.pitch)
        form.addRow("Reverb Mix:", self.rev)
        form.addRow("Delay Mix:", self.dly)
        form.addRow(self.render_btn)
        self.setLayout(form)

        # Fill voices now
        self.populate_voices()

    def populate_voices(self):
        """Populate the voice list from VoiceManager instead of hard-coding."""
        self.voice.clear()
        try:
            voices = self.vm.get_available_voices()
            # Optional: keep Robotic first if present, then the rest alphabetically
            if ROBOTIC_ID in voices:
                self.voice.addItem(ROBOTIC_ID)
                remaining = sorted(v for v in voices if v != ROBOTIC_ID)
                for v in remaining:
                    self.voice.addItem(v)
            else:
                for v in sorted(voices):
                    self.voice.addItem(v)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Voices", f"Could not load voices:\n{e}")
            # Fallback minimal list
            self.voice.addItems([ROBOTIC_ID, "Realistic Male", "Realistic Female"])

    def on_render(self):
        txt = clean_text(self.text.toPlainText())
        if not txt:
            QtWidgets.QMessageBox.warning(self, "Input required", "Enter some text to synthesize.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = OUTPUT_DIR / f"vocal_{ts}"
        raw_wav = base.with_suffix(".raw.wav")
        fx_wav = base.with_suffix(".wav")
        mp3 = base.with_suffix(".mp3")

        voice_key = self.voice.currentText()
        speed = float(self.speed.value())
        pitch = float(self.pitch.value())

        try:
            self.render_btn.setEnabled(False)
            # 1) TTS to raw WAV
            self.vm.synth(voice_key, txt, raw_wav, speed=speed)
            # 2) Effects & export WAV
            cfg = EffectSettings(
                pitch_semitones=pitch,
                speed=1.0 if voice_key != ROBOTIC_ID else speed,
                reverb_mix=float(self.rev.value()),
                delay_mix=float(self.dly.value()),
            )
            process_wav(raw_wav, fx_wav, cfg)
            # 3) MP3
            try:
                to_mp3(fx_wav, mp3)
                mp3_msg = str(mp3)
            except Exception:
                mp3_msg = "(MP3 skipped - install ffmpeg)"

            QtWidgets.QMessageBox.information(self, "Done", f"Exported:\n{fx_wav}\n{mp3_msg}")
        finally:
            self.render_btn.setEnabled(True)
