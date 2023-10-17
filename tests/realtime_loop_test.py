from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal

import sys
import os

from RealtimeTTS import TextToAudioStream, AzureEngine
from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':

    class SimpleApp(QWidget):

        update_stt_text_signal = pyqtSignal(str)
        update_tts_text_signal = pyqtSignal(str)

        def __init__(self):
            super().__init__()

            layout = QVBoxLayout()

            font = QFont()
            font.setPointSize(18)

            self.input_text = QTextEdit(self)
            self.input_text.setFont(font)
            self.input_text.setPlaceholderText("Input")
            self.input_text.setMinimumHeight(100) 
            layout.addWidget(self.input_text)

            self.button_speak_input = QPushButton("Speak and detect input text", self)
            self.button_speak_input.setFont(font)        
            self.button_speak_input.clicked.connect(self.speak_input)
            layout.addWidget(self.button_speak_input)

            self.tts_text = QTextEdit(self)
            self.tts_text.setFont(font)
            self.tts_text.setPlaceholderText("STT (final)")
            self.tts_text.setMinimumHeight(100) 
            self.tts_text.setReadOnly(True)
            layout.addWidget(self.tts_text)

            self.stt_text = QTextEdit(self)
            self.stt_text.setFont(font)
            self.stt_text.setPlaceholderText("STT (realtime)")
            self.stt_text.setMinimumHeight(100) 
            layout.addWidget(self.stt_text)

            self.button_speak_stt = QPushButton("Speak detected text again", self)
            self.button_speak_stt.setFont(font)        
            self.button_speak_stt.clicked.connect(self.speak_stt)
            layout.addWidget(self.button_speak_stt)

            self.setLayout(layout)
            self.setWindowTitle("Realtime TTS/STT Loop Test")
            self.resize(800, 600)

            self.update_stt_text_signal.connect(self.actual_update_stt_text)
            self.update_tts_text_signal.connect(self.actual_update_tts_text)

            self.stream = TextToAudioStream(AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), "germanywestcentral"), on_audio_stream_stop=self.audio_stream_stop)

            recorder_config = {
                'spinner': False,
                'model': 'large-v2',
                'language': 'en',
                'silero_sensitivity': 0.01,
                'webrtc_sensitivity': 3,
                'post_speech_silence_duration': 0.01,
                'min_length_of_recording': 0.2,
                'min_gap_between_recordings': 0,
                'enable_realtime_transcription': True,
                'realtime_processing_pause': 0,
                'realtime_model_type': 'small.en',
                'on_realtime_transcription_stabilized': self.text_detected,
            }

            self.recorder = AudioToTextRecorder(**recorder_config)

        def speak_stt(self):
            text = self.stt_text.toPlainText()
            self.speak(text)

        def speak_input(self):
            text = self.input_text.toPlainText()
            self.speak(text)

        def text_detected(self, text):
            self.update_stt_text_signal.emit(text)

        def audio_stream_stop(self):
            self.stream.stop()
            self.recorder.stop()
            detected_text = self.recorder.text()
            self.update_stt_text_signal.emit(detected_text)
            self.update_tts_text_signal.emit(detected_text)

        def speak(self, text):
            self.stt_text.clear()        
            self.stream.feed(text)

            self.recorder.start()
            self.stream.play_async()

        def actual_update_stt_text(self, text):
            self.stt_text.setText(text)

        def actual_update_tts_text(self, text):
            self.tts_text.setText(text)

        def closeEvent(self, event):
            if self.recorder:
                self.recorder.shutdown()

    app = QApplication(sys.argv)

    window = SimpleApp()
    window.show()

    sys.exit(app.exec_())