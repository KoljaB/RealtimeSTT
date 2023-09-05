from RealtimeTTS import TextToAudioStream, AzureEngine, ElevenlabsEngine, SystemEngine
from RealtimeSTT import AudioToTextRecorder

from PyQt5.QtCore import Qt, QTimer, QRect, QEvent, pyqtSignal, QThread, QPoint, QPropertyAnimation, QVariantAnimation
from PyQt5.QtGui import QPalette, QColor, QPainter, QFontMetrics, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QDesktopWidget

import os
import openai
import sys
import time
import sounddevice as sd
import numpy as np
import wavio


max_history_messages = 6
return_to_wakewords_after_silence = 12
start_with_wakeword = False
recorder_model = "large-v2"
language = "de"

openai.api_key = os.environ.get("OPENAI_API_KEY")
azure_speech_region = "germanywestcentral"

user_font_size = 22
user_color = QColor(208, 208, 208) # gray

assistant_font_size = 24
assistant_color = QColor(240, 240, 240) # white



voice = "en-GB-SoniaNeural"
prompt = "Respond helpfully, concisely, and when appropriate, with the subtle, polite irony of a butler."

if language == "de":
    voice = "de-DE-MajaNeural"
    prompt = 'Antworte hilfreich, knapp und bei Gelegenheit mit der feinen, höflichen Ironie eines Butlers.'


system_prompt_message = {
    'role': 'system',
    'content': prompt
}

def generate_response(messages):
    """Generate assistant's response using OpenAI."""
    for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True, logit_bias={35309:-100, 36661:-100}):
        text_chunk = chunk["choices"][0]["delta"].get("content")
        if text_chunk:
            yield text_chunk

history = []
MAX_WINDOW_WIDTH = 1600
MAX_WIDTH_ASSISTANT = 1200
MAX_WIDTH_USER = 1500

class AudioPlayer(QThread):
    def __init__(self, file_path):
        super(AudioPlayer, self).__init__()
        self.file_path = file_path

    def run(self):
        wav = wavio.read(self.file_path)
        sound = wav.data.astype(np.float32) / np.iinfo(np.int16).max  
        sd.play(sound, wav.rate)
        sd.wait()

class TextRetrievalThread(QThread):
    textRetrieved = pyqtSignal(str)

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.active = False  

    def run(self):
        while True:
            if self.active:  
                text = self.recorder.text()
                self.recorder.wake_word_activation_delay = return_to_wakewords_after_silence
                self.textRetrieved.emit(text)
                self.active = False
            time.sleep(0.1) 

    def activate(self):
        self.active = True 

class TransparentWindow(QWidget):
    updateUI = pyqtSignal()
    clearAssistantTextSignal = pyqtSignal()
    clearUserTextSignal = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setGeometry(1, 1, 1, 1) 

        self.setWindowTitle("Transparent Window")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        self.big_symbol_font = QFont('Arial', 32)
        self.small_symbol_font = QFont('Arial', 17)
        self.user_font = QFont('Arial', user_font_size)
        self.assistant_font = QFont('Arial', assistant_font_size)      
        self.assistant_font.setItalic(True) 

        self.big_symbol_text = ""
        self.small_symbol_text = ""
        self.user_text = ""
        self.assistant_text = ""
        self.displayed_user_text = ""
        self.displayed_assistant_text = ""
        self.stream = None
        self.text_retrieval_thread = None

        self.user_text_timer = QTimer(self)
        self.assistant_text_timer = QTimer(self)
        self.user_text_timer.timeout.connect(self.clear_user_text)
        self.assistant_text_timer.timeout.connect(self.clear_assistant_text)

        self.clearUserTextSignal.connect(self.init_clear_user_text)
        self.clearAssistantTextSignal.connect(self.init_clear_assistant_text)
        self.user_text_opacity = 255 
        self.assistant_text_opacity = 255 
        self.updateUI.connect(self.update_self)
        self.audio_player = None

        self.run_fade_user = False
        self.run_fade_assistant = False

    def init(self):
        self.stream = TextToAudioStream(
                # SystemEngine(),

                AzureEngine(
                    os.environ.get("AZURE_SPEECH_KEY"),
                    azure_speech_region,
                    voice,
                    rate=34,
                    pitch=10,
                ),

                # ElevenlabsEngine(
                #     os.environ.get("ELEVENLABS_API_KEY")
                # ),
                on_character=self.on_character,
                on_text_stream_stop=self.on_text_stream_stop,
                on_text_stream_start=self.on_text_stream_start,
                on_audio_stream_stop=self.on_audio_stream_stop,
                log_characters=True,
            )       
        self.recorder = AudioToTextRecorder(
            model=recorder_model,
            language=language,
            wake_words="Jarvis",
            spinner=True,
            silero_sensitivity=0.2,
            webrtc_sensitivity=3,
            on_recording_start=self.on_recording_start,
            on_vad_detect_start=self.on_vad_detect_start,
            on_wakeword_detection_start=self.on_wakeword_detection_start,
            on_transcription_start=self.on_transcription_start,
        )
        if not start_with_wakeword:
            self.recorder.wake_word_activation_delay = return_to_wakewords_after_silence
            
        self.text_retrieval_thread = TextRetrievalThread(self.recorder)
        self.text_retrieval_thread.textRetrieved.connect(self.process_user_text)
        self.text_retrieval_thread.start()
        self.text_retrieval_thread.activate()

    def showEvent(self, event: QEvent):
        super().showEvent(event)
        if event.type() == QEvent.Show:
            self.set_symbols("⌛", "🚀")
            QTimer.singleShot(1000, self.init) 

    def on_character(self, char):
        if self.stream:
            self.assistant_text += char
            self.updateUI.emit()

    def on_text_stream_stop(self):
        print("\"", end="", flush=True)
        if self.stream:
            assistant_response = self.stream.text()            
            self.assistant_text = assistant_response
            history.append({'role': 'assistant', 'content': assistant_response})

    def on_audio_stream_stop(self):
        if self.stream:
            self.clearAssistantTextSignal.emit()
            self.text_retrieval_thread.activate()

    def generate_answer(self):
        self.run_fade_assistant = False
        if self.assistant_text_timer.isActive():
            self.assistant_text_timer.stop()

        history.append({'role': 'user', 'content': self.user_text})
        self.remove_assistant_text()
        assistant_response = generate_response([system_prompt_message] + history[-max_history_messages:])
        self.stream.feed(assistant_response)
        self.stream.play_async(minimum_sentence_length=7,
                               buffer_threshold_seconds=3)

    def set_symbols(self, big_symbol, small_symbol):
        self.big_symbol_text = big_symbol
        self.small_symbol_text = small_symbol
        self.updateUI.emit()

    def on_text_stream_start(self):
        self.set_symbols("⌛", "👄")

    def process_user_text(self, user_text):
        user_text = user_text.strip()
        if user_text:
            self.run_fade_user = False
            if self.user_text_timer.isActive():
                self.user_text_timer.stop()

            self.user_text_opacity = 255 
            self.user_text = user_text
            self.clearUserTextSignal.emit()
            print (f"Me: \"{user_text}\"\nAI: \"", end="", flush=True)
            self.set_symbols("⌛", "🧠")
            QTimer.singleShot(100, self.generate_answer)

    def on_transcription_start(self):
        self.set_symbols("⌛", "📝")

    def on_recording_start(self):
        self.set_symbols("🎙️", "🔴")

    def on_vad_detect_start(self):
        if self.small_symbol_text == "💤" or self.small_symbol_text == "🚀":
            self.audio_player = AudioPlayer("active.wav")
            self.audio_player.start() 

        self.set_symbols("🎙️", "⚪")

    def on_wakeword_detection_start(self):
        self.audio_player = AudioPlayer("inactive.wav")
        self.audio_player.start()         

        self.set_symbols("", "💤")

    def init_clear_user_text(self):
        if self.user_text_timer.isActive():
            self.user_text_timer.stop()        
        self.user_text_timer.start(10000)

    def remove_user_text(self):
        self.user_text = ""
        self.user_text_opacity = 255 
        self.updateUI.emit()

    def fade_out_user_text(self):
        if not self.run_fade_user:
            return

        if self.user_text_opacity > 0:
            self.user_text_opacity -= 5 
            self.updateUI.emit()
            QTimer.singleShot(50, self.fade_out_user_text)
        else:
            self.run_fade_user = False
            self.remove_user_text()        

    def clear_user_text(self):
        self.user_text_timer.stop()

        if not self.user_text:
            return

        self.user_text_opacity = 255
        self.run_fade_user = True
        self.fade_out_user_text()

    def init_clear_assistant_text(self):
        if self.assistant_text_timer.isActive():
            self.assistant_text_timer.stop()        
        self.assistant_text_timer.start(10000)

    def remove_assistant_text(self):
        self.assistant_text = ""
        self.assistant_text_opacity = 255 
        self.updateUI.emit()

    def fade_out_assistant_text(self):
        if not self.run_fade_assistant:
            return
        
        if self.assistant_text_opacity > 0:
            self.assistant_text_opacity -= 5 
            self.updateUI.emit()
            QTimer.singleShot(50, self.fade_out_assistant_text)
        else:
            self.run_fade_assistant = False
            self.remove_assistant_text()        

    def clear_assistant_text(self):
        self.assistant_text_timer.stop()

        if not self.assistant_text:
            return

        self.assistant_text_opacity = 255
        self.run_fade_assistant = True
        self.fade_out_assistant_text()

    def update_self(self):

        self.blockSignals(True)
                
        self.displayed_user_text, self.user_width = self.return_text_adjusted_to_width(self.user_text, self.user_font, MAX_WIDTH_USER)
        self.displayed_assistant_text, self.assistant_width = self.return_text_adjusted_to_width(self.assistant_text, self.assistant_font, MAX_WIDTH_ASSISTANT)       

        fm_symbol = QFontMetrics(self.big_symbol_font)
        self.symbol_width = fm_symbol.width(self.big_symbol_text) + 3
        self.symbol_height = fm_symbol.height() + 8

        self.total_width = MAX_WINDOW_WIDTH

        fm_user = QFontMetrics(self.user_font)
        user_text_lines = (self.displayed_user_text.count("\n") + 1)
        self.user_height = fm_user.height() * user_text_lines + 7

        fm_assistant = QFontMetrics(self.assistant_font)
        assistant_text_lines = (self.displayed_assistant_text.count("\n") + 1)
        self.assistant_height = fm_assistant.height() * assistant_text_lines + 18

        self.total_height = sum([self.symbol_height, self.user_height, self.assistant_height])

        desktop = QDesktopWidget()
        screen_rect = desktop.availableGeometry(desktop.primaryScreen())
        self.setGeometry(screen_rect.right() - self.total_width - 50, 0, self.total_width + 50, self.total_height + 50)

        self.blockSignals(False)

        self.update()

    def drawTextWithOutline(self, painter, x, y, width, height, alignment, text, textColor, outlineColor, outline_size):
        painter.setPen(outlineColor)
        for dx, dy in [(-outline_size, 0), (outline_size, 0), (0, -outline_size), (0, outline_size),
                    (-outline_size, -outline_size), (outline_size, -outline_size),
                    (-outline_size, outline_size), (outline_size, outline_size)]:
            painter.drawText(x + dx, y + dy, width, height, alignment, text)

        painter.setPen(textColor)
        painter.drawText(x, y, width, height, alignment, text)

    def paintEvent(self, event):
        painter = QPainter(self)

        offsetX = 4
        offsetY = 5
    
        painter.setPen(QColor(255, 255, 255))

        # Draw symbol
        painter.setFont(self.big_symbol_font)
        if self.big_symbol_text:
            painter.drawText(self.total_width - self.symbol_width + 5 + offsetX, offsetY, self.symbol_width, self.symbol_height, Qt.AlignRight | Qt.AlignTop, self.big_symbol_text)
            painter.setFont(self.small_symbol_font)
            painter.drawText(self.total_width - self.symbol_width + 17 + offsetX, offsetY + 10, self.symbol_width, self.symbol_height, Qt.AlignRight | Qt.AlignBottom, self.small_symbol_text)
        else:
            painter.setFont(self.small_symbol_font)
            painter.drawText(self.total_width - 43 + offsetX, offsetY + 2, 50, 50, Qt.AlignRight | Qt.AlignBottom, self.small_symbol_text)

        # Draw User Text
        painter.setFont(self.user_font)
        user_x = self.total_width - self.user_width - 45 + offsetX
        user_y = offsetY + 15
        user_color_with_opacity = QColor(user_color.red(), user_color.green(), user_color.blue(), self.user_text_opacity)
        outline_color_with_opacity = QColor(0, 0, 0, self.user_text_opacity)
        self.drawTextWithOutline(painter, user_x, user_y, self.user_width, self.user_height, Qt.AlignRight | Qt.AlignTop, self.displayed_user_text, user_color_with_opacity, outline_color_with_opacity, 2)

        # Draw Assistant Text
        painter.setFont(self.assistant_font)
        assistant_x = self.total_width - self.assistant_width - 5  + offsetX
        assistant_y = self.user_height + offsetY + 15
        assistant_color_with_opacity = QColor(assistant_color.red(), assistant_color.green(), assistant_color.blue(), self.assistant_text_opacity)
        outline_color_with_opacity = QColor(0, 0, 0, self.assistant_text_opacity)
        self.drawTextWithOutline(painter, assistant_x, assistant_y, self.assistant_width, self.assistant_height, Qt.AlignRight | Qt.AlignTop, self.displayed_assistant_text, assistant_color_with_opacity, outline_color_with_opacity, 2)

    def return_text_adjusted_to_width(self, text, font, max_width_allowed):
        """
        Line feeds are inserted so that the text width does never exceed max_width.
        Text is only broken up on whole words.
        """
        fm = QFontMetrics(font)
        words = text.split(' ')
        adjusted_text = ''
        current_line = ''
        max_width_used = 0
        
        for word in words:
            current_width = fm.width(current_line + word)
            if current_width <= max_width_allowed:
                current_line += word + ' '
            else:
                line_width = fm.width(current_line)
                if line_width > max_width_used:
                    max_width_used = line_width
                adjusted_text += current_line + '\n'
                current_line = word + ' '
        
        line_width = fm.width(current_line)
        if line_width > max_width_used:
            max_width_used = line_width
        adjusted_text += current_line 
        return adjusted_text.rstrip(), max_width_used         

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = TransparentWindow()
    window.show()

    sys.exit(app.exec_())