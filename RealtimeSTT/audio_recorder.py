"""

The AudioToTextRecorder class in the provided code facilitates fast speech-to-text transcription.

The class employs the faster_whisper library to transcribe the recorded audio 
into text using machine learning models, which can be run either on a GPU or CPU.
Voice activity detection (VAD) is built in, meaning the software can automatically 
start or stop recording based on the presence or absence of speech.
Additionally, it uses both short-term and long-term noise analysis to determine 
when actual voice activity occurs, as opposed to ambient noise. 
It integrates wake word detection through the pvporcupine library, allowing the 
software to initiate recording when a specific word or phrase is spoken.
The system provides real-time feedback and can be further customized with multiple
parameters like wake word sensitivity, recording intervals, and buffer durations.


Features:
- Voice Activity Detection: Automatically starts/stops recording when speech is detected or when speech ends.
- Wake Word Detection: Starts recording when a specified wake word (or words) is detected.
- Buffer Management: Handles short and long term audio buffers for efficient processing.
- Event Callbacks: Customizable callbacks for when recording starts or finishes.
- Noise Level Calculation: Adjusts based on the background noise for more accurate voice activity detection.

Author: Kolja Beigel

"""

import pyaudio
import collections
import faster_whisper
import torch
import numpy as np
import struct
import pvporcupine
import threading
import time
import logging
from collections import deque

SAMPLE_RATE = 16000
BUFFER_SIZE = 512
LONG_TERM_HISTORY_BUFFERSIZE = 2.0 # seconds
SHORT_TERM_HISTORY_BUFFERSIZE = 2.0 # seconds
WAIT_AFTER_START_BEFORE_ACTIVITY_DETECTION = 0.3 # seconds
ACTIVITY_DETECTION_AFTER_START_PERCENT = 0.6

class AudioToTextRecorder:
    """
    A class responsible for capturing audio from the microphone, detecting voice activity, and then transcribing the captured audio using the `faster_whisper` model.
    """
    
    def __init__(self,
                 model: str = "tiny",
                 language: str = "",
                 wake_words: str = "",
                 wake_words_sensitivity: float = 0.5,
                 on_recording_started = None,
                 on_recording_finished = None,
                 min_recording_interval: float = 1.0,
                 interval_between_records: float = 1.0,
                 buffer_duration: float = 1.0,
                 voice_activity_threshold: float = 250,
                 voice_deactivity_sensitivity: float = 0.3,
                 voice_deactivity_silence_after_speech_end: float = 0.1,
                 long_term_smoothing_factor: float = 0.995,
                 short_term_smoothing_factor: float = 0.900,
                 level=logging.WARNING,
                 ):
        """
        Initializes an audio recorder and  transcription and wake word detection.

        Args:
            model (str): Specifies the size of the transcription model to use or the path to a converted model directory. 
                Valid options are 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'. 
                If a specific size is provided, the model is downloaded from the Hugging Face Hub.
            language (str): Language code for speech-to-text engine. If not specified, the model will attempt to detect the language automatically.
            wake_words (str): Comma-separated string of wake words to initiate recording. Supported wake words include:
                'alexa', 'americano', 'blueberry', 'bumblebee', 'computer', 'grapefruits', 'grasshopper', 'hey google', 'hey siri', 'jarvis', 'ok google', 'picovoice', 'porcupine', 'terminator'.
            wake_words_sensitivity (float): Sensitivity for wake word detection, ranging from 0 (least sensitive) to 1 (most sensitive). Default is 0.5.
            on_recording_started (callable, optional): Callback invoked when recording begins.
            on_recording_finished (callable, optional): Callback invoked when recording ends.
            min_recording_interval (float): Minimum interval (in seconds) for recording durations.
            interval_between_records (float): Interval (in seconds) between consecutive recordings.
            buffer_duration (float): Duration (in seconds) to maintain pre-roll audio in the buffer.
            voice_activity_threshold (float): Threshold level above long-term noise to determine the start of voice activity.
            voice_deactivity_sensitivity (float): Sensitivity for voice deactivation detection, ranging from 0 (least sensitive) to 1 (most sensitive). Default is 0.3.
            voice_deactivity_silence_after_speech_end (float): Duration (in seconds) of silence after speech ends to trigger voice deactivation. Default is 0.1.
            long_term_smoothing_factor (float): Exponential smoothing factor used in calculating long-term noise level.
            short_term_smoothing_factor (float): Exponential smoothing factor used in calculating short-term noise level.
            level (logging level): Desired log level for internal logging. Default is `logging.WARNING`.

        Raises:
            Exception: Errors related to initializing transcription model, wake word detection, or audio recording.
        """

        self.language = language
        self.wake_words = wake_words
        self.min_recording_interval = min_recording_interval
        self.interval_between_records = interval_between_records
        self.buffer_duration = buffer_duration
        self.voice_activity_threshold = voice_activity_threshold
        self.voice_deactivity_sensitivity = voice_deactivity_sensitivity
        self.voice_deactivity_silence_after_speech_end = voice_deactivity_silence_after_speech_end
        self.long_term_smoothing_factor = long_term_smoothing_factor
        self.short_term_smoothing_factor = short_term_smoothing_factor
        self.on_recording_started = on_recording_started
        self.on_recording_finished = on_recording_finished        
        self.level = level

        self.buffer_size = BUFFER_SIZE
        self.sample_rate = SAMPLE_RATE
        self.last_start_time = 0  # time when the recording last started
        self.last_stop_time = 0   # time when the recording last stopped
        self.speech_end_silence_start = 0 

        self.level_long_term = 0
        self.level_short_term = 0
        self.level_peak = 0
        self.level_floor = 0
        self.voice_deactivity_probability = 0
        self.long_term_noise_calculation = True
        self.state = "initializing"

        # Initialize the logging configuration with the specified level
        logging.basicConfig(format='RealTimeSTT: %(message)s', level=level)

        # Initialize the transcription model
        try:
            self.model = faster_whisper.WhisperModel(model_size_or_path=model, device='cuda' if torch.cuda.is_available() else 'cpu')

        except Exception as e:
            logging.exception(f"Error initializing faster_whisper transcription model: {e}")
            raise            

        # Setup wake word detection
        if wake_words:

            self.wake_words_list = [word.strip() for word in wake_words.split(',')]
            sensitivity_list = [float(wake_words_sensitivity) for _ in range(len(self.wake_words_list))]

            try:
                self.porcupine  = pvporcupine.create(keywords=self.wake_words_list, sensitivities=sensitivity_list)
                self.buffer_size = self.porcupine.frame_length
                self.sample_rate = self.porcupine.sample_rate

            except Exception as e:
                logging.exception(f"Error initializing porcupine wake word detection engine: {e}")
                raise

        # Setup audio recording infrastructure
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(rate=self.sample_rate, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=self.buffer_size)

        except Exception as e:
            logging.exception(f"Error initializing pyaudio audio recording: {e}")
            raise            

        # This will store the noise levels for the last x seconds
        # Assuming data is captured at the buffer size rate, determine how many entries 
        buffersize_long_term_history = int((self.sample_rate // self.buffer_size) * LONG_TERM_HISTORY_BUFFERSIZE)
        self.long_term_noise_history = deque(maxlen=buffersize_long_term_history)        
        buffersize_short_term_history = int((self.sample_rate // self.buffer_size) * SHORT_TERM_HISTORY_BUFFERSIZE)
        self.short_term_noise_history = deque(maxlen=buffersize_short_term_history)        

        self.audio_buffer = collections.deque(maxlen=int((self.sample_rate // self.buffer_size) * self.buffer_duration))
        self.frames = []

        # Recording control flags
        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        # Start the recording worker thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()


    def text(self):
        """
        Transcribes audio captured by the class instance using the `faster_whisper` model.

        - Waits for voice activity if not yet started recording 
        - Waits for voice deactivity if not yet stopped recording 
        - Transcribes the recorded audio.

        Returns:
            str: The transcription of the recorded audio or an empty string in case of an error.
        """

        try:        
            # If not yet started to record, wait for voice activity to initiate recording.
            if not self.is_recording and len(self.frames) == 0:

                self.state = "listening"
                self.start_recording_on_voice_activity = True

                while not self.is_recording:
                    time.sleep(0.1)  # Use a small sleep to prevent busy-waiting.

            # If still recording, wait for voice deactivity to finish recording.
            if self.is_recording:

                self.state = "recording"
                self.stop_recording_on_voice_deactivity = True      

                while self.is_recording:
                    time.sleep(0.1)  # Use a small sleep to prevent busy-waiting.

            # Convert the concatenated frames into text
            self.state = "transcribing"

            try:
                audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
                self.frames = []
                return " ".join(seg.text for seg in self.model.transcribe(audio_array, language=self.language if self.language else None)[0]).strip()
            except ValueError:
                logging.error("Error converting audio buffer to numpy array.")
                raise
            except faster_whisper.WhisperError as e:
                logging.error(f"Whisper transcription error: {e}")
                raise
            except Exception as e:
                logging.error(f"General transcription error: {e}")
                raise

        except Exception as e:
                print(f"Error during transcription: {e}")           
                return ""          


    def start(self):
        """
        Starts recording audio directly without waiting for voice activity.
        """

        current_time = time.time()
        
        # Ensure there's a minimum interval between stopping and starting recording
        if current_time - self.last_stop_time < self.interval_between_records:
            logging.info("Attempted to start recording too soon after stopping.")
            return self
        
        logging.info("recording started")
        self.state = "recording"
        self.frames = []
        self.is_recording = True
        self.last_start_time = current_time

        if self.on_recording_started:
            self.on_recording_started()

        return self
    

    def stop(self):
        logging.info("recording stopped")
        """
        Stops recording audio.
        """

        current_time = time.time()

        # Ensure there's a minimum interval between starting and stopping recording
        if current_time - self.last_start_time < self.interval_between_records:
            logging.info("Attempted to stop recording too soon after starting.")
            return self
                
        logging.info("recording stopped")                
        self.state = "listening"
        self.is_recording = False
        self.last_stop_time = current_time

        if self.on_recording_finished:
            self.on_recording_finished()

        return self


    def shutdown(self):
        """
        Safely shuts down the audio recording by stopping the recording worker and closing the audio stream.
        """
        self.is_recording = False
        self.is_running = False
        self.recording_thread.join()
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
        except Exception as e:
            logging.error(f"Error closing the audio stream: {e}")


    def _calculate_percentile_mean(self, buffer, percentile, upper=True):
        """
        Calculates the mean of the specified percentile from the provided buffer of 
        long_term noise levels. If upper is True, it calculates from the upper side,
        otherwise from the lower side.

        Args:
        - buffer (list): The buffer containing the history of long_term noise levels.
        - percentile (float): The desired percentile (0.0 <= percentile <= 1.0). E.g., 0.125 for 1/8.
        - upper (bool): Determines if the function considers the upper or lower portion of data.

        Returns:
        - float: The mean value of the desired portion.
        """
        sorted_buffer = sorted(buffer)
        
        index = int(len(sorted_buffer) * percentile)

        if upper:
            values = sorted_buffer[-index:]  # Get values from the top
        else:
            values = sorted_buffer[:index]   # Get values from the bottom

        if len(values) == 0:
            return 0.0
        
        return sum(values) / len(values)
    

    def _recording_worker(self):
        """
        The main worker method which constantly monitors the audio input for voice activity and accordingly starts/stops the recording.
        Uses long_term noise level measurements to determine voice activity.
        """
        
        was_recording = False
        voice_after_recording = False

        # Continuously monitor audio for voice activity
        while self.is_running:

            try:
                data = self.stream.read(self.buffer_size)
            except pyaudio.paInputOverflowed:
                logging.warning("Input overflowed. Frame dropped.")
                continue
            except Exception as e:
                logging.error(f"Error during recording: {e}")
                time.sleep(1)
                continue

            audio_level = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
            if not self.is_recording and self.long_term_noise_calculation:
                self.level_long_term = self.level_long_term * self.long_term_smoothing_factor + audio_level * (1.0 - self.long_term_smoothing_factor)
            self.level_short_term = self.level_short_term * self.short_term_smoothing_factor + audio_level * (1.0 - self.short_term_smoothing_factor)
            
            self.long_term_noise_history.append(self.level_long_term)
            self.short_term_noise_history.append(self.level_short_term)

            self.level_peak = self._calculate_percentile_mean(self.short_term_noise_history, 0.05, upper=True)
            self.level_floor = self._calculate_percentile_mean(self.short_term_noise_history, 0.1, upper=False)
            short_term_to_peak_percentage = (self.level_short_term - self.level_floor) / (self.level_peak - self.level_floor)

            if not self.is_recording:
                logging.debug(f'Level: {int(audio_level)}, long_term: {int(self.level_long_term)}, short_term: {int(self.level_short_term)}, Peak: {int(self.level_peak)}, long_term low: {int(self.level_floor)}, Percentage: {int(short_term_to_peak_percentage*100)}%')
            else:
                short_term_to_peak_percentage = (self.level_short_term - self.level_long_term) / (self.level_peak - self.level_long_term)
                logging.debug(f'Level: {int(audio_level)}, long_term: {int(self.level_long_term)}, short_term: {int(self.level_short_term)}, Peak: {int(self.level_peak)}, long_term low: {int(self.level_floor)}, Percentage: {int(short_term_to_peak_percentage*100)}%')

            # Check if we're not currently recording
            if not self.is_recording:

                voice_after_recording = False

                # Check if wake word detection is active
                if self.wake_words:

                    try:
                        pcm = struct.unpack_from("h" * self.buffer_size, data)
                        wakeword_index = self.porcupine.process(pcm)
                    except struct.error:
                        logging.error("Error unpacking audio data for wake word processing.")
                        continue
                    except Exception as e:
                        logging.error(f"Wake word processing error: {e}")
                        continue
                    
                    wakeword_detected = wakeword_index >= 0
                    
                    if wakeword_detected:
                        logging.info(f'wake word "{self.wake_words_list[wakeword_index]}" detected')
                        self.start()
                        if self.is_recording:
                            self.level_long_term = self._calculate_percentile_mean(self.long_term_noise_history, 0.125, upper=False)
                            self.start_recording_on_voice_activity = False

                # Check for voice activity to trigger the start of recording
                elif self.start_recording_on_voice_activity and self.level_short_term > self.level_long_term + self.voice_activity_threshold:

                    logging.info("voice activity detected")

                    self.start()

                    if self.is_recording:
                        self.level_long_term = self._calculate_percentile_mean(self.long_term_noise_history, 0.125, upper=False)
                        self.start_recording_on_voice_activity = False

                        # Add the buffered audio to the recording frames
                        self.frames.extend(list(self.audio_buffer))
                    
                self.speech_end_silence_start = 0

            # If we're currently recording and voice deactivity is detected, stop the recording
            else:
                current_time = time.time()

                self.state = "recording - waiting for voice end" if voice_after_recording else "recording - waiting for voice"

                # we don't detect voice in the first x seconds cause it could be fragments from the wake word
                if current_time - self.last_start_time > WAIT_AFTER_START_BEFORE_ACTIVITY_DETECTION:
                    if not voice_after_recording and self.level_short_term > self.level_long_term + (self.voice_activity_threshold * ACTIVITY_DETECTION_AFTER_START_PERCENT):
                        logging.info("voice activity after recording detected")
                        voice_after_recording = True

                # we are recording
                short_term_to_peak_percentage = (self.level_short_term - self.level_long_term) / (self.level_peak - self.level_long_term)
                logging.debug(f'short_term_to_peak_percentage: {int(short_term_to_peak_percentage*100)}%, peak: {int(self.level_peak)}, long_term: {int(self.level_long_term)}')

                if voice_after_recording and self.stop_recording_on_voice_deactivity: 
                    if short_term_to_peak_percentage < self.voice_deactivity_sensitivity:
                        # silence detected (after voice detected while recording)

                        if self.speech_end_silence_start == 0:
                            self.speech_end_silence_start = time.time()
                            self.state = "recording - voice end, silence wait"
                        
                    else:
                        self.speech_end_silence_start = 0

                    if self.speech_end_silence_start and time.time() - self.speech_end_silence_start > self.voice_deactivity_silence_after_speech_end:
                        logging.info("voice deactivity detected")
                        self.stop()
                        if not self.is_recording:
                                voice_after_recording = False

            if not self.is_recording and was_recording:
                # Reset after stopping recording to ensure clean state
                self.stop_recording_on_voice_deactivity = False

            short_term_to_peak_percentage = min(max(short_term_to_peak_percentage, 0.0), 1.0)
            self.voice_deactivity_probability = 1 - short_term_to_peak_percentage

            if self.is_recording:
                self.frames.append(data)

            self.audio_buffer.append(data)	

            was_recording = self.is_recording
            time.sleep(0.01)

    def __del__(self):
        """
        Destructor method ensures safe shutdown of the recorder when the instance is destroyed.
        """
        self.shutdown()