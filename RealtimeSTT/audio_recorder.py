"""

The AudioToTextRecorder class in the provided code facilitates fast speech-to-text transcription.

The class employs the faster_whisper library to transcribe the recorded audio 
into text using machine learning models, which can be run either on a GPU or CPU.
Voice activity detection (VAD) is built in, meaning the software can automatically 
start or stop recording based on the presence or absence of speech.
It integrates wake word detection through the pvporcupine library, allowing the 
software to initiate recording when a specific word or phrase is spoken.
The system provides real-time feedback and can be further customized.

Features:
- Voice Activity Detection: Automatically starts/stops recording when speech is detected or when speech ends.
- Wake Word Detection: Starts recording when a specified wake word (or words) is detected.
- Event Callbacks: Customizable callbacks for when recording starts or finishes.
- Fast Transcription: Returns the transcribed text from the audio as fast as possible.

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
import webrtcvad
import itertools
from collections import deque
from halo import Halo


SAMPLE_RATE = 16000
BUFFER_SIZE = 512
SILERO_SENSITIVITY = 0.6
WEBRTC_SENSITIVITY = 3
WAKE_WORDS_SENSITIVITY = 0.6
TIME_SLEEP = 0.02

class AudioToTextRecorder:
    """
    A class responsible for capturing audio from the microphone, detecting voice activity, and then transcribing the captured audio using the `faster_whisper` model.
    """
    
    def __init__(self,
                 model: str = "tiny",
                 language: str = "",
                 on_recording_start = None,
                 on_recording_stop = None,
                 on_transcription_start = None,
                 spinner = True,
                 level=logging.WARNING,

                 # Voice activation parameters
                 silero_sensitivity: float = SILERO_SENSITIVITY,
                 webrtc_sensitivity: int = WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = 0.2,
                 min_length_of_recording: float = 1.0,
                 min_gap_between_recordings: float = 1.0,
                 pre_recording_buffer_duration: float = 1,
                 on_vad_detect_start = None,
                 on_vad_detect_stop = None,

                 # Wake word parameters
                 wake_words: str = "",
                 wake_words_sensitivity: float = WAKE_WORDS_SENSITIVITY,
                 wake_word_activation_delay: float = 0,
                 wake_word_timeout: float = 5.0,
                 on_wakeword_detected = None,
                 on_wakeword_timeout = None,
                 on_wakeword_detection_start = None,
                 on_wakeword_detection_end = None,
                 ):
        """
        Initializes an audio recorder and  transcription and wake word detection.

        Args:
        - model (str, default="tiny"): Specifies the size of the transcription model to use or the path to a converted model directory. 
                Valid options are 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'. 
                If a specific size is provided, the model is downloaded from the Hugging Face Hub.
        - language (str, default=""): Language code for speech-to-text engine. If not specified, the model will attempt to detect the language automatically.
        - on_recording_start (callable, default=None): Callback function to be called when recording of audio to be transcripted starts.
        - on_recording_stop (callable, default=None): Callback function to be called when recording of audio to be transcripted stops.
        - on_transcription_start (callable, default=None): Callback function to be called when transcription of audio to text starts.       
        - spinner (bool, default=True): Show spinner animation with current state.
        - level (int, default=logging.WARNING): Logging level.
        - silero_sensitivity (float, default=SILERO_SENSITIVITY): Sensitivity for the Silero Voice Activity Detection model ranging from 0 (least sensitive) to 1 (most sensitive). Default is 0.5.
        - webrtc_sensitivity (int, default=WEBRTC_SENSITIVITY): Sensitivity for the WebRTC Voice Activity Detection engine ranging from 1 (least sensitive) to 3 (most sensitive). Default is 3.
        - post_speech_silence_duration (float, default=0.2): Duration in seconds of silence that must follow speech before the recording is considered to be completed. This ensures that any brief pauses during speech don't prematurely end the recording.
        - min_gap_between_recordings (float, default=1.0): Specifies the minimum time interval in seconds that should exist between the end of one recording session and the beginning of another to prevent rapid consecutive recordings.
        - min_length_of_recording (float, default=1.0): Specifies the minimum duration in seconds that a recording session should last to ensure meaningful audio capture, preventing excessively short or fragmented recordings.
        - pre_recording_buffer_duration (float, default=0.2): Duration in seconds for the audio buffer to maintain pre-roll audio (compensates speech activity detection latency)
        - wake_words (str, default=""): Comma-separated string of wake words to initiate recording. Supported wake words include:
                'alexa', 'americano', 'blueberry', 'bumblebee', 'computer', 'grapefruits', 'grasshopper', 'hey google', 'hey siri', 'jarvis', 'ok google', 'picovoice', 'porcupine', 'terminator'.
        - wake_words_sensitivity (float, default=0.5): Sensitivity for wake word detection, ranging from 0 (least sensitive) to 1 (most sensitive). Default is 0.5.
        - wake_word_activation_delay (float, default=0): Duration in seconds after the start of monitoring before the system switches to wake word activation if no voice is initially detected. If set to zero, the system uses wake word activation immediately.
        - wake_word_timeout (float, default=5): Duration in seconds after a wake word is recognized. If no subsequent voice activity is detected within this window, the system transitions back to an inactive state, awaiting the next wake word or voice activation.
        - on_wakeword_detected (callable, default=None): Callback function to be called when a wake word is detected.
        - on_wakeword_timeout (callable, default=None): Callback function to be called when the system goes back to an inactive state after when no speech was detected after wake word activation
        - on_wakeword_detection_start (callable, default=None): Callback function to be called when the system starts to listen for wake words
        - on_wakeword_detection_end (callable, default=None): Callback function to be called when the system stops to listen for wake words (e.g. because of timeout or wake word detected)

        Raises:
            Exception: Errors related to initializing transcription model, wake word detection, or audio recording.
        """

        self.language = language
        self.wake_words = wake_words
        self.wake_word_activation_delay = wake_word_activation_delay
        self.wake_word_timeout = wake_word_timeout
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording       
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.post_speech_silence_duration = post_speech_silence_duration
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop        
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_transcription_start = on_transcription_start
    
        self.level = level
        self.buffer_size = BUFFER_SIZE
        self.sample_rate = SAMPLE_RATE
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0 
        self.speech_end_silence_start = 0 
        self.silero_sensitivity = silero_sensitivity
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False

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

            self.wake_words_list = [word.strip() for word in wake_words.lower().split(',')]
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


        # Setup voice activity detection model WebRTC
        try:
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)

        except Exception as e:
            logging.exception(f"Error initializing WebRTC voice activity detection engine: {e}")
            raise       


        # Setup voice activity detection model Silero VAD
        try:
            self.silero_vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                verbose=False
            )

        except Exception as e:
            logging.exception(f"Error initializing Silero VAD voice activity detection engine: {e}")
            raise       

        self.audio_buffer = collections.deque(maxlen=int((self.sample_rate // self.buffer_size) * self.pre_recording_buffer_duration))
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

        self.listen_start = time.time()
        
                
        # If not yet started to record, wait for voice activity to initiate recording.
        if not self.is_recording and len(self.frames) == 0:
            self._set_state("listening")
            self._set_spinner("speak now")
            self.start_recording_on_voice_activity = True

            while not self.is_recording:
                time.sleep(TIME_SLEEP)

        # If still recording, wait for voice deactivity to finish recording.
        if self.is_recording:
            self.stop_recording_on_voice_deactivity = True      

            while self.is_recording:
                time.sleep(TIME_SLEEP)

        # Convert the concatenated frames into text
        try:
            audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            self.frames = []

            # perform transcription
            transcription = " ".join(seg.text for seg in self.model.transcribe(audio_array, language=self.language if self.language else None)[0]).strip()

            self.recording_stop_time = 0
            self.listen_start = 0

            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None
                self._set_state("inactive")

            return transcription
        
        except ValueError:
            logging.error("Error converting audio buffer to numpy array.")
            raise

        except faster_whisper.WhisperError as e:
            logging.error(f"Whisper transcription error: {e}")
            raise

        except Exception as e:
            logging.error(f"General transcription error: {e}")
            raise


    def start(self):
        """
        Starts recording audio directly without waiting for voice activity.
        """

        # Ensure there's a minimum interval between stopping and starting recording
        if time.time() - self.recording_stop_time < self.min_gap_between_recordings:
            logging.info("Attempted to start recording too soon after stopping.")
            return self
        
        logging.info("recording started")
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.frames = []
        self.is_recording = True        
        self.recording_start_time = time.time()
        self._set_spinner("recording")
        self._set_state("recording")
        if self.halo: self.halo._interval = 100

        if self.on_recording_start:
            self.on_recording_start()

        return self
    

    def stop(self):
        """
        Stops recording audio.
        """

        # Ensure there's a minimum interval between starting and stopping recording
        if time.time() - self.recording_start_time < self.min_length_of_recording:
            logging.info("Attempted to stop recording too soon after starting.")
            return self
                
        logging.info("recording stopped")                
        self.is_recording = False
        self.recording_stop_time = time.time()

        self._set_spinner("transcribing")
        self._set_state("transcribing")

        if self.on_recording_stop:
            self.on_recording_stop()

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
        noise levels. If upper is True, it calculates from the upper side,
        otherwise from the lower side.

        Args:
        - buffer (list): The buffer containing the history of noise levels.
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
        

    def _is_silero_speech(self, data):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with 16000 sample rate and 16 bits per sample)
        """

        audio_chunk = np.frombuffer(data, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / 32768.0  # Convert to float and normalize
        vad_prob = self.silero_vad_model(torch.from_numpy(audio_chunk), SAMPLE_RATE).item()
        return vad_prob > (1 - self.silero_sensitivity)


    def _is_webrtc_speech(self, data):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with 16000 sample rate and 16 bits per sample)
        """
        # Number of audio frames per millisecond
        frame_length = int(self.sample_rate * 0.01)  # for 10ms frame
        num_frames = int(len(data) / (2 * frame_length))

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = data[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, self.sample_rate):
                return True
        return False
    
    
    def _is_voice_active(self, data):
        """
        Determine if voice is active based on the provided data.

        Args:
            data: The audio data to be checked for voice activity.

        Returns:
            bool: True if voice is active, False otherwise.
        """
        # Define a constant for the time threshold
        TIME_THRESHOLD = 0.1
        
        # Check if enough time has passed to reset the Silero check time
        if time.time() - self.silero_check_time > TIME_THRESHOLD:
            self.silero_check_time = 0
        
        # First quick performing check for voice activity using WebRTC
        if self._is_webrtc_speech(data):
            
            # If silero check time not set
            if self.silero_check_time == 0:
                self.silero_check_time = time.time()
            
                # Perform a more intensive check using Silero
                if self._is_silero_speech(data):

                    return True  # Voice is active
            
        return False  # Voice is not active    


    def _set_state(self, new_state):
        """
        Update the current state of the recorder and execute corresponding state-change callbacks.

        Args:
            new_state (str): The new state to set. 

        """
        # Check if the state has actually changed
        if new_state == self.state:
            return
        
        # Store the current state for later comparison
        old_state = self.state
        
        # Update to the new state
        self.state = new_state

        # Execute callbacks based on transitioning FROM a particular state
        if old_state == "listening":
            if self.on_vad_detect_stop:
                self.on_vad_detect_stop()
        elif old_state == "wakeword":
            if self.on_wakeword_detection_end:
                self.on_wakeword_detection_end()

        # Execute callbacks based on transitioning TO a particular state
        if new_state == "listening":
            if self.on_vad_detect_start:
                self.on_vad_detect_start()
        elif new_state == "wakeword":
            if self.on_wakeword_detection_start:
                self.on_wakeword_detection_start()
        elif new_state == "transcribing":
            if self.on_transcription_start:
                self.on_transcription_start()


    def _set_spinner(self, text):
        """
        Update the spinner's text or create a new spinner with the provided text.

        Args:
            text (str): The text to be displayed alongside the spinner.
        """
        if self.spinner:
            # If the Halo spinner doesn't exist, create and start it
            if self.halo is None:
                self.halo = Halo(text=text)
                self.halo.start()
            # If the Halo spinner already exists, just update the text
            else:
                self.halo.text = text


    def _recording_worker(self):
        """
        The main worker method which constantly monitors the audio input for voice activity and accordingly starts/stops the recording.
        """
        
        was_recording = False
        delay_was_passed = False

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

            if not self.is_recording:
                # handle not recording state

                time_since_listen_start = time.time() - self.listen_start if self.listen_start else 0
                wake_word_activation_delay_passed = (time_since_listen_start > self.wake_word_activation_delay)

                # handle wake-word timeout callback
                if wake_word_activation_delay_passed and not delay_was_passed:
                    if self.wake_words and self.wake_word_activation_delay:
                        if self.on_wakeword_timeout:
                            self.on_wakeword_timeout()
                delay_was_passed = wake_word_activation_delay_passed

                # Set state and spinner text 
                if not self.recording_stop_time:
                    if self.wake_words and wake_word_activation_delay_passed and not self.wakeword_detected:
                        self._set_state("wakeword")
                        if self.spinner and self.halo:
                            self.halo.text = f"say {self.wake_words}"
                            self.halo._interval = 500
                    else:
                        if self.listen_start:
                            self._set_state("listening")
                        else:
                            self._set_state("inactive")
                        if self.spinner and self.halo:
                            self.halo.text = "speak now"
                            self.halo._interval = 200

                # Detect wake words if applicable
                if self.wake_words and wake_word_activation_delay_passed:
                    try:
                        pcm = struct.unpack_from("h" * self.buffer_size, data)
                        wakeword_index = self.porcupine.process(pcm)

                    except struct.error:
                        logging.error("Error unpacking audio data for wake word processing.")
                        continue
                    
                    except Exception as e:
                        logging.error(f"Wake word processing error: {e}")
                        continue
                    
                    # If a wake word is detected
                    if wakeword_index >= 0:

                        # Removing the wake word from the recording
                        samples_for_0_1_sec = int(self.sample_rate * 0.1)
                        start_index = max(0, len(self.audio_buffer) - samples_for_0_1_sec)
                        temp_samples = collections.deque(itertools.islice(self.audio_buffer, start_index, None))
                        self.audio_buffer.clear()
                        self.audio_buffer.extend(temp_samples)

                        self.wake_word_detect_time = time.time()
                        self.wakeword_detected = True
                        if self.on_wakeword_detected:
                            self.on_wakeword_detected()

                # Check for voice activity to trigger the start of recording
                if ((not self.wake_words or not wake_word_activation_delay_passed) and self.start_recording_on_voice_activity) or self.wakeword_detected:

                    if self._is_voice_active(data):
                        logging.info("voice activity detected")

                        self.start()

                        if self.is_recording:
                            self.start_recording_on_voice_activity = False

                            # Add the buffered audio to the recording frames
                            self.frames.extend(list(self.audio_buffer))

                        self.silero_vad_model.reset_states()

                self.speech_end_silence_start = 0

            else:
                # If we are currently recording

                # Stop the recording if silence is detected after speech
                if self.stop_recording_on_voice_deactivity:

                    if not self._is_webrtc_speech(data):

                        # Voice deactivity was detected, so we start measuring silence time before stopping recording
                        if self.speech_end_silence_start == 0:
                            self.speech_end_silence_start = time.time()
                        
                    else:
                        self.speech_end_silence_start = 0

                    # Wait for silence to stop recording after speech
                    if self.speech_end_silence_start and time.time() - self.speech_end_silence_start > self.post_speech_silence_duration:
                        logging.info("voice deactivity detected")
                        self.stop()

            if not self.is_recording and was_recording:
                # Reset after stopping recording to ensure clean state
                self.stop_recording_on_voice_deactivity = False

            if time.time() - self.silero_check_time > 0.1:
                self.silero_check_time = 0
            
            if self.wake_word_detect_time and time.time() - self.wake_word_detect_time > self.wake_word_timeout:
                self.wake_word_detect_time = 0
                if self.wakeword_detected and self.on_wakeword_timeout:
                    self.on_wakeword_timeout()
                self.wakeword_detected = False

            if self.is_recording:
                self.frames.append(data)

            self.audio_buffer.append(data)	

            was_recording = self.is_recording
            time.sleep(TIME_SLEEP)


    def __del__(self):
        """
        Destructor method ensures safe shutdown of the recorder when the instance is destroyed.
        """
        self.shutdown()