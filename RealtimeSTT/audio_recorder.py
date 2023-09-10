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
import os
import re
import collections
import halo
import traceback

INIT_MODEL_TRANSCRIPTION = "tiny"
INIT_MODEL_TRANSCRIPTION_REALTIME = "tiny"
INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0

TIME_SLEEP = 0.02
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0

class AudioToTextRecorder:
    """
    A class responsible for capturing audio from the microphone, detecting voice activity, and then transcribing the captured audio using the `faster_whisper` model.
    """
    
    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
                 language: str = "",
                 on_recording_start = None,
                 on_recording_stop = None,
                 on_transcription_start = None,
                 ensure_sentence_starting_uppercase = True,
                 ensure_sentence_ends_with_period = True,
                 spinner = True,
                 level=logging.WARNING,

                 # Realtime transcription parameters
                 enable_realtime_transcription = False,
                 realtime_model_type = INIT_MODEL_TRANSCRIPTION_REALTIME,
                 realtime_processing_pause = INIT_REALTIME_PROCESSING_PAUSE,
                 on_realtime_transcription_update = None,
                 on_realtime_transcription_stabilized = None,

                 # Voice activation parameters
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = INIT_POST_SPEECH_SILENCE_DURATION,
                 min_length_of_recording: float = INIT_MIN_LENGTH_OF_RECORDING,
                 min_gap_between_recordings: float = INIT_MIN_GAP_BETWEEN_RECORDINGS,
                 pre_recording_buffer_duration: float = INIT_PRE_RECORDING_BUFFER_DURATION,
                 on_vad_detect_start = None,
                 on_vad_detect_stop = None,

                 # Wake word parameters
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                 wake_word_activation_delay: float = INIT_WAKE_WORD_ACTIVATION_DELAY,
                 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
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
        - ensure_sentence_starting_uppercase (bool, default=True): Ensures that every sentence detected by the algorithm starts with an uppercase letter.
        - ensure_sentence_ends_with_period (bool, default=True): Ensures that every sentence that doesn't end with punctuation such as "?", "!" ends with a period
        - spinner (bool, default=True): Show spinner animation with current state.
        - level (int, default=logging.WARNING): Logging level.
        - enable_realtime_transcription (bool, default=False): Enables or disables real-time transcription of audio. When set to True, the audio will be transcribed continuously as it is being recorded.
        - realtime_model_type (str, default="tiny"): Specifies the machine learning model to be used for real-time transcription. Valid options include 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
        - realtime_processing_pause (float, default=0.1): Specifies the time interval in seconds after a chunk of audio gets transcribed. Lower values will result in more "real-time" (frequent) transcription updates but may increase computational load.
        - on_realtime_transcription_update = A callback function that is triggered whenever there's an update in the real-time transcription. The function is called with the newly transcribed text as its argument.
        - on_realtime_transcription_stabilized = A callback function that is triggered when the transcribed text stabilizes in quality. The stabilized text is generally more accurate but may arrive with a slight delay compared to the regular real-time updates.
        - silero_sensitivity (float, default=SILERO_SENSITIVITY): Sensitivity for the Silero Voice Activity Detection model ranging from 0 (least sensitive) to 1 (most sensitive). Default is 0.5.
        - webrtc_sensitivity (int, default=WEBRTC_SENSITIVITY): Sensitivity for the WebRTC Voice Activity Detection engine ranging from 0 (least aggressive / most sensitive) to 3 (most aggressive, least sensitive). Default is 3.
        - post_speech_silence_duration (float, default=0.2): Duration in seconds of silence that must follow speech before the recording is considered to be completed. This ensures that any brief pauses during speech don't prematurely end the recording.
        - min_gap_between_recordings (float, default=1.0): Specifies the minimum time interval in seconds that should exist between the end of one recording session and the beginning of another to prevent rapid consecutive recordings.
        - min_length_of_recording (float, default=1.0): Specifies the minimum duration in seconds that a recording session should last to ensure meaningful audio capture, preventing excessively short or fragmented recordings.
        - pre_recording_buffer_duration (float, default=0.2): Duration in seconds for the audio buffer to maintain pre-roll audio (compensates speech activity detection latency)
        - on_vad_detect_start (callable, default=None): Callback function to be called when the system listens for voice activity.
        - on_vad_detect_stop (callable, default=None): Callback function to be called when the system stops listening for voice activity.
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
        self.ensure_sentence_starting_uppercase = ensure_sentence_starting_uppercase
        self.ensure_sentence_ends_with_period = ensure_sentence_ends_with_period
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
        self.enable_realtime_transcription = enable_realtime_transcription
        self.realtime_model_type = realtime_model_type
        self.realtime_processing_pause = realtime_processing_pause
        self.on_realtime_transcription_update = on_realtime_transcription_update
        self.on_realtime_transcription_stabilized = on_realtime_transcription_stabilized
    
        self.level = level
        self.buffer_size = BUFFER_SIZE
        self.sample_rate = SAMPLE_RATE
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0 
        self.silero_working = False
        self.speech_end_silence_start = 0 
        self.silero_sensitivity = silero_sensitivity
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False

        # Initialize the logging configuration with the specified level
        logging.basicConfig(format='RealTimeSTT: %(name)s - %(levelname)s - %(message)s', level=level) # filename='audio_recorder.log'


        # Initialize the transcription model
        try:
            self.model = faster_whisper.WhisperModel(model_size_or_path=model, device='cuda' if torch.cuda.is_available() else 'cpu')

            if self.enable_realtime_transcription:
                self.realtime_model_type = faster_whisper.WhisperModel(model_size_or_path=self.realtime_model_type, device='cuda' if torch.cuda.is_available() else 'cpu')


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
            logging.info(f"Initializing WebRTC voice with Sensitivity {webrtc_sensitivity}")
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

        # Start the realtime transcription worker thread
        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()

        logging.debug('Constructor finished')


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
            audio_array = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
            self.frames = []

            # perform transcription
            transcription = " ".join(seg.text for seg in self.model.transcribe(audio_array, language=self.language if self.language else None)[0]).strip()

            self.recording_stop_time = 0
            self.listen_start = 0

            self._set_state("inactive")

            return self._preprocess_output(transcription)
        
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
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.frames = []
        self.is_recording = True        
        self.recording_start_time = time.time()
        self._set_state("recording")
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False

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
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0 

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


    def _is_silero_speech(self, data):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with 16000 sample rate and 16 bits per sample)
        """

        logging.debug('Performing silero speech activity check')
        self.silero_working = True
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE  # Convert to float and normalize
        # print ("S", end="", flush=True)             
        vad_prob = self.silero_vad_model(torch.from_numpy(audio_chunk), SAMPLE_RATE).item()
        is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
        if is_silero_speech_active:
            # print ("+", end="", flush=True)
            self.is_silero_speech_active = True
        # else:
            # print ("-", end="", flush=True)
        self.silero_working = False
        return is_silero_speech_active


    def _is_webrtc_speech(self, data, all_frames_must_be_true=False):
        """
        Returns true if speech is detected in the provided audio data

        Args:
            data (bytes): raw bytes of audio data (1024 raw bytes with 16000 sample rate and 16 bits per sample)
        """
        # Number of audio frames per millisecond
        frame_length = int(self.sample_rate * 0.01)  # for 10ms frame
        num_frames = int(len(data) / (2 * frame_length))
        speech_frames = 0        

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = data[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, self.sample_rate):
                speech_frames += 1
                if not all_frames_must_be_true:
                    return True
        if all_frames_must_be_true:
            return speech_frames == num_frames
        else:
            return False
    
        
    def _check_voice_activity(self, data):
        """
        Initiate check if voice is active based on the provided data.

        Args:
            data: The audio data to be checked for voice activity.
        """
        # # Define a constant for the time threshold
        # TIME_THRESHOLD = 0.1
        
        # # Check if enough time has passed to reset the Silero check time
        # if time.time() - self.silero_check_time > TIME_THRESHOLD:
        #     self.silero_check_time = 0

        self.is_webrtc_speech_active = self._is_webrtc_speech(data)
        
        # First quick performing check for voice activity using WebRTC
        if self.is_webrtc_speech_active:
            
            if not self.silero_working:
                self.silero_working = True

                # Run the intensive check in a separate thread
                threading.Thread(target=self._is_silero_speech, args=(data,)).start()

            # # If silero check time not set
            # if self.silero_check_time == 0:                
            #     self.silero_check_time = time.time()

    
    def _is_voice_active(self):
        """
        Determine if voice is active.

        Returns:
            bool: True if voice is active, False otherwise.
        """
        #print("C", end="", flush=True)
        # if not self.is_webrtc_speech_active and not self.is_silero_speech_active:
        #     print (".", end="", flush=True)
        # elif self.is_webrtc_speech_active and not self.is_silero_speech_active:
        #     print ("W", end="", flush=True)
        # elif not self.is_webrtc_speech_active and self.is_silero_speech_active:
        #     print ("S", end="", flush=True)
        # elif self.is_webrtc_speech_active and self.is_silero_speech_active:
        #     print ("#", end="", flush=True)

        return self.is_webrtc_speech_active and self.is_silero_speech_active


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
            self._set_spinner("speak now")
            if self.spinner:
                self.halo._interval = 250
        elif new_state == "wakeword":
            if self.on_wakeword_detection_start:
                self.on_wakeword_detection_start()
            self._set_spinner(f"say {self.wake_words}")
            if self.spinner:
                self.halo._interval = 500
        elif new_state == "transcribing":
            if self.on_transcription_start:
                self.on_transcription_start()
            self._set_spinner("transcribing")
            if self.spinner:
                self.halo._interval = 50
        elif new_state == "recording":
            self._set_spinner("recording")
            if self.spinner:
                self.halo._interval = 100
        elif new_state == "inactive":
            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None


    def _set_spinner(self, text):
        """
        Update the spinner's text or create a new spinner with the provided text.

        Args:
            text (str): The text to be displayed alongside the spinner.
        """
        if self.spinner:
            # If the Halo spinner doesn't exist, create and start it
            if self.halo is None:
                self.halo = halo.Halo(text=text)
                self.halo.start()
            # If the Halo spinner already exists, just update the text
            else:
                self.halo.text = text


    def _recording_worker(self):
        """
        The main worker method which constantly monitors the audio input for voice activity and accordingly starts/stops the recording.
        """

        logging.debug('Starting recording worker')
        try:
            was_recording = False
            delay_was_passed = False

            # Continuously monitor audio for voice activity
            while self.is_running:

                try:
                    data = self.stream.read(self.buffer_size)

                except OSError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        logging.warning("Input overflowed. Frame dropped.")
                    else:
                        logging.error(f"Error during recording: {e}")
                    tb_str = traceback.format_exc()
                    print (f"Traceback: {tb_str}")
                    print (f"Error: {e}")

                    continue

                except Exception as e:
                    logging.error(f"Error during recording: {e}")
                    time.sleep(1)
                    tb_str = traceback.format_exc()
                    print (f"Traceback: {tb_str}")
                    print (f"Error: {e}")
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
                        else:
                            if self.listen_start:
                                self._set_state("listening")
                            else:
                                self._set_state("inactive")

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

                        if self._is_voice_active():
                            logging.info("voice activity detected")

                            self.start()

                            if self.is_recording:
                                self.start_recording_on_voice_activity = False

                                # Add the buffered audio to the recording frames
                                self.frames.extend(list(self.audio_buffer))
                                self.audio_buffer.clear()

                            self.silero_vad_model.reset_states()
                        else:
                            data_copy = data[:]
                            self._check_voice_activity(data_copy)

                    self.speech_end_silence_start = 0

                else:
                    # If we are currently recording

                    # Stop the recording if silence is detected after speech
                    if self.stop_recording_on_voice_deactivity:

                        if not self._is_webrtc_speech(data, True):

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

                if not self.is_recording or self.speech_end_silence_start:
                    self.audio_buffer.append(data)	

                was_recording = self.is_recording
                time.sleep(TIME_SLEEP)

        except Exception as e:
            logging.error(f"Unhandled exeption in _recording_worker: {e}")
            raise


    def _preprocess_output(self, text, preview=False):
        """
        Preprocesses the output text by removing any leading or trailing whitespace,
        converting all whitespace sequences to a single space character, and capitalizing
        the first character of the text.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        text = re.sub(r'\s+', ' ', text.strip())

        if self.ensure_sentence_starting_uppercase:
            if text:
                text = text[0].upper() + text[1:]

        # Ensure the text ends with a proper punctuation if it ends with an alphanumeric character
        if not preview:
            if self.ensure_sentence_ends_with_period:
                if text and text[-1].isalnum():
                    text += '.'

        return text


    def find_tail_match_in_text(self, text1, text2, length_of_match=10):
        """
        Find the position where the last 'n' characters of text1 match with a substring in text2.
        
        This method takes two texts, extracts the last 'n' characters from text1 (where 'n' is determined
        by the variable 'length_of_match'), and searches for an occurrence of this substring in text2,
        starting from the end of text2 and moving towards the beginning.

        Parameters:
        - text1 (str): The text containing the substring that we want to find in text2.
        - text2 (str): The text in which we want to find the matching substring.
        - length_of_match(int): The length of the matching string that we are looking for

        Returns:
        int: The position (0-based index) in text2 where the matching substring starts.
            If no match is found or either of the texts is too short, returns -1.
        """
        
        # Check if either of the texts is too short
        if len(text1) < length_of_match or len(text2) < length_of_match:
            return -1
        
        # The end portion of the first text that we want to compare
        target_substring = text1[-length_of_match:]
        
        # Loop through text2 from right to left
        for i in range(len(text2) - length_of_match + 1):
            # Extract the substring from text2 to compare with the target_substring
            current_substring = text2[len(text2) - i - length_of_match:len(text2) - i]
            
            # Compare the current_substring with the target_substring
            if current_substring == target_substring:
                return len(text2) - i  # Position in text2 where the match starts
        
        return -1


    def _realtime_worker(self):
        """
        Performs real-time transcription if the feature is enabled.

        The method is responsible transcribing recorded audio frames in real-time
         based on the specified resolution interval.
        The transcribed text is stored in `self.realtime_transcription_text` and a callback
        function is invoked with this text if specified.
        """
        try:

            logging.debug('Starting realtime worker')

            # Return immediately if real-time transcription is not enabled
            if not self.enable_realtime_transcription:
                return
                
            # Continue running as long as the main process is active
            while self.is_running:

                # Check if the recording is active
                if self.is_recording:
                    
                    # Sleep for the duration of the transcription resolution
                    time.sleep(self.realtime_processing_pause)
                    
                    # Convert the buffer frames to a NumPy array
                    audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                    
                    # Normalize the array to a [-1, 1] range
                    audio_array = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

                    # Perform transcription and assemble the text
                    segments = self.realtime_model_type.transcribe(
                        audio_array,
                        language=self.language if self.language else None
                    )

                    # double check recording state because it could have changed mid-transcription
                    if self.is_recording and time.time() - self.recording_start_time > 0.5:

                        logging.debug('Starting realtime transcription')
                        self.realtime_transcription_text = " ".join(seg.text for seg in segments[0]).strip()

                        self.text_storage.append(self.realtime_transcription_text)

                        # Take the last two texts in storage, if they exist
                        if len(self.text_storage) >= 2:
                            last_two_texts = self.text_storage[-2:]
                            
                            # Find the longest common prefix between the two texts
                            prefix = os.path.commonprefix([last_two_texts[0], last_two_texts[1]])

                            # This prefix is the text that was transcripted two times in the same way
                            # Store as "safely detected text" 
                            if len(prefix) >= len(self.realtime_stabilized_safetext):
                                # Only store when longer than the previous as additional security 
                                self.realtime_stabilized_safetext = prefix

                        # Find parts of the stabilized text in the freshly transscripted text
                        matching_position = self.find_tail_match_in_text(self.realtime_stabilized_safetext, self.realtime_transcription_text)
                        if matching_position < 0:
                            if self.realtime_stabilized_safetext:
                                if self.on_realtime_transcription_stabilized:
                                    self.on_realtime_transcription_stabilized(self._preprocess_output(self.realtime_stabilized_safetext, True))
                            else:
                                if self.on_realtime_transcription_stabilized:
                                    self.on_realtime_transcription_stabilized(self._preprocess_output(self.realtime_transcription_text, True))
                        else:
                            # We found parts of the stabilized text in the transcripted text
                            # We now take the stabilized text and add only the freshly transcripted part to it
                            output_text = self.realtime_stabilized_safetext + self.realtime_transcription_text[matching_position:]

                            # This yields us the "left" text part as stabilized AND at the same time delivers fresh detected parts 
                            # on the first run without the need for two transcriptions
                            if self.on_realtime_transcription_stabilized:
                                self.on_realtime_transcription_stabilized(self._preprocess_output(output_text, True))

                        # Invoke the callback with the transcribed text
                            if self.on_realtime_transcription_update:
                                self.on_realtime_transcription_update(self._preprocess_output(self.realtime_transcription_text, True))

                # If not recording, sleep briefly before checking again
                else:
                    time.sleep(TIME_SLEEP)

        except Exception as e:
            logging.error(f"Unhandled exeption in _realtime_worker: {e}")
            raise

    def __del__(self):
        """
        Destructor method ensures safe shutdown of the recorder when the instance is destroyed.
        """
        self.shutdown()