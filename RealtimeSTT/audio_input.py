from colorama import init, Fore, Style
from scipy.signal import butter, filtfilt, resample_poly
import pyaudio
import logging

DESIRED_RATE = 16000
CHUNK_SIZE = 1024
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1

class AudioInput:
    def __init__(
            self,
            input_device_index: int = None,
            debug_mode: bool = False,
            target_samplerate: int = DESIRED_RATE,
            chunk_size: int = CHUNK_SIZE,
            audio_format: int = AUDIO_FORMAT,
            channels: int = CHANNELS,
            resample_to_target: bool = True,
        ):

        self.input_device_index = input_device_index
        self.debug_mode = debug_mode
        self.audio_interface = None
        self.stream = None
        self.device_sample_rate = None
        self.target_samplerate = target_samplerate
        self.chunk_size = chunk_size
        self.audio_format = audio_format
        self.channels = channels
        self.resample_to_target = resample_to_target

    def get_supported_sample_rates(self, device_index):
        """Test which standard sample rates are supported by the specified device."""
        standard_rates = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        supported_rates = []

        device_info = self.audio_interface.get_device_info_by_index(device_index)
        max_channels = device_info.get('maxInputChannels')  # Changed from maxOutputChannels

        for rate in standard_rates:
            try:
                if self.audio_interface.is_format_supported(
                    rate,
                    input_device=device_index,  # Changed to input_device
                    input_channels=max_channels,  # Changed to input_channels
                    input_format=self.audio_format,  # Changed to input_format
                ):
                    supported_rates.append(rate)
            except:
                continue
        return supported_rates

    def _get_best_sample_rate(self, actual_device_index, desired_rate):
        """Determines the best available sample rate for the device."""
        try:
            device_info = self.audio_interface.get_device_info_by_index(actual_device_index)
            supported_rates = self.get_supported_sample_rates(actual_device_index)

            if desired_rate in supported_rates:
                return desired_rate

            return max(supported_rates)

            # lower_rates = [r for r in supported_rates if r <= desired_rate]
            # if lower_rates:
            #     return max(lower_rates)

            # higher_rates = [r for r in supported_rates if r > desired_rate]
            # if higher_rates:
            #     return min(higher_rates)

            return int(device_info.get('defaultSampleRate', 44100))

        except Exception as e:
            logging.warning(f"Error determining sample rate: {e}")
            return 44100  # Safe fallback

    def list_devices(self):
        """List all available audio input devices with supported sample rates."""
        try:
            init()  # Initialize colorama
            self.audio_interface = pyaudio.PyAudio()
            device_count = self.audio_interface.get_device_count()

            print(f"Available audio input devices:")
            #print(f"{Fore.LIGHTBLUE_EX}Available audio input devices:{Style.RESET_ALL}")
            for i in range(device_count):
                device_info = self.audio_interface.get_device_info_by_index(i)
                device_name = device_info.get('name')
                max_input_channels = device_info.get('maxInputChannels', 0)

                if max_input_channels > 0:  # Only consider devices with input capabilities
                    supported_rates = self.get_supported_sample_rates(i)
                    print(f"{Fore.LIGHTGREEN_EX}Device {Style.RESET_ALL}{i}{Fore.LIGHTGREEN_EX}: {device_name}{Style.RESET_ALL}")
                    
                    # Format each rate in cyan
                    if supported_rates:
                        rates_formatted = ", ".join([f"{Fore.CYAN}{rate}{Style.RESET_ALL}" for rate in supported_rates])
                        print(f"  {Fore.YELLOW}Supported sample rates: {rates_formatted}{Style.RESET_ALL}")
                    else:
                        print(f"  {Fore.YELLOW}Supported sample rates: None{Style.RESET_ALL}")

        except Exception as e:
            print(f"Error listing devices: {e}")
        finally:
            if self.audio_interface:
                self.audio_interface.terminate()

    def setup(self):
        """Initialize audio interface and open stream"""
        try:
            self.audio_interface = pyaudio.PyAudio()

            if self.debug_mode:
                print(f"Input device index: {self.input_device_index}")
            actual_device_index = (self.input_device_index if self.input_device_index is not None 
                                else self.audio_interface.get_default_input_device_info()['index'])
            
            if self.debug_mode:
                print(f"Actual selected device index: {actual_device_index}")
            self.input_device_index = actual_device_index
            self.device_sample_rate = self._get_best_sample_rate(actual_device_index, self.target_samplerate)

            if self.debug_mode:
                print(f"Setting up audio on device {self.input_device_index} with sample rate {self.device_sample_rate}")

            try:
                self.stream = self.audio_interface.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.device_sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    input_device_index=self.input_device_index,
                )
                if self.debug_mode:
                    print(f"Audio recording initialized successfully at {self.device_sample_rate} Hz")
                return True
            except Exception as e:
                print(f"Failed to initialize audio stream at {self.device_sample_rate} Hz: {e}")
                return False

        except Exception as e:
            print(f"Error initializing audio recording: {e}")
            if self.audio_interface:
                self.audio_interface.terminate()
            return False

    def lowpass_filter(self, signal, cutoff_freq, sample_rate):
        """
        Apply a low-pass Butterworth filter to prevent aliasing in the signal.

        Args:
            signal (np.ndarray): Input audio signal to filter
            cutoff_freq (float): Cutoff frequency in Hz
            sample_rate (float): Sampling rate of the input signal in Hz

        Returns:
            np.ndarray: Filtered audio signal

        Notes:
            - Uses a 5th order Butterworth filter
            - Applies zero-phase filtering using filtfilt
        """
        # Calculate the Nyquist frequency (half the sample rate)
        nyquist_rate = sample_rate / 2.0

        # Normalize cutoff frequency to Nyquist rate (required by butter())
        normal_cutoff = cutoff_freq / nyquist_rate

        # Design the Butterworth filter
        b, a = butter(5, normal_cutoff, btype='low', analog=False)

        # Apply zero-phase filtering (forward and backward)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def resample_audio(self, pcm_data, target_sample_rate, original_sample_rate):
        """
        Filter and resample audio data to a target sample rate.

        Args:
            pcm_data (np.ndarray): Input audio data
            target_sample_rate (int): Desired output sample rate in Hz
            original_sample_rate (int): Original sample rate of input in Hz

        Returns:
            np.ndarray: Resampled audio data

        Notes:
            - Applies anti-aliasing filter before resampling
            - Uses polyphase filtering for high-quality resampling
        """
        if target_sample_rate < original_sample_rate:
            # Downsampling with low-pass filter
            pcm_filtered = self.lowpass_filter(pcm_data, target_sample_rate / 2, original_sample_rate)
            resampled = resample_poly(pcm_filtered, target_sample_rate, original_sample_rate)
        else:
            # Upsampling without low-pass filter
            resampled = resample_poly(pcm_data, target_sample_rate, original_sample_rate)
        return resampled

    def read_chunk(self):
        """Read a chunk of audio data"""
        return self.stream.read(self.chunk_size, exception_on_overflow=False)

    def cleanup(self):
        """Clean up audio resources"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            if self.audio_interface:
                self.audio_interface.terminate()
                self.audio_interface = None
        except Exception as e:
            print(f"Error cleaning up audio resources: {e}")
