from colorama import init, Fore, Style
import pyaudio
import logging
import time


class AudioInput:
    def __init__(self, input_device_index=None, debug_mode=False):
        self.input_device_index = input_device_index
        self.debug_mode = debug_mode
        self.audio_interface = None
        self.stream = None
        self.device_sample_rate = None

        # PyAudio constants
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16  
        self.CHANNELS = 1
        self.DESIRED_RATE = 16000

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
                    input_format=self.FORMAT,  # Changed to input_format
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
            self.device_sample_rate = self._get_best_sample_rate(actual_device_index, self.DESIRED_RATE)

            if self.debug_mode:
                print(f"Setting up audio on device {self.input_device_index} with sample rate {self.device_sample_rate}")

            try:
                self.stream = self.audio_interface.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.device_sample_rate,
                    input=True,
                    frames_per_buffer=self.CHUNK,
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

    def read_chunk(self):
        """Read a chunk of audio data"""
        return self.stream.read(self.CHUNK, exception_on_overflow=False)

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