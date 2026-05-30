"""Audio input worker for microphone capture."""

import logging
import signal as system_signal
import time
import traceback


logger = logging.getLogger("realtimestt")


def run_audio_data_worker(
    audio_queue,
    target_sample_rate,
    buffer_size,
    input_device_index,
    shutdown_event,
    interrupt_stop_event,
    use_microphone
):
    """
    Captures microphone audio and queues complete recorder chunks.

    The worker validates the selected input device, retries recoverable stream
    failures, resamples microphone audio, and exits when shutdown is requested.

    Args:
    - audio_queue: Queue receiving processed audio bytes.
    - target_sample_rate: Output sample rate expected by downstream VAD.
    - buffer_size: Number of samples expected by the Silero VAD model.
    - input_device_index: Optional input device index.
    - shutdown_event: Event that stops the worker.
    - interrupt_stop_event: Event set on keyboard interruption.
    - use_microphone: Shared flag controlling microphone reads.
    """
    import pyaudio
    import numpy as np
    from scipy import signal

    if __name__ == '__main__':
        system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

    def get_highest_sample_rate(audio_interface, device_index):
        """
        Returns the highest supported sample rate for an input device.
        """
        try:
            device_info = audio_interface.get_device_info_by_index(device_index)
            logger.debug(f"Retrieving highest sample rate for device index {device_index}: {device_info}")
            max_rate = int(device_info['defaultSampleRate'])

            if 'supportedSampleRates' in device_info:
                supported_rates = [int(rate) for rate in device_info['supportedSampleRates']]
                if supported_rates:
                    max_rate = max(supported_rates)

            logger.debug(f"Highest supported sample rate for device index {device_index} is {max_rate}")
            return max_rate
        except Exception as e:
            logger.warning(f"Failed to get highest sample rate: {e}")
            return 48000  # Fallback to a common high sample rate

    def initialize_audio_stream(audio_interface, sample_rate, chunk_size):
        """
        Initializes the audio stream with retry-friendly error handling.
        """
        nonlocal input_device_index

        def validate_device(device_index):
            """
            Checks whether an input device can be opened and read.
            """
            try:
                device_info = audio_interface.get_device_info_by_index(device_index)
                logger.debug(f"Validating device index {device_index} with info: {device_info}")
                if not device_info.get('maxInputChannels', 0) > 0:
                    logger.debug("Device has no input channels, invalid for recording.")
                    return False

                test_stream = audio_interface.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=target_sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size,
                    input_device_index=device_index,
                    start=False  # Don't start the stream yet
                )

                test_stream.start_stream()
                test_data = test_stream.read(chunk_size, exception_on_overflow=False)
                test_stream.stop_stream()
                test_stream.close()

                if len(test_data) == 0:
                    logger.debug("Device produced no data, invalid for recording.")
                    return False

                logger.debug(f"Device index {device_index} successfully validated.")
                return True

            except Exception as e:
                logger.debug(f"Device validation failed for index {device_index}: {e}")
                return False

        while not shutdown_event.is_set():
            try:
                input_devices = []
                device_count = audio_interface.get_device_count()
                logger.debug(f"Found {device_count} total audio devices on the system.")
                for i in range(device_count):
                    try:
                        device_info = audio_interface.get_device_info_by_index(i)
                        if device_info.get('maxInputChannels', 0) > 0:
                            input_devices.append(i)
                    except Exception as e:
                        logger.debug(f"Could not retrieve info for device index {i}: {e}")
                        continue

                logger.debug(f"Available input devices with input channels: {input_devices}")
                if not input_devices:
                    raise Exception("No input devices found")

                if input_device_index is None or input_device_index not in input_devices:
                    try:
                        default_device = audio_interface.get_default_input_device_info()
                        logger.debug(f"Default device info: {default_device}")
                        if validate_device(default_device['index']):
                            input_device_index = default_device['index']
                            logger.debug(f"Default device {input_device_index} selected.")
                    except Exception:
                        logger.debug("Default device validation failed, checking other devices...")
                        for device_index in input_devices:
                            if validate_device(device_index):
                                input_device_index = device_index
                                logger.debug(f"Device {input_device_index} selected.")
                                break
                        else:
                            raise Exception("No working input devices found")

                if not validate_device(input_device_index):
                    raise Exception("Selected device validation failed")

                logger.debug(f"Opening stream with device index {input_device_index}, "
                            f"sample_rate={sample_rate}, chunk_size={chunk_size}")
                stream = audio_interface.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size,
                    input_device_index=input_device_index,
                )

                logger.info(f"Microphone connected and validated (device index: {input_device_index}, "
                            f"sample rate: {sample_rate}, chunk size: {chunk_size})")
                return stream

            except Exception as e:
                logger.error(f"Microphone connection failed: {e}. Retrying...", exc_info=True)
                input_device_index = None
                time.sleep(3)  # Wait before retrying
                continue

    def preprocess_audio(chunk, original_sample_rate, target_sample_rate):
        """
        Converts one audio chunk to mono int16 bytes at the target rate.
        """
        if isinstance(chunk, np.ndarray):
            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)

            if original_sample_rate != target_sample_rate:
                logger.debug(f"Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz.")
                num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                chunk = signal.resample(chunk, num_samples)

            chunk = chunk.astype(np.int16)
        else:
            chunk = np.frombuffer(chunk, dtype=np.int16)

            if original_sample_rate != target_sample_rate:
                logger.debug(f"Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz.")
                num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                chunk = signal.resample(chunk, num_samples)
                chunk = chunk.astype(np.int16)

        return chunk.tobytes()

    audio_interface = None
    stream = None
    device_sample_rate = None
    chunk_size = 1024  # Increased chunk size for better performance

    def setup_audio():
        """
        Creates or recreates the microphone stream.
        """
        nonlocal audio_interface, stream, device_sample_rate, input_device_index
        try:
            if audio_interface is None:
                logger.debug("Creating PyAudio interface...")
                audio_interface = pyaudio.PyAudio()

            if input_device_index is None:
                try:
                    default_device = audio_interface.get_default_input_device_info()
                    input_device_index = default_device['index']
                    logger.debug(f"No device index supplied; using default device {input_device_index}")
                except OSError as e:
                    logger.debug(f"Default device retrieval failed: {e}")
                    input_device_index = None

            # Prefer the VAD-native rate, then fall back to device-supported rates.
            sample_rates_to_try = [16000]
            if input_device_index is not None:
                highest_rate = get_highest_sample_rate(audio_interface, input_device_index)
                if highest_rate != 16000:
                    sample_rates_to_try.append(highest_rate)
            else:
                sample_rates_to_try.append(48000)

            logger.debug(f"Sample rates to try for device {input_device_index}: {sample_rates_to_try}")

            for rate in sample_rates_to_try:
                try:
                    device_sample_rate = rate
                    logger.debug(f"Attempting to initialize audio stream at {device_sample_rate} Hz.")
                    stream = initialize_audio_stream(audio_interface, device_sample_rate, chunk_size)
                    if stream is not None:
                        logger.debug(
                            f"Audio recording initialized successfully at {device_sample_rate} Hz, "
                            f"reading {chunk_size} frames at a time"
                        )
                        return True
                except Exception as e:
                    logger.warning(f"Failed to initialize audio stream at {device_sample_rate} Hz: {e}")
                    continue

            raise Exception("Failed to initialize audio stream with all sample rates.")

        except Exception as e:
            logger.exception(f"Error initializing pyaudio audio recording: {e}")
            if audio_interface:
                audio_interface.terminate()
            return False

    logger.debug(f"Starting audio data worker with target_sample_rate={target_sample_rate}, "
                f"buffer_size={buffer_size}, input_device_index={input_device_index}")

    if not setup_audio():
        raise Exception("Failed to set up audio recording.")

    buffer = bytearray()
    silero_buffer_size = 2 * buffer_size  # Silero complains if too short

    time_since_last_buffer_message = 0

    try:
        while not shutdown_event.is_set():
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)

                if use_microphone.value:
                    processed_data = preprocess_audio(data, device_sample_rate, target_sample_rate)
                    buffer += processed_data

                    while len(buffer) >= silero_buffer_size:
                        to_process = buffer[:silero_buffer_size]
                        buffer = buffer[silero_buffer_size:]

                        if time_since_last_buffer_message:
                            time_passed = time.time() - time_since_last_buffer_message
                            if time_passed > 1:
                                logger.debug("_audio_data_worker writing audio data into queue.")
                                time_since_last_buffer_message = time.time()
                        else:
                            time_since_last_buffer_message = time.time()

                        audio_queue.put(to_process)

            except OSError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    logger.warning("Input overflowed. Frame dropped.")
                else:
                    logger.error(f"OSError during recording: {e}", exc_info=True)
                    logger.error("Attempting to reinitialize the audio stream...")

                    try:
                        if stream:
                            stream.stop_stream()
                            stream.close()
                    except Exception:
                        pass

                    time.sleep(1)
                    if not setup_audio():
                        logger.error("Failed to reinitialize audio stream. Exiting.")
                        break
                    else:
                        logger.error("Audio stream reinitialized successfully.")
                continue

            except Exception as e:
                logger.error(f"Unknown error during recording: {e}")
                tb_str = traceback.format_exc()
                logger.error(f"Traceback: {tb_str}")
                logger.error(f"Error: {e}")
                logger.info("Attempting to reinitialize the audio stream...")
                try:
                    if stream:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass

                time.sleep(1)
                if not setup_audio():
                    logger.error("Failed to reinitialize audio stream. Exiting.")
                    break
                else:
                    logger.info("Audio stream reinitialized successfully.")
                continue

    except KeyboardInterrupt:
        interrupt_stop_event.set()
        logger.debug("Audio data worker process finished due to KeyboardInterrupt")
    finally:
        # Preserve partial buffered audio before closing the stream.
        if buffer:
            audio_queue.put(bytes(buffer))

        try:
            if stream:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        if audio_interface:
            audio_interface.terminate()
