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
    Worker method that handles the audio recording process.

    This method runs in a separate process and is responsible for:
    - Setting up the audio input stream for recording at the highest possible sample rate.
    - Continuously reading audio data from the input stream, resampling if necessary,
    preprocessing the data, and placing complete chunks in a queue.
    - Handling errors during the recording process.
    - Gracefully terminating the recording process when a shutdown event is set.

    Args:
        audio_queue (queue.Queue): A queue where recorded audio data is placed.
        target_sample_rate (int): The desired sample rate for the output audio (for Silero VAD).
        buffer_size (int): The number of samples expected by the Silero VAD model.
        input_device_index (int): The index of the audio input device.
        shutdown_event (threading.Event): An event that, when set, signals this worker method to terminate.
        interrupt_stop_event (threading.Event): An event to signal keyboard interrupt.
        use_microphone (multiprocessing.Value): A shared value indicating whether to use the microphone.

    Raises:
        Exception: If there is an error while initializing the audio recording.
    """
    import pyaudio
    import numpy as np
    from scipy import signal

    if __name__ == '__main__':
        system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

    def get_highest_sample_rate(audio_interface, device_index):
        """Get the highest supported sample rate for the specified device."""
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
        nonlocal input_device_index

        def validate_device(device_index):
            """Validate that the device exists and is actually available for input."""
            try:
                device_info = audio_interface.get_device_info_by_index(device_index)
                logger.debug(f"Validating device index {device_index} with info: {device_info}")
                if not device_info.get('maxInputChannels', 0) > 0:
                    logger.debug("Device has no input channels, invalid for recording.")
                    return False

                # Try to actually read from the device
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

        """Initialize the audio stream with error handling."""
        while not shutdown_event.is_set():
            try:
                # First, get a list of all available input devices
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

                # If input_device_index is None or invalid, try to find a working device
                if input_device_index is None or input_device_index not in input_devices:
                    # First try the default device
                    try:
                        default_device = audio_interface.get_default_input_device_info()
                        logger.debug(f"Default device info: {default_device}")
                        if validate_device(default_device['index']):
                            input_device_index = default_device['index']
                            logger.debug(f"Default device {input_device_index} selected.")
                    except Exception:
                        # If default device fails, try other available input devices
                        logger.debug("Default device validation failed, checking other devices...")
                        for device_index in input_devices:
                            if validate_device(device_index):
                                input_device_index = device_index
                                logger.debug(f"Device {input_device_index} selected.")
                                break
                        else:
                            raise Exception("No working input devices found")

                # Validate the selected device one final time
                if not validate_device(input_device_index):
                    raise Exception("Selected device validation failed")

                # If we get here, we have a validated device
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
        """Preprocess audio chunk similar to feed_audio method."""
        if isinstance(chunk, np.ndarray):
            # Handle stereo to mono conversion if necessary
            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)

            # Resample to target_sample_rate if necessary
            if original_sample_rate != target_sample_rate:
                logger.debug(f"Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz.")
                num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                chunk = signal.resample(chunk, num_samples)

            chunk = chunk.astype(np.int16)
        else:
            # If chunk is bytes, convert to numpy array
            chunk = np.frombuffer(chunk, dtype=np.int16)

            # Resample if necessary
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

            # We'll try 16000 Hz first, then the highest rate we detect, then fallback if needed
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

            # If we reach here, none of the sample rates worked
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

                    # Check if the buffer has reached or exceeded the silero_buffer_size
                    while len(buffer) >= silero_buffer_size:
                        # Extract silero_buffer_size amount of data from the buffer
                        to_process = buffer[:silero_buffer_size]
                        buffer = buffer[silero_buffer_size:]

                        # Feed the extracted data to the audio_queue
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
                    # Attempt to reinitialize the stream
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
                # Attempt to reinitialize the stream
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
        # After recording stops, feed any remaining audio data
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
