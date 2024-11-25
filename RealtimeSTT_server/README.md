# RealtimeSTT Server and Client

This directory contains the server and client implementations for the RealtimeSTT library, providing real-time speech-to-text transcription with WebSocket interfaces. The server allows clients to connect via WebSocket to send audio data and receive real-time transcription updates. The client handles communication with the server, allowing audio recording, parameter management, and control commands.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Server Usage](#server-usage)
  - [Starting the Server](#starting-the-server)
  - [Server Parameters](#server-parameters)
- [Client Usage](#client-usage)
  - [Starting the Client](#starting-the-client)
  - [Client Parameters](#client-parameters)
- [WebSocket Interface](#websocket-interface)
- [Examples](#examples)
  - [Starting the Server and Client](#starting-the-server-and-client)
  - [Setting Parameters](#setting-parameters)
  - [Retrieving Parameters](#retrieving-parameters)
  - [Calling Server Methods](#calling-server-methods)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Real-Time Transcription**: Provides real-time speech-to-text transcription using pre-configured or user-defined STT models.
- **WebSocket Communication**: Makes use of WebSocket connections for control commands and data handling.
- **Flexible Recording Options**: Supports configurable pauses for sentence detection and various voice activity detection (VAD) methods.
- **VAD Support**: Includes support for Silero and WebRTC VAD for robust voice activity detection.
- **Wake Word Detection**: Capable of detecting wake words to initiate transcription.
- **Configurable Parameters**: Allows fine-tuning of recording and transcription settings via command-line arguments or control commands.

## Installation

Ensure you have Python 3.8 or higher installed. Install the required packages using:

```bash
pip install git+https://github.com/KoljaB/RealtimeSTT.git@dev
```

## Server Usage

### Starting the Server

Start the server using the command-line interface:

```bash
stt-server [OPTIONS]
```

The server will initialize and begin listening for WebSocket connections on the specified control and data ports.

### Server Parameters

You can configure the server using the following command-line arguments:

### Available Parameters:

#### `-m`, `--model`

- **Type**: `str`
- **Default**: `'large-v2'`
- **Description**: Path to the Speech-to-Text (STT) model or specify a model size. Options include: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, or any HuggingFace CTranslate2 STT model such as `deepdml/faster-whisper-large-v3-turbo-ct2`.

#### `-r`, `--rt-model`, `--realtime_model_type`

- **Type**: `str`
- **Default**: `'tiny.en'`
- **Description**: Model size for real-time transcription. Options are the same as for `--model`. This is used only if real-time transcription is enabled (`--enable_realtime_transcription`).

#### `-l`, `--lang`, `--language`

- **Type**: `str`
- **Default**: `'en'`
- **Description**: Language code for the STT model to transcribe in a specific language. Leave this empty for auto-detection based on input audio. Default is `'en'`. [List of supported language codes](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L11-L110).

#### `-i`, `--input-device`, `--input_device_index`

- **Type**: `int`
- **Default**: `1`
- **Description**: Index of the audio input device to use. Use this option to specify a particular microphone or audio input device based on your system.

#### `-c`, `--control`, `--control_port`

- **Type**: `int`
- **Default**: `8011`
- **Description**: The port number used for the control WebSocket connection. Control connections are used to send and receive commands to the server.

#### `-d`, `--data`, `--data_port`

- **Type**: `int`
- **Default**: `8012`
- **Description**: The port number used for the data WebSocket connection. Data connections are used to send audio data and receive transcription updates in real time.

#### `-w`, `--wake_words`

- **Type**: `str`
- **Default**: `""` (empty string)
- **Description**: Specify the wake word(s) that will trigger the server to start listening. For example, setting this to `"Jarvis"` will make the system start transcribing when it detects the wake word `"Jarvis"`.

#### `-D`, `--debug`

- **Action**: `store_true`
- **Description**: Enable debug logging for detailed server operations.

#### `-W`, `--write`

- **Metavar**: `FILE`
- **Description**: Save received audio to a WAV file.

#### `--silero_sensitivity`

- **Type**: `float`
- **Default**: `0.05`
- **Description**: Sensitivity level for Silero Voice Activity Detection (VAD), with a range from `0` to `1`. Lower values make the model less sensitive, useful for noisy environments.

#### `--silero_use_onnx`

- **Action**: `store_true`
- **Default**: `False`
- **Description**: Enable the ONNX version of the Silero model for faster performance with lower resource usage.

#### `--webrtc_sensitivity`

- **Type**: `int`
- **Default**: `3`
- **Description**: Sensitivity level for WebRTC Voice Activity Detection (VAD), with a range from `0` to `3`. Higher values make the model less sensitive, useful for cleaner environments.

#### `--min_length_of_recording`

- **Type**: `float`
- **Default**: `1.1`
- **Description**: Minimum duration of valid recordings in seconds. This prevents very short recordings from being processed, which could be caused by noise or accidental sounds.

#### `--min_gap_between_recordings`

- **Type**: `float`
- **Default**: `0`
- **Description**: Minimum time (in seconds) between consecutive recordings. Setting this helps avoid overlapping recordings when there's a brief silence between them.

#### `--enable_realtime_transcription`

- **Action**: `store_true`
- **Default**: `True`
- **Description**: Enable continuous real-time transcription of audio as it is received. When enabled, transcriptions are sent in near real-time.

#### `--realtime_processing_pause`

- **Type**: `float`
- **Default**: `0.02`
- **Description**: Time interval (in seconds) between processing audio chunks for real-time transcription. Lower values increase responsiveness but may put more load on the CPU.

#### `--silero_deactivity_detection`

- **Action**: `store_true`
- **Default**: `True`
- **Description**: Use the Silero model for end-of-speech detection. This option can provide more robust silence detection in noisy environments, though it consumes more GPU resources.

#### `--early_transcription_on_silence`

- **Type**: `float`
- **Default**: `0.2`
- **Description**: Start transcription after the specified seconds of silence. This is useful when you want to trigger transcription mid-speech when there is a brief pause. Should be lower than `post_speech_silence_duration`. Set to `0` to disable.

#### `--beam_size`

- **Type**: `int`
- **Default**: `5`
- **Description**: Beam size for the main transcription model. Larger values may improve transcription accuracy but increase the processing time.

#### `--beam_size_realtime`

- **Type**: `int`
- **Default**: `3`
- **Description**: Beam size for the real-time transcription model. A smaller beam size allows for faster real-time processing but may reduce accuracy.

#### `--initial_prompt`

- **Type**: `str`
- **Default**:

  ```
  End incomplete sentences with ellipses. Examples: 
  Complete: The sky is blue. 
  Incomplete: When the sky... 
  Complete: She walked home. 
  Incomplete: Because he...
  ```

- **Description**: Initial prompt that guides the transcription model to produce transcriptions in a particular style or format. The default provides instructions for handling sentence completions and ellipsis usage.

#### `--end_of_sentence_detection_pause`

- **Type**: `float`
- **Default**: `0.45`
- **Description**: The duration of silence (in seconds) that the model should interpret as the end of a sentence. This helps the system detect when to finalize the transcription of a sentence.

#### `--unknown_sentence_detection_pause`

- **Type**: `float`
- **Default**: `0.7`
- **Description**: The duration of pause (in seconds) that the model should interpret as an incomplete or unknown sentence. This is useful for identifying when a sentence is trailing off or unfinished.

#### `--mid_sentence_detection_pause`

- **Type**: `float`
- **Default**: `2.0`
- **Description**: The duration of pause (in seconds) that the model should interpret as a mid-sentence break. Longer pauses can indicate a pause in speech but not necessarily the end of a sentence.

#### `--wake_words_sensitivity`

- **Type**: `float`
- **Default**: `0.5`
- **Description**: Sensitivity level for wake word detection, with a range from `0` (most sensitive) to `1` (least sensitive). Adjust this value based on your environment to ensure reliable wake word detection.

#### `--wake_word_timeout`

- **Type**: `float`
- **Default**: `5.0`
- **Description**: Maximum time in seconds that the system will wait for a wake word before timing out. After this timeout, the system stops listening for wake words until reactivated.

#### `--wake_word_activation_delay`

- **Type**: `float`
- **Default**: `20`
- **Description**: The delay in seconds before the wake word detection is activated after the system starts listening. This prevents false positives during the start of a session.

#### `--wakeword_backend`

- **Type**: `str`
- **Default**: `'none'`
- **Description**: The backend used for wake word detection. You can specify different backends such as `"default"` or any custom implementations depending on your setup.

#### `--openwakeword_model_paths`

- **Type**: `str` (accepts multiple values)
- **Description**: A list of file paths to OpenWakeWord models. This is useful if you are using OpenWakeWord for wake word detection and need to specify custom models.

#### `--openwakeword_inference_framework`

- **Type**: `str`
- **Default**: `'tensorflow'`
- **Description**: The inference framework to use for OpenWakeWord models. Supported frameworks could include `"tensorflow"`, `"pytorch"`, etc.

#### `--wake_word_buffer_duration`

- **Type**: `float`
- **Default**: `1.0`
- **Description**: Duration of the buffer in seconds for wake word detection. This sets how long the system will store the audio before and after detecting the wake word.

#### `--use_main_model_for_realtime`

- **Action**: `store_true`
- **Description**: Enable this option if you want to use the main model for real-time transcription, instead of the smaller, faster real-time model. Using the main model may provide better accuracy but at the cost of higher processing time.

#### `--use_extended_logging`

- **Action**: `store_true`
- **Description**: Writes extensive log messages for the recording worker that processes the audio chunks.

#### `--logchunks`

- **Action**: `store_true`
- **Description**: Enable logging of incoming audio chunks (periods).

**Example:**

```bash
stt-server -m small.en -l en -c 9001 -d 9002
```

## Client Usage

### Starting the Client

Start the client using:

```bash
stt [OPTIONS]
```

The client connects to the STT server's control and data WebSocket URLs to facilitate real-time speech transcription and control.

### Available Parameters for STT Client:

#### `-i`, `--input-device`
- **Type**: `int`
- **Metavar**: `INDEX`
- **Description**: Audio input device index. Use `-L` to list available devices.

#### `-l`, `--language`
- **Type**: `str` 
- **Default**: `'en'`
- **Metavar**: `LANG`
- **Description**: Language code to be used for transcription.

#### `-sed`, `--speech-end-detection`
- **Action**: `store_true`
- **Description**: Enable intelligent speech end detection for better sentence boundaries.

#### `-D`, `--debug`
- **Action**: `store_true`
- **Description**: Enable debug mode for detailed logging.

#### `-n`, `--norealtime`
- **Action**: `store_true`
- **Description**: Disable real-time transcription output.

#### `-W`, `--write`
- **Metavar**: `FILE`
- **Description**: Save recorded audio to a WAV file.

#### `-s`, `--set`
- **Type**: `list`
- **Metavar**: `('PARAM', 'VALUE')`
- **Action**: `append`
- **Description**: Set a recorder parameter. Can be used multiple times with different parameters.

#### `-m`, `--method`
- **Type**: `list`
- **Metavar**: `METHOD`
- **Action**: `append`
- **Description**: Call a recorder method with optional arguments.

#### `-g`, `--get`
- **Type**: `list`
- **Metavar**: `PARAM`
- **Action**: `append`
- **Description**: Get the value of a recorder parameter.

#### `-c`, `--continous`
- **Action**: `store_true`
- **Description**: Run in continuous mode, transcribing speech without exiting.

#### `-L`, `--list`
- **Action**: `store_true`
- **Description**: List all available audio input devices and exit.

#### `--control`, `--control_url`
- **Type**: `str`
- **Default**: `ws://127.0.0.1:8011`
- **Description**: WebSocket URL for STT control connection.

#### `--data`, `--data_url`
- **Type**: `str`
- **Default**: `ws://127.0.0.1:8012`
- **Description**: WebSocket URL for STT data connection.


### Parameters only available when speech-end-detection is active:

#### `--post-silence`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Post speech silence duration in seconds.

#### `--unknown-pause` 
- **Type**: `float`
- **Default**: `1.3`
- **Description**: Unknown sentence detection pause duration in seconds.

#### `--mid-pause`
- **Type**: `float` 
- **Default**: `3.0`
- **Description**: Mid-sentence detection pause duration in seconds.

#### `--end-pause`
- **Type**: `float`
- **Default**: `0.7` 
- **Description**: End of sentence detection pause duration in seconds.

#### `--hard-break`
- **Type**: `float`
- **Default**: `3.0`
- **Description**: Hard break threshold in seconds when background noise is present.

#### `--min-texts`
- **Type**: `int`
- **Default**: `3`
- **Description**: Minimum number of texts required for hard break detection.

#### `--min-similarity`
- **Type**: `float`
- **Default**: `0.99`
- **Description**: Minimum text similarity threshold for hard break detection.

#### `--min-chars`
- **Type**: `int`
- **Default**: `15`
- **Description**: Minimum number of characters required for hard break detection.

**Examples:**

```bash
# List available audio devices
stt -L

# Use specific input device and language
stt -i 1 -l en

# Enable intelligent speech end detection and continuous mode
stt -sed -c

# Set parameter and save audio
stt -s silero_sensitivity 0.1 -W recording.wav

# Use custom WebSocket URLs
stt --control ws://localhost:9001 --data ws://localhost:9002
```

## WebSocket Interface

The server uses two WebSocket connections:

1. **Control WebSocket**: Used to send and receive control commands, such as setting parameters or invoking recorder methods.

2. **Data WebSocket**: Used to send audio data for transcription and receive real-time transcription updates.

## Examples

### Starting the Server and Client

1. **Start the Server with Default Settings:**

   ```bash
   stt-server
   ```

2. **Start the Client with Default Settings:**

   ```bash
   stt
   ```

### Setting Parameters

Set the Silero sensitivity to `0.1`:

```bash
stt -s silero_sensitivity 0.1
```

### Retrieving Parameters

Get the current Silero sensitivity value:

```bash
stt -g silero_sensitivity
```

### Calling Server Methods

Call the `set_microphone` method on the recorder:

```bash
stt -m set_microphone False
```

### Running in Debug Mode

Enable debug mode for detailed logging:

```bash
stt -D
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

# Additional Information

The server and client scripts are designed to work seamlessly together, enabling efficient real-time speech transcription with minimal latency. The flexibility in configuration allows users to tailor the system to specific needs, such as adjusting sensitivity levels for different environments or selecting appropriate STT models based on resource availability.

**Note:** Ensure that the server is running before starting the client. The client includes functionality to check if the server is running and can prompt the user to start it if necessary.

# Troubleshooting

- **Server Not Starting:** If the server fails to start, check that all dependencies are installed and that the specified ports are not in use.

- **Audio Issues:** Ensure that the correct audio input device index is specified if using a device other than the default.

- **WebSocket Connection Errors:** Verify that the control and data URLs are correct and that the server is listening on those ports.

# Contact

For questions or support, please open an issue on the [GitHub repository](https://github.com/KoljaB/RealtimeSTT/issues).

# Acknowledgments

Special thanks to the contributors of the RealtimeSTT library and the open-source community for their continuous support.

---

**Disclaimer:** This software is provided "as is", without warranty of any kind, express or implied. Use it at your own risk.