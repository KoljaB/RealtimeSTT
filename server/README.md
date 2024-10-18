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

- `--model` (str, default: `'medium.en'`): Path to the STT model or model size. Options include `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, or any Hugging Face CTranslate2 STT model like `deepdml/faster-whisper-large-v3-turbo-ct2`.

- `--realtime_model_type` (str, default: `'tiny.en'`): Model size for real-time transcription. Same options as `--model`.

- `--language` (str, default: `'en'`): Language code for the STT model. Leave empty for auto-detection.

- `--input_device_index` (int, default: `1`): Index of the audio input device to use.

- `--silero_sensitivity` (float, default: `0.05`): Sensitivity for Silero VAD. Lower values are less sensitive.

- `--webrtc_sensitivity` (float, default: `3`): Sensitivity for WebRTC VAD. Higher values are less sensitive.

- `--min_length_of_recording` (float, default: `1.1`): Minimum duration (in seconds) for a valid recording.

- `--min_gap_between_recordings` (float, default: `0`): Minimum time (in seconds) between consecutive recordings.

- `--enable_realtime_transcription` (flag, default: `True`): Enable real-time transcription of audio.

- `--realtime_processing_pause` (float, default: `0.02`): Time interval (in seconds) between processing audio chunks for real-time transcription.

- `--silero_deactivity_detection` (flag, default: `True`): Use Silero model for end-of-speech detection.

- `--early_transcription_on_silence` (float, default: `0.2`): Start transcription after specified seconds of silence.

- `--beam_size` (int, default: `5`): Beam size for the main transcription model.

- `--beam_size_realtime` (int, default: `3`): Beam size for the real-time transcription model.

- `--initial_prompt` (str): Initial prompt for the transcription model to guide its output format and style.

- `--end_of_sentence_detection_pause` (float, default: `0.45`): Duration of pause (in seconds) to consider as the end of a sentence.

- `--unknown_sentence_detection_pause` (float, default: `0.7`): Duration of pause (in seconds) to consider as an unknown or incomplete sentence.

- `--mid_sentence_detection_pause` (float, default: `2.0`): Duration of pause (in seconds) to consider as a mid-sentence break.

- `--control_port` (int, default: `8011`): Port for the control WebSocket connection.

- `--data_port` (int, default: `8012`): Port for the data WebSocket connection.

**Example:**

```bash
stt-server --model small.en --language en --control_port 9001 --data_port 9002
```

## Client Usage

### Starting the Client

Start the client using:

```bash
stt [OPTIONS]
```

The client connects to the STT server's control and data WebSocket URLs to facilitate real-time speech transcription and control.

### Client Parameters

- `--control-url` (default: `ws://localhost:8011`): The WebSocket URL for server control commands.

- `--data-url` (default: `ws://localhost:8012`): The WebSocket URL for sending audio data and receiving transcription updates.

- `--debug`: Enable debug mode, which prints detailed logs to `stderr`.

- `--nort` or `--norealtime`: Disable real-time output of transcription results.

- `--set-param PARAM VALUE`: Set a recorder parameter (e.g., `silero_sensitivity`, `beam_size`). This option can be used multiple times.

- `--get-param PARAM`: Retrieve the value of a specific recorder parameter. Can be used multiple times.

- `--call-method METHOD [ARGS]`: Call a method on the recorder with optional arguments. Can be used multiple times.

**Example:**

```bash
stt --set-param silero_sensitivity 0.1 --get-param silero_sensitivity
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
stt --set-param silero_sensitivity 0.1
```

### Retrieving Parameters

Get the current Silero sensitivity value:

```bash
stt --get-param silero_sensitivity
```

### Calling Server Methods

Call the `set_microphone` method on the recorder:

```bash
stt --call-method set_microphone
```

### Running in Debug Mode

Enable debug mode for detailed logging:

```bash
stt --debug
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