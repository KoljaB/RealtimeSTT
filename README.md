# RealtimeSTT

*Easy to use low latency speech to text library for realtime applications*

## About the project

Listens to microphone and transcribes voice into text.

Provices voice activity detection, wake word activation and lightning-fast speech-to-text transcription. Checks for voice activity with WebRTC first for a quick decision, then double-checks with Silero for better accuracy for reliable voice activity detection even amidst ambient noise.

Perfect for voice assistants or applications where solid, fast and precise speech-to-text transformation is important.

> **Hint**: In need of the inverse – turning text streams into instant voice output - dive into [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS). Together, they form a powerful realtime audio wrapper around large language model outputs.

## Features

- **Real-time Transcription**: Delivers text as fast as possible (while you speak) using faster_whisper.
- **Voice Activity Detection**: Automatically starts/stops recording when speech is detected or when speech ends.
- **Wake Word Activation**: Starts detection only after a specified wake word (or words) was detected.
- **Event Callbacks**: Customizable callbacks for when recording starts or finishes.

## Quick Start

Basic usage:

### Manual Recording

Start and stop of recording are manually triggered.

```python
recorder.start()
recorder.stop()
print(recorder.text())
```

### Automatic Recording

Recording based on voice activity detection.

```python
recorder = AudioToTextRecorder()
print(recorder.text())
```  

### Wakewords

Keyword activation before detecting voice.

```python
recorder = AudioToTextRecorder(wake_words="jarvis")

print('Say "Jarvis" then speak.')
print(recorder.text())
```

## Installation

```bash
pip install RealtimeSTT
```

## GPU Support

To significantly improve transcription speed, especially in real-time applications, I **strongly recommend** utilizing GPU acceleration via CUDA. By default, the transcription is performed on the CPU. 

1. **Install NVIDIA CUDA Toolkit 11.8**:
	- Visit [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
	- Select version 11.
	- Download and install the software.

2. **Install NVIDIA cuDNN 8.7.0 for CUDA 11.x**:
	- Visit [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
	- Click on "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x".
	- Download and install the software.
	
3. **Reconfigure PyTorch for CUDA**:
	- If you have PyTorch CPU version installed, remove it: `pip uninstall torch` (CPU gets installed with the pip install RealtimeSTT command)
	- Install PyTorch again with CUDA support: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`.

Note: To check if your NVIDIA GPU supports CUDA, visit the [official CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).

## Quick Start

Here's a basic usage example:

### Automatic Recording

```python
recorder = AudioToTextRecorder()
print(recorder.text())
```

### Manual Recording

```python
recorder.start()
recorder.stop()
print(recorder.text())
```

### Callbacks

You can set callback functions to be executed when recording starts or stops:

```python
def my_start_callback():
    print("Recording started!")

def my_stop_callback():
    print("Recording stopped!")

recorder = AudioToTextRecorder(on_recording_started=my_start_callback, on_recording_finished=my_stop_callback)
```

### Wakewords

Write the comma-separated list of your desired activation keywords into the wake_words parameter. You can choose wake words from these list: alexa, americano, blueberry, bumblebee, computer, grapefruits, grasshopper, hey google, hey siri, jarvis, ok google, picovoice, porcupine, terminator. 

```python
recorder = AudioToTextRecorder(wake_words="jarvis")

print('Say "Jarvis" then speak.')
print(recorder.text())
```

## Testing the Library

The test subdirectory contains a set of scripts to help you evaluate and understand the capabilities of the RealtimeTTS library.

- **simple_test.py**
    - **Description**: A "hello world" styled demonstration of the library's simplest usage.

- **wakeword_test.py**
    - **Description**: A demonstration of the wakeword activation.

- **translator.py**
    - **Dependencies**: Run `pip install openai realtimetts`.
    - **Description**: Real-time translations into six different languages.

- **openai_voice_interface.py**
    - **Dependencies**: Run `pip install openai realtimetts`.
    - **Description**: Wake word activated and voice based user interface to the OpenAI API.

- **advanced_talk.py**
    - **Dependencies**: Run `pip install openai keyboard realtimetts`.
    - **Description**: Choose TTS engine and voice before starting AI conversation.

- **minimalistic_talkbot.py**
    - **Dependencies**: Run `pip install openai realtimetts`.
    - **Description**: A basic talkbot in 20 lines of code.

## Configuration

- **model** (str, default="tiny"): Model size or path for transcription.
    - Options: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
    - Note: If a size is provided, the model will be downloaded from the Hugging Face Hub.

- **language** (str, default=""): Language code for transcription. If left empty, the model will try to auto-detect the language.

- **on_recording_start**: A callable function triggered when recording starts.

- **on_recording_stop**: A callable function triggered when recording ends.

- **spinner** (bool, default=True): Provides a spinner animation text with information about the current recorder state.

- **level** (int, default=logging.WARNING): Logging level.

- **silero_sensitivity** (float, default=0.6): Sensitivity for Silero's voice activity detection ranging from 0 (least sensitive) to 1 (most sensitive). Default is 0.6.

- **webrtc_sensitivity** (int, default=3): Sensitivity for the WebRTC Voice Activity Detection engine ranging from 1 (least sensitive) to 3 (most sensitive). Default is 3.

- **post_speech_silence_duration** (float, default=0.2): Duration in seconds of silence that must follow speech before the recording is considered to be completed. This ensures that any brief pauses during speech don't prematurely end the recording.

- **min_gap_between_recordings** (float, default=1.0): Specifies the minimum time interval in seconds that should exist between the end of one recording session and the beginning of another to prevent rapid consecutive recordings.

- **min_length_of_recording** (float, default=1.0): Specifies the minimum duration in seconds that a recording session should last to ensure meaningful audio capture, preventing excessively short or fragmented recordings.

- **pre_recording_buffer_duration** (float, default=0.2): The time span, in seconds, during which audio is buffered prior to formal recording. This helps counterbalancing the latency inherent in speech activity detection, ensuring no initial audio is missed.

- **wake_words** (str, default=""): Wake words for initiating the recording. Multiple wake words can be provided as a comma-separated string. Supported wake words are: alexa, americano, blueberry, bumblebee, computer, grapefruits, grasshopper, hey google, hey siri, jarvis, ok google, picovoice, porcupine, terminator

- **wake_words_sensitivity** (float, default=0.6): Sensitivity level for wake word detection (0 for least sensitive, 1 for most sensitive).

- **wake_word_activation_delay** (float, default=0): Duration in seconds after the start of monitoring before the system switches to wake word activation if no voice is initially detected. If set to zero, the system uses wake word activation immediately.

- **wake_word_timeout** (float, default=5): Duration in seconds after a wake word is recognized. If no subsequent voice activity is detected within this window, the system transitions back to an inactive state, awaiting the next wake word or voice activation.

- **on_wakeword_detected**: A callable function triggered when a wake word is detected.

- **on_wakeword_timeout**: Callback function to be called when the system goes back to an inactive state after when no speech was detected after wake word activation.

## Contribution

Contributions are always welcome! 

## License

MIT

## Author

Kolja Beigel  
Email: kolja.beigel@web.de  
[GitHub](https://github.com/KoljaB/RealtimeSTT)

---