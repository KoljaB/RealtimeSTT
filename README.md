
# RealtimeSTT

*Easy-to-use, low-latency speech-to-text library for realtime applications*

## About the Project

RealtimeSTT listens to the microphone and transcribes voice into text.  

It's ideal for:

- **Voice Assistants**
- Applications requiring **fast and precise** speech-to-text conversion

https://github.com/KoljaB/RealtimeSTT/assets/7604638/207cb9a2-4482-48e7-9d2b-0722c3ee6d14

### Features

- **Voice Activity Detection**: Automatically detects when you start and stop speaking.
- **Realtime Transcription**: Transforms speech to text in real-time.
- **Wake Word Activation**: Can activate upon detecting a designated wake word.

> **Hint**: *Check out [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS), the output counterpart of this library, for text-to-voice capabilities. Together, they form a powerful realtime audio wrapper around large language models.*

## Tech Stack

This library uses:

- **Voice Activity Detection**
  - [WebRTCVAD](https://github.com/wiseman/py-webrtcvad) for initial voice activity detection.
  - [SileroVAD](https://github.com/snakers4/silero-vad) for more accurate verification.
- **Speech-To-Text**
  - [Faster_Whisper](https://github.com/guillaumekln/faster-whisper) for instant (GPU-accelerated) transcription.
- **Wake Word Detection**
  - [Porcupine](https://github.com/Picovoice/porcupine) for wake word detection.

*These components represent the "industry standard" for cutting-edge applications, providing the most modern and effective foundation for building high-end solutions.*


## Installation

```bash
pip install RealtimeSTT
```

This will install all the necessary dependencies, including a **CPU support only** version of PyTorch.

Although it is possible to run RealtimeSTT with a CPU installation only (use a small model like "tiny" or "base" in this case) you will get way better experience using:

### GPU Support with CUDA (recommended)

Additional steps are needed for a **GPU-optimized** installation. These steps are recommended for those who require **better performance** and have a compatible NVIDIA GPU.

> **Note**: *To check if your NVIDIA GPU supports CUDA, visit the [official CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).*

To use RealtimeSTT with GPU support via CUDA please follow these steps:

1. **Install NVIDIA CUDA Toolkit 11.8**:
    - Visit [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Select version 11.
    - Download and install the software.

2. **Install NVIDIA cuDNN 8.7.0 for CUDA 11.x**:
    - Visit [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
    - Click on "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x".
    - Download and install the software.

3. **Install ffmpeg**:

    You can download an installer for your OS from the [ffmpeg Website](https://ffmpeg.org/download.html).  
    
    Or use a package manager:

    - **On Ubuntu or Debian**:
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```

    - **On Arch Linux**:
        ```bash
        sudo pacman -S ffmpeg
        ```

    - **On MacOS using Homebrew** ([https://brew.sh/](https://brew.sh/)):
        ```bash
        brew install ffmpeg
        ```

    - **On Windows using Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```bash
        choco install ffmpeg
        ```

    - **On Windows using Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ```bash
        scoop install ffmpeg
        ```    

4. **Install PyTorch with CUDA support**:
    ```bash
    pip uninstall torch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

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

Keyword activation before detecting voice. Write the comma-separated list of your desired activation keywords into the wake_words parameter. You can choose wake words from these list: alexa, americano, blueberry, bumblebee, computer, grapefruits, grasshopper, hey google, hey siri, jarvis, ok google, picovoice, porcupine, terminator. 

```python
recorder = AudioToTextRecorder(wake_words="jarvis")

print('Say "Jarvis" then speak.')
print(recorder.text())
```

### Callbacks

You can set callback functions to be executed on different events (see [Configuration](#configuration)) :

```python
def my_start_callback():
    print("Recording started!")

def my_stop_callback():
    print("Recording stopped!")

recorder = AudioToTextRecorder(on_recording_start=my_start_callback,
                               on_recording_stop=my_stop_callback)
```


## Testing the Library

The test subdirectory contains a set of scripts to help you evaluate and understand the capabilities of the RealtimeTTS library.

Test scripts depending on RealtimeTTS library may require you to enter your azure service region within the script. 
When using OpenAI-, Azure- or Elevenlabs-related demo scripts the API Keys should be provided in the environment variables OPENAI_API_KEY, AZURE_SPEECH_KEY and ELEVENLABS_API_KEY (see [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS))

- **simple_test.py**
    - **Description**: A "hello world" styled demonstration of the library's simplest usage.

- **realtimestt_test.py**
    - **Description**: Showcasing live-transcription.

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

The example_app subdirectory contains a polished user interface application for the OpenAI API based on PyQt5.

## Configuration

### Initialization Parameters for `AudioToTextRecorder`

When you initialize the `AudioToTextRecorder` class, you have various options to customize its behavior.

#### General Parameters

- **model** (str, default="tiny"): Model size or path for transcription.
    - Options: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.
    - Note: If a size is provided, the model will be downloaded from the Hugging Face Hub.

- **language** (str, default=""): Language code for transcription. If left empty, the model will try to auto-detect the language.

- **on_recording_start**: A callable function triggered when recording starts.

- **on_recording_stop**: A callable function triggered when recording ends.

- **on_transcription_start**: A callable function triggered when transcription starts.

- **ensure_sentence_starting_uppercase** (bool, default=True): Ensures that every sentence detected by the algorithm starts with an uppercase letter.

- **ensure_sentence_ends_with_period** (bool, default=True): Ensures that every sentence that doesn't end with punctuation such as "?", "!" ends with a period

- **spinner** (bool, default=True): Provides a spinner animation text with information about the current recorder state.

- **level** (int, default=logging.WARNING): Logging level.

#### Real-time Transcription Parameters

**Note**: When enabling realtime description a GPU installation is strongly advised. Using realtime transcription may create high GPU loads.

- **enable_realtime_transcription** (bool, default=False): Enables or disables real-time transcription of audio. When set to True, the audio will be transcribed continuously as it is being recorded.

- **realtime_model_type** (str, default="tiny"): Specifies the size or path of the machine learning model to be used for real-time transcription.
    - Valid options: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.

- **realtime_processing_pause** (float, default=0.2): Specifies the time interval in seconds after a chunk of audio gets transcribed. Lower values will result in more "real-time" (frequent) transcription updates but may increase computational load.

- **on_realtime_transcription_update**: A callback function that is triggered whenever there's an update in the real-time transcription. The function is called with the newly transcribed text as its argument.

- **on_realtime_transcription_stabilized**: A callback function that is triggered whenever there's an update in the real-time transcription and returns a higher quality, stabilized text as its argument.

#### Voice Activation Parameters

- **silero_sensitivity** (float, default=0.6): Sensitivity for Silero's voice activity detection ranging from 0 (least sensitive) to 1 (most sensitive). Default is 0.6.

- **webrtc_sensitivity** (int, default=3): Sensitivity for the WebRTC Voice Activity Detection engine ranging from 1 (least sensitive) to 3 (most sensitive). Default is 3.

- **post_speech_silence_duration** (float, default=0.2): Duration in seconds of silence that must follow speech before the recording is considered to be completed. This ensures that any brief pauses during speech don't prematurely end the recording.

- **min_gap_between_recordings** (float, default=1.0): Specifies the minimum time interval in seconds that should exist between the end of one recording session and the beginning of another to prevent rapid consecutive recordings.

- **min_length_of_recording** (float, default=1.0): Specifies the minimum duration in seconds that a recording session should last to ensure meaningful audio capture, preventing excessively short or fragmented recordings.

- **pre_recording_buffer_duration** (float, default=0.2): The time span, in seconds, during which audio is buffered prior to formal recording. This helps counterbalancing the latency inherent in speech activity detection, ensuring no initial audio is missed.

- **on_vad_detect_start**: A callable function triggered when the system starts to listen for voice activity.

- **on_vad_detect_stop**: A callable function triggered when the system stops to listen for voice activity.

#### Wake Word Parameters

- **wake_words** (str, default=""): Wake words for initiating the recording. Multiple wake words can be provided as a comma-separated string. Supported wake words are: alexa, americano, blueberry, bumblebee, computer, grapefruits, grasshopper, hey google, hey siri, jarvis, ok google, picovoice, porcupine, terminator

- **wake_words_sensitivity** (float, default=0.6): Sensitivity level for wake word detection (0 for least sensitive, 1 for most sensitive).

- **wake_word_activation_delay** (float, default=0): Duration in seconds after the start of monitoring before the system switches to wake word activation if no voice is initially detected. If set to zero, the system uses wake word activation immediately.

- **wake_word_timeout** (float, default=5): Duration in seconds after a wake word is recognized. If no subsequent voice activity is detected within this window, the system transitions back to an inactive state, awaiting the next wake word or voice activation.

- **on_wakeword_detected**: A callable function triggered when a wake word is detected.

- **on_wakeword_timeout**: A callable function triggered when the system goes back to an inactive state after when no speech was detected after wake word activation.

- **on_wakeword_detection_start**: A callable function triggered when the system starts to listen for wake words

- **on_wakeword_detection_end**: A callable function triggered when stopping to listen for wake words (e.g. because of timeout or wake word detected)


## Contribution

Contributions are always welcome! 

## License

MIT

## Author

Kolja Beigel  
Email: kolja.beigel@web.de  
[GitHub](https://github.com/KoljaB/RealtimeSTT)
