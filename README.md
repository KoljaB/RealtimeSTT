# RealtimeSTT
[![PyPI](https://img.shields.io/pypi/v/RealtimeSTT)](https://pypi.org/project/RealtimeSTT/)
[![Downloads](https://static.pepy.tech/badge/RealtimeSTT)](https://pepy.tech/project/KoljaB/RealtimeSTT)
[![GitHub release](https://img.shields.io/github/release/KoljaB/RealtimeSTT.svg)](https://GitHub.com/KoljaB/RealtimeSTT/releases/)
[![GitHub commits](https://badgen.net/github/commits/KoljaB/RealtimeSTT)](https://GitHub.com/Naereen/KoljaB/RealtimeSTT/commit/)
[![GitHub forks](https://img.shields.io/github/forks/KoljaB/RealtimeSTT.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/KoljaB/RealtimeSTT/network/)
[![GitHub stars](https://img.shields.io/github/stars/KoljaB/RealtimeSTT.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/KoljaB/RealtimeSTT/stargazers/)

*Easy-to-use, low-latency speech-to-text library for realtime applications*

## New

- AudioToTextRecorderClient class, which automatically starts a server if none is running and connects to it. The class shares the same interface as AudioToTextRecorder, making it easy to upgrade or switch between the two. (Work in progress, most parameters and callbacks of AudioToTextRecorder are already implemented into AudioToTextRecorderClient, but not all. Also the server can not handle concurrent (parallel) requests yet.)
- reworked CLI interface ("stt-server" to start the server, "stt" to start the client, look at "server" folder for more info)

## About the Project

RealtimeSTT listens to the microphone and transcribes voice into text.  

> **Hint:** *<strong>Check out [Linguflex](https://github.com/KoljaB/Linguflex)</strong>, the original project from which RealtimeSTT is spun off. It lets you control your environment by speaking and is one of the most capable and sophisticated open-source assistants currently available.*

It's ideal for:

- **Voice Assistants**
- Applications requiring **fast and precise** speech-to-text conversion

https://github.com/user-attachments/assets/797e6552-27cd-41b1-a7f3-e5cbc72094f5

### Updates

Latest Version: v0.3.81

See [release history](https://github.com/KoljaB/RealtimeSTT/releases).

> **Hint:** *Since we use the `multiprocessing` module now, ensure to include the `if __name__ == '__main__':` protection in your code to prevent unexpected behavior, especially on platforms like Windows. For a detailed explanation on why this is important, visit the [official Python documentation on `multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming).*

## Quick Examples

### Print everything being said:

```python
from RealtimeSTT import AudioToTextRecorder
import pyautogui

def process_text(text):
    print(text)

if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder()

    while True:
        recorder.text(process_text)
```

### Type everything being said:

```python
from RealtimeSTT import AudioToTextRecorder
import pyautogui

def process_text(text):
    pyautogui.typewrite(text + " ")

if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder()

    while True:
        recorder.text(process_text)
```
*Will type everything being said into your selected text box*

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
  - [Porcupine](https://github.com/Picovoice/porcupine) or [OpenWakeWord](https://github.com/dscripka/openWakeWord) for wake word detection.


*These components represent the "industry standard" for cutting-edge applications, providing the most modern and effective foundation for building high-end solutions.*

## Installation

```bash
pip install RealtimeSTT
```

This will install all the necessary dependencies, including a **CPU support only** version of PyTorch.

Although it is possible to run RealtimeSTT with a CPU installation only (use a small model like "tiny" or "base" in this case) you will get way better experience using:

### GPU Support with CUDA (recommended)


### Updating PyTorch for CUDA Support

To upgrade your PyTorch installation to enable GPU support with CUDA, follow these instructions based on your specific CUDA version. This is useful if you wish to enhance the performance of RealtimeSTT with CUDA capabilities.

#### For CUDA 11.8:
To update PyTorch and Torchaudio to support CUDA 11.8, use the following commands:

```bash
pip install torch==2.5.1+cu118 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.X:
To update PyTorch and Torchaudio to support CUDA 12.X, execute the following:

```bash
pip install torch==2.5.1+cu121 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Replace `2.5.1` with the version of PyTorch that matches your system and requirements.

### Steps That Might Be Necessary Before

> **Note**: *To check if your NVIDIA GPU supports CUDA, visit the [official CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).*

If you didn't use CUDA models before, some additional steps might be needed one time before installation. These steps prepare the system for CUDA support and installation of the **GPU-optimized** installation. This is recommended for those who require **better performance** and have a compatible NVIDIA GPU. To use RealtimeSTT with GPU support via CUDA please also follow these steps:

1. **Install NVIDIA CUDA Toolkit**:
    - select between CUDA 11.8 or CUDA 12.X Toolkit
        - for 12.X visit [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and select latest version.
        - for 11.8 visit [NVIDIA CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Select operating system and version.
    - Download and install the software.

2. **Install NVIDIA cuDNN**:
    - select between CUDA 11.8 or CUDA 12.X Toolkit
        - for 12.X visit [cuDNN Downloads](https://developer.nvidia.com/cudnn-downloads).
            - Select operating system and version.
            - Download and install the software.
        - for 11.8 visit [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
            - Click on "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x".
            - Download and install the software.
    
3. **Install ffmpeg**:

    > **Note**: *Installation of ffmpeg might not actually be needed to operate RealtimeSTT* <sup> *thanks to jgilbert2017 for pointing this out</sup>

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

    - **On Windows using Winget** [official documentation](https://learn.microsoft.com/en-us/windows/package-manager/winget/) :
        ```bash
        winget install Gyan.FFmpeg
        ```
        
    - **On Windows using Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```bash
        choco install ffmpeg
        ```

    - **On Windows using Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ```bash
        scoop install ffmpeg
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

#### Standalone Example:

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':
    recorder = AudioToTextRecorder()
    recorder.start()
    input("Press Enter to stop recording...")
    recorder.stop()
    print("Transcription: ", recorder.text())
```

### Automatic Recording

Recording based on voice activity detection.

```python
with AudioToTextRecorder() as recorder:
    print(recorder.text())
```

#### Standalone Example:

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':
    with AudioToTextRecorder() as recorder:
        print("Transcription: ", recorder.text())
```

When running recorder.text in a loop it is recommended to use a callback, allowing the transcription to be run asynchronously:


```python
def process_text(text):
    print (text)
    
while True:
    recorder.text(process_text)
```

#### Standalone Example:

```python
from RealtimeSTT import AudioToTextRecorder

def process_text(text):
    print(text)

if __name__ == '__main__':
    recorder = AudioToTextRecorder()

    while True:
        recorder.text(process_text)
```

### Wakewords

Keyword activation before detecting voice. Write the comma-separated list of your desired activation keywords into the wake_words parameter. You can choose wake words from these list: alexa, americano, blueberry, bumblebee, computer, grapefruits, grasshopper, hey google, hey siri, jarvis, ok google, picovoice, porcupine, terminator. 

```python
recorder = AudioToTextRecorder(wake_words="jarvis")

print('Say "Jarvis" then speak.')
print(recorder.text())
```

#### Standalone Example:

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':
    recorder = AudioToTextRecorder(wake_words="jarvis")

    print('Say "Jarvis" to start recording.')
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

#### Standalone Example:

```python
from RealtimeSTT import AudioToTextRecorder

def start_callback():
    print("Recording started!")

def stop_callback():
    print("Recording stopped!")

if __name__ == '__main__':
    recorder = AudioToTextRecorder(on_recording_start=start_callback,
                                   on_recording_stop=stop_callback)
```

### Feed chunks

If you don't want to use the local microphone set use_microphone parameter to false and provide raw PCM audiochunks in 16-bit mono (samplerate 16000) with this method:

```python
recorder.feed_audio(audio_chunk)
```

#### Standalone Example:

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':
    recorder = AudioToTextRecorder(use_microphone=False)
    with open("audio_chunk.pcm", "rb") as f:
        audio_chunk = f.read()

    recorder.feed_audio(audio_chunk)
    print("Transcription: ", recorder.text())
```

### Shutdown

You can shutdown the recorder safely by using the context manager protocol:

```python
with AudioToTextRecorder() as recorder:
    [...]
```


Or you can call the shutdown method manually (if using "with" is not feasible):

```python
recorder.shutdown()
```

#### Standalone Example:

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':
    with AudioToTextRecorder() as recorder:
        [...]
    # or manually shutdown if "with" is not used
    recorder.shutdown()
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

- **language** (str, default=""): Language code for transcription. If left empty, the model will try to auto-detect the language. Supported language codes are listed in [Whisper Tokenizer library](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py).

- **compute_type** (str, default="default"): Specifies the type of computation to be used for transcription. See [Whisper Quantization](https://opennmt.net/CTranslate2/quantization.html)

- **input_device_index** (int, default=0): Audio Input Device Index to use.

- **gpu_device_index** (int, default=0): GPU Device Index to use. The model can also be loaded on multiple GPUs by passing a list of IDs (e.g. [0, 1, 2, 3]).

- **device** (str, default="cuda"): Device for model to use. Can either be "cuda" or "cpu". 

- **on_recording_start**: A callable function triggered when recording starts.

- **on_recording_stop**: A callable function triggered when recording ends.

- **on_transcription_start**: A callable function triggered when transcription starts.

- **ensure_sentence_starting_uppercase** (bool, default=True): Ensures that every sentence detected by the algorithm starts with an uppercase letter.

- **ensure_sentence_ends_with_period** (bool, default=True): Ensures that every sentence that doesn't end with punctuation such as "?", "!" ends with a period

- **use_microphone** (bool, default=True): Usage of local microphone for transcription. Set to False if you want to provide chunks with feed_audio method.

- **spinner** (bool, default=True): Provides a spinner animation text with information about the current recorder state.

- **level** (int, default=logging.WARNING): Logging level.

- **init_logging** (bool, default=True): Whether to initialize the logging framework. Set to False to manage this yourself.

- **handle_buffer_overflow** (bool, default=True): If set, the system will log a warning when an input overflow occurs during recording and remove the data from the buffer.

- **beam_size** (int, default=5): The beam size to use for beam search decoding.

- **initial_prompt** (str or iterable of int, default=None): Initial prompt to be fed to the transcription models.

- **suppress_tokens** (list of int, default=[-1]): Tokens to be suppressed from the transcription output.

- **on_recorded_chunk**: A callback function that is triggered when a chunk of audio is recorded. Submits the chunk data as parameter.

- **debug_mode** (bool, default=False): If set, the system prints additional debug information to the console.

- **print_transcription_time** (bool, default=False): Logs the processing time of the main model transcription. This can be useful for performance monitoring and debugging.

- **early_transcription_on_silence** (int, default=0): If set, the system will transcribe audio faster when silence is detected. Transcription will start after the specified milliseconds. Keep this value lower than `post_speech_silence_duration`, ideally around `post_speech_silence_duration` minus the estimated transcription time with the main model. If silence lasts longer than `post_speech_silence_duration`, the recording is stopped, and the transcription is submitted. If voice activity resumes within this period, the transcription is discarded. This results in faster final transcriptions at the cost of additional GPU load due to some unnecessary final transcriptions.

- **allowed_latency_limit** (int, default=100): Specifies the maximum number of unprocessed chunks in the queue before discarding chunks. This helps prevent the system from being overwhelmed and losing responsiveness in real-time applications.

- **no_log_file** (bool, default=False): If set, the system will skip writing the debug log file, reducing disk I/O. Useful if logging to a file is not needed and performance is a priority.

#### Real-time Transcription Parameters

> **Note**: *When enabling realtime description a GPU installation is strongly advised. Using realtime transcription may create high GPU loads.*

- **enable_realtime_transcription** (bool, default=False): Enables or disables real-time transcription of audio. When set to True, the audio will be transcribed continuously as it is being recorded.

- **use_main_model_for_realtime** (bool, default=False): If set to True, the main transcription model will be used for both regular and real-time transcription. If False, a separate model specified by `realtime_model_type` will be used for real-time transcription. Using a single model can save memory and potentially improve performance, but may not be optimized for real-time processing. Using separate models allows for a smaller, faster model for real-time transcription while keeping a more accurate model for final transcription.

- **realtime_model_type** (str, default="tiny"): Specifies the size or path of the machine learning model to be used for real-time transcription.
    - Valid options: 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'.

- **realtime_processing_pause** (float, default=0.2): Specifies the time interval in seconds after a chunk of audio gets transcribed. Lower values will result in more "real-time" (frequent) transcription updates but may increase computational load.

- **on_realtime_transcription_update**: A callback function that is triggered whenever there's an update in the real-time transcription. The function is called with the newly transcribed text as its argument.

- **on_realtime_transcription_stabilized**: A callback function that is triggered whenever there's an update in the real-time transcription and returns a higher quality, stabilized text as its argument.

- **beam_size_realtime** (int, default=3): The beam size to use for real-time transcription beam search decoding.

#### Voice Activation Parameters

- **silero_sensitivity** (float, default=0.6): Sensitivity for Silero's voice activity detection ranging from 0 (least sensitive) to 1 (most sensitive). Default is 0.6.

- **silero_use_onnx** (bool, default=False): Enables usage of the pre-trained model from Silero in the ONNX (Open Neural Network Exchange) format instead of the PyTorch format. Default is False. Recommended for faster performance.

- **silero_deactivity_detection** (bool, default=False): Enables the Silero model for end-of-speech detection. More robust against background noise. Utilizes additional GPU resources but improves accuracy in noisy environments. When False, uses the default WebRTC VAD, which is more sensitive but may continue recording longer due to background sounds.

- **webrtc_sensitivity** (int, default=3): Sensitivity for the WebRTC Voice Activity Detection engine ranging from 0 (least aggressive / most sensitive) to 3 (most aggressive, least sensitive). Default is 3.

- **post_speech_silence_duration** (float, default=0.2): Duration in seconds of silence that must follow speech before the recording is considered to be completed. This ensures that any brief pauses during speech don't prematurely end the recording.

- **min_gap_between_recordings** (float, default=1.0): Specifies the minimum time interval in seconds that should exist between the end of one recording session and the beginning of another to prevent rapid consecutive recordings.

- **min_length_of_recording** (float, default=1.0): Specifies the minimum duration in seconds that a recording session should last to ensure meaningful audio capture, preventing excessively short or fragmented recordings.

- **pre_recording_buffer_duration** (float, default=0.2): The time span, in seconds, during which audio is buffered prior to formal recording. This helps counterbalancing the latency inherent in speech activity detection, ensuring no initial audio is missed.

- **on_vad_detect_start**: A callable function triggered when the system starts to listen for voice activity.

- **on_vad_detect_stop**: A callable function triggered when the system stops to listen for voice activity.

#### Wake Word Parameters

- **wakeword_backend** (str, default="pvporcupine"): Specifies the backend library to use for wake word detection. Supported options include 'pvporcupine' for using the Porcupine wake word engine or 'oww' for using the OpenWakeWord engine.

- **openwakeword_model_paths** (str, default=None): Comma-separated paths to model files for the openwakeword library. These paths point to custom models that can be used for wake word detection when the openwakeword library is selected as the wakeword_backend.

- **openwakeword_inference_framework** (str, default="onnx"): Specifies the inference framework to use with the openwakeword library. Can be either 'onnx' for Open Neural Network Exchange format or 'tflite' for TensorFlow Lite.

- **wake_words** (str, default=""): Initiate recording when using the 'pvporcupine' wakeword backend. Multiple wake words can be provided as a comma-separated string. Supported wake words are: alexa, americano, blueberry, bumblebee, computer, grapefruits, grasshopper, hey google, hey siri, jarvis, ok google, picovoice, porcupine, terminator. For the 'openwakeword' backend, wake words are automatically extracted from the provided model files, so specifying them here is not necessary.

- **wake_words_sensitivity** (float, default=0.6): Sensitivity level for wake word detection (0 for least sensitive, 1 for most sensitive).

- **wake_word_activation_delay** (float, default=0): Duration in seconds after the start of monitoring before the system switches to wake word activation if no voice is initially detected. If set to zero, the system uses wake word activation immediately.

- **wake_word_timeout** (float, default=5): Duration in seconds after a wake word is recognized. If no subsequent voice activity is detected within this window, the system transitions back to an inactive state, awaiting the next wake word or voice activation.

- **wake_word_buffer_duration** (float, default=0.1): Duration in seconds to buffer audio data during wake word detection. This helps in cutting out the wake word from the recording buffer so it does not falsely get detected along with the following spoken text, ensuring cleaner and more accurate transcription start triggers. Increase this if parts of the wake word get detected as text.

- **on_wakeword_detected**: A callable function triggered when a wake word is detected.

- **on_wakeword_timeout**: A callable function triggered when the system goes back to an inactive state after when no speech was detected after wake word activation.

- **on_wakeword_detection_start**: A callable function triggered when the system starts to listen for wake words

- **on_wakeword_detection_end**: A callable function triggered when stopping to listen for wake words (e.g. because of timeout or wake word detected)

## OpenWakeWord  

### Training models

Look [here](https://github.com/dscripka/openWakeWord?tab=readme-ov-file#training-new-models) for information about how to train your own OpenWakeWord models. You can use a [simple Google Colab notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing) for a start or use a [more detailed notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb) that enables more customization (can produce high quality models, but requires more development experience).

### Convert model to ONNX format

You might need to use tf2onnx to convert tensorflow tflite models to onnx format:

```bash
pip install -U tf2onnx
python -m tf2onnx.convert --tflite my_model_filename.tflite --output my_model_filename.onnx
```

### Configure RealtimeSTT

Suggested starting parameters for OpenWakeWord usage:
```python
    with AudioToTextRecorder(
        wakeword_backend="oww",
        wake_words_sensitivity=0.35,
        openwakeword_model_paths="word1.onnx,word2.onnx",
        wake_word_buffer_duration=1,
        ) as recorder:
```

## FAQ

### Q: I encountered the following error: "Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so} Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor." How do I fix this?

**A:** This issue arises from a mismatch between the version of `ctranslate2` and cuDNN. The `ctranslate2` library was updated to version 4.5.0, which uses cuDNN 9.2. There are two ways to resolve this issue:
1. **Downgrade `ctranslate2` to version 4.4.0**:
   ```bash
   pip install ctranslate2==4.4.0
   ```
2. **Upgrade cuDNN** on your system to version 9.2 or above.

## Contribution

Contributions are always welcome! 

Shoutout to [Steven Linn](https://github.com/stevenlafl) for providing docker support. 

## License

[MIT](https://github.com/KoljaB/RealtimeSTT?tab=MIT-1-ov-file)

## Author

Kolja Beigel  
Email: kolja.beigel@web.de  
[GitHub](https://github.com/KoljaB/RealtimeSTT)
