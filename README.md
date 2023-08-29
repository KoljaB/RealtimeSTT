# RealTimeSTT

A fast Voice Activity Detection and Transcription System

Listens to microphone, detects voice activity and immediately transcribes it using the `faster_whisper` model. Adapts to various environments with a ambient noise level-based voice activity detection.

Ideal for applications like voice assistants or any application where immediate speech-to-text conversion is desired with minimal latency.

## Features

1. **Voice Activity Detection**: Automatically starts/stops recording when speech is detected or when speech ends.
2. **Wake Word Detection**: Starts recording when a specified wake word (or words) is detected.
3. **Buffer Management**: Handles short and long term audio buffers for efficient processing.
4. **Event Callbacks**: Customizable callbacks for when recording starts or finishes.
5. **Noise Level Calculation**: Adjusts based on the background noise for more accurate voice activity detection.
6. **Error Handling**: Comprehensive error handling to catch and report any anomalies during execution.

## Installation

```bash
pip install RealTimeSTT
```

## GPU Support

To significantly improve transcription speed, especially in real-time applications, we **strongly recommend** utilizing GPU acceleration via CUDA. By default, the transcription is performed on the CPU. 

1. **Install NVIDIA CUDA Toolkit 11.8**:
	- Visit [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
	- Select version 11.
	- Download and install the software.

2. **Install NVIDIA cuDNN 8.7.0 for CUDA 11.x**:
	- Visit [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
	- Click on "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x".
	- Download and install the software.
	
3. **Reconfigure PyTorch for CUDA**:
	- If you have PyTorch installed, remove it: `pip uninstall torch`.
	- Install PyTorch again with CUDA support: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`.

Note: To check if your NVIDIA GPU supports CUDA, visit the [official CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).

## Usage

### Automatic Recording

```python
print(AudioToTextRecorder().text())
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

## Configuration

The class comes with numerous configurable parameters such as buffer size, activity thresholds, and smoothing factors to fine-tune the recording and transcription process based on the specific needs of your application:

* `model`: Specifies the size of the transcription model to use or the path to a converted model directory. Valid options are 'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2'. If a specific size is provided, the model is downloaded from the Hugging Face Hub.

* `language`: Defines the language code for the speech-to-text engine. If not specified, the model will attempt to detect the language automatically.

* `wake_words`: A comma-separated string of wake words to initiate recording. Supported wake words include 'alexa', 'americano', 'blueberry', 'bumblebee', 'computer', 'grapefruits', 'grasshopper', 'hey google', 'hey siri', 'jarvis', 'ok google', 'picovoice', 'porcupine', 'terminator'.

* `wake_words_sensitivity`: Determines the sensitivity for wake word detection, ranging from 0 (least sensitive) to 1 (most sensitive). The default value is 0.5.

* `on_recording_started`: A callable option which is invoked when the recording starts.

* `on_recording_finished`: A callable option invoked when the recording ends.

* `min_recording_interval`: Specifies the minimum interval (in seconds) for recording durations.

* `interval_between_records`: Determines the interval (in seconds) between consecutive recordings.

* `buffer_duration`: Indicates the duration (in seconds) to maintain pre-roll audio in the buffer.

* `voice_activity_threshold`: The threshold level above the long-term noise to detect the start of voice activity.

* `voice_deactivity_sensitivity`: Sensitivity level for voice deactivation detection, ranging from 0 (least sensitive) to 1 (most sensitive). The default value is 0.3.

* `voice_deactivity_silence_after_speech_end`: Duration (in seconds) of silence required after speech ends to trigger voice deactivation. The default is 0.1 seconds.

* `long_term_smoothing_factor`: Exponential smoothing factor utilized in calculating the long-term noise level.

* `short_term_smoothing_factor`: Exponential smoothing factor for calculating the short-term noise level.

* `level`: Sets the desired logging level for internal logging. Default is `logging.WARNING`.

## Contribution

Contributions are always welcome! 

## License

MIT

## Author

Kolja Beigel  
Email: kolja.beigel@web.de  
[GitHub](https://github.com/KoljaB/RealTimeSTT)

---