import os
import re

import setuptools
from setuptools.command.build_py import build_py as _build_py


current_version = "1.0.1"


INSTALL_GUIDE = """
RealtimeSTT lets you choose the transcription and wake-word dependencies you
want to install.

Recommended default local Whisper install:

    pip install "realtimestt[recommended]"

Main ASR backend only, without the faster packaged Silero ONNX Runtime VAD:

    pip install "realtimestt[faster-whisper]"

Core package only, without a transcription engine or wake-word backend:

    pip install realtimestt

Install multiple extras by separating them with commas:

    pip install "realtimestt[faster-whisper,porcupine]"
    pip install "realtimestt[whisper-cpp,openwakeword]"

Available extras include:

- faster-whisper: default CTranslate2 Whisper backend
- whisper-cpp: whisper.cpp backend through pywhispercpp
- openai-whisper: original OpenAI Whisper Python backend
- sherpa-onnx: sherpa-onnx CPU backends
- silero-vad: packaged Silero model assets and PyTorch wrapper
- silero-onnx/silero-onnx-cpu: fastest Silero VAD CPU ONNX Runtime backend
- silero-onnx-gpu: installs Silero's ONNX GPU runtime extra for experiments
- parakeet: NVIDIA NeMo Parakeet backend
- transformers: shared Transformers dependency for Moonshine, Granite, and Cohere
- moonshine, granite, cohere: aliases for the Transformers dependency set
- qwen: Qwen ASR backend
- qwen-vllm: Qwen ASR with vLLM extras
- kroko-builder: helper command for building/installing Kroko-ONNX
- porcupine: Porcupine wake-word backend
- openwakeword: OpenWakeWord wake-word backend
- wakewords: both wake-word backends
- recommended/default: faster-whisper backend plus fast Silero CPU ONNX VAD
- all: all PyPI-installable optional backends

The WebRTC VAD and baseline Silero/PyTorch dependencies are still part of the
core install because AudioToTextRecorder initializes both VAD paths. Install
the recommended/default or silero-onnx extra for the faster raw CPU ONNX
Runtime Silero backend.

For live Kroko-ONNX usage, install the builder helper and then build Kroko in
the same Python environment:

    pip install "realtimestt[kroko-builder]"
    stt-install-kroko --build

"""

# Get the absolute path of requirements.txt
req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

def parse_requirements(filename):
    parsed = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            package = re.split(
                r"\s*(?:===|==|>=|<=|~=|!=|>|<|;)",
                line,
                maxsplit=1,
            )[0].strip()
            parsed[package] = line
    return parsed


def requirement(name, fallback=None):
    return requirements.get(name, fallback or name)


def unique_requirements(items):
    seen = set()
    unique = []
    for item in items:
        normalized = item.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(item)
    return unique


def is_local_backup_file(path):
    filename = os.path.basename(path)
    return " - Kopie" in filename or filename.endswith((".bak", ".tmp"))


class build_py(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, module, path)
            for pkg, module, path in modules
            if not is_local_backup_file(path)
        ]


requirements = parse_requirements(req_path)

base_requirements = [
    requirement("PyAudio"),
    requirement("webrtcvad-wheels"),
    requirement("halo"),
    requirement("torch"),
    requirement("torchaudio"),
    requirement("scipy"),
    requirement("websockets"),
    requirement("websocket-client"),
    requirement("soundfile"),
]

faster_whisper_requirements = [requirement("faster-whisper")]
whisper_cpp_requirements = ["pywhispercpp"]
openai_whisper_requirements = ["openai-whisper"]
sherpa_onnx_requirements = ["sherpa-onnx"]
silero_vad_requirements = [
    "silero-vad>=6.2.1; python_version >= '3.8'",
]
silero_onnx_requirements = [
    "silero-vad[onnx-cpu]>=6.2.1; python_version >= '3.8'",
]
silero_onnx_gpu_requirements = [
    "silero-vad[onnx-gpu]>=6.2.1; python_version >= '3.8'",
]
transformers_requirements = ["transformers"]
parakeet_requirements = ["nemo_toolkit[asr]"]
qwen_requirements = ["qwen-asr"]
qwen_vllm_requirements = ["qwen-asr[vllm]"]
kroko_builder_requirements = []
porcupine_requirements = [requirement("pvporcupine")]
openwakeword_requirements = [requirement("openwakeword")]

all_optional_requirements = unique_requirements(
    faster_whisper_requirements
    + whisper_cpp_requirements
    + openai_whisper_requirements
    + sherpa_onnx_requirements
    + silero_onnx_requirements
    + transformers_requirements
    + parakeet_requirements
    + qwen_requirements
    + porcupine_requirements
    + openwakeword_requirements
)

extras_require = {
    "minimal": [],
    "faster-whisper": faster_whisper_requirements,
    "whisper-cpp": whisper_cpp_requirements,
    "whispercpp": whisper_cpp_requirements,
    "openai-whisper": openai_whisper_requirements,
    "sherpa-onnx": sherpa_onnx_requirements,
    "sherpa": sherpa_onnx_requirements,
    "silero-vad": silero_vad_requirements,
    "silero": silero_vad_requirements,
    "silero-onnx": silero_onnx_requirements,
    "silero-onnx-cpu": silero_onnx_requirements,
    "vad-onnx": silero_onnx_requirements,
    "silero-onnx-gpu": silero_onnx_gpu_requirements,
    "vad-onnx-gpu": silero_onnx_gpu_requirements,
    "transformers": transformers_requirements,
    "moonshine": transformers_requirements,
    "granite": transformers_requirements,
    "cohere": transformers_requirements,
    "parakeet": parakeet_requirements,
    "nvidia-parakeet": parakeet_requirements,
    "qwen": qwen_requirements,
    "qwen3-asr": qwen_requirements,
    "qwen-vllm": qwen_vllm_requirements,
    "kroko-builder": kroko_builder_requirements,
    "porcupine": porcupine_requirements,
    "pvporcupine": porcupine_requirements,
    "pvp": porcupine_requirements,
    "openwakeword": openwakeword_requirements,
    "oww": openwakeword_requirements,
    "wakewords": unique_requirements(
        porcupine_requirements + openwakeword_requirements
    ),
    "wake-words": unique_requirements(
        porcupine_requirements + openwakeword_requirements
    ),
    "recommended": unique_requirements(
        faster_whisper_requirements + silero_onnx_requirements
    ),
    "default": unique_requirements(
        faster_whisper_requirements + silero_onnx_requirements
    ),
    "all": all_optional_requirements,
}

# Read README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = INSTALL_GUIDE + long_description

setuptools.setup(
    name="realtimestt",
    version=current_version,
    author="Kolja Beigel",
    author_email="kolja.beigel@web.de",
    description="A fast Voice Activity Detection and Transcription System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KoljaB/RealTimeSTT",
    packages=setuptools.find_packages(
        include=[
            "RealtimeSTT",
            "RealtimeSTT.*",
            "RealtimeSTT_server",
            "RealtimeSTT_server.*",
        ]
    ),
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "Operating System :: OS Independent",
    # ],
    python_requires='>=3.6',
    license='MIT',
    install_requires=base_requirements,
    extras_require=extras_require,
    keywords="real-time, audio, transcription, speech-to-text, voice-activity-detection, VAD, real-time-transcription, ambient-noise-detection, microphone-input, faster_whisper, speech-recognition, voice-assistants, audio-processing, buffered-transcription, pyaudio, ambient-noise-level, voice-deactivity",
    package_data={"RealtimeSTT": ["warmup_audio.wav"]},
    include_package_data=True,
    cmdclass={"build_py": build_py},
    entry_points={
        'console_scripts': [
            'stt-server=RealtimeSTT_server.stt_server:main',
            'stt=RealtimeSTT_server.stt_cli_client:main',
            'stt-install-kroko=RealtimeSTT.install_kroko:main',
        ],
    },
)
