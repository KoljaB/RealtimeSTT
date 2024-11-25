import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="RealtimeSTT",
    version="0.3.81",
    author="Kolja Beigel",
    author_email="kolja.beigel@web.de",
    description="A fast Voice Activity Detection and Transcription System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KoljaB/RealTimeSTT",
    packages=setuptools.find_packages(include=["RealtimeSTT", "RealtimeSTT_server"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    keywords="real-time, audio, transcription, speech-to-text, voice-activity-detection, VAD, real-time-transcription, ambient-noise-detection, microphone-input, faster_whisper, speech-recognition, voice-assistants, audio-processing, buffered-transcription, pyaudio, ambient-noise-level, voice-deactivity",
    entry_points={
        'console_scripts': [
            'stt-server=RealtimeSTT_server.stt_server:main',
            'stt=RealtimeSTT_server.stt_cli_client:main',
        ],
    },
)
