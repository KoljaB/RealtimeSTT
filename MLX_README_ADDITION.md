# MLX-Whisper Backend Support (Apple Silicon)

## Overview

RealtimeSTT now supports **Apple Silicon acceleration** through the MLX-Whisper backend! This enables native M1/M2/M3/M4 chip support without requiring CUDA or GPU drivers.

### Key Benefits

✅ **Native Apple Silicon Support** - Runs directly on M1/M2/M3/M4 chips  
✅ **No CUDA Required** - Works out of the box on macOS  
✅ **Unified Memory Architecture** - Efficient memory usage on Apple Silicon  
✅ **Near Real-Time Performance** - RTF ~1.07x with tiny model  
✅ **Drop-in Replacement** - Compatible API with faster-whisper backend  

## Installation

```bash
# Install MLX-Whisper
pip install mlx-whisper

# Or install all requirements (includes MLX for macOS)
pip install -r requirements.txt
```

## Quick Start

```python
from RealtimeSTT import AudioToTextRecorder

# Initialize with MLX backend for Apple Silicon
recorder = AudioToTextRecorder(
    model="tiny",           # or "base", "small", "medium", etc.
    backend="mlx-whisper",  # Enable MLX backend
    language="en"           # Optional: specify language
)

# Use as normal
with recorder:
    print(recorder.text())
```

## Backend Comparison

| Feature | faster-whisper | MLX-Whisper |
|---------|---------------|-------------|
| **Platform** | CUDA GPUs / CPU | Apple Silicon (M1/M2/M3/M4) |
| **Installation** | Requires CUDA drivers | Works out of the box on macOS |
| **Performance (tiny)** | RTF ~1.07x (CPU) | RTF ~1.07x (Apple Silicon) |
| **Memory** | GPU VRAM / System RAM | Unified Memory |
| **Batch Processing** | ✅ Supported | ❌ Not supported (yet) |

## Performance Benchmarks

Tested on **Apple M2** with various audio files:

| Audio | Model | Backend | RTF | Notes |
|-------|-------|---------|-----|-------|
| 6.6s | tiny | MLX | 1.80x | Short audio |
| 167s | tiny | MLX | **1.07x** | Long audio (near real-time!) |
| 6.6s | tiny | CPU | 1.81x | Similar to MLX |
| 167s | tiny | CPU | 1.07x | Similar to MLX |

**RTF < 1.0 = Faster than real-time** (ideal for live transcription)

## Supported Models

All standard Whisper models are supported:

- `tiny` (39M params, ~150MB)
- `base` (74M params, ~280MB)
- `small` (244M params, ~900MB)  
- `medium` (769M params, ~2.8GB)
- `large-v3` (1.5B params, ~5.6GB)
- `large-v3-turbo` (809M params, ~3.0GB) - **Recommended for production**

Models are automatically downloaded from HuggingFace and cached locally.

## Migration from faster-whisper

**Before:**
```python
recorder = AudioToTextRecorder(
    model="tiny",
    device="cuda",
    compute_type="float16"
)
```

**After:**
```python
recorder = AudioToTextRecorder(
    model="tiny",
    backend="mlx-whisper"  # That's it!
)
```

## Documentation

For complete documentation, see [MLX_SUPPORT.md](MLX_SUPPORT.md)

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python** 3.8+
- **mlx-whisper** >= 0.4.3

---

*MLX-Whisper backend enables efficient, native speech-to-text on Apple Silicon without CUDA dependencies.*
