# MLX-Whisper Support for Apple Silicon

This document describes the MLX-Whisper backend support added to RealtimeSTT, which enables native Apple Silicon (M1/M2/M3/M4) acceleration for speech-to-text transcription.

## What is MLX?

[MLX](https://github.com/ml-explore/mlx) is Apple's machine learning framework specifically designed for Apple Silicon. It provides efficient computation on Apple's unified memory architecture and Neural Engine, offering excellent performance for ML workloads on Mac computers.

## Features

- **Native Apple Silicon Support**: Runs natively on M1/M2/M3/M4 chips
- **Unified Memory Architecture**: Leverages Apple's efficient memory system
- **No CUDA Required**: Works without GPU drivers or CUDA installation
- **Drop-in Replacement**: Compatible API with the existing faster-whisper backend
- **Automatic Model Caching**: Models are downloaded once and cached locally

## Installation

### Basic Installation

```bash
pip install mlx-whisper
```

### From requirements

The MLX-Whisper dependency is included in `requirements.txt` for macOS only:

```bash
pip install -r requirements.txt
```

For a dedicated MLX setup:

```bash
pip install -r requirements-mlx.txt
```

## Usage

### Basic Example

```python
from RealtimeSTT import AudioToTextRecorder

# Initialize with MLX backend
recorder = AudioToTextRecorder(
    model="tiny",  # or "base", "small", "medium", "large-v3", etc.
    backend="mlx-whisper",  # Enable MLX backend
    language="en"  # Optional: specify language
)

# Use as normal
with recorder:
    print(recorder.text())
```

### Comparison with faster-whisper

```python
# faster-whisper backend (default)
recorder_fw = AudioToTextRecorder(
    model="tiny",
    backend="faster-whisper",  # or just omit this parameter
    device="cuda"  # Requires CUDA
)

# MLX-Whisper backend (Apple Silicon)
recorder_mlx = AudioToTextRecorder(
    model="tiny",
    backend="mlx-whisper",
    # No device parameter needed - runs on Apple Silicon automatically
)
```

## Supported Models

The following Whisper model sizes are supported with automatic translation to MLX-community models:

| Model Size | MLX Model Repository | Parameters | Memory |
|-----------|---------------------|------------|--------|
| `tiny` | `mlx-community/whisper-tiny` | 39M | ~150 MB |
| `tiny.en` | `mlx-community/whisper-tiny.en` | 39M | ~150 MB |
| `base` | `mlx-community/whisper-base` | 74M | ~280 MB |
| `base.en` | `mlx-community/whisper-base.en` | 74M | ~280 MB |
| `small` | `mlx-community/whisper-small` | 244M | ~900 MB |
| `small.en` | `mlx-community/whisper-small.en` | 244M | ~900 MB |
| `medium` | `mlx-community/whisper-medium` | 769M | ~2.8 GB |
| `medium.en` | `mlx-community/whisper-medium.en` | 769M | ~2.8 GB |
| `large-v1` | `mlx-community/whisper-large-v1` | 1550M | ~5.6 GB |
| `large-v2` | `mlx-community/whisper-large-v2` | 1550M | ~5.6 GB |
| `large-v3` | `mlx-community/whisper-large-v3` | 1550M | ~5.6 GB |
| `large-v3-turbo` | `mlx-community/whisper-large-v3-turbo` | 809M | ~3.0 GB |

## Performance

### Benchmarks (Apple M2)

Real-time streaming tests (simulating microphone input):

#### Short Audio (6.6s)
| Model | Backend | First Transcription | Result |
|-------|---------|---------------------|--------|
| tiny  | CPU (faster-whisper) | 4.5s | ✓ Correct |
| tiny  | GPU (mlx-whisper) | 4.3s | ✓ **0.2s faster** |

#### Multi-Sentence (19.6s, 5 sentences with 2.5s silence)

**Tiny Model:**
| Backend | Sentences | First | All Responses |
|---------|-----------|-------|---------------|
| CPU | 4/5 | 2.4s | 2.4s, 7.1s, 11.8s, 16.0s |
| MLX | 4/5 | 2.3s | 2.3s, 7.0s, 11.7s, 15.9s |

**Medium Model:**
| Backend | Sentences | First | Quality |
|---------|-----------|-------|---------|
| CPU | 3/5 ⚠️ | 7.2s | Missing sentences |
| MLX | 4/5 ✓ | 3.0s | **4.2s faster**, complete |

### Key Performance Notes

1. **Tiny model**: MLX marginally faster (0.1-0.2s per transcription)
2. **Medium model**: MLX shows **significant advantage** (4.2s faster, better quality)
3. **Heavier models benefit more** from GPU acceleration
4. **MLX maintains quality** while CPU backend drops sentences under load
5. **No CUDA required** - works out of the box on Apple Silicon Macs
6. **First run downloads models** (~150MB for tiny, ~3GB for large-v3-turbo)
7. **Subsequent runs use cached models** (instant loading)

## Technical Details

### Model Loading

MLX-Whisper models are automatically downloaded from HuggingFace Hub and cached locally. The backend translates standard Whisper model names to MLX-community repository paths:

```python
# When you specify:
model="tiny"

# MLX backend automatically uses:
path_or_hf_repo="mlx-community/whisper-tiny"
```

### Transcription Format

MLX transcription results are automatically converted to be compatible with faster-whisper's format:

```python
# MLX returns: {"text": "...", "language": "en"}
# Converted to: segments, info (same as faster-whisper)
```

### Multiprocessing Compatibility

The MLX backend is fully compatible with RealtimeSTT's multiprocessing architecture, using pickle-safe classes for inter-process communication.

## Limitations

1. **macOS Only**: MLX is designed for Apple Silicon and only works on macOS
2. **No Batching**: Unlike faster-whisper, MLX doesn't support batch processing in this integration
3. **No VAD Filter**: The `faster_whisper_vad_filter` parameter is ignored with MLX backend
4. **Beam Size**: The `beam_size` parameter may not have the same effect as with faster-whisper

## Troubleshooting

### Import Error

```
ImportError: MLX-Whisper is not installed
```

**Solution**: Install MLX-Whisper:
```bash
pip install mlx-whisper
```

### Model Download Error

```
RepositoryNotFoundError: 401 Client Error
```

**Solution**: This usually means the model name wasn't correctly translated. Use standard names like "tiny", "base", "small", etc.

### Slow Performance

**Possible causes**:
1. **First run**: Models are being downloaded (one-time)
2. **Large model**: Try a smaller model (tiny/base) first
3. **System load**: Close other applications for better performance

## Advanced Configuration

### Custom Model Paths

You can use custom MLX model repositories:

```python
recorder = AudioToTextRecorder(
    model="mlx-community/my-custom-whisper-model",
    backend="mlx-whisper"
)
```

### Language Detection

Leave `language` empty for automatic detection:

```python
recorder = AudioToTextRecorder(
    model="base",
    backend="mlx-whisper",
    language=""  # Auto-detect
)
```

### Memory Management

For systems with limited RAM, use smaller models:

```python
# Low memory (< 8GB RAM)
model="tiny"  # ~150 MB

# Medium memory (8-16GB RAM)  
model="base"  # ~280 MB
model="small"  # ~900 MB

# High memory (16GB+ RAM)
model="medium"  # ~2.8 GB
model="large-v3-turbo"  # ~3.0 GB
```

## Migration Guide

### From faster-whisper to MLX

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
    backend="mlx-whisper"
    # device and compute_type not needed
)
```

### Switching Between Backends

```python
# Use faster-whisper on systems with CUDA
if has_cuda:
    backend = "faster-whisper"
    device = "cuda"
else:
    # Use MLX on Apple Silicon
    backend = "mlx-whisper"
    device = None  # Not used with MLX

recorder = AudioToTextRecorder(
    model="base",
    backend=backend,
    device=device if backend == "faster-whisper" else None
)
```

## Future Improvements

Potential enhancements for MLX backend:

- [ ] Batch processing support
- [ ] VAD filter integration
- [ ] Custom beam search parameters
- [ ] Quantization support (int8, int4)
- [ ] Real-time transcription model optimization
- [ ] Benchmark suite for different Apple Silicon chips

## Contributing

Contributions to improve MLX support are welcome! Areas of interest:

1. Performance optimizations
2. Additional model support
3. Better error handling
4. Documentation improvements

## References

- [MLX Framework](https://github.com/ml-explore/mlx)
- [MLX-Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [Whisper by OpenAI](https://github.com/openai/whisper)
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)

## License

This MLX integration follows the same MIT license as RealtimeSTT.

---

**Note**: MLX is under active development. Features and performance may improve in future releases.
