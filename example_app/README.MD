# GPU Support with CUDA (recommended)

Steps for a **GPU-optimized** installation:

1. **Install NVIDIA CUDA Toolkit 11.8**:
    - Visit [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Select version 11.
    - Download and install the software.

2. **Install NVIDIA cuDNN 8.7.0 for CUDA 11.x**:
    - Visit [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
    - Click on "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x".
    - Download and install the software.

3. **ElevenlabsEngine**
    - If you plan to use the `ElevenlabsEngine`, you need `mpv` is installed on your system for streaming mpeg audio

    - **macOS**:
    ```bash
    brew install mpv
    ```

    - **Linux and Windows**: Visit [mpv.io](https://mpv.io/) for installation instructions.

4. **Install PyTorch with CUDA support**:
    - run install_gpu.bat

5. **Configure script**
    - open VoiceApp.py and configure your engine, set API keys, Azure service region, language etc