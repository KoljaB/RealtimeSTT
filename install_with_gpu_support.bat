pip uninstall torch
pip install torch==2.0.1+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt