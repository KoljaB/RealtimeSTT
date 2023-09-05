@echo off
cd /d %~dp0

REM Check if the venv directory exists
if not exist test_env\Scripts\python.exe (
    echo Creating VENV
    python -m venv test_env
) else (
    echo VENV already exists
)

echo Activating VENV
start cmd /k "call test_env\Scripts\activate.bat && install_with_gpu_support.bat"
