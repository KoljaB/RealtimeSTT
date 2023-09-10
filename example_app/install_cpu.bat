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
start cmd /k "call test_env\Scripts\activate.bat && pip install --upgrade RealtimeSTT==0.1.4 && pip install --upgrade RealtimeTTS==0.1.3 && pip install pysoundfile==0.9.0.post1 openai==0.27.8 keyboard==0.13.5 PyQt5==5.15.9 sounddevice==0.4.6 wavio==0.0.7"