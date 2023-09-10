@echo off
cd /d %~dp0

REM Check if the venv directory exists
if not exist test_env\Scripts\python.exe (
    echo Creating VENV
    python -m venv test_env
) else (
    echo VENV already exists
)


:: OpenAI API Key  https://platform.openai.com/
set OPENAI_API_KEY=

:: Microsoft Azure API Key  https://portal.azure.com/
set AZURE_SPEECH_KEY=

:: Elevenlabs API Key  https://www.elevenlabs.io/Elevenlabs
set ELEVENLABS_API_KEY=


echo Activating VENV
start cmd /k "call test_env\Scripts\activate.bat && python ui_openai_voice_interface.py"