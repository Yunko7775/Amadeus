@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [ERROR] The project virtual environment was not found.
    echo Run the setup commands in README.md first.
    pause
    exit /b 1
)

"%PYTHON%" "%~dp0main.py"
if errorlevel 1 (
    echo.
    echo Amadeus failed to start. Review the error above.
    pause
    exit /b 1
)
