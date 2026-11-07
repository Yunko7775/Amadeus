import json
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from shared.settings import (
    BACKEND_BASE_URL,
    BACKEND_HOST,
    BACKEND_PORT,
    GEMINI_API_KEY,
    GPT_SOVITS_DIR,
    ROOT_DIR,
)


VENV_DIR = ROOT_DIR / ".venv" / "Scripts"
PYTHON = VENV_DIR / "python.exe"
PYTHONW = VENV_DIR / "pythonw.exe"


def start_console(cmd, cwd):
    print("Start:", cmd)
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )


def start_gui(cmd, cwd):
    print("Start GUI:", cmd)
    return subprocess.Popen(
        cmd,
        cwd=str(cwd)
    )


def wait_for_backend(process, timeout=30):
    health_url = f"{BACKEND_BASE_URL}/health"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "The Flask backend exited before it became ready. "
                "Check the backend console for its error."
            )

        try:
            with urlopen(health_url, timeout=1) as response:
                status = json.load(response)
                if response.status == 200 and status.get("status") == "ok":
                    return
        except (OSError, TimeoutError, URLError, json.JSONDecodeError):
            time.sleep(0.5)

    raise RuntimeError(
        f"The Flask backend did not become ready within {timeout} seconds "
        f"at {BACKEND_HOST}:{BACKEND_PORT}. Check its console for errors."
    )


def main():
    if not GEMINI_API_KEY:
        raise SystemExit(
            "Gemini API key is not configured.\n"
            "1. Copy .env.example to .env in the project folder.\n"
            "2. Open .env and replace the example value with your key.\n"
            "3. Save the file and run main.py again."
        )

    api_bat = GPT_SOVITS_DIR / "api.bat"
    if not api_bat.is_file():
        raise SystemExit(
            f"GPT-SoVITS api.bat was not found:\n{api_bat}\n"
            "Check GPT_SOVITS_DIR in your .env file."
        )

    if not PYTHON.is_file() or not PYTHONW.is_file():
        raise SystemExit(
            "The project virtual environment was not found. Create it and "
            "install the dependencies from requirements.txt first."
        )

    start_console(["cmd", "/c", str(api_bat)], GPT_SOVITS_DIR)
    time.sleep(2)

    backend_process = start_console(
        [str(PYTHON), "-m", "backend.flask_backend"],
        ROOT_DIR,
    )
    try:
        wait_for_backend(backend_process)
    except RuntimeError as error:
        backend_process.terminate()
        raise SystemExit(str(error)) from error

    start_gui([str(PYTHONW), "-m", "desktop.desktop_app"], ROOT_DIR)


if __name__ == "__main__":
    main()
