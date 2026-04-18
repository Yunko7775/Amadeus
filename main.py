import subprocess
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv", "Scripts")

PYTHON = os.path.join(VENV_DIR, "python.exe")
PYTHONW = os.path.join(VENV_DIR, "pythonw.exe")

def start_console(cmd, cwd):
    print("Start:", cmd)
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def start_gui(cmd, cwd):
    print("Start GUI:", cmd)
    return subprocess.Popen(
        cmd,
        cwd=cwd
    )


# 1️⃣ gpt-soviet
GPT_SOVIET_DIR = os.path.join(BASE_DIR, "GPT-SoVITS-v2pro-20250604-nvidia50")
API_BAT = os.path.join(GPT_SOVIET_DIR, "api.bat")

start_console(["cmd", "/c", API_BAT], GPT_SOVIET_DIR)
time.sleep(2)

# 2️⃣ Flask
start_console([PYTHON, "flask_backend.py"], BASE_DIR)
time.sleep(2)

# 3️⃣ GUI
start_gui([PYTHONW, "Live2d_TK.py"], BASE_DIR)


