import ipaddress
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _configured_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


_LOCAL_APP_DATA = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))

DATA_DIR = _configured_path("AMADEUS_DATA_DIR", _LOCAL_APP_DATA / "Amadeus")
GPT_SOVITS_DIR = _configured_path(
    "GPT_SOVITS_DIR",
    ROOT_DIR / "GPT-SoVITS-v2pro-20250604-nvidia50",
)
LIVE2D_MODEL_PATH = _configured_path(
    "LIVE2D_MODEL_PATH",
    ROOT_DIR / "Resources" / "kurisu_cubism3" / "kurisu.model3.json",
)
TTS_REFERENCE_AUDIO = _configured_path(
    "TTS_REFERENCE_AUDIO",
    GPT_SOVITS_DIR / "Voice" / "Ref" / "ref.wav",
)
TTS_OUTPUT_DIR = _configured_path(
    "TTS_OUTPUT_DIR",
    GPT_SOVITS_DIR / "Voice" / "Out",
)
SCREENSHOT_DEBUG_PATH = DATA_DIR / "screenshots" / "Debug.png"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_MODEL_FALLBACK = os.getenv(
    "GEMINI_MODEL_FALLBACK",
    (
        "gemini-3.1-flash-lite"
        if GEMINI_MODEL == "gemini-3.5-flash-lite"
        else "gemini-3.5-flash-lite"
    ),
)
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-2",
)
GEMINI_EMBEDDING_MODEL_FALLBACK = os.getenv(
    "GEMINI_EMBEDDING_MODEL_FALLBACK",
    (
        "gemini-embedding-001"
        if GEMINI_EMBEDDING_MODEL == "gemini-embedding-2"
        else "gemini-embedding-2"
    ),
)
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BACKEND_HOST = os.getenv("AMADEUS_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("AMADEUS_PORT", "5000"))
BACKEND_BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
WEB_ACCESS_CODE = os.getenv("AMADEUS_WEB_ACCESS_CODE", "").strip()
if WEB_ACCESS_CODE and len(WEB_ACCESS_CODE) < 16:
    raise ValueError("AMADEUS_WEB_ACCESS_CODE must be at least 16 characters.")

try:
    _backend_host_is_loopback = ipaddress.ip_address(BACKEND_HOST).is_loopback
except ValueError:
    _backend_host_is_loopback = BACKEND_HOST.lower() == "localhost"
if not WEB_ACCESS_CODE and not _backend_host_is_loopback:
    raise ValueError(
        "Set AMADEUS_WEB_ACCESS_CODE before binding the backend beyond localhost."
    )
BACKEND_BIND_HOST = "0.0.0.0" if WEB_ACCESS_CODE else BACKEND_HOST
BACKEND_REQUEST_TIMEOUT = int(os.getenv("AMADEUS_REQUEST_TIMEOUT", "180"))
PROACTIVE_IDLE_MINUTES = int(os.getenv("AMADEUS_PROACTIVE_IDLE_MINUTES", "120"))
PROACTIVE_REQUEST_TIMEOUT_SECONDS = 90
TTS_API_URL = os.getenv("TTS_API_URL", "http://127.0.0.1:9880")
PUSH_TO_TALK_KEY = (
    os.getenv("AMADEUS_PUSH_TO_TALK_KEY", "mouse_x2").strip().lower()
)
