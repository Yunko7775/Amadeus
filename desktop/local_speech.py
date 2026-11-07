import re
import threading

import numpy as np

from shared.settings import ROOT_DIR


_MODEL_LOCK = threading.Lock()
_MODEL = None
_MODEL_CACHE_DIR = ROOT_DIR / "Resources" / "models"
_MODEL_DIRECTORY = _MODEL_CACHE_DIR / "faster-whisper-small"
_INITIAL_PROMPT = ""


def transcribe_chinese(audio: np.ndarray) -> str:
    global _MODEL

    if audio.ndim != 1 or audio.size == 0:
        return ""
    with _MODEL_LOCK:
        if _MODEL is None:
            import ctranslate2
            from faster_whisper import WhisperModel

            model_path = (
                str(_MODEL_DIRECTORY) if _MODEL_DIRECTORY.is_dir() else "small"
            )
            has_cuda = ctranslate2.get_cuda_device_count() > 0
            _MODEL = WhisperModel(
                model_path,
                device="cuda" if has_cuda else "cpu",
                compute_type="float16" if has_cuda else "int8",
                download_root=str(_MODEL_CACHE_DIR),
            )
        segments, _info = _MODEL.transcribe(
            audio,
            language="zh",
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
            initial_prompt=_INITIAL_PROMPT,
        )
        text = "".join(segment.text for segment in segments)
    return re.sub(r"\s+", "", text).strip()
