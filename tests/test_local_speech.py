import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from desktop import local_speech


class LocalSpeechTests(unittest.TestCase):
    def setUp(self):
        local_speech._MODEL = None

    def tearDown(self):
        local_speech._MODEL = None

    def test_empty_audio_does_not_load_model(self):
        with patch.dict(sys.modules, {"faster_whisper": None}):
            self.assertEqual(
                local_speech.transcribe_chinese(np.array([], dtype=np.float32)),
                "",
            )

    def test_transcribes_traditional_chinese_with_domain_prompt(self):
        model = Mock()
        model.transcribe.return_value = (
            iter(
                [
                    SimpleNamespace(text=" 你好， "),
                    SimpleNamespace(text=" 世界。 "),
                ]
            ),
            SimpleNamespace(language="zh"),
        )
        model_class = Mock(return_value=model)
        faster_whisper = types.SimpleNamespace(WhisperModel=model_class)
        ctranslate2 = types.SimpleNamespace(get_cuda_device_count=Mock(return_value=1))

        with patch.dict(
            sys.modules,
            {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2},
        ):
            result = local_speech.transcribe_chinese(
                np.ones(16_000, dtype=np.float32)
            )

        self.assertEqual(result, "你好，世界。")
        model_class.assert_called_once_with(
            (
                str(local_speech._MODEL_DIRECTORY)
                if local_speech._MODEL_DIRECTORY.is_dir()
                else "small"
            ),
            device="cuda",
            compute_type="float16",
            download_root=str(local_speech._MODEL_CACHE_DIR),
        )
        model.transcribe.assert_called_once()
        self.assertEqual(
            model.transcribe.call_args.kwargs,
            {
                "language": "zh",
                "vad_filter": True,
                "beam_size": 5,
                "condition_on_previous_text": False,
                "initial_prompt": local_speech._INITIAL_PROMPT,
            },
        )
        self.assertIn("牧瀬紅莉栖", local_speech._INITIAL_PROMPT)
        self.assertIn("Gemini", local_speech._INITIAL_PROMPT)
        self.assertNotIn("請依照錄音原話", local_speech._INITIAL_PROMPT)
        self.assertNotIn("逐字轉錄", local_speech._INITIAL_PROMPT)

    def test_falls_back_to_cpu_when_cuda_is_unavailable(self):
        model = Mock()
        model.transcribe.return_value = (iter([]), SimpleNamespace(language="zh"))
        model_class = Mock(return_value=model)
        faster_whisper = types.SimpleNamespace(WhisperModel=model_class)
        ctranslate2 = types.SimpleNamespace(get_cuda_device_count=Mock(return_value=0))

        with patch.dict(
            sys.modules,
            {"faster_whisper": faster_whisper, "ctranslate2": ctranslate2},
        ):
            local_speech.transcribe_chinese(np.ones(16_000, dtype=np.float32))

        model_class.assert_called_once_with(
            (
                str(local_speech._MODEL_DIRECTORY)
                if local_speech._MODEL_DIRECTORY.is_dir()
                else "small"
            ),
            device="cpu",
            compute_type="int8",
            download_root=str(local_speech._MODEL_CACHE_DIR),
        )


if __name__ == "__main__":
    unittest.main()
