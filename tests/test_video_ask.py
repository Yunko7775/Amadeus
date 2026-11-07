import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shared import config

config.API_KEY = "test-key"

from backend import flask_backend
from backend.video_reader import PreparedVideo, VideoFrame, VideoProcessingError


class VideoAskTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        flask_backend.conversation_history.clear()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _prepared_video(self, video):
        @contextmanager
        def fake_prepare_video(_url, _ffmpeg_path):
            yield video

        return fake_prepare_video

    def _post_with_video(self, video):
        with patch.object(flask_backend, "prepare_video", self._prepared_video(video)):
            with patch.object(
                flask_backend,
                "build_full_prompt",
                return_value="BASE PROMPT",
            ):
                with patch.object(flask_backend, "update_conversation_history"):
                    with patch.object(
                        flask_backend,
                        "translate_to_japanese",
                        return_value="日本語。",
                    ):
                        with patch.object(
                            flask_backend,
                            "_schedule_memory_consolidation",
                        ):
                            with patch.object(flask_backend.threading, "Thread"):
                                with patch.object(
                                    flask_backend.client.models,
                                    "generate_content",
                                    return_value=SimpleNamespace(
                                        text=(
                                            "【中文回覆】\n"
                                            "影片內容。（neutral）\n"
                                            "【日文語音稿】\n"
                                            "動画の内容です。"
                                        )
                                    ),
                                ) as generate_content:
                                    with patch.object(
                                        flask_backend.client.interactions,
                                        "create",
                                    ) as interaction:
                                        response = flask_backend.app.test_client().post(
                                            "/ask",
                                            json={
                                                "user_id": "video-test",
                                                "message": (
                                                    "幫我看看這個影片 "
                                                    "https://youtu.be/abcd1234"
                                                ),
                                            },
                                        )
        return response, generate_content, interaction

    def test_video_request_sends_subtitles_and_sampled_frames_to_gemini(self):
        video = PreparedVideo(
            title="A public video",
            transcript="字幕內容",
            frames=[
                VideoFrame(timestamp="00:00:20", data=b"jpeg-frame"),
            ],
            audio_path=None,
        )

        response, generate_content, interaction = self._post_with_video(video)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["sources"],
            [{"title": "A public video", "url": "https://youtu.be/abcd1234"}],
        )
        interaction.assert_not_called()
        generate_content.assert_called_once()
        parts = generate_content.call_args.kwargs["contents"][0].parts
        self.assertIn("字幕內容", parts[0].text)
        self.assertEqual(parts[1].text, "影片畫面時間：00:00:20")
        self.assertEqual(parts[2].inline_data.data, b"jpeg-frame")
        self.assertEqual(parts[2].inline_data.mime_type, "image/jpeg")

    def test_audio_is_sent_when_video_has_no_captions(self):
        audio_path = Path(self.temp_dir.name) / "audio.mp3"
        audio_path.write_bytes(b"compressed-audio")
        video = PreparedVideo(
            title="A video without captions",
            transcript="",
            frames=[VideoFrame(timestamp="00:00:20", data=b"jpeg-frame")],
            audio_path=audio_path,
        )

        response, generate_content, _interaction = self._post_with_video(video)

        self.assertEqual(response.status_code, 200)
        parts = generate_content.call_args.kwargs["contents"][0].parts
        audio_part = parts[-1]
        self.assertEqual(audio_part.inline_data.data, b"compressed-audio")
        self.assertEqual(audio_part.inline_data.mime_type, "audio/mpeg")

    def test_video_processing_errors_are_reported_without_model_fallback(self):
        @contextmanager
        def fail_prepare_video(_url, _ffmpeg_path):
            raise VideoProcessingError(
                "video_too_long",
                "目前一次最多分析 10 分鐘的影片。",
            )
            yield

        with patch.object(flask_backend, "prepare_video", fail_prepare_video):
            with patch.object(
                flask_backend,
                "build_full_prompt",
                return_value="BASE PROMPT",
            ):
                with patch.object(flask_backend.client.models, "generate_content") as generate:
                    response = flask_backend.app.test_client().post(
                        "/ask",
                        json={
                            "user_id": "video-test",
                            "message": "https://youtu.be/abcd1234",
                        },
                    )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"], "video_too_long")
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
