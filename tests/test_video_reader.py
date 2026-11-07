import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.video_reader import (
    VideoProcessingError,
    extract_supported_video_url,
    parse_vtt_transcript,
    prepare_video,
)


class VideoReaderTests(unittest.TestCase):
    def test_extracts_supported_video_urls_only(self):
        self.assertEqual(
            extract_supported_video_url(
                "請看 https://www.bilibili.com/video/BV123abc。"
            ),
            "https://www.bilibili.com/video/BV123abc",
        )
        self.assertEqual(
            extract_supported_video_url("https://youtu.be/abcd1234"),
            "https://youtu.be/abcd1234",
        )
        for url in (
            "http://www.youtube.com/watch?v=abc",
            "https://youtube.com.example.org/watch?v=abc",
            "https://example.com/video",
        ):
            with self.subTest(url=url):
                self.assertIsNone(extract_supported_video_url(url))

    def test_parses_vtt_cues_and_removes_duplicate_lines(self):
        transcript = parse_vtt_transcript(
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\n"
            "<v speaker>你好 &amp; 世界</v>\n\n"
            "00:02.000 --> 00:03.000\n"
            "你好 &amp; 世界\n\n"
            "00:00:04.000 --> 00:00:05.000\n"
            "下一句\n"
        )

        self.assertEqual(transcript, "你好 & 世界\n下一句")

    def test_rejects_unsupported_urls_before_downloading(self):
        with self.assertRaises(VideoProcessingError) as context:
            with prepare_video("https://example.com/video", "ffmpeg.exe"):
                pass

        self.assertEqual(context.exception.code, "unsupported_video_url")

    def test_rejects_overlong_videos_without_downloading(self):
        class FakeYoutubeDL:
            def __init__(self, _options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                self.assert_download_false = not download
                return {"duration": 601, "title": "Long video"}

            def download(self, _urls):
                raise AssertionError("overlong video must not be downloaded")

        fake_yt_dlp = types.SimpleNamespace(
            YoutubeDL=FakeYoutubeDL,
            utils=types.SimpleNamespace(DownloadError=RuntimeError),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            ffmpeg_path = Path(temp_dir) / "ffmpeg.exe"
            ffmpeg_path.touch()
            with patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}):
                with self.assertRaises(VideoProcessingError) as context:
                    with prepare_video(
                        "https://youtu.be/abcd1234",
                        ffmpeg_path,
                    ):
                        pass

        self.assertEqual(context.exception.code, "video_too_long")

    def test_rejects_downloads_over_the_size_limit(self):
        class FakeYoutubeDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                return {"duration": 60, "title": "Large video"}

            def download(self, _urls):
                try:
                    self.options["progress_hooks"][0](
                        {
                            "status": "downloading",
                            "downloaded_bytes": 100 * 1024 * 1024 + 1,
                            "filename": "large.mp4",
                        }
                    )
                except RuntimeError:
                    return 1
                return 0

        fake_yt_dlp = types.SimpleNamespace(
            YoutubeDL=FakeYoutubeDL,
            utils=types.SimpleNamespace(DownloadError=RuntimeError),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            ffmpeg_path = Path(temp_dir) / "ffmpeg.exe"
            ffmpeg_path.touch()
            with patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}):
                with self.assertRaises(VideoProcessingError) as context:
                    with prepare_video(
                        "https://youtu.be/abcd1234",
                        ffmpeg_path,
                    ):
                        pass

        self.assertEqual(context.exception.code, "video_download_limit_exceeded")

    def test_extracts_frames_and_uses_audio_when_captions_are_missing(self):
        class FakeYoutubeDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def extract_info(self, _url, download):
                return {"duration": 60, "title": "Test video"}

            def download(self, _urls):
                output_path = Path(self.options["outtmpl"].replace("%(id)s.%(ext)s", "video.mp4"))
                output_path.write_bytes(b"video")

        fake_yt_dlp = types.SimpleNamespace(
            YoutubeDL=FakeYoutubeDL,
            utils=types.SimpleNamespace(DownloadError=RuntimeError),
        )

        def fake_ffmpeg(_ffmpeg, arguments, timeout=30):
            output_path = Path(arguments[-1])
            output_path.write_bytes(b"media")

        with tempfile.TemporaryDirectory() as temp_dir:
            ffmpeg_path = Path(temp_dir) / "ffmpeg.exe"
            ffmpeg_path.touch()
            with patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}):
                with patch(
                    "backend.video_reader._run_ffmpeg",
                    side_effect=fake_ffmpeg,
                ):
                    with prepare_video(
                        "https://youtu.be/abcd1234",
                        ffmpeg_path,
                    ) as video:
                        self.assertEqual(video.title, "Test video")
                        self.assertEqual(len(video.frames), 2)
                        self.assertTrue(all(frame.data == b"media" for frame in video.frames))
                        self.assertIsNotNone(video.audio_path)
                        self.assertTrue(video.audio_path.is_file())


if __name__ == "__main__":
    unittest.main()
