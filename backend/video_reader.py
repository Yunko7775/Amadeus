import html
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse


MAX_VIDEO_DURATION_SECONDS = 10 * 60
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024
MAX_VIDEO_DOWNLOAD_SECONDS = 120
MAX_FRAME_COUNT = 6
MAX_TRANSCRIPT_CHARACTERS = 16_000
VIDEO_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
VTT_TIMESTAMP_PATTERN = re.compile(
    r"^(?:\d{1,2}:)?\d{2}:\d{2}\.\d{3}\s+-->\s+"
)


class VideoProcessingError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class VideoFrame:
    timestamp: str
    data: bytes


@dataclass(frozen=True)
class PreparedVideo:
    title: str
    transcript: str
    frames: list[VideoFrame]
    audio_path: Path | None


def extract_supported_video_url(message: str) -> str | None:
    for candidate in VIDEO_URL_PATTERN.findall(message):
        url = candidate.rstrip(".,;:!?)]}，。；：！？、")
        try:
            parsed = urlparse(url)
        except ValueError:
            continue
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme != "https" or not hostname:
            continue
        if (
            hostname in {"bilibili.com", "b23.tv", "youtube.com", "youtu.be"}
            or hostname.endswith(".bilibili.com")
            or hostname.endswith(".b23.tv")
            or hostname.endswith(".youtube.com")
        ):
            return url
    return None


def parse_vtt_transcript(contents: str) -> str:
    transcript_lines: list[str] = []
    for line in contents.splitlines():
        line = line.strip()
        if (
            not line
            or line == "WEBVTT"
            or line.startswith(("NOTE", "STYLE", "REGION"))
            or VTT_TIMESTAMP_PATTERN.match(line)
            or line.isdigit()
        ):
            continue
        text = html.unescape(re.sub(r"<[^>]*>", "", line)).strip()
        if text and (not transcript_lines or transcript_lines[-1] != text):
            transcript_lines.append(text)
    transcript = "\n".join(transcript_lines)
    if len(transcript) > MAX_TRANSCRIPT_CHARACTERS:
        transcript = (
            transcript[:MAX_TRANSCRIPT_CHARACTERS]
            + "\n[字幕過長，後續內容已截斷]"
        )
    return transcript


def _resolve_ffmpeg(ffmpeg_path: str | Path | None) -> str:
    if ffmpeg_path:
        resolved = Path(ffmpeg_path)
        if resolved.is_file():
            return str(resolved)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    raise VideoProcessingError(
        "ffmpeg_missing",
        "找不到 FFmpeg。請確認 GPT-SoVITS 的 runtime\\ffmpeg.exe 存在，"
        "或將 FFmpeg 加入 PATH。",
        500,
    )


def _run_ffmpeg(ffmpeg: str, arguments: list[str], timeout: int = 30) -> None:
    try:
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *arguments],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise VideoProcessingError(
            "video_processing_timeout",
            "影片處理超時，請稍後再試或提供較短的影片。",
            504,
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        print(f"[video] FFmpeg failed: {detail[:500]}")
        raise VideoProcessingError(
            "video_processing_failed",
            "無法從影片擷取畫面或音訊；影片可能無法解碼。",
        ) from error


def _format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


@contextmanager
def prepare_video(
    url: str,
    ffmpeg_path: str | Path | None = None,
) -> Iterator[PreparedVideo]:
    if extract_supported_video_url(url) != url:
        raise VideoProcessingError(
            "unsupported_video_url",
            "目前只支援 HTTPS 的 Bilibili 與 YouTube 影片連結。",
        )
    ffmpeg = _resolve_ffmpeg(ffmpeg_path)
    try:
        import yt_dlp
    except ImportError as error:
        raise VideoProcessingError(
            "video_downloader_missing",
            "尚未安裝影片讀取元件。請在 Amadeus 環境執行 pip install -r requirements.txt。",
            500,
        ) from error

    with tempfile.TemporaryDirectory(prefix="amadeus-video-") as temp_dir:
        work_dir = Path(temp_dir)
        download_started_at = 0.0
        download_limit_hit = False
        downloaded_by_file: dict[str, int] = {}

        def check_download_limits(progress: dict[str, object]) -> None:
            nonlocal download_limit_hit
            if progress.get("status") != "downloading":
                return
            downloaded = progress.get("downloaded_bytes", 0)
            filename = progress.get("filename")
            if isinstance(downloaded, (int, float)):
                if isinstance(filename, str):
                    downloaded_by_file[filename] = int(downloaded)
                total_downloaded = sum(downloaded_by_file.values())
                if (
                    downloaded > MAX_VIDEO_SIZE_BYTES
                    or total_downloaded > MAX_VIDEO_SIZE_BYTES
                ):
                    download_limit_hit = True
                    raise yt_dlp.utils.DownloadError(
                        "video download size limit exceeded"
                    )
            if time.monotonic() - download_started_at > MAX_VIDEO_DOWNLOAD_SECONDS:
                download_limit_hit = True
                raise yt_dlp.utils.DownloadError("video download time limit exceeded")

        options = {
            "format": (
                "best[height<=720][ext=mp4]/best[height<=720]/"
                "bestvideo[height<=720]+bestaudio"
            ),
            "outtmpl": str(work_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "max_filesize": MAX_VIDEO_SIZE_BYTES,
            "socket_timeout": 20,
            "retries": 1,
            "fragment_retries": 1,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["zh-Hant", "zh-Hans", "zh", "zh-TW", "zh-CN", "en", "ja"],
            "subtitlesformat": "vtt/best",
            "ffmpeg_location": ffmpeg,
            "progress_hooks": [check_download_limits],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=False)
                if not isinstance(info, dict) or info.get("_type") == "playlist":
                    raise VideoProcessingError(
                        "video_playlist_unsupported",
                        "目前一次只能讀取一支影片，不能處理播放清單。",
                    )
                duration = info.get("duration")
                if not isinstance(duration, (int, float)) or duration <= 0:
                    raise VideoProcessingError(
                        "video_duration_unavailable",
                        "無法確認影片長度；目前只支援長度不超過 10 分鐘的影片。",
                    )
                if duration > MAX_VIDEO_DURATION_SECONDS:
                    raise VideoProcessingError(
                        "video_too_long",
                        "目前一次最多分析 10 分鐘的影片。",
                    )
                download_started_at = time.monotonic()
                download_result = downloader.download([url])
                if download_limit_hit:
                    raise VideoProcessingError(
                        "video_download_limit_exceeded",
                        "影片下載超過 100 MB 或 120 秒限制；請提供較短或較低畫質的影片。",
                    )
                if download_result not in (None, 0):
                    raise VideoProcessingError(
                        "video_download_failed",
                        "無法取得這支公開影片。請確認連結有效、影片可公開觀看，"
                        "且不需要登入或會員權限。",
                    )
        except VideoProcessingError:
            raise
        except yt_dlp.utils.DownloadError as error:
            if download_limit_hit:
                raise VideoProcessingError(
                    "video_download_limit_exceeded",
                    "影片下載超過 100 MB 或 120 秒限制；請提供較短或較低畫質的影片。",
                ) from error
            print(f"[video] Download failed: {error}")
            raise VideoProcessingError(
                "video_download_failed",
                "無法取得這支公開影片。請確認連結有效、影片可公開觀看，"
                "且不需要登入或會員權限。",
            ) from error

        downloaded_size = sum(
            file.stat().st_size
            for file in work_dir.iterdir()
            if file.is_file()
        )
        if downloaded_size > MAX_VIDEO_SIZE_BYTES:
            raise VideoProcessingError(
                "video_download_limit_exceeded",
                "影片相關下載檔案超過 100 MB 限制；請提供較短或較低畫質的影片。",
            )

        subtitle_files = sorted(work_dir.glob("*.vtt"))
        transcript = ""
        if subtitle_files:
            subtitle_file = min(subtitle_files, key=lambda item: item.stat().st_size)
            transcript = parse_vtt_transcript(
                subtitle_file.read_text(encoding="utf-8", errors="replace")
            )

        video_files = [
            file
            for file in work_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() not in {".vtt", ".json", ".description"}
        ]
        if not video_files:
            raise VideoProcessingError(
                "video_download_failed",
                "下載完成後找不到可分析的影片檔。",
            )
        video_file = max(video_files, key=lambda item: item.stat().st_size)

        frame_count = min(MAX_FRAME_COUNT, max(1, int(duration // 30)))
        frames: list[VideoFrame] = []
        for index in range(frame_count):
            timestamp_seconds = duration * (index + 1) / (frame_count + 1)
            frame_path = work_dir / f"frame-{index}.jpg"
            _run_ffmpeg(
                ffmpeg,
                [
                    "-ss",
                    f"{timestamp_seconds:.3f}",
                    "-i",
                    str(video_file),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=1024:1024:force_original_aspect_ratio=decrease",
                    "-q:v",
                    "5",
                    "-y",
                    str(frame_path),
                ],
            )
            if frame_path.is_file():
                frames.append(
                    VideoFrame(
                        timestamp=_format_timestamp(timestamp_seconds),
                        data=frame_path.read_bytes(),
                    )
                )
        if not frames:
            raise VideoProcessingError(
                "video_frames_unavailable",
                "無法從影片擷取代表畫面。",
            )

        audio_path = None
        if not transcript:
            extracted_audio = work_dir / "audio.mp3"
            try:
                _run_ffmpeg(
                    ffmpeg,
                    [
                        "-i",
                        str(video_file),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-b:a",
                        "64k",
                        "-f",
                        "mp3",
                        "-y",
                        str(extracted_audio),
                    ],
                    timeout=60,
                )
            except VideoProcessingError as error:
                if error.code != "video_processing_failed":
                    raise
                print("[video] Audio could not be extracted; continuing with sampled frames.")
            if extracted_audio.is_file() and extracted_audio.stat().st_size > 0:
                audio_path = extracted_audio

        yield PreparedVideo(
            title=str(info.get("title") or "線上影片"),
            transcript=transcript,
            frames=frames,
            audio_path=audio_path,
        )
