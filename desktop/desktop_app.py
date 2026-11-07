import io
import json
import queue
import re
import sqlite3
import threading
import time
from datetime import datetime, time as datetime_time, timedelta
from pathlib import Path
import numpy as np
import pygame
import requests
import tkinter as tk
from tkinter import filedialog
from PIL import ImageGrab
import live2d
from backend.memory_manager import get_application_state, set_application_state
from desktop.live2d_view import Live2DView
from desktop.local_speech import transcribe_chinese
from desktop.voice_input import PushToTalk, format_hotkey_label
from shared.expression_manager import extract_expression_and_text, remove_source_section
from shared.greetings import daily_greeting_context
from shared.proactive import should_send_proactive
from shared.screen_observation import has_significant_screen_change
from shared.settings import (
    BACKEND_BASE_URL,
    BACKEND_REQUEST_TIMEOUT,
    LIVE2D_MODEL_PATH,
    PROACTIVE_IDLE_MINUTES,
    PROACTIVE_REQUEST_TIMEOUT_SECONDS,
    PUSH_TO_TALK_KEY,
    TTS_API_URL,
    TTS_OUTPUT_DIR,
    TTS_REFERENCE_AUDIO,
)

print(live2d.__file__)

def remove_generated_audio_file(filepath):
    output_directory = Path(TTS_OUTPUT_DIR).resolve()
    audio_path = Path(filepath).resolve()
    if (
        audio_path.parent != output_directory
        or not audio_path.name.startswith("output_")
        or audio_path.suffix.lower() != ".wav"
    ):
        return False

    try:
        audio_path.unlink()
    except FileNotFoundError:
        return False
    except OSError as error:
        print(f"Generated audio cleanup failed for {audio_path}: {error}")
        return False
    return True


try:
    import live2d.utils.lipsync as lipsync

    _orig_update = lipsync.WavHandler.Update


    def _patched_update(self, *args, **kwargs):
        if hasattr(self, "pcmData") and isinstance(self.pcmData, np.ndarray):
            if self.pcmData.ndim == 1:
                self.pcmData = np.expand_dims(self.pcmData, axis=0)
        return _orig_update(self, *args, **kwargs)


    lipsync.WavHandler.Update = _patched_update
except Exception as patch_err:
    print()
# =========================================================================

def split_text_jp(text, max_len=30):
    raw_sentences = re.split(r'(?<=[。！？\?\!])', text)
    chunks = []

    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_len:
            sub_s = re.split(r'(?<=[、、,])', s)
            temp = ""
            for sub in sub_s:
                if len(temp) + len(sub) <= max_len:
                    temp += sub
                else:
                    if temp.strip():
                        chunks.append(temp.strip())
                    temp = sub
            if temp.strip():
                chunks.append(temp.strip())
        else:
            chunks.append(s)

    valid_chunks = []
    for chunk in chunks:
        text_only = re.sub(r'[。！？\?\!、、,,\s\-_]', '', chunk)
        if text_only:
            valid_chunks.append(chunk)

    return valid_chunks


def normalize_japanese_tts_text(text: str) -> str:
    text = re.sub(
        r"牧[瀬濑][\s・･·]*[紅红]莉栖",
        "まきせくりす",
        text,
    )
    return re.sub(r"[紅红]莉栖", "くりす", text)


class Live2DChatApp:
    def __init__(self, master):
        self.master = master
        master.title("Amadeus - Desk Pet")

        master.geometry("450x830+100+100")
        master.overrideredirect(True)
        master.attributes("-topmost", True)

        self.transparent_color = '#000000'
        master.config(bg=self.transparent_color)
        master.wm_attributes('-transparentcolor', self.transparent_color)

        pygame.mixer.init()
        self.current_audio = None
        self.bubble_timer = None

        self.auto_screen_active = False  # 陪玩开关
        self.screen_interval = 60  # 截屏检测间隔（秒）
        self.is_screen_processing = False  # 锁，防止请求重叠
        self.screen_observation_after_id = None
        self.last_observation_image = None
        self.screen_observation_generation = 0
        self.last_interaction_at = time.monotonic()
        self.last_proactive_at = self.last_interaction_at
        self.proactive_request_active = False
        self.proactive_status_after_id = None
        self.proactive_started_at = None
        self.speech_active = False
        self.speech_generation = 0
        self.chat_generation = 0
        self.chat_request_active = False
        self.voice_input_active = False
        self.voice_recording_active = False
        self.voice_input_generation = 0
        self._active_stream_timing = None
        self.activity_generation = 0
        self.closing = False
        self._daily_greeting_calendar_date = (
            datetime.now().astimezone().date().isoformat()
        )
        self._greeting_waiting_for_first_chat = False

        self.setup_ui()
        self.setup_interactions()
        self.voice_input = PushToTalk(
            self.on_voice_recording_changed,
            self.on_voice_audio_ready,
            self.on_voice_input_error,
            hotkey=PUSH_TO_TALK_KEY,
            on_utterance_audio=self.on_voice_utterance_audio,
            should_pause_utterance_detection=self.should_pause_voice_detection,
        )
        try:
            self.voice_input.start_listener()
        except Exception as error:
            print(f"[voice input] Global push-to-talk shortcut unavailable: {error}")
            self.master.after(
                1_000,
                lambda: self.show_voice_status(
                f"{format_hotkey_label(PUSH_TO_TALK_KEY)}無法啟用，可點擊麥克風按鈕切換聆聽"
                ),
            )
        self.master.bind_all("<KeyPress>", self.record_user_activity, add="+")
        self.master.bind_all("<ButtonPress>", self.record_user_activity, add="+")
        self.master.after(1_000, self.start_daily_greeting)
        self.schedule_daily_greeting_calendar_check()
        self.master.after(60_000, self.check_idle_for_proactive)

    def schedule_daily_greeting_calendar_check(self):
        if self.closing:
            return
        now = datetime.now().astimezone()
        next_midnight = datetime.combine(
            now.date() + timedelta(days=1),
            datetime_time.min,
            tzinfo=now.tzinfo,
        )
        delay_ms = max(
            1_000,
            int((next_midnight - now).total_seconds() * 1_000),
        )
        self.master.after(delay_ms, self.check_daily_greeting_calendar)

    def check_daily_greeting_calendar(self):
        if self.closing:
            return
        self.refresh_daily_greeting_calendar()
        self.schedule_daily_greeting_calendar_check()

    def refresh_daily_greeting_calendar(self, now=None):
        current_time = now or datetime.now().astimezone()
        today = current_time.date().isoformat()
        if today == self._daily_greeting_calendar_date:
            return False
        self._daily_greeting_calendar_date = today
        self._greeting_waiting_for_first_chat = True
        context = daily_greeting_context(current_time)
        print(
            f"[greeting] Local calendar updated: date={today}, "
            f"holiday={context['holiday'] or 'none'}"
        )
        return True

    def start_daily_greeting(self):
        if self.closing:
            return
        today = datetime.now().astimezone().date().isoformat()
        try:
            greeted_date = get_application_state("daily_greeting_date")
        except sqlite3.Error as error:
            print(f"[greeting] Failed to read greeting state: {error}")
            self.master.after(60_000, self.start_daily_greeting)
            return
        if greeted_date == today:
            return
        if self.is_proactive_busy():
            self.master.after(60_000, self.start_daily_greeting)
            return
        self.start_proactive_request(force=True, greeting=True)

    def record_user_activity(self, _event=None):
        self.last_interaction_at = time.monotonic()
        self.activity_generation += 1

    def check_idle_for_proactive(self):
        if self.closing:
            return

        now = time.monotonic()
        busy = self.is_proactive_busy()
        if should_send_proactive(
            now=now,
            last_interaction=self.last_interaction_at,
            idle_seconds=PROACTIVE_IDLE_MINUTES * 60,
            busy=busy,
            last_proactive_at=self.last_proactive_at,
        ):
            self.start_proactive_request()
        elif (
            busy
            and now - max(self.last_interaction_at, self.last_proactive_at)
            >= PROACTIVE_IDLE_MINUTES * 60
        ):
            print(
                "[proactive] Automatic call is due but deferred: "
                f"screen_processing={self.is_screen_processing}, "
                f"speech_active={self.speech_active}, "
                f"request_active={self.proactive_request_active}, "
                f"input_pending={bool(self.user_input.get().strip())}"
            )

        self.master.after(60_000, self.check_idle_for_proactive)

    def is_proactive_busy(self):
        return (
            self.is_screen_processing
            or self.proactive_request_active
            or self.speech_active
            or getattr(self, "chat_request_active", False)
            or getattr(self, "voice_input_active", False)
            or bool(self.user_input.get().strip())
        )

    def start_proactive_request(self, force=False, greeting=False):
        if self.closing:
            return
        if self.proactive_request_active:
            return
        if not should_send_proactive(
            now=time.monotonic(),
            last_interaction=self.last_interaction_at,
            idle_seconds=PROACTIVE_IDLE_MINUTES * 60,
            busy=self.is_proactive_busy(),
            last_proactive_at=self.last_proactive_at,
            force=force,
        ):
            if force:
                self.show_voice_status("目前正在忙碌，稍後再試")
            return

        self.last_proactive_at = time.monotonic()
        self.proactive_request_active = True
        self.set_proactive_button_busy(True)
        if force:
            self.proactive_started_at = time.monotonic()
            self.show_voice_status(
                "正在準備今日問候..." if greeting else "正在想一個適合分享的話題..."
            )
            self.proactive_status_after_id = self.master.after(
                15_000,
                self.update_proactive_status,
            )
        generation = self.activity_generation
        threading.Thread(
            target=self.request_proactive_message,
            args=(generation, force, greeting),
            daemon=True,
        ).start()

    def trigger_proactive_now(self):
        self.start_proactive_request(force=True)

    def update_proactive_status(self):
        self.proactive_status_after_id = None
        if not self.proactive_request_active or self.proactive_started_at is None:
            return
        elapsed = int(time.monotonic() - self.proactive_started_at)
        self.show_voice_status(f"還在處理主動訊息（已等待 {elapsed} 秒）...")
        self.proactive_status_after_id = self.master.after(
            15_000,
            self.update_proactive_status,
        )

    def clear_proactive_status_timer(self):
        if self.proactive_status_after_id is not None:
            self.master.after_cancel(self.proactive_status_after_id)
            self.proactive_status_after_id = None
        self.proactive_started_at = None

    def set_proactive_button_busy(self, busy):
        if hasattr(self, "btn_proactive"):
            self.btn_proactive.config(
                state=tk.DISABLED if busy else tk.NORMAL,
                text="處理中..." if busy else "主動聊聊",
            )

    def request_proactive_message(self, generation, force=False, greeting=False):
        try:
            payload = {"user_id": "user_001", "force": force}
            if greeting:
                payload["greeting"] = True
            response = requests.post(
                f"{BACKEND_BASE_URL}/proactive",
                json=payload,
                timeout=(5, PROACTIVE_REQUEST_TIMEOUT_SECONDS),
            )
            if response.status_code == 504:
                raise requests.Timeout("The proactive request exceeded its time limit")
            response.raise_for_status()
            data = response.json()
            self.master.after(
                0,
                lambda: self.show_proactive_message(
                    data,
                    generation,
                    force,
                    greeting,
                ),
            )
        except requests.Timeout as e:
            print(f"[proactive] Request timed out: {e}")
            try:
                self.master.after(
                    0,
                    lambda: self.finish_proactive_request(
                        force=force,
                        error="timeout",
                        greeting=greeting,
                    ),
                )
            except tk.TclError:
                pass
        except Exception as e:
            print(f"[proactive] Request failed: {e}")
            try:
                self.master.after(
                    0,
                    lambda error=str(e): self.finish_proactive_request(
                        force=force,
                        error=error,
                        greeting=greeting,
                    ),
                )
            except tk.TclError:
                pass

    def show_proactive_message(self, data, generation, force=False, greeting=False):
        self.proactive_request_active = False
        self.clear_proactive_status_timer()
        self.set_proactive_button_busy(False)
        if (
            self.closing
            or (not force and generation != self.activity_generation)
            or (greeting and generation != self.activity_generation)
            or (
                not force
                and time.monotonic() - self.last_interaction_at
                < PROACTIVE_IDLE_MINUTES * 60
            )
        ):
            print("[proactive] Ignoring stale automatic message")
            if greeting and not self.closing:
                self.master.after(60_000, self.start_daily_greeting)
            return

        if not data.get("reply"):
            if force:
                self.show_voice_status(
                    "今日問候未能產生，稍後重試"
                    if greeting
                    else "這次沒有找到值得分享的話題"
                )
            if greeting:
                self.master.after(60_000, self.start_daily_greeting)
            return

        reply = data["reply"]
        print(
            f"[proactive] Showing {'manual' if force else 'automatic'} reply "
            f"({len(reply)} characters)"
        )
        japanese_reply = data.get("japanese_reply", "")
        expression = data.get("expression", "neutral")
        try:
            self.live2d.set_expression(expression)
        except Exception as e:
            print(f"[proactive] Failed to set expression: {e}")
        self.display_message(reply, auto_hide=False)
        if greeting:
            today = datetime.now().astimezone().date().isoformat()
            try:
                set_application_state("daily_greeting_date", today)
            except sqlite3.Error as error:
                print(f"[greeting] Failed to save greeting state: {error}")
        if japanese_reply:
            self.show_voice_status("🎤 语音生成中...")
            self.process_and_stream(reply, japanese_reply)
        else:
            self.display_message(reply, auto_hide=True)

    def finish_proactive_request(self, force=False, error=None, greeting=False):
        self.proactive_request_active = False
        self.clear_proactive_status_timer()
        self.set_proactive_button_busy(False)
        if force:
            if error:
                if greeting:
                    message = (
                        "今日問候處理逾時，稍後重試"
                        if error == "timeout"
                        else "今日問候失敗，稍後重試"
                    )
                else:
                    message = (
                        "主動訊息處理逾時，請稍後再試"
                        if error == "timeout"
                        else "主動訊息失敗，請檢查後端與 API 狀態"
                    )
                self.show_voice_status(message)
                if greeting:
                    self.master.after(60_000, self.start_daily_greeting)

    def setup_ui(self):
        self.live2d = Live2DView(
            self.master,
            model_versions=3,
            model_path=str(LIVE2D_MODEL_PATH),
            width=450,
            height=550,
            offset_x=0.0,
            background_path=""
        )
        self.live2d.config(bg=self.transparent_color)
        self.live2d.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.bubble_slot = tk.Frame(
            self.master,
            bg=self.transparent_color,
            height=146,
        )
        self.bubble_slot.pack(
            side=tk.TOP,
            fill=tk.X,
            before=self.live2d,
        )
        self.bubble_slot.pack_propagate(False)

        self.bubble_frame = tk.Frame(self.bubble_slot, bg="#ffffff", bd=2, relief="solid")
        self.bubble_text = tk.Text(
            self.bubble_frame, bg="#ffffff", fg="#333333",
            font=("Microsoft YaHei", 9), wrap="char", bd=0, highlightthickness=0
        )
        self.bubble_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.bubble_text.bind("<MouseWheel>",
                              lambda e: self.bubble_text.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        self.bubble_frame.configure(height=130)
        self.bubble_frame.pack_propagate(False)

        # 底部区域
        self.input_frame = tk.Frame(self.master, bg=self.transparent_color)
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=20)

        self.voice_status_label = tk.Label(
            self.input_frame, text="", fg="#00a2ff", bg=self.transparent_color, font=("Microsoft YaHei", 9, "bold")
        )
        self.voice_status_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        input_container = tk.Frame(self.input_frame, bg=self.transparent_color)
        input_container.pack(fill=tk.X)
        input_container.columnconfigure(0, weight=1)

        self.user_input = tk.Entry(
            input_container,
            font=("Microsoft YaHei", 10),
            bg="#f0f0f0",
            relief="flat",
        )
        self.user_input.grid(row=0, column=0, sticky="ew", ipady=3)
        self.user_input.bind("<Return>", lambda e: self.send_message())

        self.btn_voice = tk.Button(
            input_container,
            text="🎙",
            command=self.toggle_voice_recording,
            font=("Microsoft YaHei", 9),
            bg="#ff9800",
            fg="white",
            relief="flat",
            padx=6,
            pady=0,
        )
        self.btn_voice.grid(row=0, column=1, padx=(5, 0))

        btn_send = tk.Button(
            input_container,
            text="发送",
            command=self.send_message,
            font=("Microsoft YaHei", 9),
            bg="#4caf50",
            fg="white",
            relief="flat",
            padx=6,
            pady=0,
        )
        btn_send.grid(row=0, column=2, padx=(5, 0))

        controls_container = tk.Frame(self.input_frame, bg=self.transparent_color)
        controls_container.pack(fill=tk.X, pady=(5, 0))

        self.btn_proactive = tk.Button(
            controls_container, text="主動聊聊", command=self.trigger_proactive_now,
            font=("Microsoft YaHei", 9), bg="#2196f3", fg="white", relief="flat", padx=6
        )
        self.btn_proactive.pack(side=tk.RIGHT)

        # 陪玩控制按钮
        self.btn_auto_screen = tk.Button(
            controls_container, text="开启观测", command=self.toggle_auto_screen,
            font=("Microsoft YaHei", 9), bg="#673ab7", fg="white", relief="flat", padx=6
        )
        self.btn_auto_screen.pack(side=tk.RIGHT, padx=(0, 5))

    def setup_interactions(self):
        self.live2d.bind("<ButtonPress-1>", self.start_move)
        self.live2d.bind("<B1-Motion>", self.do_move)

        self.menu = tk.Menu(self.master, tearoff=0, font=("Microsoft YaHei", 10))
        self.menu.add_command(label="🖼️ 上传图片", command=self.send_image)
        self.menu.add_separator()
        self.menu.add_command(label="❌ 退出系统", command=self.on_closing)

        self.live2d.bind("<Button-3>", self.show_menu)

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.master.winfo_x() + deltax
        y = self.master.winfo_y() + deltay
        self.master.geometry(f"+{x}+{y}")

    def toggle_voice_recording(self):
        self.voice_input.toggle_recording()

    def update_voice_button(self, recording):
        self.btn_voice.config(
            text="🔴" if recording else "🎙",
            bg="#f44336" if recording else "#ff9800",
        )

    def _schedule_ui(self, callback, *args, **kwargs):
        if self.closing:
            return
        try:
            self.master.after(0, lambda: callback(*args, **kwargs))
        except tk.TclError:
            pass

    def on_voice_recording_changed(self, recording):
        self._schedule_ui(self.update_voice_button, recording)
        if recording:
            self.voice_recording_active = True
            started_at = time.perf_counter()
            self._voice_timing = {
                "started_at": started_at,
                "recording_started_at": started_at,
                "recording_ms": 0.0,
                "transcription_ms": 0.0,
                "tts_generation_ms": 0.0,
                "tts_playback_ms": 0.0,
            }
            self.speech_generation += 1
            self.chat_generation += 1
            self.chat_request_active = False
            self.voice_input_generation += 1
            self.voice_input_active = True
            self.speech_active = False
            self._schedule_ui(self.interrupt_speech_playback)
            self._schedule_ui(self.record_user_activity)
            self._schedule_ui(
                self.show_voice_status,
                f"🎙 持續聆聽中：停頓約 0.5 秒開始回覆，再點{format_hotkey_label(PUSH_TO_TALK_KEY)}停止"
            )
        else:
            self.voice_recording_active = False
            timing = getattr(self, "_voice_timing", None)
            if timing is not None:
                timing["recording_ms"] = (
                    time.perf_counter() - timing["recording_started_at"]
                ) * 1000
            self.voice_input_active = False
            if not self.chat_request_active and not self.speech_active:
                self._schedule_ui(self.hide_voice_status)

    def should_pause_voice_detection(self):
        return self.chat_request_active or self.speech_active

    def on_voice_audio_ready(self, audio):
        self.voice_input_active = True
        generation = self.voice_input_generation
        timing = getattr(self, "_voice_timing", None) or {
            "started_at": time.perf_counter(),
            "recording_ms": 0.0,
            "transcription_ms": 0.0,
            "tts_generation_ms": 0.0,
            "tts_playback_ms": 0.0,
        }
        self._voice_timing = None
        try:
            threading.Thread(
                target=self.transcribe_voice_input,
                args=(audio, generation, self.chat_generation, timing),
                daemon=True,
            ).start()
        except Exception as error:
            self.voice_input_active = False
            self.log_voice_timing(timing, "transcription_start_failed")
            self.on_voice_input_error(f"無法啟動本機語音辨識：{error}")

    def on_voice_input_error(self, message):
        self._schedule_ui(self.show_voice_status, message)

    def on_voice_partial_audio(self, audio):
        if not self.voice_recording_active:
            return
        transcript = transcribe_chinese(audio)
        if not transcript or not self.voice_recording_active:
            return
        self._schedule_ui(
            self.show_voice_status,
            f"🎙 本機辨識中：{transcript[-48:]}",
        )

    def on_voice_utterance_audio(
        self,
        audio,
        recording_id,
        started_at,
        ended_at,
    ):
        if self.closing or recording_id != self.voice_input_generation:
            return
        self.voice_input_active = True
        self._schedule_ui(self.show_voice_status, "本機辨識中...")
        timing = {
            "started_at": started_at,
            "recording_ms": max(0.0, (ended_at - started_at) * 1000),
            "transcription_ms": 0.0,
            "tts_generation_ms": 0.0,
            "tts_playback_ms": 0.0,
        }
        self.transcribe_voice_input(
            audio,
            recording_id,
            self.chat_generation,
            timing,
        )

    def transcribe_voice_input(
        self,
        audio,
        voice_generation,
        chat_generation,
        timing=None,
    ):
        transcription_started_at = time.perf_counter()
        try:
            transcript = transcribe_chinese(audio)
        except Exception as error:
            if timing is not None:
                timing["transcription_ms"] = (
                    time.perf_counter() - transcription_started_at
                ) * 1000
                self.log_voice_timing(timing, "transcription_failed")
            print(f"[voice input] Local transcription failed: {error}")
            self._schedule_ui(
                self.finish_voice_transcription,
                voice_generation,
                "本機語音辨識失敗；請確認模型已下載且麥克風正常",
            )
            return
        if timing is not None:
            timing["transcription_ms"] = (
                time.perf_counter() - transcription_started_at
            ) * 1000
        if not transcript:
            self._schedule_ui(
                self.finish_empty_voice_utterance,
                voice_generation,
            )
            return
        self._schedule_ui(
            self.send_voice_transcript,
            transcript,
            voice_generation,
            chat_generation,
            timing,
        )

    def finish_voice_transcription(self, generation, message):
        if generation != self.voice_input_generation:
            return
        self.voice_input_active = False
        self.show_voice_status(message)

    def finish_empty_voice_utterance(self, generation):
        if generation != self.voice_input_generation:
            return
        self.voice_input_active = self.voice_recording_active
        if self.voice_recording_active:
            self.show_voice_status(
                "🎙 持續聆聽中：停頓約 0.5 秒開始回覆"
            )
        else:
            self.hide_voice_status()

    def send_voice_transcript(
        self,
        transcript,
        voice_generation,
        chat_generation,
        timing=None,
    ):
        if self.closing or voice_generation != self.voice_input_generation:
            if timing is not None:
                self.log_voice_timing(timing, "interrupted_before_chat")
            return
        if chat_generation != self.chat_generation:
            self.voice_input_active = False
            if timing is not None:
                self.log_voice_timing(timing, "interrupted_before_chat")
            return
        self.voice_input_active = self.voice_recording_active
        self.user_input.delete(0, tk.END)
        self.start_streaming_voice_reply(transcript, timing)

    def start_streaming_voice_reply(self, transcript, timing):
        self.chat_request_active = True
        self.speech_active = True
        self._active_stream_timing = timing
        chat_generation = self.chat_generation
        speech_generation = self.speech_generation
        speech_queue = queue.Queue()
        audio_queue = queue.Queue()
        result = {"final": None, "error": None, "fallback": False}
        self.show_voice_status("Gemini 正在生成串流語音回覆...")

        def produce_audio():
            index = 0
            try:
                while speech_generation == self.speech_generation:
                    speech_text = speech_queue.get()
                    if speech_text is None:
                        break
                    chunks = split_text_jp(
                        normalize_japanese_tts_text(speech_text),
                        max_len=30,
                    )
                    for chunk in chunks:
                        if speech_generation != self.speech_generation:
                            break
                        generation_started_at = time.perf_counter()
                        audio_path = self.generate_single_audio(chunk, index)
                        timing["tts_generation_ms"] += (
                            time.perf_counter() - generation_started_at
                        ) * 1000
                        index += 1
                        if audio_path:
                            audio_queue.put(audio_path)
            except Exception as error:
                result["error"] = error
                print(f"[voice stream] TTS producer failed: {error}")
            finally:
                audio_queue.put(None)

        def consume_audio():
            pipeline_started_at = time.perf_counter()
            try:
                while speech_generation == self.speech_generation:
                    audio_path = audio_queue.get()
                    if audio_path is None:
                        break
                    try:
                        playback_started_at = time.perf_counter()
                        self.play_audio(audio_path)
                        while (
                            speech_generation == self.speech_generation
                            and pygame.mixer.music.get_busy()
                        ):
                            time.sleep(0.05)
                        timing["tts_playback_ms"] += (
                            time.perf_counter() - playback_started_at
                        ) * 1000
                    finally:
                        remove_generated_audio_file(audio_path)
            finally:
                timing["tts_pipeline_ms"] = round(
                    (time.perf_counter() - pipeline_started_at) * 1000,
                    1,
                )
                final = result["final"]
                completed = (
                    final is not None
                    and result["error"] is None
                    and speech_generation == self.speech_generation
                )
                if result["error"] is not None:
                    self.chat_request_active = False
                    self._schedule_ui(
                        self.show_voice_status,
                        "即時語音回覆失敗，請檢查後端與 Gemini 連線",
                    )
                elif completed:
                    if self.voice_recording_active:
                        self._schedule_ui(
                            self.show_voice_status,
                            "🎙 持續聆聽中：停頓約 0.5 秒開始回覆",
                        )
                    else:
                        self._schedule_ui(self.hide_voice_status)
                    self._schedule_ui(self.schedule_bubble_hide)
                if speech_generation == self.speech_generation:
                    self.speech_active = False
                if completed or result["error"] is not None:
                    if getattr(self, "_active_stream_timing", None) is timing:
                        self._active_stream_timing = None
                    self.log_voice_timing(
                        timing,
                        "completed" if completed else "stream_failed",
                    )
                self._schedule_ui(self.finish_speech, speech_generation)

        def request_stream():
            request_started_at = time.perf_counter()
            try:
                with requests.post(
                    f"{BACKEND_BASE_URL}/ask_stream",
                    json={"user_id": "user_001", "message": transcript},
                    stream=True,
                    timeout=(5, BACKEND_REQUEST_TIMEOUT),
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(
                        chunk_size=1,
                        decode_unicode=True,
                    ):
                        if speech_generation != self.speech_generation:
                            break
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        event_type = event.get("type")
                        if event_type == "speech":
                            speech_queue.put(event["text"])
                        elif event_type == "final":
                            result["final"] = event
                            timing["ask_server_ms"] = event.get("timings", {})
                            self._schedule_ui(
                                self.display_stream_reply,
                                event,
                                chat_generation,
                                speech_generation,
                            )
                        elif event_type == "fallback":
                            result["fallback"] = True
                            break
                        elif event_type == "error":
                            raise RuntimeError(event.get("message", "Gemini stream failed"))
                timing["ask_request_ms"] = (
                    time.perf_counter() - request_started_at
                ) * 1000
                if result["fallback"] and speech_generation == self.speech_generation:
                    speech_queue.put(None)
                    self._schedule_ui(
                        self.fallback_to_regular_voice_ask,
                        transcript,
                        timing,
                        chat_generation,
                    )
                    return
                if result["final"] is None and speech_generation == self.speech_generation:
                    raise RuntimeError("Gemini stream ended without a final reply")
            except Exception as error:
                result["error"] = error
                timing["ask_request_ms"] = (
                    time.perf_counter() - request_started_at
                ) * 1000
                print(f"[voice stream] Gemini request failed: {error}")
            finally:
                speech_queue.put(None)

        threading.Thread(target=produce_audio, daemon=True).start()
        threading.Thread(target=consume_audio, daemon=True).start()
        threading.Thread(target=request_stream, daemon=True).start()

    def display_stream_reply(self, event, chat_generation, speech_generation):
        if (
            self.closing
            or chat_generation != self.chat_generation
            or speech_generation != self.speech_generation
        ):
            return
        try:
            self.live2d.set_expression(event.get("expression", "neutral"))
        except Exception as error:
            print(f"[voice stream] Could not update expression: {error}")
        self.display_message(event["reply"], auto_hide=False)
        self.chat_request_active = False

    def fallback_to_regular_voice_ask(
        self,
        transcript,
        timing,
        chat_generation,
    ):
        if chat_generation != self.chat_generation or self.closing:
            return
        self.user_input.insert(0, transcript)
        self._active_voice_timing = timing
        self.send_message()

    @staticmethod
    def log_voice_timing(timing, outcome):
        elapsed_ms = (time.perf_counter() - timing["started_at"]) * 1000
        server_timings = timing.get("ask_server_ms", {})
        metrics = {
            "recording_ms": round(timing.get("recording_ms", 0.0), 1),
            "transcription_ms": round(
                timing.get("transcription_ms", 0.0), 1
            ),
            "ask_request_ms": round(
                timing.get("ask_request_ms", 0.0), 1
            ),
            "gemini_ms": server_timings.get("gemini_ms"),
            "translation_ms": server_timings.get("translation_ms"),
            "tts_generation_ms": round(
                timing.get("tts_generation_ms", 0.0), 1
            ),
            "tts_playback_ms": round(
                timing.get("tts_playback_ms", 0.0), 1
            ),
            "tts_pipeline_ms": timing.get("tts_pipeline_ms"),
            "end_to_end_ms": round(elapsed_ms, 1),
        }
        threading.Thread(
            target=Live2DChatApp._publish_voice_timing,
            args=({"outcome": outcome, "timings": metrics},),
            daemon=True,
        ).start()

    @staticmethod
    def _publish_voice_timing(payload):
        try:
            response = requests.post(
                f"{BACKEND_BASE_URL}/voice_timing",
                json=payload,
                timeout=(1, 3),
            )
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"[voice timing] Could not send timing to backend: {error}")

    def interrupt_speech_playback(self):
        try:
            self.live2d.stop_audio()
        except Exception as error:
            print(f"[voice input] Could not stop current speech: {error}")

    def show_menu(self, event):
        self.menu.tk_popup(event.x_root, event.y_root)

    def show_voice_status(self, msg):
        self.voice_status_label.config(text=msg)
        self.voice_status_label.update_idletasks()

    def hide_voice_status(self):
        self.voice_status_label.config(text="")
        self.voice_status_label.update_idletasks()

    def display_message(self, message, auto_hide=True):
        self.bubble_text.config(state="normal")
        self.bubble_text.delete("1.0", tk.END)
        self.bubble_text.insert(tk.END, message)
        self.bubble_text.yview_moveto(0)
        self.bubble_text.config(state="disabled")

        if not self.bubble_frame.winfo_manager():
            self.bubble_frame.pack(
                fill=tk.BOTH,
                expand=True,
                padx=40,
                pady=8,
            )

        if self.bubble_timer:
            self.master.after_cancel(self.bubble_timer)
            self.bubble_timer = None

        if auto_hide:
            self.bubble_timer = self.master.after(15000, self.hide_bubble)

    def hide_bubble(self):
        self.bubble_frame.pack_forget()

    def schedule_bubble_hide(self):
        if self.bubble_timer:
            self.master.after_cancel(self.bubble_timer)
        self.bubble_timer = self.master.after(15000, self.hide_bubble)

    # ==================== 陪玩功能核心逻辑 ====================
    def toggle_auto_screen(self):
        self.auto_screen_active = not self.auto_screen_active
        self.screen_observation_generation += 1
        if self.auto_screen_active:
            self.last_observation_image = None
            self.btn_auto_screen.config(text="⏹停止观测", bg="#f44336")
            self.show_voice_status("屏幕观测已开启")
            self.schedule_screen_observation(delay_ms=0)
        else:
            if self.screen_observation_after_id is not None:
                self.master.after_cancel(self.screen_observation_after_id)
                self.screen_observation_after_id = None
            self.btn_auto_screen.config(text="开启观测", bg="#673ab7")
            self.show_voice_status("⏹观测模式已关闭")

    def schedule_screen_observation(self, delay_ms=None):
        if not self.auto_screen_active or self.closing:
            return
        if delay_ms is None:
            delay_ms = self.screen_interval * 1_000
        self.screen_observation_after_id = self.master.after(
            delay_ms,
            self.capture_and_analyze_screen,
        )

    def capture_and_analyze_screen(self):
        self.screen_observation_after_id = None
        if not self.auto_screen_active or self.closing:
            return
        if (
            self.is_screen_processing
            or self.proactive_request_active
            or self.speech_active
            or getattr(self, "voice_input_active", False)
        ):
            self.schedule_screen_observation()
            return

        focused_widget = self.master.focus_get()
        try:
            self.master.withdraw()
            self.master.update_idletasks()
            screenshot = ImageGrab.grab()
        except Exception as error:
            print(f"[screen observation] Screenshot capture failed: {error}")
            self.show_voice_status("觀測截圖失敗，請稍後再試")
            self.schedule_screen_observation()
            return
        finally:
            try:
                self.master.deiconify()
                if (
                    focused_widget is not None
                    and focused_widget.winfo_exists()
                ):
                    focused_widget.focus_set()
            except tk.TclError:
                pass

        if not has_significant_screen_change(self.last_observation_image, screenshot):
            self.schedule_screen_observation()
            return

        self.show_voice_status("👀 正在分析畫面...")
        self.last_observation_image = screenshot.copy()
        self.is_screen_processing = True
        self.schedule_screen_observation()
        try:
            threading.Thread(
                target=self.send_screen_observation,
                args=(
                    screenshot,
                    self.screen_observation_generation,
                    self.activity_generation,
                ),
                daemon=True,
            ).start()
        except Exception as error:
            print(f"[screen observation] Could not start analysis request: {error}")
            self.is_screen_processing = False
            self.last_observation_image = None
            self.show_voice_status("觀測分析無法啟動，請稍後再試")

    def send_screen_observation(self, screenshot, generation, activity_generation):
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        files = {"image": ("screen.png", img_byte_arr, "image/png")}
        data = {
            "user_id": "user_001",
            "mode": "screen_observation",
            "question": (
                "（持續屏幕觀測中）自然接續你正在陪我做的事，"
                "直接回應畫面裡的內容。"
            ),
        }
        try:
            response = requests.post(
                f"{BACKEND_BASE_URL}/ask_with_image",
                files=files,
                data=data,
                timeout=(5, BACKEND_REQUEST_TIMEOUT),
            )
            response.raise_for_status()
            result = response.json()
            self.master.after(
                0,
                lambda result=result: self.finish_screen_observation(
                    result,
                    generation=generation,
                    activity_generation=activity_generation,
                ),
            )
        except Exception as error:
            print(f"[screen observation] Analysis request failed: {error}")
            try:
                self.master.after(
                    0,
                    lambda error=error: self.finish_screen_observation(
                        None,
                        error,
                        generation,
                        activity_generation,
                    ),
                )
            except tk.TclError:
                self.is_screen_processing = False

    def finish_screen_observation(
        self,
        result,
        error=None,
        generation=None,
        activity_generation=None,
    ):
        self.is_screen_processing = False
        if (
            generation != self.screen_observation_generation
            or not self.auto_screen_active
            or self.closing
            or activity_generation != self.activity_generation
        ):
            return
        if error is not None:
            self.last_observation_image = None
            self.show_voice_status("觀測分析失敗，請檢查後端與 API 狀態")
            return
        if not isinstance(result, dict):
            self.last_observation_image = None
            self.show_voice_status("觀測分析沒有收到有效回覆")
            return

        reply = result.get("reply", "")
        japanese_reply = result.get("japanese_reply", "")
        expression = result.get("expression", "neutral")
        if not reply:
            self.last_observation_image = None
            self.show_voice_status("觀測分析沒有收到有效回覆")
            return

        try:
            self.live2d.set_expression(expression)
        except Exception as error:
            print(f"[screen observation] Could not update expression: {error}")

        _, reply = extract_expression_and_text(reply)
        if japanese_reply:
            self.display_message(reply, auto_hide=False)
            self.show_voice_status("🎤 語音生成中...")
            self.process_and_stream(reply, japanese_reply)
        else:
            self.display_message(reply, auto_hide=True)
            self.hide_voice_status()

    # ==================== 普通文本 & 图像发送 ====================
    def send_message(self):
        message = self.user_input.get().strip()
        if not message:
            return

        voice_timing = getattr(self, "_active_voice_timing", None)
        self._active_voice_timing = None
        self.chat_generation += 1
        generation = self.chat_generation
        self.chat_request_active = True
        self.user_input.delete(0, tk.END)
        self.refresh_daily_greeting_calendar()
        if self._greeting_waiting_for_first_chat:
            self._greeting_waiting_for_first_chat = False
            self.master.after(0, self.start_daily_greeting)
        self.show_voice_status("⌛ 思考中...")

        def fetch_reply():
            ask_started_at = time.perf_counter()
            try:
                response = requests.post(
                    f"{BACKEND_BASE_URL}/ask",
                    json={"user_id": "user_001", "message": message},
                    timeout=(5, BACKEND_REQUEST_TIMEOUT)
                )
                data = response.json()
                if voice_timing is not None:
                    voice_timing["ask_request_ms"] = (
                        time.perf_counter() - ask_started_at
                    ) * 1000
                    server_timings = data.get("timings", {})
                    if isinstance(server_timings, dict):
                        voice_timing["ask_server_ms"] = server_timings
                if generation != self.chat_generation:
                    if voice_timing is not None:
                        self.log_voice_timing(voice_timing, "superseded")
                    return
                self.chat_request_active = False
                reply = data.get("reply", "")
                jp_text = data.get("japanese_reply", "")
                expression = data.get("expression", "neutral")

                try:
                    self.live2d.set_expression(expression)
                except Exception as e:
                    print(f"设置表情失败: {e}")

                _, reply = extract_expression_and_text(reply)
                reply = remove_source_section(reply)

                self.master.after(0, lambda: self.display_message(reply, auto_hide=False))

                if jp_text:
                    self.master.after(0, lambda: self.show_voice_status("🎤 语音生成中..."))
                    self.process_and_stream(reply, jp_text, voice_timing)
                else:
                    if voice_timing is not None:
                        self.log_voice_timing(voice_timing, "no_japanese_reply")
                    self.master.after(0, lambda: self.display_message(reply, auto_hide=True))
                    self.master.after(0, self.hide_voice_status)

            except Exception as e:
                if voice_timing is not None:
                    voice_timing["ask_request_ms"] = (
                        time.perf_counter() - ask_started_at
                    ) * 1000
                    self.log_voice_timing(voice_timing, "ask_failed")
                if generation == self.chat_generation:
                    self.chat_request_active = False
                    self.master.after(
                        0,
                        lambda: self.display_message(f"系统错误: {str(e)}"),
                    )
                    self.master.after(0, self.hide_voice_status)

        threading.Thread(target=fetch_reply, daemon=True).start()


    def send_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        question = self.user_input.get().strip() or "请描述这张图片"
        self.user_input.delete(0, tk.END)
        self.show_voice_status("📤 图片上传并分析中...")

        def upload_process():
            try:
                with open(file_path, "rb") as f:
                    files = {"image": f}
                    data = {"user_id": "user_001", "question": question}
                    response = requests.post(
                        f"{BACKEND_BASE_URL}/ask_with_image",
                        files=files,
                        data=data,
                        timeout=(5, BACKEND_REQUEST_TIMEOUT)
                    )

                res = response.json()
                reply = res.get("reply", "")
                jp_text = res.get("japanese_reply", "")
                expression = res.get("expression", "neutral")

                try:
                    self.live2d.set_expression(expression)
                except Exception:
                    pass

                _, reply = extract_expression_and_text(reply)

                self.master.after(0, lambda: self.display_message(reply, auto_hide=False))

                if jp_text:
                    self.master.after(0, lambda: self.show_voice_status("🎤 语音生成中..."))
                    self.process_and_stream(reply, jp_text)
                else:
                    self.master.after(0, lambda: self.display_message(reply, auto_hide=True))
                    self.master.after(0, self.hide_voice_status)

            except Exception as e:
                self.master.after(0, lambda: self.display_message(f"图片发送失败: {e}"))
                self.master.after(0, self.hide_voice_status)

        threading.Thread(target=upload_process, daemon=True).start()

    def process_and_stream(self, zh_reply, jp_text, voice_timing=None):
        self.speech_generation += 1
        generation = self.speech_generation
        self.speech_active = True
        pipeline_started_at = time.perf_counter()
        chunks = split_text_jp(
            normalize_japanese_tts_text(jp_text),
            max_len=30,
        )
        audio_queue = queue.Queue()

        def producer():
            try:
                for idx, chunk in enumerate(chunks):
                    if generation != self.speech_generation:
                        break
                    generation_started_at = time.perf_counter()
                    audio_path = self.generate_single_audio(chunk, idx)
                    if voice_timing is not None:
                        voice_timing["tts_generation_ms"] += (
                            time.perf_counter() - generation_started_at
                        ) * 1000
                    if not audio_path:
                        continue
                    if generation != self.speech_generation:
                        remove_generated_audio_file(audio_path)
                        break
                    audio_queue.put(audio_path)
            finally:
                audio_queue.put(None)

        def consumer():
            try:
                while generation == self.speech_generation:
                    try:
                        audio_path = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if audio_path is None:
                        break

                    try:
                        if generation != self.speech_generation:
                            continue
                        playback_started_at = time.perf_counter()
                        self.play_audio(audio_path)
                        self._schedule_ui(self.hide_voice_status)
                        while (
                            generation == self.speech_generation
                            and pygame.mixer.music.get_busy()
                        ):
                            time.sleep(0.05)
                        if voice_timing is not None:
                            voice_timing["tts_playback_ms"] += (
                                time.perf_counter() - playback_started_at
                            ) * 1000
                    except Exception as error:
                        print(f"音频播放等待失败: {error}")
                    finally:
                        remove_generated_audio_file(audio_path)

                if generation == self.speech_generation:
                    self._schedule_ui(
                        self.display_message,
                        zh_reply,
                        auto_hide=True,
                    )
                    self._schedule_ui(self.hide_voice_status)
            finally:
                if voice_timing is not None:
                    voice_timing["tts_pipeline_ms"] = round(
                        (time.perf_counter() - pipeline_started_at) * 1000,
                        1,
                    )
                    self.log_voice_timing(
                        voice_timing,
                        "completed"
                        if generation == self.speech_generation
                        else "interrupted",
                    )
                self._schedule_ui(self.finish_speech, generation)

        threading.Thread(target=producer, daemon=True).start()
        threading.Thread(target=consumer, daemon=True).start()

    def finish_speech(self, generation=None):
        if generation is not None and generation != self.speech_generation:
            return
        self.speech_active = False

    def generate_single_audio(self, chunk, idx):
        data = {
            "refer_wav_path": str(TTS_REFERENCE_AUDIO),
            "prompt_text": "そうなったのが自分のせいかもって責任を感じて ",
            "prompt_language": "ja",
            "text": chunk,
            "text_language": "ja"
        }
        try:
            response = requests.post(TTS_API_URL, json=data, timeout=30)
            if response.status_code == 200:
                TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = TTS_OUTPUT_DIR / f"output_{timestamp}_{idx}.wav"
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return str(output_path)
        except Exception as e:
            print(f"语音合成错误: {e}")
        return None

    def play_audio(self, filepath):
        try:
            self.live2d.play_audio(filepath)
        except Exception as e:
            print(f"播放音频错误: {e}")

    def on_closing(self):
        self.closing = True
        self.auto_screen_active = False
        self.voice_input.close()
        self.speech_generation += 1
        if self.screen_observation_after_id is not None:
            self.master.after_cancel(self.screen_observation_after_id)
            self.screen_observation_after_id = None
        if self.current_audio and self.current_audio.is_playing():
            self.current_audio.stop()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = Live2DChatApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()