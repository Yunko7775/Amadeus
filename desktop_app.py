import tkinter as tk
import requests
from datetime import datetime
import pygame
from live2d_view import Live2DView
import re
import time
import threading
import queue
from tkinter import filedialog
import numpy as np
import io
from PIL import ImageGrab
import live2d
from expression_manager import extract_expression_and_text, remove_source_section
from proactive import should_send_proactive
from settings import (
    BACKEND_BASE_URL,
    BACKEND_REQUEST_TIMEOUT,
    LIVE2D_MODEL_PATH,
    PROACTIVE_IDLE_MINUTES,
    TTS_API_URL,
    TTS_OUTPUT_DIR,
    TTS_REFERENCE_AUDIO,
)

print(live2d.__file__)


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


class Live2DChatApp:
    def __init__(self, master):
        self.master = master
        master.title("Amadeus - Desk Pet")

        master.geometry("450x650+100+100")
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
        self.last_interaction_at = time.monotonic()
        self.proactive_sent_for_idle_period = False
        self.proactive_request_active = False
        self.speech_active = False
        self.activity_generation = 0
        self.closing = False

        self.setup_ui()
        self.setup_interactions()
        self.master.bind_all("<KeyPress>", self.record_user_activity, add="+")
        self.master.bind_all("<ButtonPress>", self.record_user_activity, add="+")
        self.master.after(60_000, self.check_idle_for_proactive)

    def record_user_activity(self, _event=None):
        self.last_interaction_at = time.monotonic()
        self.proactive_sent_for_idle_period = False
        self.activity_generation += 1

    def check_idle_for_proactive(self):
        if self.closing:
            return

        idle_seconds = PROACTIVE_IDLE_MINUTES * 60
        busy = (
            self.auto_screen_active
            or self.is_screen_processing
            or self.proactive_request_active
            or self.speech_active
            or bool(self.user_input.get().strip())
        )
        if should_send_proactive(
            now=time.monotonic(),
            last_interaction=self.last_interaction_at,
            idle_seconds=idle_seconds,
            already_sent=self.proactive_sent_for_idle_period,
            busy=busy,
        ):
            self.proactive_sent_for_idle_period = True
            self.proactive_request_active = True
            generation = self.activity_generation
            threading.Thread(
                target=self.request_proactive_message,
                args=(generation,),
                daemon=True,
            ).start()

        self.master.after(60_000, self.check_idle_for_proactive)

    def request_proactive_message(self, generation):
        try:
            response = requests.post(
                f"{BACKEND_BASE_URL}/proactive",
                json={"user_id": "user_001"},
                timeout=(5, BACKEND_REQUEST_TIMEOUT),
            )
            response.raise_for_status()
            data = response.json()
            self.master.after(
                0,
                lambda: self.show_proactive_message(data, generation),
            )
        except Exception as e:
            print(f"[proactive] Request failed: {e}")
            try:
                self.master.after(0, self.finish_proactive_request)
            except tk.TclError:
                pass

    def show_proactive_message(self, data, generation):
        self.proactive_request_active = False
        if (
            self.closing
            or generation != self.activity_generation
            or time.monotonic() - self.last_interaction_at < PROACTIVE_IDLE_MINUTES * 60
            or not data.get("reply")
        ):
            return

        reply = data["reply"]
        japanese_reply = data.get("japanese_reply", "")
        expression = data.get("expression", "neutral")
        try:
            self.live2d.set_expression(expression)
        except Exception as e:
            print(f"[proactive] Failed to set expression: {e}")
        self.display_message(reply, auto_hide=False)
        if japanese_reply:
            self.show_voice_status("🎤 语音生成中...")
            self.process_and_stream(reply, japanese_reply)
        else:
            self.display_message(reply, auto_hide=True)

    def finish_proactive_request(self):
        self.proactive_request_active = False

    def setup_ui(self):
        self.live2d = Live2DView(
            self.master,
            model_versions=2,
            model_path=str(LIVE2D_MODEL_PATH),
            width=450,
            height=550,
            offset_x=-0.5,
            background_path=""
        )
        self.live2d.config(bg=self.transparent_color)
        self.live2d.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 聊天气泡框
        self.bubble_frame = tk.Frame(self.master, bg="#ffffff", bd=2, relief="solid")
        self.bubble_text = tk.Text(
            self.bubble_frame, bg="#ffffff", fg="#333333",
            font=("Microsoft YaHei", 9), wrap="char", bd=0, highlightthickness=0
        )
        self.bubble_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.bubble_text.bind("<MouseWheel>",
                              lambda e: self.bubble_text.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # 底部区域
        self.input_frame = tk.Frame(self.master, bg=self.transparent_color)
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=20)

        self.voice_status_label = tk.Label(
            self.input_frame, text="", fg="#00a2ff", bg=self.transparent_color, font=("Microsoft YaHei", 9, "bold")
        )
        self.voice_status_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        input_container = tk.Frame(self.input_frame, bg=self.transparent_color)
        input_container.pack(fill=tk.X)

        self.user_input = tk.Entry(input_container, font=("Microsoft YaHei", 10), bg="#f0f0f0", relief="flat")
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        self.user_input.bind("<Return>", lambda e: self.send_message())

        btn_send = tk.Button(
            input_container, text="发送", command=self.send_message,
            font=("Microsoft YaHei", 10), bg="#4caf50", fg="white", relief="flat", padx=8
        )
        btn_send.pack(side=tk.RIGHT, padx=(5, 0))

        # 陪玩控制按钮
        self.btn_auto_screen = tk.Button(
            input_container, text="开启观测", command=self.toggle_auto_screen,
            font=("Microsoft YaHei", 9), bg="#673ab7", fg="white", relief="flat", padx=6
        )
        self.btn_auto_screen.pack(side=tk.RIGHT, padx=(5, 0))

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

        self.bubble_frame.place(relx=0.5, rely=0.02, relwidth=0.82, height=130, anchor="n")
        self.bubble_frame.lift()

        if self.bubble_timer:
            self.master.after_cancel(self.bubble_timer)
            self.bubble_timer = None

        if auto_hide:
            self.bubble_timer = self.master.after(15000, self.hide_bubble)

    def hide_bubble(self):
        self.bubble_frame.place_forget()

    # ==================== 陪玩功能核心逻辑 ====================
    def toggle_auto_screen(self):
        self.auto_screen_active = not self.auto_screen_active
        if self.auto_screen_active:
            self.btn_auto_screen.config(text="⏹停止观测", bg="#f44336")
            self.show_voice_status("屏幕观测已开启")
            threading.Thread(target=self.auto_screen_loop, daemon=True).start()
        else:
            self.btn_auto_screen.config(text="开启观测", bg="#673ab7")
            self.show_voice_status("⏹观测模式已关闭")

    def auto_screen_loop(self):
        while self.auto_screen_active:
            if not self.is_screen_processing:
                self.capture_and_analyze_screen()

            for _ in range(self.screen_interval):
                if not self.auto_screen_active:
                    break
                time.sleep(1)

    def capture_and_analyze_screen(self):
        self.is_screen_processing = True
        try:
            self.master.after(0, lambda: self.show_voice_status("👀 观察屏幕中..."))
            self.master.withdraw()
            self.master.update()  # 强制刷新界面

            screenshot = ImageGrab.grab()

            img_byte_arr = io.BytesIO()
            screenshot.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # 4. 立即恢复显示窗口
            self.master.deiconify()

            # 5. 准备发送给后端的 /ask_with_image 接口
            files = {'image': ('screen.png', img_byte_arr, 'image/png')}
            data = {
                'user_id': 'user_001',
                'question': '（连线观察中）快看看我当前屏幕上在看什么、在干嘛？用你的风格（傲娇、带点学术吐槽）简单点评两句，绝对不准提窗口或程序。'
            }

            response = requests.post(
                f"{BACKEND_BASE_URL}/ask_with_image",
                files=files,
                data=data,
                timeout=(5, BACKEND_REQUEST_TIMEOUT)
            )

            if response.status_code == 200:
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

                if jp_text and self.auto_screen_active:
                    self.master.after(0, lambda: self.show_voice_status("🎤 语音生成中..."))
                    self.process_and_stream(reply, jp_text)
                else:
                    self.master.after(0, lambda: self.display_message(reply, auto_hide=True))
                    self.master.after(0, self.hide_voice_status)

        except Exception as e:
            print(f"❌ 陪玩屏幕分析出错: {e}")
            self.master.deiconify()  # 确保出错时窗口也能恢复显示
            self.master.after(0, self.hide_voice_status)
        finally:
            self.is_screen_processing = False

    # ==================== 普通文本 & 图像发送 ====================
    def send_message(self):
        message = self.user_input.get().strip()
        if not message:
            return

        self.user_input.delete(0, tk.END)
        self.show_voice_status("⌛ 思考中...")

        def fetch_reply():
            try:
                response = requests.post(
                    f"{BACKEND_BASE_URL}/ask",
                    json={"user_id": "user_001", "message": message},
                    timeout=(5, BACKEND_REQUEST_TIMEOUT)
                )
                data = response.json()
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
                    self.process_and_stream(reply, jp_text)
                else:
                    self.master.after(0, lambda: self.display_message(reply, auto_hide=True))
                    self.master.after(0, self.hide_voice_status)

            except Exception as e:
                self.master.after(0, lambda: self.display_message(f"系统错误: {str(e)}"))
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

    def process_and_stream(self, zh_reply, jp_text):
        self.speech_active = True
        chunks = split_text_jp(jp_text, max_len=30)
        audio_queue = queue.Queue()

        def producer():
            for idx, chunk in enumerate(chunks):
                audio_path = self.generate_single_audio(chunk, idx)
                if audio_path:
                    audio_queue.put(audio_path)
            audio_queue.put(None)

        def consumer():
            first_sentence = True
            while True:
                audio_path = audio_queue.get()
                if audio_path is None:
                    break

                if first_sentence:
                    self.master.after(0, self.hide_voice_status)
                    first_sentence = False

                if audio_path:
                    self.play_audio(audio_path)
                    try:
                        sound = pygame.mixer.Sound(audio_path)
                        duration = sound.get_length()
                        time.sleep(duration)
                    except Exception as e:
                        print(f"音频播放等待失败: {e}")

            self.master.after(0, lambda: self.display_message(zh_reply, auto_hide=True))
            self.master.after(0, self.hide_voice_status)
            self.master.after(0, self.finish_speech)

        threading.Thread(target=producer, daemon=True).start()
        threading.Thread(target=consumer, daemon=True).start()

    def finish_speech(self):
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
        if self.current_audio and self.current_audio.is_playing():
            self.current_audio.stop()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = Live2DChatApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()