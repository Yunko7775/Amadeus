import tkinter as tk
from tkinter import scrolledtext
import requests
import os
from datetime import datetime
import pygame
from Live2dTK import Live2dFrame  # 你自定义的 Live2D 显示类
import re
import time
import threading
from tkinter import filedialog

# 多语言表情关键词映射
EXPR_MAP = {
    "happy":      ["happy", "幸せ", "ハッピー", "満足しています"],
    "angry":      ["angry", "怒り"],
    "neutral":    ["neutral", "ニュートラル", "ニュートン", "中立"],
    "surprised":  ["surprised", "びっくり", "驚いた"],
    "sad":        ["sad", "悲しい"],
    "disgust":    ["disgust", "嫌悪感"],
}
EXPRESSION_KEYWORDS = [kw for v in EXPR_MAP.values() for kw in v]

def remove_expression_keywords(text, keywords):
    for kw in keywords:
        text = re.sub(fr"(\(|（)\s*{re.escape(kw)}\s*(\)|）)", "", text, flags=re.IGNORECASE)
    return text.strip()

def split_text_jp(text, max_len=40):
    sentences = re.split(r'(?<=[。！？\?\.])', text)
    result = []
    for s in sentences:
        s = s.strip()
        if s:
            while len(s) > max_len:
                result.append(s[:max_len])
                s = s[max_len:]
            if s:
                result.append(s)
    return result

def normalize_expression(expr, expr_map):
    for key, variants in expr_map.items():
        if expr in variants:
            return key
    return "neutral"

class Live2DChatApp:
    def __init__(self, master):
        self.master = master
        master.title("Amadeus")
        master.geometry("1200x750")
        pygame.mixer.init()
        self.current_audio = None
        self.voice_status_label = None
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        chat_frame = tk.Frame(main_frame, width=240, bg="white")
        chat_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, width=30, font=("Microsoft YaHei", 10)
        )
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_display.insert(tk.END, "OS：欢迎进入 Amadeus 聊天系统\n")
        self.chat_display.config(state=tk.DISABLED)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        model_frame = tk.Frame(right_frame)
        model_frame.pack(fill=tk.BOTH, expand=True)

        self.live2d = Live2dFrame(
            model_frame,
            model_versions=2,
            model_path=r"D:\Amadeus\Resources\kurisu\kurisu.model.json",
            width=900,
            height=700,
            offset_x=-0.3,
            background_path=r"D:\Amadeus\\Resources\bg.jpg"
        )
        self.live2d.pack(fill=tk.BOTH, expand=True)

        input_frame = tk.Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=10, padx=10)

        self.user_input = tk.Entry(input_frame, font=("Microsoft YaHei", 10))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", lambda e: self.send_message())

        tk.Button(
            input_frame, text="发送", command=self.send_message,
            font=("Microsoft YaHei", 10)
        ).pack(side=tk.RIGHT)

        # Voice maked Label
        self.voice_status_label = tk.Label(right_frame, text="", fg="#0078d7", font=("Microsoft YaHei", 10))
        self.voice_status_label.pack(fill=tk.X, pady=(0, 10))

        # Picture
        tk.Button(input_frame, text="上传图片", command=self.send_image, font=("Microsoft YaHei", 10)
                  ).pack(side=tk.RIGHT, padx=(5, 0))

    def show_voice_status(self, msg):
        self.voice_status_label.config(text=msg)
        self.voice_status_label.update_idletasks()

    def hide_voice_status(self):
        self.voice_status_label.config(text="")
        self.voice_status_label.update_idletasks()

    def send_to_sovits(self, text, split_long=True):
        if split_long:
            chunks = split_text_jp(text, max_len=40)
        else:
            chunks = [text]
        audio_paths = []
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            data = {
                "refer_wav_path": "D:/Amadeus/GPT-SoVITS-v2pro-20250604-nvidia50/Voice/Ref/ref.wav",
                "prompt_text": "そうなったのが自分のせいかもって責任を感じて ",
                "prompt_language": "ja",
                "text": chunk,
                "text_language": "ja"
            }
            try:
                response = requests.post("http://127.0.0.1:9880", json=data, timeout=30)
                if response.status_code == 200:
                    output_dir = "D:/Amadeus/GPT-SoVITS-v2pro-20250604-nvidia50/Voice/Out"
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    output_path = os.path.join(output_dir, f"output_{timestamp}_{idx}.wav")
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    audio_paths.append(output_path)
            except Exception as e:
                print(f"语音合成错误: {e}")
        return audio_paths

    def play_audio(self, filepath):
        try:
            self.live2d.play_audio(filepath)
        except Exception as e:
            print(f"播放音频错误: {e}")

    def play_audios_in_order(self, filepaths):
        for wav in filepaths:
            self.play_audio(wav)
            try:
                sound = pygame.mixer.Sound(wav)
                duration = sound.get_length()
                time.sleep(duration)
            except Exception as e:
                print(f"顺序播放失败: {e}")

    def display_message(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def send_message(self):
        message = self.user_input.get().strip()
        if not message:
            return

        # Display message
        self.display_message("你", message)
        self.user_input.delete(0, tk.END)

        # Show loading
        self.show_voice_status("⌛ 文本加载中...")

        def fetch_reply():
            try:
                response = requests.post(
                    "http://localhost:5000/ask",
                    json={
                        "user_id": "user_001",
                        "message": message
                    },
                    timeout=60
                )
                data = response.json()
                reply = data.get("reply", "")
                jp_text = data.get("japanese_reply", "")
                expression = data.get("expression", "neutral")

                #  Live2D expression
                try:
                    self.live2d.set_expression(expression)
                except Exception as e:
                    print(f"设置表情失败: {e}")


                reply = remove_expression_keywords(reply, EXPRESSION_KEYWORDS)

                # 显示中文回复在主线程
                self.master.after(0, lambda: self.display_message("Amadeus", reply))

                # 如果有日语文本，播放语音
                if jp_text:
                    print(f"合成语音文本: [{jp_text}]")
                    self.master.after(0, lambda: self.show_voice_status("🎤 Voice Loading..."))
                    threading.Thread(target=self.sovits_and_play, args=(jp_text,), daemon=True).start()
                else:
                    self.master.after(0, lambda: self.hide_voice_status())

            except Exception as e:
                self.master.after(0, lambda: self.display_message("系统", f"发生错误: {str(e)}"))
                self.master.after(0, self.hide_voice_status)

        threading.Thread(target=fetch_reply, daemon=True).start()
    def sovits_and_play(self, jp_text):
        audio_paths = self.send_to_sovits(jp_text, split_long=True)
        if audio_paths:
            self.play_audios_in_order(audio_paths)
        self.master.after(0, self.hide_voice_status)

    def on_closing(self):
        if self.current_audio and self.current_audio.is_playing():
            self.current_audio.stop()
        self.master.destroy()

    def send_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        question = self.user_input.get().strip() or "请描述这张图片"


        self.display_message("你", f"[发送图片] {os.path.basename(file_path)}\n提示：{question}")
        self.user_input.delete(0, tk.END)

        try:
            self.show_voice_status("📤 图片上传中...")

            with open(file_path, "rb") as f:
                files = {"image": f}
                data = {"question": question}

                #  Flask /ask_with_image 接口
                response = requests.post(
                    "http://localhost:5000/ask_with_image",
                    files=files,
                    data=data,
                    timeout=60
                )

            res = response.json()
            reply = res.get("reply", "")
            jp_text = res.get("japanese_reply", "")
            expression = res.get("expression", "neutral")

            try:
                self.live2d.set_expression(expression)
            except Exception as e:
                print(f"设置表情失败: {e}")

            # 显示中文回复
            self.display_message("Amadeus", reply)

            # Voice Maked
            if jp_text:
                print(f"合成语音文本: [{jp_text}]")
                self.show_voice_status("🎤 Voice Loading...")
                threading.Thread(target=self.sovits_and_play, args=(jp_text,), daemon=True).start()

        except Exception as e:
            self.display_message("系统", f"图片发送失败: {e}")
            self.hide_voice_status()


if __name__ == "__main__":
    root = tk.Tk()
    app = Live2DChatApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()