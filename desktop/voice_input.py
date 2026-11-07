import queue
import threading
import time
from collections.abc import Callable

import numpy as np


SAMPLE_RATE = 16_000
MAX_RECORDING_SECONDS = 60
MIN_RECORDING_SAMPLES = int(SAMPLE_RATE * 0.3)
VOICE_ACTIVITY_RMS_THRESHOLD = 0.012
MIN_UTTERANCE_SPEECH_SAMPLES = int(SAMPLE_RATE * 0.15)
VOICE_ENDPOINT_SILENCE_SAMPLES = int(SAMPLE_RATE * 0.5)
_MOUSE_FORWARD_HOTKEYS = {"mouse_x2", "x2"}


def format_hotkey_label(hotkey: str) -> str:
    if hotkey.lower() in _MOUSE_FORWARD_HOTKEYS:
        return "滑鼠前進側鍵"
    return hotkey.upper()


class PushToTalk:
    def __init__(
        self,
        on_recording_changed: Callable[[bool], None],
        on_audio_ready: Callable[[np.ndarray], None],
        on_error: Callable[[str], None],
        hotkey: str = "mouse_x2",
        on_partial_audio: Callable[[np.ndarray], None] | None = None,
        on_utterance_audio: Callable[[np.ndarray, int, float, float], None]
        | None = None,
        should_pause_utterance_detection: Callable[[], bool] | None = None,
    ):
        self.on_recording_changed = on_recording_changed
        self.on_audio_ready = on_audio_ready
        self.on_error = on_error
        self.on_partial_audio = on_partial_audio
        self.on_utterance_audio = on_utterance_audio
        self.should_pause_utterance_detection = should_pause_utterance_detection
        self.hotkey = hotkey.lower()
        self.hotkey_label = format_hotkey_label(self.hotkey)
        self._lock = threading.Lock()
        self._pressed = False
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._sample_count = 0
        self._hit_limit = False
        self._listener = None
        self._key = None
        self._button = None
        self._closed = False
        self._key_down = False
        self._partial_queue = None
        self._partial_sample_count = 0
        self._utterance_queue = None
        self._recording_id = 0
        self._utterance_chunks: list[np.ndarray] = []
        self._utterance_started_at = None
        self._silence_sample_count = 0
        self._utterance_speech_sample_count = 0

    def start_listener(self) -> None:
        if self.hotkey in _MOUSE_FORWARD_HOTKEYS:
            from pynput import mouse

            self._button = mouse.Button.x2
            self._listener = mouse.Listener(on_click=self._on_mouse_click)
            self._listener.start()
            return

        from pynput import keyboard

        self._key = getattr(keyboard.Key, self.hotkey, None)
        if self._key is None:
            raise ValueError(f"Unsupported push-to-talk key: {self.hotkey}")
        self._listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._listener.start()

    def _on_key_press(self, key) -> None:
        if key == self._key:
            with self._lock:
                if self._key_down:
                    return
                self._key_down = True
            self.toggle_recording()

    def _on_key_release(self, key) -> None:
        if key == self._key:
            with self._lock:
                self._key_down = False

    def _on_mouse_click(self, _x, _y, button, pressed) -> None:
        if button == self._button and pressed:
            self.toggle_recording()

    def toggle_recording(self) -> None:
        with self._lock:
            is_recording = self._pressed
        if is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        with self._lock:
            if self._closed or self._pressed:
                return
            self._pressed = True
            self._chunks = []
            self._sample_count = 0
            self._hit_limit = False
            self._recording_id += 1
            self._utterance_chunks = []
            self._utterance_started_at = None
            self._silence_sample_count = 0
            self._utterance_speech_sample_count = 0
            self._partial_sample_count = SAMPLE_RATE * 2
            if self.on_utterance_audio is not None:
                self._utterance_queue = queue.Queue()
                utterance_queue = self._utterance_queue
                recording_id = self._recording_id
                threading.Thread(
                    target=self._process_utterances,
                    args=(utterance_queue, recording_id),
                    daemon=True,
                ).start()
            if self.on_partial_audio is not None:
                self._partial_queue = queue.Queue(maxsize=1)
                partial_queue = self._partial_queue
                threading.Thread(
                    target=self._process_partial_audio,
                    args=(partial_queue,),
                    daemon=True,
                ).start()

        self.on_recording_changed(True)
        try:
            import sounddevice

            stream = sounddevice.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._capture_audio,
            )
            with self._lock:
                self._stream = stream
            stream.start()
        except Exception as error:
            with self._lock:
                self._pressed = False
                self._stream = None
                self._chunks = []
                partial_queue = self._partial_queue
                self._partial_queue = None
                utterance_queue = self._utterance_queue
                self._utterance_queue = None
            if partial_queue is not None:
                while True:
                    try:
                        partial_queue.get_nowait()
                    except queue.Empty:
                        break
                partial_queue.put_nowait(None)
            if utterance_queue is not None:
                utterance_queue.put_nowait(None)
            self.on_recording_changed(False)
            self.on_error(f"無法開啟麥克風：{error}")

    def _capture_audio(self, input_data, frame_count, _time_info, status) -> None:
        import sounddevice

        if status:
            self.on_error(f"麥克風收音狀態異常：{status}")

        completed_utterance = None
        with self._lock:
            if not self._pressed:
                return
            continuous_mode = self.on_utterance_audio is not None
            remaining = (
                frame_count
                if continuous_mode
                else SAMPLE_RATE * MAX_RECORDING_SECONDS - self._sample_count
            )
            if remaining <= 0:
                self._hit_limit = True
                raise sounddevice.CallbackStop
            captured = input_data[: min(frame_count, remaining), 0].copy()
            if not continuous_mode or self.on_partial_audio is not None:
                self._chunks.append(captured)
            self._sample_count += len(captured)
            pause_detection = (
                continuous_mode
                and self.should_pause_utterance_detection is not None
                and self.should_pause_utterance_detection()
            )
            if pause_detection:
                self._utterance_chunks = []
                self._utterance_started_at = None
                self._silence_sample_count = 0
                self._utterance_speech_sample_count = 0
                return
            rms = float(np.sqrt(np.mean(np.square(captured, dtype=np.float64))))
            is_speech = rms >= VOICE_ACTIVITY_RMS_THRESHOLD
            if is_speech and self._utterance_started_at is None:
                self._utterance_started_at = time.perf_counter()
                self._utterance_chunks = []
                self._utterance_speech_sample_count = 0
            if self._utterance_started_at is not None:
                self._utterance_chunks.append(captured)
                if is_speech:
                    self._silence_sample_count = 0
                    self._utterance_speech_sample_count += len(captured)
                else:
                    self._silence_sample_count += len(captured)
                if self._silence_sample_count >= VOICE_ENDPOINT_SILENCE_SAMPLES:
                    if (
                        self._utterance_speech_sample_count
                        >= MIN_UTTERANCE_SPEECH_SAMPLES
                    ):
                        completed_utterance = (
                            np.concatenate(self._utterance_chunks),
                            self._recording_id,
                            self._utterance_started_at,
                            time.perf_counter(),
                        )
                        if self._utterance_queue is not None:
                            self._utterance_queue.put(completed_utterance)
                    self._utterance_chunks = []
                    self._utterance_started_at = None
                    self._silence_sample_count = 0
                    self._utterance_speech_sample_count = 0
            if (
                self._partial_queue is not None
                and self._sample_count >= self._partial_sample_count
            ):
                partial_audio = np.concatenate(self._chunks)[-SAMPLE_RATE * 8 :]
                try:
                    self._partial_queue.put_nowait(partial_audio)
                except queue.Full:
                    try:
                        self._partial_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._partial_queue.put_nowait(partial_audio)
                self._partial_sample_count += SAMPLE_RATE * 2
            if (
                not continuous_mode
                and self._sample_count >= SAMPLE_RATE * MAX_RECORDING_SECONDS
            ):
                self._hit_limit = True
                raise sounddevice.CallbackStop

    def _process_utterances(self, utterance_queue, recording_id: int) -> None:
        while True:
            item = utterance_queue.get()
            if item is None:
                return
            audio, item_recording_id, started_at, ended_at = item
            if item_recording_id != recording_id:
                continue
            try:
                if self.on_utterance_audio is not None:
                    self.on_utterance_audio(
                        audio,
                        recording_id,
                        started_at,
                        ended_at,
                    )
            except Exception as error:
                self.on_error(f"即時語音辨識失敗：{error}")

    def _process_partial_audio(self, partial_queue) -> None:
        while True:
            audio = partial_queue.get()
            if audio is None:
                return
            try:
                if self.on_partial_audio is not None:
                    self.on_partial_audio(audio)
            except Exception as error:
                self.on_error(f"即時語音辨識失敗：{error}")

    def stop_recording(self, discard: bool = False) -> None:
        with self._lock:
            if not self._pressed:
                return
            self._pressed = False
            stream = self._stream
            chunks = self._chunks
            sample_count = self._sample_count
            hit_limit = self._hit_limit
            recording_id = self._recording_id
            final_utterance = None
            if (
                not discard
                and self._utterance_started_at is not None
                and self._utterance_speech_sample_count
                >= MIN_UTTERANCE_SPEECH_SAMPLES
            ):
                final_utterance = (
                    np.concatenate(self._utterance_chunks),
                    recording_id,
                    self._utterance_started_at,
                    time.perf_counter(),
                )
            utterance_queue = self._utterance_queue
            partial_queue = self._partial_queue
            self._stream = None
            self._chunks = []
            self._sample_count = 0
            self._hit_limit = False
            self._partial_queue = None
            self._partial_sample_count = 0
            self._utterance_queue = None
            self._utterance_chunks = []
            self._utterance_started_at = None
            self._silence_sample_count = 0
            self._utterance_speech_sample_count = 0

        if utterance_queue is not None:
            if discard:
                while True:
                    try:
                        utterance_queue.get_nowait()
                    except queue.Empty:
                        break
            elif final_utterance is not None:
                utterance_queue.put(final_utterance)
            utterance_queue.put(None)
        if partial_queue is not None:
            while True:
                try:
                    partial_queue.get_nowait()
                except queue.Empty:
                    break
            partial_queue.put_nowait(None)

        if stream is not None:
            try:
                if stream.active:
                    stream.stop()
            except Exception as error:
                self.on_error(f"停止麥克風收音失敗：{error}")
            finally:
                try:
                    stream.close()
                except Exception as error:
                    self.on_error(f"關閉麥克風收音失敗：{error}")

        self.on_recording_changed(False)
        if discard:
            return
        if hit_limit:
            self.on_error("單次錄音已達 60 秒上限")
        if self.on_utterance_audio is not None:
            return
        if sample_count < MIN_RECORDING_SAMPLES:
            if sample_count:
                self.on_error(
                    f"錄音太短了，請按住{self.hotkey_label}多說一點"
                )
            return

        self.on_audio_ready(np.concatenate(chunks))

    def close(self) -> None:
        with self._lock:
            self._closed = True
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self.stop_recording(discard=True)
