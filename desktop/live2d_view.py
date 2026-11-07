from OpenGL.GL import *
from pyopengltk import OpenGLFrame
from pyautogui import position
from os import path
import live2d.v3 as live2d
import live2d.v2 as live2d_v2
from live2d.utils.lipsync import WavHandler
from live2d.utils.image import Image
from shared.settings import ROOT_DIR

import pygame

RESOURCES_DIRECTORY = str(ROOT_DIR / "Resources")


# 自测开关：True = 强制歪头 25°；验证完改回 False
SELF_TEST = False


# 情绪 → 头部/身体表演偏移（只用 moc 里确定存在的角度/身体/眼球参数）
EXPRESSION_ACTING = {
    "happy":     {"ParamAngleX": -4.0},
    "angry":     {"ParamAngleX": 8.0, "ParamBodyAngleX": 4.0},
    "sad":       {"ParamAngleX": -6.0},
    "surprised": {"ParamAngleX": -8.0},
    "disgust":   {},
    "neutral":   {},
}

MODEL_EXPRESSIONS = {
    "neutral": "neutral",
    "happy": "joy",
    "smile1": "smile1",
    "smile2": "smile2",
    "angry": "anger",
    "sad": "sadness",
    "surprised": "surprise",
    "disgust": "unhappy",
    "shy": "shy",
    "bashful": "shy2",
    "disdain": "disdain",
}

MODEL_MOTIONS = {
    "happy": "joy",
    "smile1": "smile1",
    "smile2": "smile2",
    "angry": "anger",
    "sad": "sadness",
    "surprised": "surprise",
    "disgust": "unhappy",
    "shy": "shy",
    "bashful": "shy2",
    "disdain": "disdain",
}


class Live2DView(OpenGLFrame):
    def __init__(self, *args, model_versions=3, model_path=None, background_path=None, fps=60, offset_x=0.0, offset_y=0.0, **kwargs):
        self.live2d_model_version = model_versions
        self.live2d = live2d_v2 if model_versions == 2 else live2d
        self.live2d_model_path = model_path
        self.fps = fps

        if not self.live2d_model_path:
            raise ValueError("Live2DView requires model_path parameter")

        OpenGLFrame.__init__(self, *args, **kwargs)

        self.model = None
        self.wav_handler = WavHandler()
        self.lip_sync_factor = 3.0
        self.mouth_param_name = "ParamMouthOpenY"
        self.last_mouth_value = 0.0
        self.smooth_factor = 0.3

        self.background_path = background_path
        self.background_image = None

        self.offset_x = offset_x
        self.offset_y = offset_y

        self.acting = {}
        self.motion_group = None
        self.reset_motion_transients = False

        pygame.mixer.init()

    def initgl(self):
        self.animate = max(1, round(1000 / self.fps))
        self.live2d.init()
        self.live2d.setLogEnable(True)

        if self.live2d_model_version == 3:
            self.live2d.glInit()

        self.model = self.live2d.LAppModel()
        print("Live2D module:", self.live2d.__file__)
        full_model_path = self.live2d_model_path if path.isabs(self.live2d_model_path) else path.join(RESOURCES_DIRECTORY, self.live2d_model_path)
        self.model.LoadModelJson(full_model_path)
        if self.live2d_model_version == 3:
            self.model.SetExpression("neutral")
            self.model.SetAutoBlinkEnable(True)
            self.model.SetAutoBreathEnable(True)
            self.model.StartRandomMotion("idle", self.live2d.MotionPriority.IDLE)
            self.motion_group = "idle"
        self.model.Resize(self.width, self.height)

        self.acting = {}

        print("Live2D model loaded:", full_model_path)

        if hasattr(self.model, "SetOffset"):
            try:
                self.model.SetOffset(self.offset_x, self.offset_y)
            except Exception as e:
                print(f"Offset failed: {e}")

        if self.background_path and path.exists(self.background_path):
            try:
                self.background_image = Image(self.background_path)
            except Exception as e:
                print(f"Failed to load background image: {e}")
        else:
            print("No background image or file not found.")

    def tkResize(self, event):
        self.width, self.height = event.width, event.height
        if not self.winfo_ismapped():
            return
        if not self.context_created or self.model is None:
            super().tkResize(event)
            return

        self.tkMakeCurrent()
        glViewport(0, 0, self.width, self.height)
        self.model.Resize(self.width, self.height)

    def _live2d_(self):
        try:
            screen_x, screen_y = position()
            x = screen_x - self.winfo_rootx()
            y = screen_y - self.winfo_rooty()
            self.model.Drag(x, y)
        except Exception:
            pass

        self.live2d.clearBuffer()

        if self.background_image:
            self.background_image.Draw()

        self._apply_acting()
        self._update_motion()
        self.model.Update()
        self._apply_motion_safety()

        if self.wav_handler.Update():
            rms = self.wav_handler.GetRms()
            target_value = rms * self.lip_sync_factor
            smooth_value = (self.smooth_factor * target_value + (1 - self.smooth_factor) * self.last_mouth_value)
            try:
                self.model.SetParameterValue(self.mouth_param_name, smooth_value)
            except Exception as e:
                print(f"Failed to set mouth parameter: {e}")
            self.last_mouth_value = smooth_value

        self.model.Draw()

    def redraw(self):
        glClear(GL_COLOR_BUFFER_BIT)
        self._live2d_()

    def play_audio(self, audio_file):
        full_audio_path = audio_file if path.isabs(audio_file) else path.join(RESOURCES_DIRECTORY, audio_file)
        if not path.exists(full_audio_path):
            print("Audio file not found:", full_audio_path)
            return

        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(full_audio_path)
            pygame.mixer.music.play()
            self.wav_handler.Start(full_audio_path)
        except Exception as e:
            print(f"Audio playback failed: {e}")

    def stop_audio(self):
        try:
            pygame.mixer.music.stop()
            self.model.SetParameterValue(self.mouth_param_name, 0.0)
            self.last_mouth_value = 0.0
        except Exception as e:
            print(f"Failed to reset mouth position: {e}")

    def set_expression(self, expression_name):
        model_expression = MODEL_EXPRESSIONS.get(expression_name)
        if model_expression is None:
            return

        self.model.SetExpression(model_expression)
        if self.live2d_model_version == 3:
            self.model.StopAllMotions()
            motion_group = MODEL_MOTIONS.get(expression_name, "idle")
            self.model.StartRandomMotion(motion_group, self.live2d.MotionPriority.NORMAL)
            self.motion_group = motion_group
            self.reset_motion_transients = True

        self.acting = EXPRESSION_ACTING.get(expression_name, {})
        print(f"[set_expression] {expression_name} acting={list(self.acting)}")

    def _update_motion(self):
        if self.live2d_model_version != 3 or not self.motion_group:
            return

        if self.motion_group == "idle" or not self.model.IsMotionFinished(self.motion_group):
            return

        self.reset_motion_transients = True
        self.motion_group = "idle"
        self.model.StartRandomMotion("idle", self.live2d.MotionPriority.IDLE)

    def _apply_motion_safety(self):
        if self.live2d_model_version != 3:
            return

        self.model.SetParameterValue("ParamAngleZ", 0.0)
        self.model.SetParameterValue("ParamBodyAngleZ", 0.0)
        if self.reset_motion_transients:
            self.model.SetParameterValue("ParamCheek", 0.0)
            self.reset_motion_transients = False

    def _apply_acting(self):
        if SELF_TEST:
            self.model.SetParameterValue("ParamAngleX", 25.0)
            return
        for pid, v in self.acting.items():
            self.model.SetParameterValue(pid, v)

    def end(self):
        self.live2d.dispose()
