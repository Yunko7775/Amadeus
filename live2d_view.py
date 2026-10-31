from OpenGL.GL import *
from pyopengltk import OpenGLFrame
from pyautogui import position
from os import path
from time import sleep
import live2d.v3 as live2d
import live2d.v2 as live2d_v2
from live2d.utils.lipsync import WavHandler
from live2d.utils.image import Image
from settings import ROOT_DIR

import pygame

RESOURCES_DIRECTORY = str(ROOT_DIR / "Resources")


# 自测开关：True = 强制歪头 25° 并关闭跟随（用来验证引擎挂点是否生效）；验证完改回 False
SELF_TEST = False


# 情绪 → 头部/身体表演偏移（只用 moc 里确定存在的角度/身体/眼球参数）
EXPRESSION_ACTING = {
    "happy":     {"PARAM_ANGLE_Z": 6.0, "PARAM_ANGLE_X": -4.0},
    "angry":     {"PARAM_ANGLE_X": 8.0, "PARAM_ANGLE_Z": -6.0, "PARAM_BODY_ANGLE_X": 4.0},
    "sad":       {"PARAM_ANGLE_X": -6.0, "PARAM_ANGLE_Z": 4.0},
    "surprised": {"PARAM_ANGLE_X": -8.0},
    "disgust":   {"PARAM_ANGLE_Z": -5.0},
    "neutral":   {},
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
        self.mouth_param_name = 'PARAM_MOUTH_OPEN_Y'
        self.last_mouth_value = 0.0
        self.smooth_factor = 0.3

        self.background_path = background_path
        self.background_image = None

        self.offset_x = offset_x
        self.offset_y = offset_y

        # 表演层状态
        self.acting = {}
        self._eye_factor = 1.0

        self.expression_parameters = {
            "neutral": {
                "PARAM_MOUTH_FORM": 0.0,
                "PARAM_MOUTH_OPEN_Y": 0.0,
                "PARAM_EYE_L_OPEN": 1.0,
                "PARAM_EYE_R_OPEN": 1.0,
                "PARAM_BROW_L_Y": 0.0,
                "PARAM_BROW_R_Y": 0.0,
                "PARAM_EYE_BALL_X": 0.0,
                "PARAM_EYE_BALL_Y": 0.0,
            },
            "happy": {
                "PARAM_MOUTH_FORM": 1.0,
                "PARAM_MOUTH_OPEN_Y": 0.3,
                "PARAM_EYE_L_SMILE": 1.0,
                "PARAM_EYE_R_SMILE": 1.0,
                "PARAM_EYE_L_OPEN": 1.0,
                "PARAM_EYE_R_OPEN": 1.0,
                "PARAM_BROW_L_Y": 0.5,
                "PARAM_BROW_R_Y": 0.5,
            },
            "surprised": {
                "PARAM_MOUTH_FORM": 0.0,
                "PARAM_MOUTH_OPEN_Y": 1.0,
                "PARAM_EYE_L_OPEN": 1.0,
                "PARAM_EYE_R_OPEN": 1.0,
                "PARAM_BROW_L_Y": 1.0,
                "PARAM_BROW_R_Y": 1.0,
            },
            "sad": {
                "PARAM_MOUTH_FORM": -0.5,
                "PARAM_MOUTH_OPEN_Y": 0.2,
                "PARAM_EYE_L_OPEN": 0.3,
                "PARAM_EYE_R_OPEN": 0.3,
                "PARAM_BROW_L_Y": -1.0,
                "PARAM_BROW_R_Y": -1.0,
                "PARAM_BROW_L_FORM": -1.0,
                "PARAM_BROW_R_FORM": -1.0,
            },
            "angry": {
                "PARAM_MOUTH_FORM": -0.5,
                "PARAM_MOUTH_OPEN_Y": 0.3,
                "PARAM_EYE_L_OPEN": 0.7,
                "PARAM_EYE_R_OPEN": 0.7,
                "PARAM_BROW_L_FORM": 1.0,
                "PARAM_BROW_R_FORM": 1.0,
                "PARAM_BROW_L_Y": -0.5,
                "PARAM_BROW_R_Y": -0.5,
            },
            "disgust": {
                "PARAM_MOUTH_FORM": -1.0,
                "PARAM_MOUTH_OPEN_Y": 0.0,
                "PARAM_EYE_L_OPEN": 0.3,
                "PARAM_EYE_R_OPEN": 0.3,
                "PARAM_BROW_L_Y": -1.0,
                "PARAM_BROW_R_Y": -1.0,
            },
        }

        pygame.mixer.init()

    def initgl(self):
        self.animate = 1
        self.live2d.init()
        self.live2d.setLogEnable(True)

        if self.live2d_model_version == 3:
            self.live2d.glewInit()

        self.model = self.live2d.LAppModel()
        print("live2d_v2 module:", live2d_v2.__file__)
        full_model_path = self.live2d_model_path if path.isabs(self.live2d_model_path) else path.join(RESOURCES_DIRECTORY, self.live2d_model_path)
        self.model.LoadModelJson(full_model_path)
        self.model.Resize(self.width, self.height)

        # 接上引擎挂点 + 跟随权重 + 表演状态
        self.model.onBeforePhysics = self._apply_acting
        self.model.dragWeight = 0.0 if SELF_TEST else 1.0
        self.acting = {}
        self._eye_factor = 1.0

        print("physics loaded:", self.model.physics is not None)

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

        self.model.Update()

        if self.wav_handler.Update():
            rms = self.wav_handler.GetRms()
            target_value = rms * self.lip_sync_factor
            smooth_value = (self.smooth_factor * target_value + (1 - self.smooth_factor) * self.last_mouth_value)
            try:
                self.model.SetParameterValue(self.mouth_param_name, smooth_value)
            except Exception as e:
                print(f"Failed to set mouth parameter: {e}")
            self.last_mouth_value = smooth_value

        self._apply_eye_factor()

        self.model.Draw()

    def redraw(self):
        glClear(GL_COLOR_BUFFER_BIT)
        self._live2d_()
        if self.fps < 120:
            sleep(1 / self.fps)

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
        if expression_name not in self.expression_parameters:
            return

        preset = self.expression_parameters[expression_name]
        for param, value in preset.items():
            try:
                self.model.SetParameterValue(param, value)
            except Exception as e:
                print(f"Failed to set parameter {param}: {e}")

        # 表情 → 头部/身体表演动作（这些参数一定存在）
        self.acting = EXPRESSION_ACTING.get(expression_name, {})
        # 眼睛开合用因子（眨眼每帧会覆盖绝对值，乘因子保留眨眼）
        self._eye_factor = float(preset.get("PARAM_EYE_L_OPEN", 1.0))
        # 自测时强制关跟随；否则非中性表情压到 0.3，让表演可见
        if SELF_TEST:
            self.model.dragWeight = 0.0
        else:
            self.model.dragWeight = 0.3 if self.acting else 1.0
        print(f"[set_expression] {expression_name} acting={list(self.acting)} dragWeight={self.model.dragWeight} eye={self._eye_factor}")

    def _apply_acting(self, model):
        if SELF_TEST:
            model.live2DModel.addToParamFloat("PARAM_ANGLE_X", 25.0, 1.0)
            return
        for pid, v in self.acting.items():
            model.live2DModel.addToParamFloat(pid, v, 1.0)

    def _apply_eye_factor(self):
        if self._eye_factor == 1.0:
            return
        try:
            m = self.model.live2DModel
            for pid in ("PARAM_EYE_L_OPEN", "PARAM_EYE_R_OPEN"):
                idx = m.modelContext.getParamIndex(pid)
                if idx is None or idx < 0:
                    continue
                v = m.getParamFloat(idx)
                m.setParamFloat(pid, v * self._eye_factor, 1.0)
        except Exception:
            pass

    def end(self):
        self.live2d.dispose()
