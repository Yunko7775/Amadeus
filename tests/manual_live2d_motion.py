import json
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pygame

ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = ROOT_DIRECTORY / "Resources" / "kurisu_cubism3"
MODEL_CONFIG_PATH = MODEL_DIRECTORY / "kurisu.model3.json"
sys.path.insert(0, str(ROOT_DIRECTORY))

MOTION_LABELS = {
    "neutral": "Neutral (reset expression)",
    "anger": "Anger",
    "joy": "Joy",
    "sadness": "Sadness",
    "shy": "Shy",
    "shy2": "Shy 2",
    "smile1": "Smile 1",
    "smile2": "Smile 2",
    "surprise": "Surprise",
    "unhappy": "Unhappy",
    "random1": "Gesture 1",
    "random3": "Gesture 3",
    "random4": "Gesture 4 (Disdain)",
    "random5": "Gesture 5",
}
MOTION_SEQUENCE = tuple(f"manual_{name}" for name in MOTION_LABELS)
MOTION_EXPRESSIONS = {
    "neutral": "neutral",
    "anger": "angry",
    "joy": "happy",
    "sadness": "sad",
    "shy": "shy",
    "shy2": "bashful",
    "smile1": "smile1",
    "smile2": "smile2",
    "surprise": "surprised",
    "unhappy": "disgust",
    "random3": "neutral",
    "random4": "disdain",
}
HAND_POSES = {
    "crossed": ("Hands crossed", 1.0),
    "relaxed": ("Hands relaxed", 0.0),
}


def take_next_motion(sequence: list[str]) -> str | None:
    return sequence.pop(0) if sequence else None


def create_manual_model_config():
    model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    test_motions = {}
    neutral_motion = json.loads(
        (MODEL_DIRECTORY / "neutral.motion3.json").read_text(encoding="utf-8")
    )
    neutral_motion["Meta"]["Loop"] = False

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".motion3.json",
        prefix="_manual_neutral_",
        dir=MODEL_DIRECTORY,
        delete=False,
    ) as temporary_motion:
        json.dump(neutral_motion, temporary_motion, ensure_ascii=False, indent=2)
        neutral_motion_path = Path(temporary_motion.name)

    for group_motions in model_config["FileReferences"]["Motions"].values():
        for motion in group_motions:
            motion_file = motion["File"]
            motion_name = Path(motion_file).name.removesuffix(".motion3.json")
            if motion_name == "neutral":
                motion_file = f"./{neutral_motion_path.name}"
            test_motions[f"manual_{motion_name}"] = [{"File": motion_file}]

    test_motions["idle"] = [{"File": "./neutral.motion3.json"}]
    model_config["FileReferences"]["Motions"] = test_motions
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".model3.json",
        prefix="_manual_motion_",
        dir=MODEL_DIRECTORY,
        delete=False,
    ) as temporary_config:
        json.dump(model_config, temporary_config, ensure_ascii=False, indent=2)
        return Path(temporary_config.name), neutral_motion_path


def run_manual_motion_viewer():
    from desktop.live2d_view import Live2DView, MODEL_EXPRESSIONS

    class ManualMotionView(Live2DView):
        def __init__(self, *args, **kwargs):
            self.pending_test_motion = None
            self.pending_hand_pose = None
            self.pending_stop = False
            self.motion_queue = []
            self.is_playing_all = False
            self.hand_pose = None
            self.status = None
            super().__init__(*args, **kwargs)

        def queue_test_motion(self, group):
            self.motion_queue.clear()
            self.is_playing_all = False
            self.pending_stop = False
            self.pending_test_motion = group
            self.pending_hand_pose = None
            label = MOTION_LABELS[group.removeprefix("manual_")]
            self.status.set(f"Playing: {label}")

        def queue_all_motions(self):
            self.motion_queue = list(MOTION_SEQUENCE)
            self.is_playing_all = True
            self.pending_stop = False
            self.pending_hand_pose = None
            self.pending_test_motion = take_next_motion(self.motion_queue)
            self.status.set(
                f"Playing all motions once (1/{len(MOTION_SEQUENCE)})"
            )

        def queue_hand_pose(self, value, label):
            self.motion_queue.clear()
            self.is_playing_all = False
            self.pending_stop = False
            self.pending_test_motion = None
            self.pending_hand_pose = value
            self.status.set(f"Showing: {label}")

        def stop_test(self):
            self.motion_queue.clear()
            self.is_playing_all = False
            self.pending_test_motion = None
            self.pending_hand_pose = None
            self.pending_stop = True
            self.status.set("Stopping and resetting to neutral")

        def _update_motion(self):
            if self.pending_stop:
                self.pending_stop = False
                self.model.StopAllMotions()
                self.model.SetExpression("neutral")
                self.model.StartRandomMotion("idle", self.live2d.MotionPriority.IDLE)
                self.motion_group = "idle"
                self.hand_pose = None
                self.reset_motion_transients = True
                return

            if self.pending_test_motion is not None:
                group = self.pending_test_motion
                self.pending_test_motion = None
                self.hand_pose = None
                motion_name = group.removeprefix("manual_")
                expression = MOTION_EXPRESSIONS.get(motion_name, "neutral")
                self.model.StopAllMotions()
                self.model.SetExpression(MODEL_EXPRESSIONS[expression])
                self.model.SetParameterValue("Param16", 0.0)
                self.model.SetParameterValue("ParamCheek", 0.0)
                self.model.StartRandomMotion(
                    group,
                    self.live2d.MotionPriority.FORCE,
                )
                self.motion_group = group

            if self.pending_hand_pose is not None:
                self.model.StopAllMotions()
                self.motion_group = None
                self.reset_motion_transients = True
                self.hand_pose = self.pending_hand_pose
                self.pending_hand_pose = None

            if (
                self.motion_group
                and self.motion_group.startswith("manual_")
                and self.model.IsMotionFinished(self.motion_group)
            ):
                finished_group = self.motion_group
                self.motion_group = None
                self.reset_motion_transients = True
                next_motion = take_next_motion(self.motion_queue)
                if next_motion is not None:
                    self.pending_test_motion = next_motion
                    motion_number = len(MOTION_SEQUENCE) - len(self.motion_queue)
                    motion_name = next_motion.removeprefix("manual_")
                    self.status.set(
                        f"Playing all motions once "
                        f"({motion_number}/{len(MOTION_SEQUENCE)}): "
                        f"{MOTION_LABELS[motion_name]}"
                    )
                else:
                    self.model.StopAllMotions()
                    self.model.SetExpression("neutral")
                    self.model.StartRandomMotion(
                        "idle",
                        self.live2d.MotionPriority.IDLE,
                    )
                    self.motion_group = "idle"
                    if self.is_playing_all:
                        self.is_playing_all = False
                        self.status.set("Finished: all motions played once")
                    else:
                        self.status.set(
                            f"Finished: "
                            f"{MOTION_LABELS[finished_group.removeprefix('manual_')]}"
                        )

        def _apply_motion_safety(self):
            super()._apply_motion_safety()
            if self.hand_pose is not None:
                self.model.SetParameterValue("Param16", self.hand_pose)

    model_config_path, neutral_motion_path = create_manual_model_config()
    root = tk.Tk()
    root.title("Amadeus - Manual Live2D Motion Test")
    root.geometry("1000x760")
    root.minsize(900, 650)

    viewer = None
    try:
        viewer = ManualMotionView(
            root,
            model_versions=3,
            model_path=str(model_config_path),
            width=680,
            height=740,
            fps=60,
        )
        viewer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls = ttk.Frame(root, padding=12)
        controls.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(
            controls,
            text="Click to play each motion",
            font=("", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(
            controls,
            text="Each button plays only its named motion.\n"
            "The model has one hand-pose control (crossed/relaxed), not "
            "separate finger animations. Gesture 3 is a general motion, and "
            "Gesture 4 uses Disdain. Neutral resets the face. random2 is excluded "
            "because it can crash the native runtime.",
            wraplength=250,
        ).pack(anchor=tk.W, pady=(0, 12))

        status = tk.StringVar(value="Choose a motion")
        viewer.status = status
        ttk.Button(
            controls,
            text=f"Play all {len(MOTION_SEQUENCE)} motions once",
            command=viewer.queue_all_motions,
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            controls,
            text="Stop / reset",
            command=viewer.stop_test,
        ).pack(fill=tk.X, pady=2)
        ttk.Label(controls, text="Hand poses", font=("", 10, "bold")).pack(
            anchor=tk.W,
            pady=(0, 4),
        )
        for pose_label, value in HAND_POSES.values():
            ttk.Button(
                controls,
                text=pose_label,
                command=lambda selected_value=value, label=pose_label: (
                    viewer.queue_hand_pose(selected_value, label)
                ),
            ).pack(fill=tk.X, pady=2)

        ttk.Label(
            controls,
            text="Motions",
            font=("", 10, "bold"),
        ).pack(anchor=tk.W, pady=(12, 4))
        for motion_name, label in MOTION_LABELS.items():
            group = f"manual_{motion_name}"
            ttk.Button(
                controls,
                text=label,
                command=lambda selected_group=group: viewer.queue_test_motion(selected_group),
            ).pack(fill=tk.X, pady=2)

        ttk.Separator(controls).pack(fill=tk.X, pady=10)
        ttk.Label(controls, textvariable=status, wraplength=250).pack(anchor=tk.W)

        def close_viewer():
            if viewer is not None and viewer.model is not None:
                viewer.tkMakeCurrent()
                viewer.end()
            root.destroy()
            pygame.quit()

        root.protocol("WM_DELETE_WINDOW", close_viewer)
        root.mainloop()
    finally:
        model_config_path.unlink(missing_ok=True)
        neutral_motion_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run_manual_motion_viewer()
