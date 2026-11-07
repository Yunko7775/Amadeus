import json
from pathlib import Path

from desktop.live2d_view import MODEL_EXPRESSIONS, MODEL_MOTIONS


MODEL_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "Resources" / "kurisu_cubism3"
)


def test_model_config_enables_both_eye_blink_parameters():
    model_path = MODEL_DIRECTORY / "kurisu.model3.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    groups = {group["Name"]: group for group in model["Groups"]}

    assert groups["EyeBlink"]["Ids"] == ["ParamEyeLOpen", "ParamEyeROpen"]


def test_model_motion_files_are_available_and_one_shot():
    model_path = MODEL_DIRECTORY / "kurisu.model3.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    motions = model["FileReferences"]["Motions"]

    assert len(motions["gesture"]) == 4
    assert "tsundere" not in motions
    assert [motion["File"] for motion in motions["disdain"]] == [
        "./random4.motion3.json"
    ]
    assert len(motions["smile1"]) == 1
    assert len(motions["smile2"]) == 1
    assert [motion["File"] for motion in motions["joy"]] == ["./joy.motion3.json"]
    assert all(
        "random2.motion3.json" not in motion["File"]
        for group in motions.values()
        for motion in group
    )

    for group, files in motions.items():
        for file in files:
            motion_path = MODEL_DIRECTORY / file["File"].removeprefix("./")
            motion = json.loads(motion_path.read_text(encoding="utf-8"))
            assert motion["Meta"]["Loop"] is (group == "idle")


def test_model_expressions_and_motions_are_individually_selectable():
    expected = {
        "smile1": ("smile1", "smile1"),
        "smile2": ("smile2", "smile2"),
        "disdain": ("disdain", "disdain"),
    }
    for expression, (model_expression, motion_group) in expected.items():
        assert MODEL_EXPRESSIONS[expression] == model_expression
        assert MODEL_MOTIONS[expression] == motion_group


def test_disdain_has_a_separate_smirk_expression_from_disgust():
    model_path = MODEL_DIRECTORY / "kurisu.model3.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    expressions = {
        expression["Name"]: expression["File"]
        for expression in model["FileReferences"]["Expressions"]
    }

    assert expressions["disdain"] == "./disdain.exp3.json"
    disdain = json.loads(
        (MODEL_DIRECTORY / "disdain.exp3.json").read_text(encoding="utf-8")
    )
    disgust = json.loads(
        (MODEL_DIRECTORY / "unhappy.exp3.json").read_text(encoding="utf-8")
    )
    assert disdain["Parameters"] != disgust["Parameters"]
    assert disdain["Parameters"] == [
        {"Id": "ParamMouthForm", "Value": 0.35, "Blend": "Add"}
    ]
