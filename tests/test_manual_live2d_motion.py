import json

from manual_live2d_motion import MOTION_LABELS, MOTION_SEQUENCE, take_next_motion
from manual_live2d_motion import create_manual_model_config


def test_play_all_sequence_contains_each_manual_motion_once():
    assert len(MOTION_SEQUENCE) == len(MOTION_LABELS)
    assert len(set(MOTION_SEQUENCE)) == len(MOTION_SEQUENCE)
    assert {
        group.removeprefix("manual_") for group in MOTION_SEQUENCE
    } == set(MOTION_LABELS)


def test_play_all_sequence_stops_after_one_pass():
    remaining = list(MOTION_SEQUENCE)
    played = []

    while True:
        motion = take_next_motion(remaining)
        if motion is None:
            break
        played.append(motion)

    assert played == list(MOTION_SEQUENCE)
    assert take_next_motion(remaining) is None


def test_manual_neutral_motion_is_a_one_shot_and_idle_stays_looping():
    model_config_path, neutral_motion_path = create_manual_model_config()
    try:
        model = json.loads(model_config_path.read_text(encoding="utf-8"))
        motions = model["FileReferences"]["Motions"]
        manual_neutral_path = (
            model_config_path.parent
            / motions["manual_neutral"][0]["File"].removeprefix("./")
        )
        manual_neutral = json.loads(
            manual_neutral_path.read_text(encoding="utf-8")
        )
        idle = json.loads(
            (model_config_path.parent / motions["idle"][0]["File"].removeprefix("./"))
            .read_text(encoding="utf-8")
        )

        assert manual_neutral["Meta"]["Loop"] is False
        assert idle["Meta"]["Loop"] is True
        assert neutral_motion_path == manual_neutral_path
    finally:
        model_config_path.unlink(missing_ok=True)
        neutral_motion_path.unlink(missing_ok=True)
