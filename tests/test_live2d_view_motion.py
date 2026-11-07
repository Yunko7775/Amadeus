from types import SimpleNamespace
from unittest.mock import Mock, call

from desktop.live2d_view import EXPRESSION_ACTING, Live2DView


def test_expression_acting_does_not_roll_the_head_or_body():
    assert all(
        "ParamAngleZ" not in acting and "ParamBodyAngleZ" not in acting
        for acting in EXPRESSION_ACTING.values()
    )


def make_view(motion_group):
    return SimpleNamespace(
        live2d_model_version=3,
        live2d=SimpleNamespace(
            MotionPriority=SimpleNamespace(IDLE=1, NORMAL=2),
        ),
        model=Mock(),
        motion_group=motion_group,
        reset_motion_transients=False,
    )


def test_idle_does_not_start_random_gestures():
    view = make_view("idle")

    Live2DView._update_motion(view)

    view.model.IsMotionFinished.assert_not_called()
    view.model.StartRandomMotion.assert_not_called()


def test_active_expression_motion_is_not_interrupted():
    view = make_view("joy")
    view.model.IsMotionFinished.return_value = False

    Live2DView._update_motion(view)

    view.model.StartRandomMotion.assert_not_called()


def test_neutral_expression_resets_the_native_expression_and_motion():
    view = make_view("joy")

    Live2DView.set_expression(view, "neutral")

    view.model.SetExpression.assert_called_once_with("neutral")
    view.model.StopAllMotions.assert_called_once_with()
    view.model.StartRandomMotion.assert_called_once_with("idle", 2)
    assert view.motion_group == "idle"


def test_finished_expression_motion_returns_to_idle_without_gesture():
    view = make_view("joy")
    view.model.IsMotionFinished.return_value = True

    Live2DView._update_motion(view)

    view.model.StartRandomMotion.assert_called_once_with("idle", 1)
    assert view.motion_group == "idle"
    assert view.reset_motion_transients


def test_motion_safety_keeps_model_upright_and_resets_blush():
    view = make_view("gesture")
    view.reset_motion_transients = True

    Live2DView._apply_motion_safety(view)

    assert view.model.SetParameterValue.call_args_list == [
        call("ParamAngleZ", 0.0),
        call("ParamBodyAngleZ", 0.0),
        call("ParamCheek", 0.0),
    ]
    assert not view.reset_motion_transients


def test_motion_safety_keeps_blush_until_motion_finishes():
    view = make_view("gesture")
    view.reset_motion_transients = False

    Live2DView._apply_motion_safety(view)

    assert view.model.SetParameterValue.call_args_list == [
        call("ParamAngleZ", 0.0),
        call("ParamBodyAngleZ", 0.0),
    ]
