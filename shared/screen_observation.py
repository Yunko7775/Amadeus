from PIL import Image, ImageChops, ImageStat


_COMPARISON_SIZE = (160, 90)
_MINIMUM_MEAN_DIFFERENCE = 0.035


def has_significant_screen_change(previous: Image.Image | None, current: Image.Image) -> bool:
    if previous is None or previous.size != current.size:
        return True

    previous_small = previous.convert("L").resize(_COMPARISON_SIZE, Image.Resampling.BILINEAR)
    current_small = current.convert("L").resize(_COMPARISON_SIZE, Image.Resampling.BILINEAR)
    difference = ImageChops.difference(previous_small, current_small)
    mean_difference = ImageStat.Stat(difference).mean[0] / 255
    return mean_difference >= _MINIMUM_MEAN_DIFFERENCE
