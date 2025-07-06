from typing import Tuple

def check_input_hsv(hsv : Tuple[float, float, float]):
    h, s, v = hsv
    assert 0 <= h and h <= 179
    assert 0 <= s and s <= 255
    assert 0 <= v and v <= 255

def normalize_hsv(hsv : Tuple[float, float, float]):
    h, s, v = hsv
    return h / 179., s / 255., v / 255.