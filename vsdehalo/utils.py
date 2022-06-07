from __future__ import annotations

from math import floor, ceil
from functools import partial

import vapoursynth as vs

core = vs.core


def cround(x: float) -> int:
    return floor(x + 0.5) if x > 0 else ceil(x - 0.5)


def clamp(val: float, min_val: float, max_val: float) -> float:
    return min_val if val < min_val else max_val if val > max_val else val


def mod_x(val: int | float, x: int) -> int:
    return max(x * x, cround(val / x) * x)


mod2 = partial(mod_x, x=2)

mod4 = partial(mod_x, x=4)
