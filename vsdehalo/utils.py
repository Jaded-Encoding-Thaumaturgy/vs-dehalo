from __future__ import annotations

from functools import partial
from math import ceil, floor

import vapoursynth as vs
from vskernels import Point

core = vs.core


def cround(x: float) -> int:
    return floor(x + 0.5) if x > 0 else ceil(x - 0.5)


def clamp(val: float, min_val: float, max_val: float) -> float:
    return min_val if val < min_val else max_val if val > max_val else val


def mod_x(val: int | float, x: int) -> int:
    return max(x * x, cround(val / x) * x)


mod2 = partial(mod_x, x=2)

mod4 = partial(mod_x, x=4)


def pad_reflect(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    d_width, d_height = clip.width + left + right, clip.height + top + bottom

    return Point(src_width=d_width, src_height=d_height).scale(clip, d_width, d_height, (-top, -left))
