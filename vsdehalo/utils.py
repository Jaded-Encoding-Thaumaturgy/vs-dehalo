from __future__ import annotations

import vapoursynth as vs
from vskernels import Point
from vsutil import disallow_variable_format, disallow_variable_resolution

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def pad_reflect(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    d_width, d_height = clip.width + left + right, clip.height + top + bottom

    return Point(src_width=d_width, src_height=d_height).scale(clip, d_width, d_height, (-top, -left))
