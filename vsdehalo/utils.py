from __future__ import annotations

import vapoursynth as vs
from vskernels import Point

core = vs.core


def pad_reflect(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    d_width, d_height = clip.width + left + right, clip.height + top + bottom

    return Point(src_width=d_width, src_height=d_height).scale(clip, d_width, d_height, (-top, -left))
