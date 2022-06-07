from __future__ import annotations

import vapoursynth as vs
from vsmask.better_vsutil import join, split
from vsmask.edge import EdgeDetect, PrewittStd
from vsmask.types import ensure_format as _ensure_format
from vsrgtools import removegrain, repair
from vsutil import Dither
from vsutil import Range as CRange
from vsutil import depth, get_peak_value, scale_value

from .utils import cround, pad_reflect

core = vs.core


def edge_cleaner(
    clip: vs.VideoNode, strength: float = 10, rmode: int = 17,
    hot: bool = False, smode: int = 0, edgemask: EdgeDetect = PrewittStd()
) -> vs.VideoNode:
    clip = _ensure_format(clip)

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('edge_cleaner: format not supported')

    bits = clip.format.bits_per_sample

    peak = get_peak_value(clip)

    clip_y, *chroma = split(clip)

    if smode > 0:
        strength += 4

    main = pad_reflect(clip_y, 6, 6, 6, 6)

    # warpsf is way too slow
    main = depth(main, 16, vs.INTEGER, dither_type=Dither.NONE) if clip.format.sample_type == vs.FLOAT else main
    main = main.warp.AWarpSharp2(blur=1, depth=cround(strength / 2)).std.Crop(6, 6, 6, 6)
    main = depth(main, bits, clip.format.sample_type, dither_type=Dither.NONE)

    main = repair(main, clip_y, rmode)

    mask = edgemask.edgemask(clip_y).std.Expr(
        f'x {scale_value(4, 8, bits, CRange.FULL)} < 0 x {scale_value(32, 8, bits, CRange.FULL)} > {peak} x ? ?'
    ).std.InvertMask().std.Convolution([1] * 9)

    final = clip_y.std.MaskedMerge(main, mask)

    if hot:
        final = repair(final, clip_y, 2)

    if smode:
        clean = removegrain(clip_y, 17)

        diff = clip_y.std.MakeDiff(clean)

        expr = f'x {scale_value(4, 8, bits, CRange.FULL)} < 0 x {scale_value(16, 8, bits, CRange.FULL)} > {peak} x ? ?'

        mask = edgemask.edgemask(
            diff.std.Levels(scale_value(40, 8, bits, CRange.FULL), scale_value(168, 8, bits, CRange.FULL), 0.35)
        )
        mask = removegrain(mask, 7).std.Expr(expr)

        final = final.std.MaskedMerge(clip_y, mask)

    return join([final, *chroma], clip.format.color_family)
