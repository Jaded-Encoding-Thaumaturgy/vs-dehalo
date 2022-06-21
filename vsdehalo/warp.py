from __future__ import annotations
from typing import Sequence

import vapoursynth as vs
from math import sqrt
from vsmask.better_vsutil import join, split
from vsmask.edge import EdgeDetect, PrewittStd
from vsmask.types import ensure_format as _ensure_format
from vsrgtools import boxblur, minblur, removegrain, repair
from vsrgtools.util import wmean_matrix, normalise_planes
from vsutil import Dither
from vsutil import Range as CRange
from vsutil import depth as vdepth
from vsutil import get_peak_value, scale_value, get_y

from .utils import cround, pad_reflect

core = vs.core


def edge_cleaner(
    clip: vs.VideoNode, strength: float = 10, rmode: int = 17,
    hot: bool = False, smode: bool = False, edgemask: EdgeDetect = PrewittStd()
) -> vs.VideoNode:
    clip = _ensure_format(clip)

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('edge_cleaner: format not supported')

    bits = clip.format.bits_per_sample

    peak = get_peak_value(clip)

    clip_y, *chroma = split(clip)

    if smode:
        strength += 4

    main = pad_reflect(clip_y, 6, 6, 6, 6)

    # warpsf is way too slow
    main = vdepth(main, 16, vs.INTEGER, dither_type=Dither.NONE) if clip.format.sample_type == vs.FLOAT else main
    main = main.warp.AWarpSharp2(blur=1, depth=cround(strength / 2)).std.Crop(6, 6, 6, 6)
    main = vdepth(main, bits, clip.format.sample_type, dither_type=Dither.NONE)

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


def YAHR(
    clip: vs.VideoNode, blur: int = 2,
    depth: int = 32, expand: float = 5,
    planes: int | Sequence[int] | None = None
) -> vs.VideoNode:
    clip = _ensure_format(clip)

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('edge_cleaner: format not supported')

    planes = normalise_planes(clip, planes)

    bits = clip.format.bits_per_sample

    clip_y = get_y(clip)

    main = pad_reflect(clip, 6, 6, 6, 6)

    # warpsf is way too slow
    main = vdepth(main, 16, vs.INTEGER, dither_type=Dither.NONE) if clip.format.sample_type == vs.FLOAT else main
    main = main.warp.AWarpSharp2(blur=blur, depth=depth, planes=planes).std.Crop(6, 6, 6, 6)
    main = vdepth(main, bits, clip.format.sample_type, dither_type=Dither.NONE)

    blur_diff, blur_yahr_diff = [
        c.std.MakeDiff(boxblur(minblur(c, 2), wmean_matrix, planes=planes), planes=planes) for c in (clip, main)
    ]

    rep_diff = repair(blur_diff, blur_yahr_diff, [13 if i in planes else 0 for i in range(clip.format.num_planes)])

    yahr = clip.std.MakeDiff(blur_diff.std.MakeDiff(rep_diff, planes=planes), planes=planes)

    vEdge = core.std.Expr(
        [clip_y, clip_y.std.Maximum().std.Maximum()],
        f'y x - {8 * get_peak_value(clip_y) / 255} - 128 *'
    )

    mask1 = core.tcanny.TCanny(vEdge, sqrt(expand * 2), mode=-1)

    mask2 = boxblur(vEdge, wmean_matrix).std.Invert()

    mask = core.std.Expr([mask1, mask2], 'x 16 * y min')

    return clip.std.MaskedMerge(yahr, mask, planes=planes)
