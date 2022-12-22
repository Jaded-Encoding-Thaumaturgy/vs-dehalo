"""
Modern rewrite of IFeelBloated's package with scene detection.
"""
from __future__ import annotations

from functools import partial
from math import log
from typing import Any

from vsdenoise import (
    CCDMode, CCDPoints, ChannelMode, MotionMode, MVTools, PelType, Prefilter, SearchMode, ccd, knl_means_cl
)
from vsexprtools import norm_expr_planes
from vskernels import Bicubic
from vsmasktools import Morpho
from vsrgtools import contrasharpening_dehalo, gauss_blur, gauss_fmtc_blur, lehmer_diff_merge
from vstools import (
    Matrix, PlanesT, check_ref_clip, core, disallow_variable_format, disallow_variable_resolution, get_y, join,
    normalize_planes, split, vs
)

from .alpha import fine_dehalo

__all__ = [
    'super_clip', 'smooth_clip', 'dehalo'
]


@disallow_variable_format
@disallow_variable_resolution
def super_clip(src: vs.VideoNode, pel: int = 1, planes: PlanesT = 0) -> vs.VideoNode:
    assert src.format

    if planes == [1, 2] and (src.format.subsampling_w or src.format.subsampling_h):
        src = src.resize.Bicubic(
            src.width // (1 << src.format.subsampling_w),
            src.height // (1 << src.format.subsampling_h),
            format=src.format.replace(subsampling_w=0, subsampling_h=0).id
        )
    elif planes == [0]:
        src = get_y(src)

    return PelType.NNEDI3(src, pel, nns=4, qual=2, etype=1)


@disallow_variable_format
@disallow_variable_resolution
def smooth_clip(
    src: vs.VideoNode, sr: int = 32, strength: float = 12.5,
    sharp: float = 0.80, cutoff: int = 4,
    aggressive: bool = True, fast: bool = False,
    matrix: int | Matrix | None = None,
    pel_type: PelType = PelType.BICUBIC,
    planes: PlanesT = 0
) -> vs.VideoNode:
    assert src.format

    if sr < 1:
        raise RuntimeError('vine.smooth_clip: sr has to be greater than 0!')

    if strength <= 0:
        raise RuntimeError('vine.smooth_clip: strength has to be greater than 0!')

    if sharp <= 0.0:
        raise RuntimeError('vine.smooth_clip: sharp has to be greater than 0!')

    if cutoff < 1 or cutoff > 100:
        raise RuntimeError('vine.smooth_clip: cutoff must fall in (0, 100]!')

    csp = src.format.color_family
    planes = normalize_planes(src, planes)

    if fast and csp == vs.GRAY:
        raise ValueError('vine.smooth_clip: fast=True is available only for YUV and RGB input!')

    work_clip = src

    if csp == vs.RGB:
        work_clip = work_clip.bm3d.RGB2OPP(1)

    work_clip, *chroma = split(work_clip) if planes == [0] and not fast else (work_clip, )
    assert work_clip.format

    c1 = 0.3926327792690057290863679493724 * sharp
    c2 = 18.880334973195822973214959957208
    c3 = 0.5862453661304626725671053478676

    weight = c1 * log(1.0 + 1.0 / (c1))

    h_smoothine = c2 * pow(strength / c2, c3)

    upsampled = pel_type(work_clip, 2)

    knl_channels = ChannelMode.from_planes(planes)

    if fast:
        upsampled = ccd(
            upsampled, strength * 2, 0, None,
            CCDMode.BICUBIC_LUMA, 25 / (2 * sr + 1), matrix,
            CCDPoints.MEDIUM, False, planes
        )
        upsampled = gauss_blur(upsampled, 1.45454, planes=planes)
    else:
        upsampled = knl_means_cl(upsampled, strength, 0, sr, 0, knl_channels)

    c = 0 if pel_type == PelType.NNEDI3 else (1.0 - abs(sharp)) / 2

    resampled = Bicubic(b=-sharp, c=c).scale(upsampled, work_clip.width, work_clip.height)

    if fast:
        resampled = contrasharpening_dehalo(resampled, work_clip, 2.5, planes=planes)

    clean = knl_means_cl(work_clip, strength, 0, sr, 0, knl_channels)

    clean = resampled.std.Merge(
        clean, [weight if i in planes else 0 for i in range(work_clip.format.num_planes)]
    )

    blur_func = partial(
        gauss_fmtc_blur, sigma=cutoff, sharp=100, strict=False, planes=planes
    )

    if aggressive:
        clean = core.std.Expr(
            [blur_func(work_clip), clean, blur_func(clean)],
            norm_expr_planes(work_clip, 'y z - x +', planes)
        )
    else:
        clean = lehmer_diff_merge(
            gauss_blur(work_clip, 0.5), clean, blur_func, planes=planes  # type: ignore
        )

    diff = work_clip.std.MakeDiff(clean, planes)

    diff = knl_means_cl(diff, h_smoothine, 0, sr, 1, knl_channels, rclip=clean)

    smooth = clean.std.MergeDiff(diff, planes)

    if chroma:
        smooth = join([smooth, *chroma], vs.YUV)

    if csp == vs.RGB:
        return smooth.bm3d.OPP2RGB(1)

    return smooth


@disallow_variable_format
@disallow_variable_resolution
def dehalo(
    src: vs.VideoNode, smooth: vs.VideoNode | None = None,
    tr: int = 0, refine: int = 3, pel: int = 1, thSAD: int = 400,
    super_clips: tuple[vs.VideoNode | None, vs.VideoNode | None] = (None, None),
    planes: PlanesT = 0, mask: bool | vs.VideoNode = True
) -> vs.VideoNode:
    assert src.format

    if smooth:
        assert smooth.format

        if src.format.id != smooth.format.id or (src.width, src.height) != (smooth.width, smooth.height):
            raise TypeError('vine.dehalo: smooth must have the same format and size as src!')

    if pel not in {1, 2, 4}:
        raise RuntimeError('vine.dehalo: pel has to be 1, 2 or 4!')

    if thSAD <= 0:
        raise RuntimeError('vine.dehalo: sad has to be greater than 0!')

    csp = src.format.color_family
    planes = normalize_planes(src, planes)

    if any(sclip and sclip.format and sclip.format.num_planes == 1 for sclip in super_clips):
        planes = [0]

    if smooth is None:
        smooth = smooth_clip(src, planes=planes)

    work_clip, smooth_wclip = src, smooth

    if csp == vs.RGB:
        work_clip = work_clip.bm3d.RGB2OPP(1)
        smooth_wclip = smooth_wclip.bm3d.RGB2OPP(1)

    work_clip, *chroma = split(work_clip) if planes == [0] else (work_clip, )
    smooth_wclip = get_y(smooth_wclip) if planes == [0] else smooth_wclip

    constant = 0.0009948813682897925944723492342

    me_sad = constant * pow(thSAD, 2.0) * log(1.0 + 1.0 / (constant * thSAD))

    if isinstance(mask, vs.VideoNode):
        check_ref_clip(src, (halo_mask := mask))  # type: ignore
    elif mask:
        halo_mask = fine_dehalo(work_clip, 2.1, ss=1, edgeproc=0.5, show_mask=True, planes=planes)
        halo_mask = Morpho.dilation(halo_mask, 1, planes)
    else:
        halo_mask = None

    smooth_wclip = smooth_wclip.std.MakeDiff(work_clip, planes)

    src_super, smooth_super = super_clips

    if pel == 1:
        if not src_super:
            src_super = work_clip
        if not smooth_super:
            smooth_super = smooth_wclip

    if src_super and smooth_super:
        smooth_super = smooth_super.std.MakeDiff(src_super)

    class CustomSubPelClipsMVTools(MVTools):
        def get_subpel_clips(self, *args: Any) -> tuple[vs.VideoNode | None, vs.VideoNode | None]:
            return (smooth_super, src_super)

    mv = CustomSubPelClipsMVTools(
        smooth_wclip, tr, refine, prefilter=Prefilter.NONE,
        pel=pel, rfilter=4, planes=planes
    )

    mv.analyze_args |= dict[str, Any](trymany=True, badrange=-24, divide=0)
    mv.recalculate_args |= dict[str, Any](smooth=1, divide=0, thsad=me_sad)

    mv.analyze(ref=work_clip, search=SearchMode.EXHAUSTIVE, motion=MotionMode.HIGH_SAD)

    averaged_dif = mv.degrain(thSAD=thSAD)

    averaged_dif = core.std.Expr(
        [averaged_dif, smooth_wclip], norm_expr_planes(work_clip, 'x y min', planes)
    )

    clean = work_clip.std.MergeDiff(averaged_dif, planes)

    if halo_mask:
        clean = work_clip.std.MaskedMerge(clean, halo_mask, planes)  # type: ignore

    if chroma:
        clean = join([clean, *chroma], vs.YUV)

    if csp == vs.RGB:
        return clean.bm3d.OPP2RGB(1)

    return clean
