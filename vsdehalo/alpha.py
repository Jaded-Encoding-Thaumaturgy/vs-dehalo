from __future__ import annotations

from functools import partial

import vapoursynth as vs
from vsexprtools.util import PlanesT, aka_expr_available, clamp, cround, mod4, norm_expr_planes, normalise_planes
from vskernels import BSpline, Lanczos, Mitchell
from vsmask.better_vsutil import join, split
from vsmask.edge import EdgeDetect, Robinson3
from vsmask.util import XxpandMode, expand, inpand
from vsrgtools import ConvMode, box_blur, contrasharpening, contrasharpening_dehalo, repair
from vsutil import Range as CRange
from vsutil import disallow_variable_format, disallow_variable_resolution, get_peak_value, get_y, scale_value

from . import masks

__all__ = [
    'fine_dehalo', 'fine_dehalo2', 'dehalo_alpha'
]

core = vs.core

bspline = BSpline()
mitchell = Mitchell()


@disallow_variable_format
@disallow_variable_resolution
def fine_dehalo(
    clip: vs.VideoNode, /, ref: vs.VideoNode | None = None,
    rx: float = 2.0, ry: float | None = None,
    darkstr: float = 0.0, brightstr: float = 1.0,
    lowsens: int = 50, highsens: int = 50,
    thmi: int | float = 80, thma: int | float = 128,
    thlimi: int | float = 50, thlima: int | float = 100,
    ss: float = 1.25,
    contra: int | float | bool = 0.0, excl: bool = True,
    edgeproc: float = 0.0, planes: PlanesT = 0,
    edgemask: EdgeDetect = Robinson3(), show_mask: int = 0
) -> vs.VideoNode:
    """
    Halo removal script that uses dehalo_alpha with a few masks and optional contra-sharpening
    to try remove halos without removing important details (like line edges).
    :param clip:        Source clip
    :param ref:         Dehaloed reference. Replace dehalo_alpha call
    :param rx:          X radius for halo removal in :py:func:`dehalo_alpha`
    :param ry:          Y radius for halo removal in :py:func:`dehalo_alpha`. If none ry = rx
    :param darkstr:     Strength factor for processing dark halos
    :param brightstr:   Strength factor for processing bright halos
    :param lowsens:     Low sensitivity settings. Define how weak the dehalo has to be to get fully accepted
    :param highsens:    High sensitivity settings. Define how wtrong the dehalo has to be to get fully discarded
    :param thmi:        Minimum threshold for sharp edges; keep only the sharpest edges (line edges).
    :param thma:        Maximum threshold for sharp edges; keep only the sharpest edges (line edges).
    :param thlimi:      Minimum limiting threshold; includes more edges than previously, but ignores simple details.
    :param thlima:      Maximum limiting threshold; includes more edges than previously, but ignores simple details.
    :param ss:          Supersampling factor, to avoid creation of aliasing, defaults to 1.25
    :param contra:      Contrasharpening. If True, will use :py:func:`contrasharpening`
                        otherwise use :py:func:`contrasharpening_fine_dehalo`
    :param excl:        If True, add an addionnal step to exclude edges close to each other
    :param edgeproc:    If > 0, it will add the edgemask to the processing, defaults to 0.0
    :param edgemask:    Internal mask used for detecting the edges, defaults to Robinson3()
    :param show_mask:    1 - 7
    :return:            Dehaloed clip
    """
    assert clip.format

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('fine_dehalo: format not supported')

    if not all(x >= 1 for x in (ss, rx, ry) if x is not None):
        raise ValueError('fine_dehalo: rfactor, rx, and ry must all be bigger than 1.0!')

    if not 0 <= darkstr <= 1:
        raise ValueError('fine_dehalo: darkstr must be between 0.0 and 1.0!')

    if not all(0 <= sens < 100 for sens in (lowsens, highsens)):
        raise ValueError('fine_dehalo: lowsens and highsens must be between 0 and 100!')

    if show_mask is not False and not (0 < int(show_mask) <= 7):
        raise ValueError('fine_dehalo: Valid values for show_mask are 0–7!')

    thmi, thma, thlimi, thlima = [
        scale_value(x, 8, clip.format.bits_per_sample, CRange.FULL)
        for x in [thmi, thma, thlimi, thlima]
    ]

    peak = get_peak_value(clip)
    planes = normalise_planes(clip, planes)

    ry = rx if ry is None else ry
    rx_i = cround(rx)
    ry_i = cround(ry)

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )

    norm_expr = partial(norm_expr_planes, work_clip, planes=planes)

    if ref:
        dehaloed = get_y(ref) if planes == [0] else ref
    else:
        dehaloed = dehalo_alpha(
            work_clip, rx, ry, darkstr, brightstr, lowsens, highsens, ss=ss, planes=planes
        )

    if contra:
        if isinstance(contra, (int, bool)):
            dehaloed = contrasharpening(
                dehaloed, work_clip, contra if isinstance(contra, int) else None, planes=planes
            )
        else:
            dehaloed = contrasharpening_dehalo(dehaloed, work_clip, contra, planes)

    # Main edges #
    # Basic edge detection, thresholding will be applied later.
    edges = edgemask.edgemask(work_clip)

    # Keeps only the sharpest edges (line edges)
    strong = edges.std.Expr(norm_expr(f'x {thmi} - {thma - thmi} / {peak} *'))

    # Extends them to include the potential halos
    large = expand(strong, rx_i, ry_i, planes=planes)

    # Exclusion zones #
    # When two edges are close from each other (both edges of a single
    # line or multiple parallel color bands), the halo removal
    # oversmoothes them or makes seriously bleed the bands, producing
    # annoying artifacts. Therefore we have to produce a mask to exclude
    # these zones from the halo removal.

    # Includes more edges than previously, but ignores simple details
    light = edges.std.Expr(norm_expr(f'x {thlimi} - {thlima - thlimi} / {peak} *'))

    # To build the exclusion zone, we make grow the edge mask, then shrink
    # it to its original shape. During the growing stage, close adjacent
    # edge masks will join and merge, forming a solid area, which will
    # remain solid even after the shrinking stage.
    # Mask growing
    shrink = expand(light, rx_i, ry_i, XxpandMode.ELLIPSE, planes=planes)

    # At this point, because the mask was made of a shades of grey, we may
    # end up with large areas of dark grey after shrinking. To avoid this,
    # we amplify and saturate the mask here (actually we could even
    # binarize it).
    shrink = shrink.std.Expr(norm_expr('x 4 *'))
    shrink = inpand(shrink, rx_i, rx_i, XxpandMode.ELLIPSE, planes=planes)

    # This mask is almost binary, which will produce distinct
    # discontinuities once applied. Then we have to smooth it.
    shrink = box_blur(shrink, 1, 2, planes)

    # Final mask building #

    # Previous mask may be a bit weak on the pure edge side, so we ensure
    # that the main edges are really excluded. We do not want them to be
    # smoothed by the halo removal.
    shr_med = core.std.Expr([strong, shrink], norm_expr('x y max')) if excl else strong

    # Substracts masks and amplifies the difference to be sure we get 255
    # on the areas to be processed.
    mask = core.std.Expr([large, shr_med], norm_expr('x y - 2 *'))

    # If edge processing is required, adds the edgemask
    if edgeproc > 0:
        mask = core.std.Expr([mask, strong], norm_expr(f'x y {edgeproc} 0.66 * * +'))

    # Smooth again and amplify to grow the mask a bit, otherwise the halo
    # parts sticking to the edges could be missed.
    # Also clamp to legal ranges
    mask = mask.std.Convolution([1] * 9, planes=planes)
    mask = mask.std.Expr(norm_expr(f'x 2 * 0 max {peak} min'))

    # Masking #
    if show_mask:
        if show_mask == 1:
            return mask
        if show_mask == 2:
            return shrink
        if show_mask == 3:
            return edges
        if show_mask == 4:
            return strong
        if show_mask == 5:
            return light
        if show_mask == 6:
            return large
        if show_mask == 7:
            return shr_med

    y_merge = work_clip.std.MaskedMerge(dehaloed, mask, planes)

    if chroma:
        return join([y_merge, *chroma], clip.format.color_family)

    return y_merge


def fine_dehalo2(
    clip: vs.VideoNode, mode: ConvMode = ConvMode.SQUARE,
    radius: int = 2, planes: PlanesT = 0, show_mask: bool = False
) -> vs.VideoNode:
    """
    Halo removal function for 2nd order halos.

    :param clip:        Source clip.
    :param mode:        Horizontal/Vertical or both way.
    :param radius:      Radius for mask growing.
    :param planes:      Planes to process.
    :param show_mask:    Whether to return the computed mask.

    :return:            Dehaloed clip.
    """
    assert clip.format

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('fine_dehalo2: format not supported')

    peak = get_peak_value(clip)
    planes = normalise_planes(clip, planes)

    is_float = clip.format.sample_type == vs.FLOAT

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )

    norm_expr = partial(norm_expr_planes, work_clip, planes=planes)

    mask_h = mask_v = None

    # intended to be reversed
    if mode in {ConvMode.SQUARE, ConvMode.VERTICAL}:
        mask_h = work_clip.std.Convolution([1, 2, 1, 0, 0, 0, -1, -2, -1], None, 4, planes, False)

    if mode in {ConvMode.SQUARE, ConvMode.HORIZONTAL}:
        mask_v = work_clip.std.Convolution([1, 0, -1, 2, 0, -2, 1, 0, -1], None, 4, planes, False)

    fix_h = work_clip.std.Convolution([-1, -2, 0, 0, 40, 0, 0, -2, -1], mode=ConvMode.HORIZONTAL)
    fix_v = work_clip.std.Convolution([-2, -1, 0, 0, 40, 0, 0, -1, -2], mode=ConvMode.VERTICAL)

    if mask_h and mask_v:
        mask_h = core.std.Expr([mask_h, mask_v], norm_expr('x 3 * y -'))
        mask_v = core.std.Expr([mask_v, mask_h], norm_expr('x 3 * y -'))
    elif mask_h:
        mask_h = mask_h.std.Expr(norm_expr('x 3 *'))
    elif mask_v:
        mask_v = mask_v.std.Expr(norm_expr('x 3 *'))

    if is_float:
        mask_h = mask_h and mask_h.std.Limiter(planes=planes)
        mask_v = mask_v and mask_v.std.Limiter(planes=planes)

    mask_h, mask_v = [
        masks.grow_mask(mask, radius, 1.8, planes, coordinates=coord) if mask else None
        for mask, coord in [
            (mask_h, [0, 1, 0, 0, 0, 0, 1, 0]), (mask_v, [0, 0, 0, 1, 1, 0, 0, 0])
        ]
    ]

    if is_float:
        mask_h = mask_h and mask_h.std.Limiter()
        mask_v = mask_v and mask_v.std.Limiter()

    if show_mask:
        if mask_h and mask_v:
            return core.std.Expr([mask_h, mask_v], 'x y max')
        else:
            assert (mask := mask_h or mask_v)
            return mask
    else:
        if aka_expr_available and mask_h and mask_v:
            dehaloed = core.akarin.Expr(
                [work_clip, fix_h, fix_v, mask_h, mask_v],
                f'a {peak} / MH! b {peak} / MV! x 1 MH@ - * y MH@ * + 1 MV@ - * z MV@ * +'
            )
        else:
            dehaloed = work_clip
            if mask_v:
                dehaloed = dehaloed.std.MaskedMerge(fix_h, mask_v)
            if mask_h:
                dehaloed = dehaloed.std.MaskedMerge(fix_v, mask_h)

    if chroma:
        return join([dehaloed, *chroma], clip.format.color_family)

    return dehaloed


@disallow_variable_format
@disallow_variable_resolution
def dehalo_alpha(
    clip: vs.VideoNode,
    rx: float = 2.0, ry: float | None = None,
    darkstr: float = 0.0, brightstr: float = 1.0,
    lowsens: float = 50, highsens: float = 50,
    sigma_mask: float = 0.0, ss: float = 1.5,
    planes: PlanesT = 0, show_mask: bool = False
) -> vs.VideoNode:
    """
    Reduce halo artifacts by nuking everything around edges (and also the edges actually)
    :param clip:            Source clip
    :param rx:              Horizontal radius for halo removal, defaults to 2.0
    :param ry:              Vertical radius for halo removal, defaults to 2.0
    :param darkstr:         Strength factor for dark halos, defaults to 1.0
    :param brightstr:       Strength factor for bright halos, defaults to 1.0
    :param lowsens:         Sensitivity setting, defaults to 50
    :param highsens:        Sensitivity setting, defaults to 50
    :param sigma_mask:      Blurring strength for the mask, defaults to 0.25
    :param ss:              Supersampling factor, to avoid creation of aliasing., defaults to 1.5
    :return:                Dehaloed clip
    """
    assert clip.format

    if clip.format.color_family not in {vs.GRAY, vs.YUV}:
        raise ValueError('dehalo_alpha: only GRAY and YUV formats are supported')

    peak = get_peak_value(clip)
    planes = normalise_planes(clip, planes)

    ry = rx if ry is None else ry

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )

    dehalo = mitchell.scale(work_clip, mod4(clip.width / rx), mod4(clip.height / ry))
    dehalo = bspline.scale(dehalo, clip.width, clip.height)

    norm_expr = partial(norm_expr_planes, work_clip, planes=planes)

    org_minmax = core.std.Expr([work_clip.std.Maximum(planes), work_clip.std.Minimum(planes)], norm_expr('x y -'))
    dehalo_minmax = core.std.Expr([dehalo.std.Maximum(planes), dehalo.std.Minimum(planes)], norm_expr('x y -'))

    mask = core.std.Expr(
        [org_minmax, dehalo_minmax], norm_expr(
            f'x 0 = 1.0 x y - x / ? {lowsens / 255} - x {peak} / 256 255 / + 512 255 / / {highsens / 100} + * '
            # f'{lowsens / 255} - x {peak} / 1.003921568627451 + 2.007843137254902 / {highsens / 100} + * '
            # f'{lowsens / 255} - x {peak} / 0.498046862745098 * 0.5 + {highsens / 100} + * '
            f'0.0 max 1.0 min {peak} *',
        )
    )

    sig_mask = bool(sigma_mask)
    conv_values = [float(sig_mask)] * 9
    conv_values[5] = 1 / clamp(sigma_mask, 0, 1) if sig_mask else 1

    mask = mask.std.Convolution(conv_values, planes=planes)

    if show_mask:
        return mask

    dehalo = dehalo.std.MaskedMerge(work_clip, mask, planes=planes)

    if ss > 1:
        w, h = mod4(clip.width * ss), mod4(clip.height * ss)
        ss_clip = core.std.Expr([
            Lanczos(3).scale(work_clip, w, h),
            mitchell.scale(dehalo.std.Maximum(), w, h),
            mitchell.scale(dehalo.std.Minimum(), w, h)
        ], norm_expr('x y min z max'))
        dehalo = Lanczos(3).scale(ss_clip, clip.width, clip.height)
    else:
        dehalo = repair(work_clip, dehalo, [int(i in planes) for i in range(clip.format.num_planes)])

    dehalo = core.std.Expr(
        [work_clip, dehalo], norm_expr(f'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?')
    )

    if chroma:
        return join([dehalo, *chroma], clip.format.color_family)

    return dehalo
