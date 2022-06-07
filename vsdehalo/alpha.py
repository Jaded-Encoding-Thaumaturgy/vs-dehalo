from __future__ import annotations

import vapoursynth as vs
from vskernels import BSpline, Lanczos, Mitchell
from vsmask.better_vsutil import join, split
from vsmask.edge import EdgeDetect, Robinson3
from vsmask.types import ensure_format as _ensure_format
from vsmask.util import XxpandMode, expand, inpand
from vsrgtools import contrasharpening, contrasharpening_dehalo, repair
from vsutil import Range as CRange
from vsutil import disallow_variable_format, get_peak_value, get_y, scale_value

from .utils import clamp, cround, mod4

core = vs.core

bspline = BSpline()
mitchell = Mitchell()


@disallow_variable_format
def fine_dehalo(
    clip: vs.VideoNode, /, ref: vs.VideoNode | None = None,
    rx: float = 2.0, ry: float | None = None,
    darkstr: float = 1.0, brightstr: float = 1.0,
    lowsens: int = 50, highsens: int = 50,
    thmi: int | float = 80, thma: int | float = 128,
    thlimi: int | float = 50, thlima: int | float = 100,
    ss: float = 1.25,
    contra: float | bool = 0.0, excl: bool = True,
    edgeproc: float = 0.0,
    edgemask: EdgeDetect = Robinson3(), showmask: int = 0
) -> vs.VideoNode:
    """
    Halo removal script that uses DeHalo_alpha with a few masks and optional contra-sharpening
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
    :param showmask:    1 - 7
    :return:            Dehaloed clip
    """
    clip = _ensure_format(clip)

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('fine_dehalo: format not supported')

    peak = get_peak_value(clip)

    thmi, thma, thlimi, thlima = [
        scale_value(x, 8, clip.format.bits_per_sample, CRange.FULL)
        for x in [thmi, thma, thlimi, thlima]
    ]

    ry = rx if ry is None else ry
    rx_i = cround(rx)
    ry_i = cround(ry)

    clip_y, *chroma = split(clip)

    if ref:
        dehaloed = get_y(ref)
    else:
        dehaloed = dehalo_alpha(clip_y, rx, ry, darkstr, brightstr, lowsens, highsens, ss=ss)

    if contra:
        if isinstance(contra, bool):
            dehaloed = contrasharpening(dehaloed, clip_y)
        else:
            dehaloed = contrasharpening_dehalo(dehaloed, clip_y, contra)

    # Main edges #
    # Basic edge detection, thresholding will be applied later.
    edges = edgemask.edgemask(clip_y)

    # Keeps only the sharpest edges (line edges)
    strong = edges.std.Expr(f'x {thmi} - {thma - thmi} / {peak} *')

    # Extends them to include the potential halos
    large = expand(strong, rx_i, ry_i)

    # Exclusion zones #
    # When two edges are close from each other (both edges of a single
    # line or multiple parallel color bands), the halo removal
    # oversmoothes them or makes seriously bleed the bands, producing
    # annoying artifacts. Therefore we have to produce a mask to exclude
    # these zones from the halo removal.

    # Includes more edges than previously, but ignores simple details
    light = edges.std.Expr(f'x {thlimi} - {thlima - thlimi} / {peak} *')

    # To build the exclusion zone, we make grow the edge mask, then shrink
    # it to its original shape. During the growing stage, close adjacent
    # edge masks will join and merge, forming a solid area, which will
    # remain solid even after the shrinking stage.
    # Mask growing
    shrink = expand(light, rx_i, ry_i, XxpandMode.ELLIPSE)

    # At this point, because the mask was made of a shades of grey, we may
    # end up with large areas of dark grey after shrinking. To avoid this,
    # we amplify and saturate the mask here (actually we could even
    # binarize it).
    shrink = shrink.std.Expr(expr="x 4 *")
    shrink = inpand(shrink, rx_i, rx_i, XxpandMode.ELLIPSE)

    # This mask is almost binary, which will produce distinct
    # discontinuities once applied. Then we have to smooth it.
    shrink = shrink.std.BoxBlur(hradius=1, vradius=1, hpasses=2, vpasses=2)

    # Final mask building #

    # Previous mask may be a bit weak on the pure edge side, so we ensure
    # that the main edges are really excluded. We do not want them to be
    # smoothed by the halo removal.
    shr_med = core.std.Expr([strong, shrink], 'x y max') if excl else strong

    # Substracts masks and amplifies the difference to be sure we get 255
    # on the areas to be processed.
    mask = core.std.Expr([large, shr_med], 'x y - 2 *')

    # If edge processing is required, adds the edgemask
    if edgeproc > 0:
        mask = core.std.Expr([mask, strong], f'x y {edgeproc} 0.66 * * +')

    # Smooth again and amplify to grow the mask a bit, otherwise the halo
    # parts sticking to the edges could be missed.
    # Also clamp to legal ranges
    mask = mask.std.Convolution([1] * 9)
    mask = mask.std.Expr(f'x 2 * 0 max {peak} min')

    # Masking #
    if showmask:
        if showmask == 1:
            return mask
        if showmask == 2:
            return shrink
        if showmask == 3:
            return edges
        if showmask == 4:
            return strong
        if showmask == 5:
            return light
        if showmask == 6:
            return large
        if showmask == 7:
            return shr_med

    y_merge = clip_y.std.MaskedMerge(dehaloed, mask)

    return join([y_merge, *chroma], clip.format.color_family)


@disallow_variable_format
def dehalo_alpha(
    clip: vs.VideoNode,
    rx: float = 2.0, ry: float | None = None,
    darkstr: float = 1.0, brightstr: float = 1.0,
    lowsens: float = 50, highsens: float = 50,
    sigma_mask: float = 0.0, ss: float = 1.5, show_mask: bool = False
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
    clip = _ensure_format(clip)
    if clip.format.color_family not in {vs.GRAY, vs.YUV}:
        raise ValueError('dehalo_alpha: only GRAY and YUV formats are supported')

    peak = get_peak_value(clip)

    ry = rx if ry is None else ry

    clip_y, *chroma = split(clip)

    dehalo = mitchell.scale(clip_y, mod4(clip.width / rx), mod4(clip.height / ry))
    dehalo = bspline.scale(dehalo, clip.width, clip.height)

    org_minmax = core.std.Expr([clip_y.std.Maximum(), clip_y.std.Minimum()], 'x y -')
    dehalo_minmax = core.std.Expr([dehalo.std.Maximum(), dehalo.std.Minimum()], 'x y -')

    mask = core.std.Expr(
        [org_minmax, dehalo_minmax],
        f'x 0 = 1.0 x y - x / ? {lowsens / 255} - x {peak} / 256 255 / + 512 255 / / {highsens / 100} + * '
        # f'{lowsens / 255} - x {peak} / 1.003921568627451 + 2.007843137254902 / {highsens / 100} + * '
        # f'{lowsens / 255} - x {peak} / 0.498046862745098 * 0.5 + {highsens / 100} + * '
        f'0.0 max 1.0 min {peak} *',
    )

    sig_mask = bool(sigma_mask)
    conv_values = [float(sig_mask)] * 9
    conv_values[5] = 1 / clamp(sigma_mask, 0, 1) if sig_mask else 1

    mask = mask.std.Convolution(conv_values)

    if show_mask:
        return mask

    dehalo = dehalo.std.MaskedMerge(clip_y, mask)

    if ss > 1:
        w, h = mod4(clip.width * ss), mod4(clip.height * ss)
        ss_clip = core.std.Expr([
            Lanczos(3).scale(clip_y, w, h),
            mitchell.scale(dehalo.std.Maximum(), w, h),
            mitchell.scale(dehalo.std.Minimum(), w, h)
        ], 'x y min z max')
        dehalo = Lanczos(3).scale(ss_clip, clip.width, clip.height)
    else:
        dehalo = repair(clip_y, dehalo, 1)

    dehalo = core.std.Expr([clip_y, dehalo], f'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?')

    return join([dehalo] + chroma, clip.format.color_family)
