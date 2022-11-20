from __future__ import annotations

from vsexprtools import ExprOp, aka_expr_available, combine, norm_expr
from vskernels import BSpline, Lanczos, Mitchell, Scaler, ScalerT
from vsmask.edge import EdgeDetect, Robinson3
from vsmask.util import XxpandMode, expand, inpand
from vsrgtools import box_blur, contrasharpening, contrasharpening_dehalo, repair
from vsrgtools.util import norm_rmode_planes
from vstools import (
    ColorRange, ConvMode, FuncExceptT, InvalidColorFamilyError, PlanesT, check_variable, clamp, cround,
    disallow_variable_format, disallow_variable_resolution, fallback, get_peak_value, get_y, join, mod4,
    normalize_planes, normalize_seq, scale_value, split, to_arr, vs
)

from . import masks

__all__ = [
    'fine_dehalo',
    'fine_dehalo2',
    'dehalo_alpha'
]


@disallow_variable_format
@disallow_variable_resolution
def fine_dehalo(
    clip: vs.VideoNode, /, ref: vs.VideoNode | None = None,
    rx: float = 2.0, ry: float | None = None,
    darkstr: float = 0.0, brightstr: float = 1.0,
    lowsens: int = 50, highsens: int = 50,
    thmi: float = 80.0, thma: float = 128.0,
    thlimi: float = 50.0, thlima: float = 100.0,
    ss: float = 1.25,
    contra: int | float | bool = 0.0, excl: bool = True,
    edgeproc: float = 0.0, planes: PlanesT = 0,
    edgemask: EdgeDetect = Robinson3(), show_mask: int = False
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
    :param contra:      Contrasharpening. If True or int, will use :py:func:`contrasharpening`
                        otherwise uses :py:func:`contrasharpening_fine_dehalo` with specified level.
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

    if not all(0 <= sens <= 100 for sens in (lowsens, highsens)):
        raise ValueError('fine_dehalo: lowsens and highsens must be between 0 and 100!')

    if show_mask is not False and not (0 < int(show_mask) <= 7):
        raise ValueError('fine_dehalo: Valid values for show_mask are 0–7!')

    thmi, thma, thlimi, thlima = [
        scale_value(x, 8, clip.format.bits_per_sample, ColorRange.FULL)
        for x in [thmi, thma, thlimi, thlima]
    ]

    peak = get_peak_value(clip)
    planes = normalize_planes(clip, planes)

    ry = rx if ry is None else ry
    rx_i, ry_i = cround(rx), cround(ry)

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )

    if ref:
        dehaloed = get_y(ref) if planes == [0] else ref
    else:
        dehaloed = dehalo_alpha(
            work_clip, rx, ry, darkstr, brightstr, lowsens, highsens, ss=ss, planes=planes
        )

    if contra:
        if isinstance(contra, (int, bool)):
            dehaloed = contrasharpening(
                dehaloed, work_clip, None if contra is True else contra, planes=planes
            )
        else:
            dehaloed = contrasharpening_dehalo(dehaloed, work_clip, contra, planes=planes)

    # Main edges #
    # Basic edge detection, thresholding will be applied later.
    edges = edgemask.edgemask(work_clip)

    # Keeps only the sharpest edges (line edges)
    strong = norm_expr(edges, f'x {thmi} - {thma - thmi} / {peak} *', planes)

    # Extends them to include the potential halos
    large = expand(strong, rx_i, ry_i, planes=planes)

    # Exclusion zones #
    # When two edges are close from each other (both edges of a single
    # line or multiple parallel color bands), the halo removal
    # oversmoothes them or makes seriously bleed the bands, producing
    # annoying artifacts. Therefore we have to produce a mask to exclude
    # these zones from the halo removal.

    # Includes more edges than previously, but ignores simple details
    light = norm_expr(edges, f'x {thlimi} - {thlima - thlimi} / {peak} *', planes)

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
    shrink = norm_expr(shrink, 'x 4 *', planes)
    shrink = inpand(shrink, rx_i, rx_i, XxpandMode.ELLIPSE, planes=planes)

    # This mask is almost binary, which will produce distinct
    # discontinuities once applied. Then we have to smooth it.
    shrink = box_blur(shrink, 1, 2, planes)

    # Final mask building #

    # Previous mask may be a bit weak on the pure edge side, so we ensure
    # that the main edges are really excluded. We do not want them to be
    # smoothed by the halo removal.
    shr_med = combine([strong, shrink], ExprOp.MAX, planes=planes) if excl else strong

    # Substracts masks and amplifies the difference to be sure we get 255
    # on the areas to be processed.
    mask = norm_expr([large, shr_med], 'x y - 2 *', planes)

    # If edge processing is required, adds the edgemask
    if edgeproc > 0:
        mask = norm_expr([mask, strong], f'x y {edgeproc} 0.66 * * +', planes)

    # Smooth again and amplify to grow the mask a bit, otherwise the halo
    # parts sticking to the edges could be missed.
    # Also clamp to legal ranges
    mask = box_blur(mask, planes=planes)

    if aka_expr_available:
        clamp_expr = f'0 {peak} clamp'
    else:
        clamp_expr = f'0 max {peak} min'

    mask = norm_expr(mask, f'x 2 * {clamp_expr}', planes)

    # Masking #
    if show_mask:
        return [mask, shrink, edges, strong, light, large, shr_med][show_mask - 1]

    y_merge = work_clip.std.MaskedMerge(dehaloed, mask, planes)

    if chroma:
        return join([y_merge, *chroma], clip.format.color_family)

    return y_merge


def fine_dehalo2(
    clip: vs.VideoNode, mode: ConvMode = ConvMode.SQUARE,
    radius: int = 2, dark: bool | None = True, planes: PlanesT = 0, show_mask: bool = False
) -> vs.VideoNode:
    """
    Halo removal function for 2nd order halos.

    :param clip:        Source clip.
    :param mode:        Horizontal/Vertical or both ways.
    :param radius:      Radius for mask growing.
    :param dark:        Whether to filter for dark or bright haloing.
                        None for disable merging with source clip.
    :param planes:      Planes to process.
    :param show_mask:   Whether to return the computed mask.

    :return:            Dehaloed clip.
    """
    assert clip.format

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('fine_dehalo2: format not supported')

    planes = normalize_planes(clip, planes)

    is_float = clip.format.sample_type == vs.FLOAT

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )

    mask_h = mask_v = None

    # intended to be reversed
    if aka_expr_available:
        h_mexpr, v_mexpr = [
            ExprOp.convolution('x', coord, None, 4, False)
            for coord in [
                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
            ]
        ]

        if mode == ConvMode.SQUARE:
            do_mv = do_mh = True
        else:
            do_mv, do_mh = [
                mode == m for m in {ConvMode.HORIZONTAL, ConvMode.VERTICAL}
            ]

        mask_args = (h_mexpr, do_mv, do_mh, v_mexpr)

        mask_h, mask_v = [
            norm_expr(work_clip, [
                mexpr, 3, ExprOp.MUL, [omexpr, ExprOp.SUB] if do_om else None, ExprOp.clamp(0, 1) if is_float else None
            ], planes) if do_m else None
            for mexpr, do_m, do_om, omexpr in [mask_args, mask_args[::-1]]
        ]
    else:
        if mode in {ConvMode.SQUARE, ConvMode.VERTICAL}:
            mask_h = work_clip.std.Convolution([1, 2, 1, 0, 0, 0, -1, -2, -1], None, 4, planes, False)

        if mode in {ConvMode.SQUARE, ConvMode.HORIZONTAL}:
            mask_v = work_clip.std.Convolution([1, 0, -1, 2, 0, -2, 1, 0, -1], None, 4, planes, False)

        if mask_h and mask_v:
            mask_h = norm_expr([mask_h, mask_v], 'x 3 * y -', planes)
            mask_v = norm_expr([mask_v, mask_h], 'x 3 * y -', planes)
        elif mask_h:
            mask_h = norm_expr(mask_h, 'x 3 *', planes)
        elif mask_v:
            mask_v = norm_expr(mask_v, 'x 3 *', planes)

        if is_float:
            mask_h = mask_h and mask_h.std.Limiter(planes=planes)
            mask_v = mask_v and mask_v.std.Limiter(planes=planes)

    fix_h, fix_v = [
        norm_expr(work_clip, ExprOp.convolution('x', coord, mode=mode), planes)
        if aka_expr_available else
        work_clip.std.Convolution(coord, planes=planes, mode=mode)
        for coord, mode in [
            ([-1, -2, 0, 0, 40, 0, 0, -2, -1], ConvMode.HORIZONTAL),
            ([-2, -1, 0, 0, 40, 0, 0, -1, -2], ConvMode.VERTICAL)
        ]
    ]

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
            return combine([mask_h, mask_v], ExprOp.MAX)

        assert (ret_mask := mask_h or mask_v)

        return ret_mask

    dehaloed = work_clip
    op = '' if dark is None else ExprOp.MAX if dark else ExprOp.MIN

    if aka_expr_available and mask_h and mask_v and clip.format.sample_type is vs.FLOAT:
        dehaloed = norm_expr(
            [work_clip, fix_h, fix_v, mask_h, mask_v, clip], f'x 1 a - * y a * + 1 b - * z b * + c {op}', planes
        )
    else:
        for fix, mask in [(fix_h, mask_v), (fix_v, mask_h)]:
            if mask:
                dehaloed = dehaloed.std.MaskedMerge(fix, mask)

        if op:
            dehaloed = combine([work_clip, dehaloed], ExprOp(op))

    if not chroma:
        return dehaloed

    return join([dehaloed, *chroma], clip.format.color_family)


@disallow_variable_format
@disallow_variable_resolution
def dehalo_alpha(
    clip: vs.VideoNode,
    rx: float | list[float] = 2.0, ry: float | list[float] | None = None,
    darkstr: float | list[float] = 0.0, brightstr: float | list[float] = 1.0,
    lowsens: float | list[float] = 50.0, highsens: float | list[float] = 50.0,
    sigma_mask: float = 0.0, ss: float | list[float] = 1.5,
    planes: PlanesT = 0, show_mask: bool = False, mask_radius: int = 1,
    downscaler: ScalerT = Mitchell, upscaler: ScalerT = BSpline,
    supersampler: ScalerT = Lanczos(3), supersampler_ref: ScalerT = Mitchell,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Reduce halo artifacts by nuking everything around edges (and also the edges actually).

    :param clip:                Source clip.
    :param rx:                  Horizontal radius for halo removal.
    :param ry:                  Vertical radius for halo removal.
    :param darkstr:             Strength factor for dark halos.
    :param brightstr:           Strength factor for bright halos.
    :param lowsens:             Sensitivity setting for lower bound.
    :param highsens:            Sensitivity setting for higher bound.
    :param sigma_mask:          Blurring strength for the mask.
    :param ss:                  Supersampling factor, to avoid creation of aliasing.
    :param planes:              Planes to process.
    :param show_mask:           Whether to show the computed halo mask.
    :param downscaler:          Scaler used to downscale the clip.
    :param upscaler:            Scaler used to upscale the downscaled clip.
    :param supersampler:        Scaler used to supersampler the rescaled clip to `ss` factor.
    :param supersampler_ref:    Reference scaler used to clamp the supersampled clip. Has to be blurrier.
    :param func:                Function from where this function was called.

    :return:                    Dehaloed clip.
    """

    func = func or dehalo_alpha

    assert check_variable(clip, func)

    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

    peak = get_peak_value(clip)
    planes = normalize_planes(clip, planes)

    ry = fallback(ry, rx)  # type: ignore

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )
    assert work_clip.format

    downscaler = Scaler.ensure_obj(downscaler, func)
    upscaler = Scaler.ensure_obj(upscaler, func)
    supersampler = Scaler.ensure_obj(supersampler, func)
    supersampler_ref = Scaler.ensure_obj(supersampler_ref, func)

    def _rescale(clip: vs.VideoNode, rx: float, ry: float) -> vs.VideoNode:
        return upscaler.scale(downscaler.scale(  # type: ignore
            clip, mod4(clip.width / rx), mod4(clip.height / ry)
        ), clip.width, clip.height)

    rx, ry = normalize_seq(rx), normalize_seq(ry)  # type: ignore

    if len(set(rx)) == len(set(ry)) == 1 or planes == [0] or work_clip.format.num_planes == 1:
        rx, ry = rx[0], ry[0]

    if isinstance(rx, list) or isinstance(ry, list):
        dehalo = join([
            _rescale(plane, rxp, ryp)
            for plane, rxp, ryp in zip(split(work_clip), normalize_seq(rx), normalize_seq(ry))
        ])
    else:
        dehalo = _rescale(work_clip, rx, ry)

    org_minmax = masks.gradient(work_clip, mask_radius, planes)
    dehalo_minmax = masks.gradient(dehalo, mask_radius, planes)

    mask = norm_expr(
        [org_minmax, dehalo_minmax],
        'x 0 = 1.0 x y - x / ? {lowsens} - x {peak} / 256 255 / + 512 255 / / {highsens} + * '
        '0.0 max 1.0 min {peak} *', planes, peak=peak,
        lowsens=[lo / 255 for lo in to_arr(lowsens)], highsens=[hi / 100 for hi in to_arr(highsens)]
    )

    sig_mask = bool(sigma_mask)
    conv_values = [float(sig_mask)] * 9
    conv_values[5] = 1 / clamp(sigma_mask, 0, 1) if sig_mask else 1

    mask = mask.std.Convolution(conv_values, planes=planes)

    if show_mask:
        return mask

    dehalo = dehalo.std.MaskedMerge(work_clip, mask, planes=planes)

    def _supersample(work_clip: vs.VideoNode, dehalo: vs.VideoNode, ss: float) -> vs.VideoNode:
        if ss <= 1.0:
            return repair(work_clip, dehalo, norm_rmode_planes(work_clip, 1, planes))

        w, h = mod4(work_clip.width * ss), mod4(work_clip.height * ss)
        ss_clip = norm_expr([
            supersampler.scale(work_clip, w, h),  # type: ignore
            supersampler_ref.scale(dehalo.std.Maximum(), w, h),  # type: ignore
            supersampler_ref.scale(dehalo.std.Minimum(), w, h)  # type: ignore
        ], 'x y min z max', planes)

        return supersampler.scale(ss_clip, work_clip.width, work_clip.height)  # type: ignore

    ss = normalize_seq(ss)

    if len(set(ss)) == 1 or planes == [0] or work_clip.format.num_planes == 1:
        ss = ss[0]

    if isinstance(ss, list):
        dehalo = join([
            _supersample(work_clip, dehalo, ssp)
            for wplane, dplane, ssp in zip(split(work_clip), split(dehalo), ss)
        ])
    else:
        dehalo = _supersample(work_clip, dehalo, ss)

    dehalo = norm_expr(
        [work_clip, dehalo], 'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?', planes,
        darkstr=darkstr, brightstr=brightstr
    )

    if not chroma:
        return dehalo

    return join([dehalo, *chroma], clip.format.color_family)
