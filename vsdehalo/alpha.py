from __future__ import annotations

from vsaa import Nnedi3
from vsexprtools import ExprOp, aka_expr_available, combine, norm_expr
from vskernels import BSpline, Lanczos, Mitchell, NoShift, Point, Scaler, ScalerT
from vsmasktools import EdgeDetect, Robinson3, XxpandMode, grow_mask, Morpho
from vsrgtools import box_blur, contrasharpening, contrasharpening_dehalo, repair
from vsrgtools.util import norm_rmode_planes
from vstools import (
    ColorRange, ConvMode, CustomIndexError, CustomValueError, FuncExceptT, InvalidColorFamilyError, PlanesT,
    check_variable, clamp, cround, disallow_variable_format, disallow_variable_resolution, fallback, get_peak_value,
    join, mod4, normalize_planes, normalize_seq, scale_value, split, to_arr, vs
)

__all__ = [
    'fine_dehalo',
    'fine_dehalo2',
    'dehalo_alpha'
]


FloatIterArr = float | list[float] | tuple[float | list[float], ...]


@disallow_variable_format
@disallow_variable_resolution
def fine_dehalo(
    clip: vs.VideoNode, rx: FloatIterArr = 2.0, ry: FloatIterArr | None = None, darkstr: FloatIterArr = 0.0,
    brightstr: FloatIterArr = 1.0, lowsens: FloatIterArr = 50.0, highsens: FloatIterArr = 50.0,
    thmi: int = 80, thma: int = 128, thlimi: int = 50, thlima: int = 100, sigma_mask: float = 0.0,
    ss: FloatIterArr = 1.5, contra: int | float | bool = 0.0, exclude: bool = True,
    edgeproc: float = 0.0, edgemask: EdgeDetect = Robinson3(), planes: PlanesT = 0, show_mask: int | bool = False,
    mask_radius: int = 1, downscaler: ScalerT = Mitchell, upscaler: ScalerT = BSpline,
    supersampler: ScalerT = Lanczos(3), supersampler_ref: ScalerT = Mitchell, pre_ss: float = 1.0,
    pre_supersampler: ScalerT = Nnedi3(0, field=0, shifter=NoShift), pre_downscaler: ScalerT = Point,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Halo removal script that uses ``dehalo_alpha`` with a few masks and optional contra-sharpening
    to try removing halos without nuking important details like line edges.

    **For ``rx``, ``ry``, only the first value will be used for calculating the mask.**

    ``rx``, ``ry``, ``darkstr``, ``brightstr``, ``lowsens``, ``highsens``, ``ss`` are all
    configurable per plane and iteration. `tuple` means iteration, `list` plane.

    `rx=(2.0, [2.0, 2.4], [2.2, 2.0, 2.1])` means three iterations.
     * 1st => 2.0 for all planes
     * 2nd => 2.0 for luma, 2.4 for chroma
     * 3rd => 2.2 for luma, 2.0 for u, 2.1 for v


    :param clip:                Source clip.
    :param rx:                  Horizontal radius for halo removal.
    :param ry:                  Vertical radius for halo removal.
    :param darkstr:             Strength factor for dark halos.
    :param brightstr:           Strength factor for bright halos.
    :param lowsens:             Sensitivity setting for defining how weak the dehalo has to be to get fully accepted.
    :param highsens:            Sensitivity setting for define how strong the dehalo has to be to get fully discarded.
    :param thmi:                Minimum threshold for sharp edges; keep only the sharpest edges (line edges).
    :param thma:                Maximum threshold for sharp edges; keep only the sharpest edges (line edges).
    :param thlimi:              Minimum limiting threshold; includes more edges than previously ignoring simple details.
    :param thlima:              Maximum limiting threshold; includes more edges than previously ignoring simple details.
    :param sigma_mask:          Blurring strength for the mask.
    :param ss:                  Supersampling factor, to avoid creation of aliasing.
    :param contra:              Contrasharpening. If True or int, will use :py:func:`contrasharpening`
                                otherwise uses :py:func:`contrasharpening_fine_dehalo` with specified level.
    :param exclude:             If True, add an addionnal step to exclude edges close to each other
    :param edgeproc:            If > 0, it will add the edgemask to the processing, defaults to 0.0
    :param edgemask:            Internal mask used for detecting the edges, defaults to Robinson3()
    :param planes:              Planes to process.
    :param show_mask:           Whether to show the computed halo mask. 1-7 values to select intermediate masks.
    :param mask_radius:         Mask expanding radius with ``gradient``.
    :param downscaler:          Scaler used to downscale the clip.
    :param upscaler:            Scaler used to upscale the downscaled clip.
    :param supersampler:        Scaler used to supersampler the rescaled clip to `ss` factor.
    :param supersampler_ref:    Reference scaler used to clamp the supersampled clip. Has to be blurrier.
    :param pre_ss:              Supersampling rate used before anything else.
    :param pre_supersampler:    Supersampler used for ``pre_ss``.
    :param pre_downscaler:      Downscaler used for undoing the upscaling done by ``pre_supersampler``.
    :param func:                Function from where this function was called.

    :return:                    Dehaloed clip.


    """
    assert clip.format

    func = func or fine_dehalo

    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

    if show_mask is not False and not (0 < int(show_mask) <= 7):
        raise CustomValueError('valid values for show_mask are 1–7!', func)

    thmif, thmaf, thlimif, thlimaf = [
        scale_value(x, 8, clip, ColorRange.FULL)
        for x in [thmi, thma, thlimi, thlima]
    ]

    peak = get_peak_value(clip)
    planes = normalize_planes(clip, planes)

    rx_i, ry_i = cround(to_arr(to_arr(rx)[0])[0]), cround(to_arr(to_arr(fallback(ry, rx))[0])[0])  # type: ignore

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )

    # Main edges #
    # Basic edge detection, thresholding will be applied later.
    edges = edgemask.edgemask(work_clip)

    # Keeps only the sharpest edges (line edges)
    strong = norm_expr(edges, f'x {thmif} - {thmaf - thmif} / {peak} *', planes)

    # Extends them to include the potential halos
    large = Morpho.expand(strong, rx_i, ry_i, planes=planes)

    # Exclusion zones #
    # When two edges are close from each other (both edges of a single
    # line or multiple parallel color bands), the halo removal
    # oversmoothes them or makes seriously bleed the bands, producing
    # annoying artifacts. Therefore we have to produce a mask to exclude
    # these zones from the halo removal.

    # Includes more edges than previously, but ignores simple details
    light = norm_expr(edges, f'x {thlimif} - {thlimaf - thlimif} / {peak} *', planes)

    # To build the exclusion zone, we make grow the edge mask, then shrink
    # it to its original shape. During the growing stage, close adjacent
    # edge masks will join and merge, forming a solid area, which will
    # remain solid even after the shrinking stage.
    # Mask growing
    shrink = Morpho.expand(light, rx_i, ry_i, XxpandMode.ELLIPSE, planes=planes)

    # At this point, because the mask was made of a shades of grey, we may
    # end up with large areas of dark grey after shrinking. To avoid this,
    # we amplify and saturate the mask here (actually we could even
    # binarize it).
    shrink = norm_expr(shrink, 'x 4 *', planes)
    shrink = Morpho.inpand(shrink, rx_i, rx_i, XxpandMode.ELLIPSE, planes=planes)

    # This mask is almost binary, which will produce distinct
    # discontinuities once applied. Then we have to smooth it.
    shrink = box_blur(shrink, 1, 2, planes)

    # Final mask building #

    # Previous mask may be a bit weak on the pure edge side, so we ensure
    # that the main edges are really excluded. We do not want them to be
    # smoothed by the halo removal.
    shr_med = combine([strong, shrink], ExprOp.MAX, planes=planes) if exclude else strong

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

    mask = norm_expr(mask, f'x 2 * {ExprOp.clamp(0, peak)}', planes)

    # Masking #
    if show_mask:
        return [mask, shrink, edges, strong, light, large, shr_med][show_mask - 1]

    dehaloed = dehalo_alpha(
        work_clip, rx, ry, darkstr, brightstr, lowsens, highsens, sigma_mask, ss, planes, False, mask_radius,
        downscaler, upscaler, supersampler, supersampler_ref, pre_ss, pre_supersampler, pre_downscaler, func
    )

    if isinstance(contra, float):
        dehaloed = contrasharpening_dehalo(dehaloed, work_clip, contra, planes=planes)
    elif contra:
        dehaloed = contrasharpening(
            dehaloed, work_clip, None if contra is True else contra, planes=planes
        )

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
        grow_mask(mask, radius, 1.8, planes, coordinates=coord) if mask else None
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
    clip: vs.VideoNode, rx: FloatIterArr = 2.0, ry: FloatIterArr | None = None, darkstr: FloatIterArr = 0.0,
    brightstr: FloatIterArr = 1.0, lowsens: FloatIterArr = 50.0, highsens: FloatIterArr = 50.0,
    sigma_mask: float = 0.0, ss: FloatIterArr = 1.5, planes: PlanesT = 0, show_mask: bool = False,
    mask_radius: int = 1, downscaler: ScalerT = Mitchell, upscaler: ScalerT = BSpline,
    supersampler: ScalerT = Lanczos(3), supersampler_ref: ScalerT = Mitchell, pre_ss: float = 1.0,
    pre_supersampler: ScalerT = Nnedi3(0, field=0, shifter=NoShift), pre_downscaler: ScalerT = Point,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Reduce halo artifacts by nuking everything around edges (and also the edges actually).

    ``rx``, ``ry``, ``darkstr``, ``brightstr``, ``lowsens``, ``highsens``, ``ss`` are all
    configurable per plane and iteration. `tuple` means iteration, `list` plane.

    `rx=(2.0, [2.0, 2.4], [2.2, 2.0, 2.1])` means three iterations.
     * 1st => 2.0 for all planes
     * 2nd => 2.0 for luma, 2.4 for chroma
     * 3rd => 2.2 for luma, 2.0 for u, 2.1 for v

    :param clip:                Source clip.
    :param rx:                  Horizontal radius for halo removal.
    :param ry:                  Vertical radius for halo removal.
    :param darkstr:             Strength factor for dark halos.
    :param brightstr:           Strength factor for bright halos.
    :param lowsens:             Sensitivity setting for defining how weak the dehalo has to be to get fully accepted.
    :param highsens:            Sensitivity setting for define how strong the dehalo has to be to get fully discarded.
    :param sigma_mask:          Blurring strength for the mask.
    :param ss:                  Supersampling factor, to avoid creation of aliasing.
    :param planes:              Planes to process.
    :param show_mask:           Whether to show the computed halo mask.
    :param mask_radius:         Mask expanding radius with ``gradient``.
    :param downscaler:          Scaler used to downscale the clip.
    :param upscaler:            Scaler used to upscale the downscaled clip.
    :param supersampler:        Scaler used to supersampler the rescaled clip to `ss` factor.
    :param supersampler_ref:    Reference scaler used to clamp the supersampled clip. Has to be blurrier.
    :param pre_ss:              Supersampling rate used before anything else.
    :param pre_supersampler:    Supersampler used for ``pre_ss``.
    :param pre_downscaler:      Downscaler used for undoing the upscaling done by ``pre_supersampler``.
    :param func:                Function from where this function was called.

    :return:                    Dehaloed clip.
    """

    func = func or dehalo_alpha

    assert check_variable(clip, func)

    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), func)

    peak = get_peak_value(clip)
    planes = normalize_planes(clip, planes)

    downscaler = Scaler.ensure_obj(downscaler, func)
    upscaler = Scaler.ensure_obj(upscaler, func)
    supersampler = Scaler.ensure_obj(supersampler, func)
    supersampler_ref = Scaler.ensure_obj(supersampler_ref, func)
    pre_supersampler = Scaler.ensure_obj(pre_supersampler, func)
    pre_downscaler = Scaler.ensure_obj(pre_downscaler, func)

    ry = fallback(ry, rx)  # type: ignore

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )

    if pre_ss > 1.0:
        work_clip = pre_supersampler.scale(
            work_clip, mod4(work_clip.width * pre_ss), mod4(work_clip.height * pre_ss)
        )

    assert work_clip.format

    def _rescale(clip: vs.VideoNode, rx: float, ry: float) -> vs.VideoNode:
        return upscaler.scale(downscaler.scale(  # type: ignore
            clip, mod4(clip.width / rx), mod4(clip.height / ry)
        ), clip.width, clip.height)

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

    values = [rx, ry, darkstr, brightstr, lowsens, highsens, ss]

    iterations = max([(len(x) if isinstance(x, tuple) else 1) for x in values])

    values_norm: list[tuple[list[float], ...]] = zip(*[  # type: ignore[assignment]
        tuple(normalize_seq(x) for x in y)
        for y in [
            (*x, *((x[-1], ) * (len(x) - iterations))) if isinstance(x, tuple) else ((x, ) * iterations)
            for x in values
        ]
    ])

    for rx_i, ry_i, darkstr_i, brightstr_i, lowsens_i, highsens_i, ss_i in values_norm:
        if not all(x >= 1 for x in (*ss_i, *rx_i, *ry_i)):
            raise CustomIndexError('ss, rx, and ry must all be bigger than 1.0!', func)

        if not all(0 <= x <= 1 for x in (*brightstr_i, *darkstr_i)):
            raise CustomIndexError('brightstr, darkstr must be between 0.0 and 1.0!', func)

        if not all(0 <= x <= 100 for x in (*lowsens_i, *highsens_i)):
            raise CustomIndexError('lowsens and highsens must be between 0 and 100!', func)

        if len(set(rx_i)) == len(set(ry_i)) == 1 or planes == [0] or work_clip.format.num_planes == 1:
            dehalo = _rescale(work_clip, rx_i[0], ry_i[0])
        else:
            dehalo = join([
                _rescale(plane, rxp, ryp)
                for plane, rxp, ryp in zip(split(work_clip), rx_i, ry_i)
            ])

        mask = norm_expr(
            [Morpho.gradient(work_clip, mask_radius, planes), Morpho.gradient(dehalo, mask_radius, planes)],
            'x 0 = 1.0 x y - x / ? {lowsens} - x {peak} / 256 255 / + 512 255 / / {highsens} + * '
            '0.0 max 1.0 min {peak} *', planes, peak=peak,
            lowsens=[lo / 255 for lo in lowsens_i], highsens=[hi / 100 for hi in highsens_i]
        )

        conv_values = [float((sig_mask := bool(sigma_mask)))] * 9
        conv_values[5] = 1 / clamp(sigma_mask, 0, 1) if sig_mask else 1

        mask = mask.std.Convolution(conv_values, planes=planes)

        if show_mask:
            return mask

        dehalo = dehalo.std.MaskedMerge(work_clip, mask, planes=planes)

        if len(set(ss_i)) == 1 or planes == [0] or work_clip.format.num_planes == 1:
            dehalo = _supersample(work_clip, dehalo, ss_i[0])
        else:
            dehalo = join([
                _supersample(wplane, dplane, ssp)
                for wplane, dplane, ssp in zip(split(work_clip), split(dehalo), ss_i)
            ])

        dehalo = norm_expr(
            [work_clip, dehalo], 'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?', planes,
            darkstr=darkstr_i, brightstr=brightstr_i
        )

        work_clip = dehalo

    if (dehalo.width, dehalo.height) != (clip.width, clip.height):
        dehalo = pre_downscaler.scale(work_clip, clip.width, clip.height)

    if not chroma:
        return dehalo

    return join([dehalo, *chroma], clip.format.color_family)
