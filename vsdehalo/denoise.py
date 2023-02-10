from __future__ import annotations

from math import ceil

from vsaa import Nnedi3
from vsdenoise import Prefilter
from vsexprtools import ExprOp, ExprToken, norm_expr
from vskernels import NoShift, Point, Scaler, ScalerT
from vsmasktools import Morpho, Prewitt
from vsrgtools import LimitFilterMode, bilateral, contrasharpening, contrasharpening_dehalo, limit_filter, repair
from vstools import (
    FunctionUtil, KwargsT, MatrixT, PlanesT, check_ref_clip, check_variable, core, fallback, get_y, mod4, scale_value,
    vs
)

__all__ = [
    'bidehalo',
    'smooth_dering'
]


def bidehalo(
    clip: vs.VideoNode, sigma: float = 1.5, radius: float = 7,
    sigma_final: float | None = None, radius_final: float | None = None,
    tr: int = 2, gpu: bool | None = None, matrix: MatrixT | None = None, planes: PlanesT = 0,
    bm3d_args: KwargsT | None = None, bilateral_args: KwargsT | None = None
) -> vs.VideoNode:
    """
    Simple dehalo function that uses ``bilateral`` and ``BM3D`` to remove bright haloing around edges.

    This works by utilising the ``ref`` parameter in ``bilateral`` to limit the areas that get damaged,
    and how much it gets damaged. You should use this function in conjunction with a halo mask.

    If a ref clip is passed, that will be used as a ref for the second bilateral pass instead of a blurred clip.
    Both clips will be resampled to 16bit internally, and returned in the input bitdepth.

    Recommend values for `sigma` are between 0.8 and 2.0.
    Recommend values for `radius` are between 5 and 15.

    :param clip:                Clip to process.
    :param sigma:               ``Bilateral`` spatial weight sigma.
    :param radius:              ``Bilateral`` radius weight sigma.
    :param sigma_final:         Final ``Bilateral`` call's spatial weight sigma.
                                You'll want this to be much weaker than the initial `sigma`.
                                If `None`, 1/3rd of `sigma`.
    :param radius_final:        Final ``Bilateral`` radius weight sigma.
                                if `None`, same as `radius`.
    :param tr:                  Temporal radius for BM3D
    :param gpu:                 Whether to process with GPU or not. None is auto.
    :param matrix:              Matrix parameter for BM3D YUV/RGB/OPP conversion.
    :param planes:              Specifies which planes will be processed.
    :param bm3d_args:           Additional parameters to pass to BM3D.
    :param bilateral_args:      Additional parameters to pass to Bilateral.

    :return:                    Dehalo'd clip using ``BM3D`` and ``Bilateral``.
    """

    assert check_variable(clip, bidehalo)

    bm3d_args, bilateral_args = bm3d_args or KwargsT(), bilateral_args or KwargsT()

    sigma_final = fallback(sigma_final, sigma / 3)
    radius_final = fallback(radius_final, radius)

    cuda = fallback(gpu, hasattr(core, 'bm3dcuda'))

    den = Prefilter.BM3D(
        clip, planes, sigma=[10.0, 8.0] if cuda else [8.0, 6.4], radius=tr, gpu=cuda, matrix=matrix, **bm3d_args
    )

    ref = bilateral(den, sigma, radius / 255, planes=planes, **bilateral_args)
    bidh = bilateral(den, sigma_final, radius_final / 255, ref, planes=planes, **bilateral_args)

    return ExprOp.MIN(clip, bidh, planes=planes)


def smooth_dering(
    clip: vs.VideoNode,
    smooth: vs.VideoNode | Prefilter = Prefilter.MINBLUR1,
    ringmask: vs.VideoNode | None = None,
    mrad: int = 1, msmooth: int = 1, minp: int = 1, mthr: int = 60, incedge: bool = False,
    thr: int = 12, elast: float = 2.0, darkthr: int | None = None,
    contra: int | float | bool = 1.2, drrep: int = 13, pre_ss: float = 1.0,
    pre_supersampler: ScalerT = Nnedi3(0, field=0, shifter=NoShift),
    pre_downscaler: ScalerT = Point, planes: PlanesT = 0, show_mask: bool = False
) -> vs.VideoNode:
    """
    :param clip:        Clip to process.
    :param smooth:      Already smoothed clip, or a Prefilter, tuple for [luma, chroma] prefilter.
    :param ringmask:    Custom ringing mask.
    :param mrad:        Expanding iterations of edge mask, higher value means more aggressive processing.
    :param msmooth:     Inflating iterations of edge mask, higher value means smoother edges of mask.
    :param minp:        Inpanding iterations of prewitt edge mask, higher value means more aggressive processing.
    :param mthr:        Threshold of prewitt edge mask, lower value means more aggressive processing
                        but for strong ringing, lower value will treat some ringing as edge,
                        which "protects" this ringing from being processed.
    :param incedge:     Whether to include edge in ring mask, by default ring mask only include area near edges.
    :param thr:         Threshold (8-bit scale) to limit filtering diff.
                        Smaller thr will result in more pixels being taken from processed clip.
                        Larger thr will result in less pixels being taken from input clip.
                            PDiff: pixel value diff between processed clip and input clip
                            ODiff: pixel value diff between output clip and input clip
                            PDiff, thr and elast is used to calculate ODiff:
                            ODiff = PDiff when [PDiff <= thr]
                            ODiff gradually smooths from thr to 0 when [thr <= PDiff <= thr * elast]
                            For elast>2.0, ODiff reaches maximum when [PDiff == thr * elast / 2]
                            ODiff = 0 when [PDiff >= thr * elast]
    :param elast:       Elasticity of the soft threshold.
                        Larger "elast" will result in more pixels being blended from.
    :param darkthr:     Threshold (8-bit scale) for darker area near edges, for filtering diff
                        that brightening the image by default equals to thr/4.
                        Set it lower if you think de-ringing destroys too much lines, etc.
                        When darkthr is not equal to ``thr``, ``thr`` limits darkening,
                        while ``darkthr`` limits brightening.
                        This is useful to limit the overshoot/undershoot/blurring introduced in deringing.
                        Examples:
                            ``thr=0``,   ``darkthr=0``  : no limiting
                            ``thr=255``, ``darkthr=255``: no limiting
                            ``thr=8``,   ``darkthr=2``  : limit darkening with 8, brightening is limited to 2
                            ``thr=8``,   ``darkthr=0``  : limit darkening with 8, brightening is limited to 0
                            ``thr=255``, ``darkthr=0``  : limit darkening with 255, brightening is limited to 0
                            For the last two examples, output will remain unchanged. (0/255: no limiting)
    :param contra:      Whether to use contra-sharpening to resharp deringed clip:
                            False: no contrasharpening
                            True: auto radius for contrasharpening
                            int 1-3: represents radius for contrasharpening
                            float: represents level for contrasharpening_dehalo
    :param drrep:       Use repair for details retention, recommended values are 13/12/1.
    :param planes:      Planes to be processed.
    :param show_mask:   Show the computed ringing mask.
    :param kwargs:      Kwargs to be passed to the prefilter function.

    :return:            Deringed clip.
    """
    func = FunctionUtil(clip, smooth_dering, planes, (vs.GRAY, vs.YUV))
    planes = func.norm_planes
    work_clip = func.work_clip

    pre_supersampler = Scaler.ensure_obj(pre_supersampler, smooth_dering)
    pre_downscaler = Scaler.ensure_obj(pre_downscaler, smooth_dering)

    if pre_ss > 1.0:
        work_clip = pre_supersampler.scale(  # type: ignore
            work_clip, mod4(work_clip.width * pre_ss), mod4(work_clip.height * pre_ss)
        )

    darkthr = fallback(darkthr, thr // 4)

    rep_dr = [drrep if i in planes else 0 for i in range(work_clip.format.num_planes)]

    if not isinstance(smooth, vs.VideoNode):
        smoothed = smooth(work_clip, planes)  # type: ignore
    else:
        check_ref_clip(clip, smooth)  # type: ignore

        smoothed = get_y(smooth) if func.luma_only else smooth  # type: ignore

    if contra:
        if isinstance(contra, int):
            smoothed = contrasharpening(smoothed, work_clip, contra, 13, planes)
        else:
            smoothed = contrasharpening_dehalo(smoothed, work_clip, contra, planes=planes)

    if set(rep_dr) != {0}:
        repclp = repair(work_clip, smoothed, drrep)
    else:
        repclp = work_clip

    limitclp = limit_filter(
        repclp, work_clip, None, LimitFilterMode.CLAMPING, planes, thr, elast, darkthr
    )

    if ringmask is None:
        prewittm = Prewitt.edgemask(work_clip, scale_value(mthr, 8, work_clip))

        fmask = prewittm.std.Median(planes).misc.Hysteresis(prewittm, planes)

        omask = Morpho.expand(fmask, mrad, mrad, planes=planes) if mrad > 0 else fmask

        if msmooth > 0:
            omask = Morpho.inflate(omask, msmooth, planes)

        if incedge:
            ringmask = omask
        else:
            if minp <= 0:
                imask = fmask
            elif minp % 2 == 0:
                imask = Morpho.inpand(fmask, minp // 2, planes=planes)
            else:
                imask = Morpho.inpand(Morpho.inflate(fmask, 1, planes), ceil(minp / 2), planes=planes)

            ringmask = norm_expr([omask, imask], f'x {ExprToken.RangeMax} y - * {ExprToken.RangeMax} /')

    dering = work_clip.std.MaskedMerge(limitclp, ringmask, planes)

    if show_mask:
        return ringmask

    if (dering.width, dering.height) != (clip.width, clip.height):
        dering = pre_downscaler.scale(work_clip, clip.width, clip.height)

    return func.return_clip(dering)
