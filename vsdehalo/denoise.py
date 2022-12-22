from __future__ import annotations

from functools import partial
from math import ceil
from typing import Any, Literal, cast

from vsdenoise import BM3D, BM3DCPU, BM3DCuda, BM3DCudaRTC, Prefilter
from vsexprtools import norm_expr_planes
from vsmasktools import Morpho, Prewitt
from vsrgtools import LimitFilterMode, contrasharpening, contrasharpening_dehalo, limit_filter, repair
from vstools import (
    PlanesT, core, depth, disallow_variable_format, disallow_variable_resolution, fallback, get_depth, get_peak_value,
    get_y, iterate, join, normalize_planes, normalize_seq, scale_value, split, vs, check_ref_clip
)

__all__ = [
    'bidehalo', 'HQDeringmod'
]


@disallow_variable_format
@disallow_variable_resolution
def bidehalo(
    clip: vs.VideoNode,
    sigma: float = 1.5, radius: float = 7,
    sigma_final: float | None = None, radius_final: float | None = None,
    tr: int = 2, cuda: bool | Literal['rtc'] = False,
    planes: PlanesT = 0, matrix: int | vs.MatrixCoefficients | None = None,
    bm3d_args: dict[str, Any] | None = None, bilateral_args: dict[str, Any] | None = None
) -> vs.VideoNode:
    """
    Simple dehalo function that uses ``bilateral`` and ``BM3D`` to remove bright haloing around edges.

    This works by utilising the ``ref`` parameter in ``bilateral`` to limit the areas that get damaged,
    and how much it gets damaged. You should use this function in conjunction with a halo mask.

    If a ref clip is passed, that will be used as a ref for the second bilateral pass instead of a blurred clip.
    Both clips will be resampled to 16bit internally, and returned in the input bitdepth.

    Recommend values for `sigma` are between 0.8 and 2.0.
    Recommend values for `radius` are between 5 and 15.

    Dependencies:

    * VapourSynth-Bilateral (Default)
    * VapourSynth-BilateralGPU (Cuda)
    * VapourSynth-BilateralGPU_RTC (RTC)
    * vsdenoise

    :param clip:                Clip to process.
    :param sigma:               ``Bilateral`` spatial weight sigma.
    :param sigma_final:         Final ``Bilateral`` call's spatial weight sigma.
                                You'll want this to be much weaker than the initial `sigma`.
                                If `None`, 1/3rd of `sigma`.
    :param radius:              ``Bilateral`` radius weight sigma.
    :param radius_final:        Final ``Bilateral`` radius weight sigma.
                                if `None`, same as `radius`.
    :param tr:                  Temporal radius for BM3D
    :param cuda:                Use ``BM3DCUDA`` and `BilateralGPU` if True, else ``BM3DCPU`` and `Bilateral`.
                                Also accepts 'rtc' for ``BM3DRTC`` and `BilateralGPU_RTC`.
                                Notice: final pass of bilateral will always be on cpu since the
                                gpu implementation doesn't support passing ``ref``.
    :param planes:              Specifies which planes will be processed.
                                Any unprocessed planes will be simply copied.
    :param bm3d_args:           Additional parameters to pass to BM3D.
    :param bilateral_args:      Additional parameters to pass to Bilateral.

    :return:                    Dehalo'd clip using ``BM3D`` and ``Bilateral``.
    """
    assert clip.format

    bits = get_depth(clip)

    bm3d_args = bm3d_args or dict[str, Any]()
    bilateral_args = bilateral_args or dict[str, Any]()

    sigma_final = fallback(sigma_final, sigma / 3)
    radius_final = fallback(radius_final, radius)

    planes = normalize_planes(clip, planes)

    if matrix:
        clip = clip.std.SetFrameProp('_Matrix', int(matrix))

    process_chroma = 1 in planes or 2 in planes

    if not cuda:
        sigma_luma, sigma_chroma = 8, process_chroma and 6.4
    else:
        sigma_luma, sigma_chroma = 10, process_chroma and 8

    bm3d_pargs = (depth(clip, 16), [sigma_luma, sigma_chroma], tr)

    if cuda is False:
        try:
            den = BM3D(*bm3d_pargs, **bm3d_args).clip
        except AttributeError:
            den = BM3DCPU(*bm3d_pargs, **bm3d_args).clip

        ref = den.bilateral.Bilateral(None, sigma, radius / 255, planes, **bilateral_args)
    else:
        bil_gpu_args = dict[str, Any](sigma_spatial=sigma, **bilateral_args)

        if cuda is True:
            den = BM3DCuda(*bm3d_pargs, **bm3d_args).clip
            ref = den.bilateralgpu.Bilateral(**bil_gpu_args)
        elif cuda == 'rtc':
            den = BM3DCudaRTC(*bm3d_pargs, **bm3d_args).clip
            ref = den.bilateralgpu_rtc.Bilateral(**bil_gpu_args)
        else:
            raise ValueError(f'bidehalo: Invalid cuda selection ({cuda})!')

    bidh = den.bilateral.Bilateral(ref, sigma_final, radius_final / 255, planes, **bilateral_args)
    bidh = depth(bidh, bits)

    return core.std.Expr([clip, bidh], norm_expr_planes(clip, 'x y min', planes))


@disallow_variable_format
@disallow_variable_resolution
def HQDeringmod(
    clip: vs.VideoNode,
    smooth: vs.VideoNode | Prefilter | tuple[Prefilter, Prefilter] = Prefilter.MINBLUR1,
    ringmask: vs.VideoNode | None = None,
    mrad: int = 1, msmooth: int = 1, minp: int = 1, mthr: int = 60, incedge: bool = False,
    thr: int = 12, elast: float = 2.0, darkthr: int | None = None,
    sigma: float = 128.0, sigma2: float | None = None,
    sbsize: int | None = None, sosize: int | None = None,
    contra: int | float | bool = 1.2, drrep: int = 13,
    planes: PlanesT = 0, show: bool = False, **kwargs: Any
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
    :param sigma:       DFTTest Prefilter only: sigma for medium frequecies
    :param sigma2:      DFTTest Prefilter only: sigma for low&high frequecies
    :param sbsize:      DFTTest Prefilter only: length of the sides of the spatial window
    :param sosize:      DFTTest Prefilter only: spatial overlap amount
    :param contra:      Whether to use contra-sharpening to resharp deringed clip:
                            False: no contrasharpening
                            True: auto radius for contrasharpening
                            int 1-3: represents radius for contrasharpening
                            float: represents level for contrasharpening_dehalo
    :param drrep:       Use repair for details retention, recommended values are 13/12/1.
    :param planes:      Planes to be processed.
    :param show:        Show the computed ringing mask.
    :param kwargs:      Kwargs to be passed to the prefilter function.

    :return:            Deringed clip.
    """
    assert clip.format

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('HQDeringmod: format not supported')

    peak = get_peak_value(clip)
    bits = clip.format.bits_per_sample
    planes = normalize_planes(clip, planes)
    work_clip, *chroma = split(clip) if planes == [0] else (clip, )
    assert work_clip.format

    is_HD = clip.width >= 1280 or clip.height >= 720

    # Parameters for deringing kernel
    sigma2 = fallback(sigma2, sigma / 16)
    sbsize = fallback(sbsize, 8 if is_HD else 6)
    sosize = fallback(sosize, 6 if is_HD else 4)

    darkthr = fallback(darkthr, thr // 4)

    rep_dr = [drrep if i in planes else 0 for i in range(work_clip.format.num_planes)]

    # Kernel: Smoothing
    if not isinstance(smooth, vs.VideoNode):
        smoothy, smoothc = cast(tuple[Prefilter, Prefilter], normalize_seq(smooth, 2))

        def _get_kwargs(pref: Prefilter) -> dict[str, Any]:
            if pref != Prefilter.DFTTEST:
                return kwargs
            return kwargs | dict(
                sbsize=sbsize, sosize=sosize, tbsize=1, slocation=[
                    0.0, sigma2, 0.05, sigma, 0.5, sigma, 0.75, sigma2, 1.0, 0.0
                ])

        if smoothy == smoothc or work_clip.format.num_planes == 1:
            smoothed = smoothy(work_clip, planes, **_get_kwargs(smoothy))
        else:
            smoothed = core.std.ShufflePlanes([
                smoothy(work_clip, 0, **_get_kwargs(smoothy)),
                smoothc(work_clip, list(set(planes) - {0}), **_get_kwargs(smoothc))
            ], [0, 1, 2], vs.YUV)
    else:
        check_ref_clip(clip, smooth)  # type: ignore

        smoothed = get_y(smooth) if planes == [0] else smooth  # type: ignore

    # Post-Process: Contra-Sharpening
    if contra:
        if isinstance(contra, int):
            smoothed = contrasharpening(smoothed, work_clip, contra, 13, planes)
        else:
            smoothed = contrasharpening_dehalo(smoothed, work_clip, contra, planes=planes)

    # Post-Process: Repairing
    if set(rep_dr) != {0}:
        repclp = repair(work_clip, smoothed, drrep)
    else:
        repclp = work_clip

    # Post-Process: Limiting
    limitclp = limit_filter(
        repclp, work_clip, None, LimitFilterMode.CLAMPING, planes, thr, elast, darkthr
    )

    if ringmask is None:
        # FIXME: <= instead of < for lthr, Vardë?
        prewittm = Prewitt.edgemask(work_clip, scale_value(mthr, 8, bits) + 1)

        fmask = prewittm.std.Median(planes).misc.Hysteresis(prewittm, planes)  # type: ignore

        omask = Morpho.expand(fmask, mrad, mrad, planes=planes) if mrad > 0 else fmask

        if msmooth > 0:
            omask = iterate(omask, partial(core.std.Inflate, planes=planes), msmooth)

        if incedge:
            ringmask = omask
        else:
            if minp <= 0:
                imask = fmask
            elif minp % 2 == 0:
                imask = Morpho.inpand(fmask, minp // 2, planes=planes)
            else:
                imask = Morpho.inpand(fmask.std.Inflate(planes), ceil(minp / 2), planes=planes)

            ringmask = core.std.Expr([omask, imask], f'x {peak} y - * {peak} /')

    dering = work_clip.std.MaskedMerge(limitclp, ringmask, planes)

    if show:
        return ringmask

    if chroma:
        return join([dering, *chroma], clip.format.color_family)

    return dering
