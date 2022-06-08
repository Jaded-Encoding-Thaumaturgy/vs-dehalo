from __future__ import annotations

from typing import Any, Dict, Literal, Sequence

import vapoursynth as vs
from vsdenoise import BM3D, BM3DCPU, BM3DCuda, BM3DCudaRTC
from vsrgtools.util import norm_expr_planes, normalise_planes
from vsutil import depth, disallow_variable_format, fallback, get_depth

core = vs.core


@disallow_variable_format
def bidehalo(
    clip: vs.VideoNode,
    sigma: float = 1.5, radius: float = 7,
    sigma_final: float | None = None,
    radius_final: float | None = None,
    tr: int = 2, cuda: bool | Literal['rtc'] = False,
    planes: int | Sequence[int] | None = None,
    matrix: vs.MatrixCoefficients | int | None = None,
    bm3d_args: Dict[str, Any] = {},
    bilateral_args: Dict[str, Any] = {}
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

    sigma_final = fallback(sigma_final, sigma / 3)
    radius_final = fallback(radius_final, radius)

    planes = normalise_planes(clip, planes)

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
        bil_gpu_args: Dict[str, Any] = dict(sigma_spatial=sigma, **bilateral_args)

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
