from __future__ import annotations

from typing import Any, Dict, Literal, Sequence, Type

import vapoursynth as vs
from vsutil import depth, disallow_variable_format, fallback, get_depth

core = vs.core


@disallow_variable_format
def bidehalo(clip: vs.VideoNode, ref: vs.VideoNode | None = None,
             sigma: float = 1.5, sigma_final: float | None = None,
             radius: float = 5 / 255, radius_final: float | None = None,
             cuda: bool | Literal['rtc'] = False,
             planes: int | Sequence[int] | None = None,
             bm3d_args: Dict[str, Any] = {},
             bilateral_args: Dict[str, Any] = {},
             ) -> vs.VideoNode:
    """
    Simple dehalo function that uses ``bilateral`` and ``BM3D`` to remove bright haloing around edges.

    This works by utilising the ``ref`` parameter in ``bilateral`` to limit the areas that get damaged,
    and how much it gets damaged. You should use this function in conjunction with a halo mask.

    If a ref clip is passed, that will be used as a ref for the second bilateral pass instead of a blurred clip.
    Both clips will be resampled to 16bit internally, and returned in the input bitdepth.

    Recommend values for `sigma` are between 0.8 and 2.0.
    Recommend values for `radius` are between 5 / 255 and 15 / 255.

    Dependencies:

    * VapourSynth-Bilateral (Default)
    * VapourSynth-BilateralGPU (Cuda)
    * VapourSynth-BilateralGPU_RTC (RTC)
    * vsdenoise

    :param clip:                Clip to process.
    :param ref:                 Reference clip.
    :param sigma:               ``Bilateral`` spatial weight sigma.
    :param sigma_final:         Final ``Bilateral`` call's spatial weight sigma.
                                You'll want this to be much weaker than the initial `sigma`.
                                If `None`, 1/3rd of `sigma`.
    :param radius:              ``Bilateral`` radius weight sigma.
    :param radius_final:        Final ``Bilateral`` radius weight sigma.
                                if `None`, same as `radius`.
    :param cuda:                Use ``BM3DCUDA`` and `BilateralGPU` if True, else ``BM3DCPU`` and `Bilateral`.
                                Also accepts 'rtc' for ``BM3DRTC`` and `BilateralGPU_RTC`.
    :param planes:              Specifies which planes will be processed.
                                Any unprocessed planes will be simply copied.
    :param bm3d_args:           Additional parameters to pass to BM3D.
    :param bilateral_args:      Additional parameters to pass to Bilateral.

    :return:                    Dehalo'd clip using ``BM3D`` and ``Bilateral``.
    """
    try:
        from vsdenoise import BM3DCPU, BM3DCuda, BM3DCudaRTC
    except ModuleNotFoundError:
        raise ModuleNotFoundError("bidehalo: 'Missing dependency `vsdenoise`!'")

    assert clip.format

    if planes is None:
        planes = list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    bits = get_depth(clip)

    sigma_final = fallback(sigma_final, sigma / 3)
    radius_final = fallback(radius_final, radius)

    bm3d_in_args: Dict[str, Any] = dict(radius=2, planes=planes)
    bm3d_func: Type[BM3DCPU] | Type[BM3DCuda] | Type[BM3DCudaRTC]
    # bilateral_func: [WhatTypeGoesHere?]

    if cuda is False:
        bm3d_func = BM3DCPU
        bilateral_func = core.bilateral.Bilateral
        bm3d_in_args |= dict(sigma=10)
    elif cuda is True:
        bm3d_func = BM3DCuda
        bilateral_func = core.bilateralgpu.Bilateral  # type:ignore[assignment]
        bm3d_in_args |= dict(sigma=8)
    elif cuda == 'rtc':
        bm3d_func = BM3DCudaRTC
        bilateral_func = core.bilateralgpu_rtc.Bilateral  # type:ignore[assignment]
        bm3d_in_args |= dict(sigma=8)
    else:
        raise ValueError(f"bidehalo: 'Invalid cuda selection ({cuda})!'")

    # Override with user's input settings
    bm3d_in_args |= bm3d_args

    if ref is None:
        den = depth(bm3d_func(clip, **bm3d_in_args).clip, 16)

        ref = bilateral_func(den, sigmaS=sigma, sigmaR=radius, planes=planes, **bilateral_args)
    else:
        ref = depth(ref, 16)

    bidh = bilateral_func(clip, ref=ref, sigmaS=sigma_final, sigmaR=radius_final, planes=planes, **bilateral_args)
    bidh = depth(bidh, bits)

    return core.std.Expr([clip, bidh], "x y min")
