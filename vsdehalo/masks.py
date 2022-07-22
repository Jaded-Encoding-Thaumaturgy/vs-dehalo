from typing import Any

import vapoursynth as vs
from vsmask.edge import TriticalTCanny
from vsrgtools.util import PlanesT, iterate, norm_expr_planes, wmean_matrix
from vsutil import disallow_variable_format, disallow_variable_resolution

__all__ = [
    'TritSigmaTCanny',
    # morpho functions
    'dilation', 'erosion', 'closing', 'opening', 'gradient', 'top_hat', 'black_hat'
]

core = vs.core


class TritSigmaTCanny(TriticalTCanny):
    sigma: float = 0

    def __init__(self, sigma: float = 0) -> None:
        super().__init__()
        self.sigma = sigma

    def _compute_edge_mask(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.tcanny.TCanny(self.sigma, mode=1, op=0)


@disallow_variable_format
@disallow_variable_resolution
def dilation(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.dilation: radius has to be greater than 0!')

    return iterate(src, core.std.Maximum, radius, planes, **kwargs)


@disallow_variable_format
@disallow_variable_resolution
def erosion(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.erosion: radius has to be greater than 0!')

    return iterate(src, core.std.Minimum, radius, planes, **kwargs)


@disallow_variable_format
@disallow_variable_resolution
def closing(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.closing: radius has to be greater than 0!')

    dilated = dilation(src, radius, planes, **kwargs)
    eroded = erosion(dilated, radius, planes, **kwargs)

    return eroded


@disallow_variable_format
@disallow_variable_resolution
def opening(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.opening: radius has to be greater than 0!')

    eroded = erosion(src, radius, planes, **kwargs)
    dilated = dilation(eroded, radius, planes)

    return dilated


@disallow_variable_format
@disallow_variable_resolution
def gradient(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.gradient: radius has to be greater than 0!')

    eroded = erosion(src, radius, planes)
    dilated = dilation(src, radius, planes, **kwargs)

    return core.std.Expr([dilated, eroded], norm_expr_planes(src, 'x y -', planes))


@disallow_variable_format
@disallow_variable_resolution
def top_hat(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.top_hat: radius has to be greater than 0!')

    opened = opening(src, radius, planes, **kwargs)

    return core.std.Expr([src, opened], norm_expr_planes(src, 'x y -', planes))


@disallow_variable_format
@disallow_variable_resolution
def black_hat(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.black_hat: radius has to be greater than 0!')

    closed = closing(src, radius, planes, **kwargs)

    return core.std.Expr([closed, src], norm_expr_planes(src, 'x y -', planes))


@disallow_variable_format
@disallow_variable_resolution
def outer_hat(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.outer_hat: radius has to be greater than 0!')

    dilated = dilation(src, radius, planes, **kwargs)

    return core.std.Expr([dilated, src], norm_expr_planes(src, 'x y -', planes))


@disallow_variable_format
@disallow_variable_resolution
def inner_hat(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.inner_hat: radius has to be greater than 0!')

    eroded = erosion(src, radius, planes, **kwargs)

    return core.std.Expr([eroded, src], norm_expr_planes(src, 'x y -', planes))


@disallow_variable_format
@disallow_variable_resolution
def grow_mask(
    mask: vs.VideoNode, radius: int = 1, multiply: float = 1.0,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode:
    closed = closing(mask, **kwargs)
    dilated = dilation(closed, **kwargs)
    outer = outer_hat(dilated, radius, **kwargs)

    blurred = outer.std.Convolution(wmean_matrix, planes=planes)

    if multiply != 1.0:
        return blurred.std.Expr(f'x {multiply} *')

    return blurred
