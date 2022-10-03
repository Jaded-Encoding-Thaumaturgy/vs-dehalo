from typing import Any

from vsexprtools import norm_expr
from vsmask.edge import TriticalTCanny
from vsrgtools.util import wmean_matrix
from vstools import PlanesT, core, disallow_variable_format, disallow_variable_resolution, iterate, vs

__all__ = [
    # Masking kernels
    'TritSigmaTCanny',
    # Morpho functions
    'dilation', 'erosion', 'closing', 'opening', 'gradient',
    'top_hat', 'black_hat', 'outer_hat', 'inner_hat',
    # General functions
    'grow_mask'
]


class TritSigmaTCanny(TriticalTCanny):
    sigma: float = 0.0

    def __init__(self, sigma: float = 0.0) -> None:
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

    return norm_expr([dilated, eroded], 'x y -', planes)


@disallow_variable_format
@disallow_variable_resolution
def top_hat(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.top_hat: radius has to be greater than 0!')

    opened = opening(src, radius, planes, **kwargs)

    return norm_expr([src, opened], 'x y -', planes)


@disallow_variable_format
@disallow_variable_resolution
def black_hat(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.black_hat: radius has to be greater than 0!')

    closed = closing(src, radius, planes, **kwargs)

    return norm_expr([closed, src], 'x y -', planes)


@disallow_variable_format
@disallow_variable_resolution
def outer_hat(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.outer_hat: radius has to be greater than 0!')

    dilated = dilation(src, radius, planes, **kwargs)

    return norm_expr([dilated, src], 'x y -', planes)


@disallow_variable_format
@disallow_variable_resolution
def inner_hat(src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, **kwargs: Any) -> vs.VideoNode:
    if radius < 1:
        raise RuntimeError('mask.inner_hat: radius has to be greater than 0!')

    eroded = erosion(src, radius, planes, **kwargs)

    return norm_expr([eroded, src], 'x y -', planes)


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
